"""
Learned expected-return forecaster — an OPTIONAL replacement for
PyPortfolioOpt's historical-mean estimator, which is what `optimizer.py`
uses by default via `expected_returns.mean_historical_return(prices)`.

Deliberately decoupled from optimizer.py and backtest.py: this module is
validated standalone first — information coefficient and hit rate against
the naive historical-mean baseline, on a held-out validation split — before
it is ever wired into a live rebalancing decision. See the EDA notebook's
"ML Forecaster Validation Gate" section for that check. Until that gate
passes for a given configuration, the engine keeps using
`mean_historical_return` and this module is inert — nothing here is called
anywhere else in the codebase yet.

Design choices, and why:
  * Pooled across tickers, not one model per ticker. With only 5 assets,
    5 separate models each see ~2,700 training rows; pooled with `ticker`
    as a feature, one model sees ~13,500 rows — more to regularize against.
  * Ridge is the default, not LightGBM. Weekly-horizon equity return
    prediction has a very low signal-to-noise ratio; a regularized linear
    model is harder to overfit than a gradient-boosted tree at this scale,
    and should be the real baseline candidate, not a strawman LightGBM beats.
  * Walk-forward only. No k-fold CV — financial time series leaks badly
    under random splits. `walk_forward_oos_predictions` retrains
    periodically on an expanding window and only ever predicts dates
    strictly after the data it was trained on.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature columns this module expects, using the same naming convention as
# the EDA notebook's §6 `features` frame: "{ticker}_{feature}". Add
# "sentiment" here once the FinBERT module is validated — no other change
# needed elsewhere in this file to pick it up.
FEATURE_COLUMNS: List[str] = ["vol21", "vol63", "mom12m", "drawdown", "beta_spy"]


@dataclass
class ReturnForecaster:
    """Predicts next-`horizon_days` return, pooled across tickers."""

    model_type: Literal["ridge", "lightgbm"] = "ridge"
    horizon_days: int = 5          # weekly horizon, matches the engine's calendar cadence
    ridge_alpha: float = 1.0
    lgbm_params: dict = field(default_factory=lambda: dict(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    ))

    # Annualizing a short-horizon (e.g. 5-day) point prediction by scaling it up
    # ~50x amplifies its noise enormously — a perfectly reasonable 2-3% raw
    # 5-day prediction becomes a nonsensical 100-280% "annualized" number.
    # This clip is a safety rail, not a tuning knob: if this module is ever
    # wired into optimizer.py, mu_ml must ALSO be shrunk/blended toward the
    # historical mean first (same discipline as the sentiment tilt cap) —
    # clipping here only prevents an absurd value from reaching that point,
    # it does not make the raw annualized prediction trustworthy.
    annualized_clip: float = 1.00  # +/-100%, deliberately generous — see note above

    def __post_init__(self):
        self._model = None
        self._scaler = None
        self._tickers: List[str] = []
        self._X_columns: Optional[pd.Index] = None
        self._n_clipped_last_call = 0

    # -- data assembly --------------------------------------------------
    def _build_training_frame(
        self, features: pd.DataFrame, prices: pd.DataFrame, tickers: List[str]
    ) -> pd.DataFrame:
        """
        features: wide frame indexed by date, columns "{ticker}_{feature}".
        prices:   wide adjusted-close frame indexed by date, one column per ticker.
        Returns a long (date, ticker) frame: FEATURE_COLUMNS + "ticker" + "fwd_return".
        fwd_return is the realized simple return over the NEXT horizon_days —
        i.e. a row's label is always strictly in the future relative to its features.
        """
        fwd_return = prices[tickers].shift(-self.horizon_days) / prices[tickers] - 1.0

        rows = []
        for t in tickers:
            sub = pd.DataFrame(index=features.index)
            for feat in FEATURE_COLUMNS:
                col = f"{t}_{feat}"
                sub[feat] = features[col] if col in features.columns else np.nan
            sub["ticker"] = t
            sub["fwd_return"] = fwd_return[t]
            rows.append(sub)
        return pd.concat(rows).dropna()

    # -- fit / predict ----------------------------------------------------
    def fit(self, features: pd.DataFrame, prices: pd.DataFrame, tickers: List[str]) -> "ReturnForecaster":
        self._tickers = list(tickers)
        long_df = self._build_training_frame(features, prices, tickers)
        if long_df.empty:
            raise ValueError("No training rows after feature/target alignment — check feature coverage.")

        X = pd.get_dummies(long_df[FEATURE_COLUMNS + ["ticker"]], columns=["ticker"], prefix="tk")
        y = long_df["fwd_return"].values
        self._X_columns = X.columns

        if self.model_type == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler().fit(X.values)
            self._model = Ridge(alpha=self.ridge_alpha).fit(self._scaler.transform(X.values), y)
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            self._scaler = None
            self._model = lgb.LGBMRegressor(**self.lgbm_params, verbosity=-1).fit(X.values, y)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        logger.info(
            "Fitted %s forecaster on %d pooled rows (%d tickers, horizon=%dd).",
            self.model_type, len(long_df), len(tickers), self.horizon_days,
        )
        return self

    def predict(self, features_row: pd.Series, tickers: Optional[List[str]] = None) -> pd.Series:
        """
        features_row: a single row (e.g. `features.loc[date]`) from the same wide
                      feature frame used in fit().
        Returns ANNUALIZED predicted return per ticker, unit-compatible with
        PyPortfolioOpt's `mean_historical_return` output — this is what lets
        the result drop straight into `optimizer.py`'s `mu` input later.
        """
        if self._model is None:
            raise RuntimeError("Call .fit() before .predict().")
        tickers = tickers or self._tickers
        rows = [{**{feat: features_row.get(f"{t}_{feat}", np.nan) for feat in FEATURE_COLUMNS}, "ticker": t}
                for t in tickers]
        X = pd.get_dummies(pd.DataFrame(rows), columns=["ticker"], prefix="tk")
        X = X.reindex(columns=self._X_columns, fill_value=0)

        if self.model_type == "ridge":
            preds_period = self._model.predict(self._scaler.transform(X.values))
        else:
            preds_period = self._model.predict(X.values)

        annualized = preds_period * (252 / self.horizon_days)
        clipped = np.clip(annualized, -self.annualized_clip, self.annualized_clip)
        # Tracked, not logged per-call — this fires often enough (see the gate
        # notebook section) that per-call warnings would just be noise. Callers
        # that care (walk_forward_oos_predictions) aggregate this into a clip
        # rate reported once, in the summary table.
        self._n_clipped_last_call = int(np.sum(clipped != annualized))
        return pd.Series(clipped, index=tickers, name="mu_ml")


# =========================================================================
# Baseline + evaluation utilities — used by the validation-gate notebook
# section, not by the live engine.
# =========================================================================

def naive_historical_mu(prices_window: pd.DataFrame, tickers: List[str]) -> pd.Series:
    """The exact estimator optimizer.py uses today — the baseline this module must beat."""
    from pypfopt import expected_returns
    return expected_returns.mean_historical_return(prices_window[tickers])


def information_coefficient(pred: pd.Series, realized: pd.Series) -> float:
    """Spearman rank correlation between predicted and realized return. The standard
    quant diagnostic for 'does this forecaster have any skill at all' — near 0 means no."""
    paired = pd.concat([pred, realized], axis=1).dropna()
    if len(paired) < 3:
        return float("nan")
    return paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman")


def hit_rate(pred: pd.Series, realized: pd.Series) -> float:
    """Sign-agreement rate between predicted and realized return. Compare against 0.5
    (coin flip) — this is the plainest-language version of 'is it even directionally right.'"""
    paired = pd.concat([pred, realized], axis=1).dropna()
    if paired.empty:
        return float("nan")
    return float((np.sign(paired.iloc[:, 0]) == np.sign(paired.iloc[:, 1])).mean())


def walk_forward_oos_predictions(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: List[str],
    train_start,
    oos_start,
    oos_end,
    model_type: str = "ridge",
    horizon_days: int = 5,
    retrain_every_days: int = 21,     # ~monthly retrain; the model doesn't need daily refitting
    min_train_days: int = 252,
    **model_kwargs,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation. Retrains every `retrain_every_days`
    on all data strictly before the current date, predicts that date, slides on.
    No date is ever predicted using a model fit on data from that date or later —
    this is the check against lookahead leakage, not just a comment.

    Also computes the naive historical-mean baseline on the SAME dates, so the
    comparison in the notebook is apples-to-apples rather than the ML model being
    judged on a different (easier or harder) set of periods than the baseline.

    **model_kwargs: forwarded to ReturnForecaster (e.g. ridge_alpha=3.0, or
    lgbm_params={...}) — this is what a hyperparameter search varies, scored on
    the validation split only, same walk-forward discipline as everywhere else.

    Returns a long (date, ticker) frame: mu_ml, mu_naive, realized_fwd_ann.
    """
    oos_dates = features.index[(features.index >= oos_start) & (features.index < oos_end)]
    records = []
    model: Optional[ReturnForecaster] = None
    last_train_date = None
    realized_fwd_ann = (prices[tickers].shift(-horizon_days) / prices[tickers] - 1.0) * (252 / horizon_days)

    for date in oos_dates:
        if last_train_date is None or (date - last_train_date).days >= retrain_every_days:
            train_feat = features.loc[(features.index >= train_start) & (features.index < date)]
            train_px = prices.loc[(prices.index >= train_start) & (prices.index < date)]
            if len(train_feat) < min_train_days:
                continue
            model = ReturnForecaster(model_type=model_type, horizon_days=horizon_days, **model_kwargs).fit(train_feat, train_px, tickers)
            last_train_date = date

        if model is None:
            continue

        naive_window = prices.loc[(prices.index >= train_start) & (prices.index < date), tickers]
        mu_naive = naive_historical_mu(naive_window, tickers) if len(naive_window) >= min_train_days else pd.Series(np.nan, index=tickers)
        mu_ml = model.predict(features.loc[date], tickers)
        n_clipped_this_date = model._n_clipped_last_call

        for t in tickers:
            records.append({
                "date": date, "ticker": t,
                "mu_ml": mu_ml[t], "mu_naive": mu_naive[t],
                "realized_fwd_ann": realized_fwd_ann.loc[date, t],
                "clipped_this_date": n_clipped_this_date > 0,
            })

    df = pd.DataFrame(records).dropna(subset=["mu_ml", "mu_naive", "realized_fwd_ann"])
    if len(df):
        clip_rate = df.drop_duplicates("date")["clipped_this_date"].mean()
        logger.info(
            "Walk-forward complete: %d (date, ticker) rows. %.0f%% of rebalance dates had at "
            "least one ticker's annualized mu_ml clipped at +/-%.0f%% — a high rate here is "
            "itself evidence the raw forecast needs shrinkage before ever reaching an optimizer.",
            len(df), clip_rate * 100, model.annualized_clip * 100,
        )
    return df


def evaluate_walk_forward(oos_df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes a walk_forward_oos_predictions() result into the gate's scorecard:
    IC and hit rate for the ML forecaster vs. the naive baseline, on identical dates."""
    rows = []
    for label, col in [("ML forecaster", "mu_ml"), ("Naive historical mean", "mu_naive")]:
        ic = information_coefficient(oos_df[col], oos_df["realized_fwd_ann"])
        hr = hit_rate(oos_df[col], oos_df["realized_fwd_ann"])
        rows.append({"Model": label, "Information coefficient": ic, "Hit rate": hr, "N (date, ticker) obs": len(oos_df)})
    return pd.DataFrame(rows)
