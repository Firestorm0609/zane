"""ScoringEngine: combines formula + ML."""
import json
import logging
import math
import os
from contextlib import closing
from typing import Callable, Optional

from .config import (
    BUY_THRESHOLD_DEFAULT, MAX_ML_WEIGHT, MIN_TRAIN_SAMPLES, ML_AVAILABLE,
    ML_LABEL_WINDOW, MODEL_PATH, MODEL_VERSION, PUMP_THRESHOLD_PCT,
    SCALER_PATH, WATCH_THRESHOLD_DEFAULT,
)
from .db import db_conn, get_state, set_state
from .features import CoinContext, FeatureRegistry
from .keywords import KeywordModel
from .market import MarketContext
from .utils import now_ts, safe_float, safe_int

log = logging.getLogger(__name__)


def _make_normalizer(name: str) -> Callable[[float], float]:
    if "percentile" in name:
        return lambda f: max(0.0, min(1.0, f))
    if "zscore" in name:
        return lambda f: max(0.0, min(1.0, (f + 4.0) / 8.0))
    if "log" in name:
        return lambda f: max(0.0, min(1.0, f / 15.0))
    if name in {"has_twitter", "has_telegram", "has_website",
                "name_is_allcaps", "name_has_numbers", "desc_has_url"}:
        return lambda f: max(0.0, min(1.0, f))
    if "progress" in name or "curve" in name:
        return lambda f: max(0.0, min(1.0, f))
    if "sin" in name or "cos" in name:
        return lambda f: max(0.0, min(1.0, (f + 1.0) / 2.0))
    if "ratio" in name or "density" in name or "completeness" in name:
        return lambda f: max(0.0, min(1.0, abs(f)))
    if "safety" in name or "clean" in name or "distribution" in name or "freshness" in name:
        return lambda f: max(0.0, min(1.0, f))
    if "count" in name:
        return lambda f: max(0.0, min(1.0, f / 5.0))
    return lambda f: max(0.0, min(1.0, abs(f) / 10.0))


class ScoringEngine:
    FORMULA_WEIGHTS: dict[str, float] = {
        "replies_per_kmc":        3.0,
        "reply_velocity":         3.0,
        "social_completeness":    2.5,
        "mc_momentum_pct":        2.5,
        "bonding_curve_progress": 2.0,
        "keyword_pump_score":     2.0,
        "reply_percentile":       1.5,
        "mc_percentile":          1.5,
        "desc_unique_word_ratio": 1.2,
        "coin_age_capped":        0.5,
        "mint_auth_safety":       2.0,
        "freeze_auth_safety":     1.5,
        "holder_distribution":    2.5,
        "bundle_clean":           3.0,
        "creator_freshness":      1.2,
        "creator_blacklisted":    4.0,
    }

    # Drift detection threshold (AUC drop >= this triggers warning)
    DRIFT_THRESHOLD = 0.05

    def __init__(self, features: FeatureRegistry,
                 keyword_model: KeywordModel,
                 market_ctx: MarketContext):
        self.features      = features
        self.keyword_model = keyword_model
        self.market_ctx    = market_ctx

        self._norm_dispatch = [_make_normalizer(n) for n in features.names]
        self._feature_idx   = {n: i for i, n in enumerate(features.names)}

        self._model = None
        self._scaler = None
        self._importances: dict[str, float] = {}
        self._n_train_samples = 0
        self._cv_auc = 0.0
        self._cv_auc_std = 0.0
        self._pump_rate = 0.5
        self._trained_at = 0
        self._buy_threshold = BUY_THRESHOLD_DEFAULT
        self._watch_threshold = WATCH_THRESHOLD_DEFAULT
        self._model_version = ""

    # ---------- public ----------

    def score(self, coin: dict) -> dict:
        ctx     = CoinContext(coin=coin, market_ctx=self.market_ctx,
                              keyword_model=self.keyword_model)
        fvec    = self.features.extract(ctx)
        f_prob  = self._formula_prob(fvec)
        ml_prob = self._ml_prob(fvec)
        w       = self._ml_weight()

        if ml_prob is not None and w > 0.0:
            blended = f_prob * (1.0 - w) + ml_prob * w
            mode    = f"ML+formula ({w:.0%} ML)"
        else:
            blended = f_prob
            ml_prob = None
            mode    = "formula"

        score_10 = max(1, min(10, round(blended * 10)))

        if blended >= self._buy_threshold:     rec = "BUY"
        elif blended >= self._watch_threshold: rec = "WATCH"
        else:                                  rec = "PASS"

        red_flags = self._red_flags(ctx, fvec)
        mc   = safe_float(coin.get("usd_market_cap"))
        reps = safe_int(coin.get("reply_count"))

        return {
            "score":               score_10,
            "probability":         round(blended, 4),
            "ml_probability":      round(ml_prob, 4) if ml_prob is not None else None,
            "ml_cv_auc_std":       round(self._cv_auc_std, 4),
            "formula_probability": round(f_prob, 4),
            "ml_weight":           round(w, 3),
            "mode":                mode,
            "recommendation":      rec,
            "red_flags":           red_flags,
            "feature_vector":      fvec,
            "summary": (
                f"Score {score_10}/10 ({blended:.1%}) | "
                f"MC ${mc:,.0f} | replies {reps} | mode: {mode}"
            ),
        }

    def train(self) -> bool:
        if not ML_AVAILABLE:
            log.warning("sklearn not installed — skipping training")
            return False

        # Learn keywords first (cheap)
        self.keyword_model.learn_from_db()

        with closing(db_conn()) as conn:
            rows = conn.execute("""
                SELECT s.feature_vector, lb.pct_change
                FROM signals s
                JOIN lookbacks lb ON lb.signal_id = s.id
                WHERE lb.window_label = ?
                  AND lb.checked = 1
                  AND lb.pct_change IS NOT NULL
                  AND s.feature_vector IS NOT NULL
            """, (ML_LABEL_WINDOW,)).fetchall()

        if len(rows) < MIN_TRAIN_SAMPLES:
            log.info("Need %d labeled samples, have %d",
                     MIN_TRAIN_SAMPLES, len(rows))
            return False

        # Lazy imports — only loaded when training is actually attempted
        import numpy as np
        import joblib
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler

        X_list, y_list = [], []
        for row in rows:
            try:
                fvec = json.loads(row["feature_vector"])
                if not isinstance(fvec, list) or len(fvec) == 0:
                    continue
                if len(fvec) < len(self.features):
                    fvec = fvec + [0.0] * (len(self.features) - len(fvec))
                elif len(fvec) > len(self.features):
                    fvec = fvec[:len(self.features)]
                fvec = [float(v) if (v is not None and math.isfinite(float(v))) else 0.0
                        for v in fvec]
                X_list.append(fvec)
                y_list.append(1 if float(row["pct_change"]) >= PUMP_THRESHOLD_PCT else 0)
            except Exception:
                continue

        n = len(X_list)
        if n < MIN_TRAIN_SAMPLES:
            log.info("After filtering, only %d usable samples (need %d)",
                     n, MIN_TRAIN_SAMPLES)
            return False

        X = np.array(X_list)
        y = np.array(y_list)

        if len(np.unique(y)) < 2:
            log.warning("Only one class in training data — skipping")
            return False

        pump_rate = float(y.mean())
        log.info("Training on %d samples | pump rate %.1f%% | label window: %s",
                 n, pump_rate * 100, ML_LABEL_WINDOW)

        n_pos = int(y.sum())
        n_neg = n - n_pos
        scale_pos = n_neg / max(n_pos, 1)
        sample_weight = np.where(y == 1, scale_pos, 1.0)
        log.info("Class imbalance correction: scale_pos_weight=%.2f (pos=%d neg=%d)",
                 scale_pos, n_pos, n_neg)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        base_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42,
        )
        model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)

        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_aucs = cross_val_score(model, Xs, y, cv=cv, scoring="roc_auc")
            cv_mean = float(cv_aucs.mean())
            cv_std  = float(cv_aucs.std())
            log.info("CV ROC-AUC: %.3f ± %.3f", cv_mean, cv_std)
        except Exception as e:
            log.warning("CV failed (%s) — training without CV", e)
            cv_mean = 0.5
            cv_std  = 0.0

        model.fit(Xs, y, sample_weight=sample_weight)

        importances: dict[str, float] = {}
        try:
            cal  = model.calibrated_classifiers_[0]
            base = (getattr(cal, "estimator", None)
                    or getattr(cal, "base_estimator", None))
            if base is not None and hasattr(base, "feature_importances_"):
                importances = dict(zip(self.features.names,
                                       base.feature_importances_.tolist()))
        except Exception as e:
            log.warning("Could not extract feature importances: %s", e)

        buy_th   = min(0.85, max(0.5,  pump_rate * 2.0))
        watch_th = min(0.65, max(0.35, pump_rate * 1.3))

        try:
            joblib.dump(model,  MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
        except Exception as e:
            log.error("Failed to persist model: %s", e)

        # ===== DRIFT DETECTION =====
        # IMPORTANT: read previous AUC BEFORE writing the new one.
        # Otherwise we'd be comparing the just-saved value to itself.
        prev_auc = float(get_state("model_cv_auc", "0"))
        drift_detected = (
            prev_auc > 0
            and (prev_auc - cv_mean) >= self.DRIFT_THRESHOLD
        )
        if drift_detected:
            delta = prev_auc - cv_mean
            log.warning(
                "MODEL DRIFT DETECTED: AUC dropped from %.3f → %.3f "
                "(Δ=%.3f, threshold=%.3f) — investigate data quality",
                prev_auc, cv_mean, delta, self.DRIFT_THRESHOLD,
            )
            set_state("model_drift_detected", "1")
            set_state("model_drift_delta", f"{delta:.4f}")
            set_state("model_drift_prev_auc", f"{prev_auc:.4f}")
            set_state("model_drift_at", str(now_ts()))
        else:
            set_state("model_drift_detected", "0")
            set_state("model_drift_delta", "0")
        # ===== END DRIFT DETECTION =====

        # Now safe to overwrite stored AUC
        self._model = model
        self._scaler = scaler
        self._importances = importances
        self._n_train_samples = n
        self._cv_auc = cv_mean
        self._cv_auc_std = cv_std
        self._pump_rate = pump_rate
        self._trained_at = now_ts()
        self._buy_threshold = buy_th
        self._watch_threshold = watch_th
        self._model_version = MODEL_VERSION

        set_state("model_trained_at",      str(self._trained_at))
        set_state("model_samples",         str(n))
        set_state("model_cv_auc",          f"{cv_mean:.4f}")
        set_state("model_cv_auc_std",      f"{cv_std:.4f}")
        set_state("model_pump_rate",       f"{pump_rate:.4f}")
        set_state("model_importances",     json.dumps(importances))
        set_state("model_buy_threshold",   f"{buy_th:.4f}")
        set_state("model_watch_threshold", f"{watch_th:.4f}")
        set_state("model_feature_count",   str(len(self.features)))
        set_state("model_version",         MODEL_VERSION)

        log.info("Training complete | CV AUC=%.3f±%.3f | BUY≥%.2f | WATCH≥%.2f",
                 cv_mean, cv_std, buy_th, watch_th)
        return True

    def load(self) -> None:
        if not ML_AVAILABLE:
            return
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
            log.info("No ML model files found — formula mode until data accumulates")
            return
        try:
            import joblib
            self._model  = joblib.load(MODEL_PATH)
            self._scaler = joblib.load(SCALER_PATH)

            saved_count = int(get_state("model_feature_count", "0"))
            if saved_count and saved_count != len(self.features):
                log.warning(
                    "Saved model has %d features, registry has %d "
                    "— disabled until retrain",
                    saved_count, len(self.features),
                )
                self._model = self._scaler = None
                return

            self._n_train_samples = int(get_state("model_samples", "0"))
            self._cv_auc          = float(get_state("model_cv_auc", "0"))
            self._cv_auc_std      = float(get_state("model_cv_auc_std", "0"))
            self._pump_rate       = float(get_state("model_pump_rate", "0.5"))
            self._buy_threshold   = float(get_state(
                "model_buy_threshold", str(BUY_THRESHOLD_DEFAULT)))
            self._watch_threshold = float(get_state(
                "model_watch_threshold", str(WATCH_THRESHOLD_DEFAULT)))
            imp_json              = get_state("model_importances", "{}")
            self._importances     = json.loads(imp_json) if imp_json else {}
            self._trained_at      = int(get_state("model_trained_at", "0"))
            self._model_version   = get_state("model_version", "")

            log.info(
                "ML model loaded | version=%s samples=%d | "
                "CV AUC=%.3f±%.3f | ML weight=%.0f%%",
                self._model_version or "?", self._n_train_samples,
                self._cv_auc, self._cv_auc_std, self._ml_weight() * 100,
            )
        except Exception as e:
            self._model = self._scaler = None
            log.warning("Failed to load ML model: %s — formula mode", e)

    def status(self) -> dict:
        return {
            "mode":            self.mode_label,
            "n_features":      len(self.features),
            "n_train_samples": self._n_train_samples,
            "min_train":       MIN_TRAIN_SAMPLES,
            "cv_auc":          self._cv_auc,
            "cv_auc_std":      self._cv_auc_std,
            "pump_rate":       self._pump_rate,
            "ml_weight":       self._ml_weight(),
            "buy_threshold":   self._buy_threshold,
            "watch_threshold": self._watch_threshold,
            "trained_at":      self._trained_at,
            "version":         self._model_version,
            "top_features":    sorted(self._importances.items(),
                                      key=lambda x: x[1], reverse=True)[:10],
        }

    @property
    def mode_label(self) -> str:
        if self._model is None:
            return "formula only"
        w = self._ml_weight()
        if w >= MAX_ML_WEIGHT * 0.95: return f"ML dominant ({w:.0%})"
        if w > 0:                     return f"blended ({w:.0%} ML)"
        return "formula only"

    # ---------- private ----------

    def _ml_weight(self) -> float:
        if self._model is None or self._n_train_samples < MIN_TRAIN_SAMPLES:
            return 0.0
        sample_ramp = min(
            1.0,
            (self._n_train_samples - MIN_TRAIN_SAMPLES) / max(MIN_TRAIN_SAMPLES, 1),
        )
        auc_factor = max(0.0, min(1.0, (self._cv_auc - 0.5) / 0.25))
        return sample_ramp * auc_factor * MAX_ML_WEIGHT

    def _formula_prob(self, fvec: list[float]) -> float:
        normed = self._normalize_for_formula(fvec)
        names = self.features.names
        total_w = 0.0
        total_v = 0.0
        for name, val in zip(names, normed):
            w = self.FORMULA_WEIGHTS.get(name, 1.0)
            total_w += w
            total_v += val * w
        if total_w == 0:
            return 0.0
        return max(0.0, min(1.0, total_v / total_w))

    def _normalize_for_formula(self, fvec: list[float]) -> list[float]:
        result = []
        for fn, val in zip(self._norm_dispatch, fvec):
            try:
                f = float(val) if val is not None else 0.0
                result.append(fn(f) if math.isfinite(f) else 0.0)
            except (TypeError, ValueError):
                result.append(0.0)
        return result

    def _ml_prob(self, fvec: list[float]) -> Optional[float]:
        if self._model is None or self._scaler is None or not ML_AVAILABLE:
            return None
        try:
            import numpy as np
            X = np.array([fvec])
            Xs = self._scaler.transform(X)
            return float(self._model.predict_proba(Xs)[0][1])
        except Exception as e:
            log.error("ml_prob: %s", e)
            return None

    def _red_flags(self, ctx: CoinContext, fvec: list[float]) -> list[str]:
        flags = []

        def fv(name: str) -> float:
            idx = self._feature_idx.get(name)
            if idx is None or idx >= len(fvec):
                return 0.0
            return fvec[idx]

        if ctx.replies == 0:                      flags.append("no engagement")
        if fv("mc_percentile") > 0.92:            flags.append("top 8% MC today")
        if fv("social_completeness") < 0.34:      flags.append("weak socials")
        if fv("keyword_negative_hits") > 2:       flags.append("rug keywords detected")
        if fv("desc_unique_word_ratio") < 0.4:    flags.append("low-quality description")
        if fv("replies_per_kmc") < 0.1 and ctx.mc > 20_000:
            flags.append("low engagement for MC")
        if self._cv_auc_std > 0.05:
            flags.append(f"high model uncertainty (±{self._cv_auc_std:.2f} AUC std)")

        if ctx.coin.get("_rpc_mint_auth_revoked") is False:
            flags.append("mint authority NOT revoked")
        if ctx.coin.get("_rpc_freeze_auth_revoked") is False:
            flags.append("freeze authority NOT revoked")

        conc = ctx.coin.get("_rpc_top5_concentration")
        if conc is not None and safe_float(conc) > 0.75:
            flags.append(f"top-5 holders own {safe_float(conc):.0%} supply")

        bundle = ctx.coin.get("_rpc_bundle_score")
        if bundle is not None:
            b = safe_float(bundle)
            if b >= 1.0:
                flags.append("bundled launch detected")
            elif b >= 0.5:
                flags.append("possible bundle activity")

        creator_txs = ctx.coin.get("_rpc_creator_tx_count")
        if creator_txs is not None and safe_int(creator_txs) >= 150:
            flags.append(f"serial launcher wallet ({safe_int(creator_txs)} txs)")

        return flags
