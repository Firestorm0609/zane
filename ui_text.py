"""All text generators for command/callback responses."""
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone

from .config import (
    DB_BACKUP_PATH, DB_PATH, DEAD_LETTER_MAX_RETRIES, LOG_PATH,
    MIN_TRAIN_SAMPLES, ML_LABEL_WINDOW, PAPER_FEE_PCT, PAPER_MAX_CONCURRENT,
    PAPER_POSITION_SIZE_USD, PAPER_SLIPPAGE_PCT, PAPER_STOP_LOSS_PCT,
    PAPER_TAKE_PROFIT_PCT, PAPER_TIME_STOP_SEC, PUMP_FRONT,
    PUMP_THRESHOLD_PCT, RUG_THRESHOLD_PCT, SNAPSHOT_COUNT,
    STREAM_DEAD_ALERT_SEC,
)
from .db import db_conn, get_state
from .market import MarketContext
from .scoring import ScoringEngine
from .state import BotState
from .trading import adaptive_sl_tp, paper_stats
from .utils import (
    REC_EMOJI, fmt_duration, fmt_pct, fmt_prob, fmt_usd,
    mdbold, mdcode, mditalic, now_ts, safe_float, safe_int, score_emoji,
)
from .wallet import PaperWallet, daily_pnl_usd, recent_loss_streak


# ---------- Status ----------

def text_monitor_status(cid: int, state: BotState) -> str:
    if cid in state.alerts:
        return (f"🟢 Alerts {mdbold('ON')} — threshold "
                f"{mdcode(f'{state.alerts[cid]}/10')}")
    return f"🔴 Alerts {mdbold('OFF')}"


# ---------- Scoring ----------

def text_scoring_mode(engine: ScoringEngine) -> str:
    s = engine.status()
    lines = [
        f"⚙️ {mdbold('Scoring Mode')}",
        "",
        f"Mode: {mdcode(s['mode'])}",
        f"Features: {mdcode(s['n_features'])}",
    f"ML weight: {mdcode(str(round(s['ml_weight']*100))+'%')}",
    f"Samples: {mdcode(str(s['n_train_samples'])+'/'+str(s['min_train']))}",
    ]
    if s["cv_auc"]:
        cv_str = f"{s['cv_auc']:.3f}"
        if s["cv_auc_std"]:
            cv_str += f" ±{s['cv_auc_std']:.3f}"
        lines.append(f"CV AUC: {mdcode(cv_str)}")
    if s["pump_rate"]:
        lines.append(f"Pump rate: {mdcode(fmt_prob(s['pump_rate']))}")
    lines += [
        f"BUY ≥ {mdcode(fmt_prob(s['buy_threshold']))}",
        f"WATCH ≥ {mdcode(fmt_prob(s['watch_threshold']))}",
    ]
    return "\n".join(lines)


def text_features(engine: ScoringEngine) -> str:
    s = engine.status()
    lines = [
        f"🧬 {mdbold('Feature Importances')}",
        mditalic(f"{s['n_features']} features total"),
        "",
    ]
    if not s["top_features"]:
        lines.append(mditalic("Model not trained yet — all features carry equal weight"))
    else:
        for i, (name, imp) in enumerate(s["top_features"], 1):
            bar = "█" * max(1, min(30, round(imp * 100)))
            lines.append(
                f"{mdcode(f'{i:2d}. {name:<30}')} "
                f"{mdcode(f'{imp:.3f}')} {bar}"
            )
    return "\n".join(lines)


def text_keywords(engine: ScoringEngine) -> str:
    km = engine.keyword_model
    s = km.status()
    lines = [
        f"🔤 {mdbold('Dynamic Keyword Model')}",
        f"Words learned: {mdcode(s['n_words'])}",
        f"Base pump rate: {mdcode(fmt_prob(s['base_rate']))}",
        f"Training samples: {mdcode(s['n_samples'])}",
        "",
        f"📈 {mdbold('Top pump signals:')}",
    ]
    if not s["top_positive"]:
        lines.append(mditalic("No data yet"))
    else:
        for w, v in s["top_positive"]:
            lines.append(f"{mdcode(w)} lift {mdcode(f'{v:+.2f}')}")
    lines.append(f"\n📉 {mdbold('Top rug signals:')}")
    if not s["top_negative"]:
        lines.append(mditalic("No data yet"))
    else:
        for w, v in s["top_negative"]:
            lines.append(f"{mdcode(w)} lift {mdcode(f'{v:+.2f}')}")
    lines.append("")
    lines.append(mditalic("Updates automatically on every model retrain"))
    return "\n".join(lines)


def text_market(market_ctx: MarketContext) -> str:
    s = market_ctx.summary()
    if s["samples"] == 0:
        return mditalic("No market data yet")
    return "\n".join([
        f"📈 {mdbold('Market Context (24hr window)')}",
        f"Coins seen: {mdcode(s['samples'])}",
        "MC p25 / median / p75:",
        f"{mdcode(fmt_usd(s['mc_p25']))} / "
        f"{mdcode(fmt_usd(s['mc_median']))} / "
        f"{mdcode(fmt_usd(s['mc_p75']))}",
        f"MC mean: {mdcode(fmt_usd(s['mc_mean']))}",
        f"Replies median: {mdcode(str(round(s['replies_median'])))}",
    ])


# ---------- Outcomes ----------

def query_outcomes_data() -> dict:
    with closing(db_conn()) as conn:
        total_sig = conn.execute("SELECT COUNT(*) AS c FROM signals").fetchone()["c"]
        checked = conn.execute(
            "SELECT COUNT(*) AS c FROM lookbacks WHERE window_label=? AND checked=1",
            (ML_LABEL_WINDOW,)
        ).fetchone()["c"]
        pending = conn.execute(
            "SELECT COUNT(*) AS c FROM lookbacks WHERE checked=0"
        ).fetchone()["c"]
        by_out = conn.execute("""
            SELECT outcome, COUNT(*) AS cnt, AVG(pct_change) AS avg_pct
            FROM lookbacks WHERE window_label=? AND checked=1 AND outcome IS NOT NULL
            GROUP BY outcome ORDER BY cnt DESC
        """, (ML_LABEL_WINDOW,)).fetchall()
    return {
        "total_signals": int(total_sig or 0),
        "checked":       int(checked or 0),
        "pending":       int(pending or 0),
        "by_outcome":    [dict(r) for r in by_out],
    }


def text_outcomes() -> str:
    data = query_outcomes_data()
    by_out  = data["by_outcome"]
    checked = data["checked"]
    pump = sum(r["cnt"] for r in by_out if r["outcome"] in ("PUMP", "MOON"))
    rug  = sum(r["cnt"] for r in by_out if r["outcome"] == "RUG")
    lines = [
        f"📤 {mdbold('Outcome Tracker')}",
        f"Signals: {mdcode(data['total_signals'])} \\| "
        f"Labeled \\({ML_LABEL_WINDOW}\\): {mdcode(checked)} \\| "
        f"Pending: {mdcode(data['pending'])}",
        "",
        mdbold(f"{ML_LABEL_WINDOW} breakdown \\(ML training window\\):"),
    ]
    if not by_out:
        lines.append(mditalic("No labeled outcomes yet — check back later"))
    else:
        for r in by_out:
            outcome = r.get("outcome") or "?"
            cnt     = safe_int(r.get("cnt"))
            avg_pct = r.get("avg_pct")
            avg_str = fmt_pct(avg_pct, 1, signed=True) if avg_pct is not None else "—"
            lines.append(
                f"• {mdcode(outcome)}  {mdcode(cnt)} "
                f"\\(avg {mdcode(avg_str)}\\)"
            )
        if checked:
            lines.append(
                f"\nPump rate {mdcode(f'{pump/checked*100:.1f}%')} \\| "
                f"Rug rate {mdcode(f'{rug/checked*100:.1f}%')}"
            )
    lines.append(
        f"\n{mditalic(f'ML label window: {ML_LABEL_WINDOW} | unlocks at {MIN_TRAIN_SAMPLES} | progress: {checked}/{MIN_TRAIN_SAMPLES}')}"
    )
    return "\n".join(lines)


# ---------- Model ----------

def text_model(engine: ScoringEngine) -> str:
    from .config import ML_AVAILABLE
    if not ML_AVAILABLE:
        return (f"⚠️ {mdbold('ML not available')}\n"
                f"{mdcode('pip install scikit-learn joblib numpy')}")

    s = engine.status()
    if s["n_train_samples"] == 0:
        labeled = 0
        with closing(db_conn()) as conn:
            labeled = conn.execute(
                "SELECT COUNT(*) AS c FROM lookbacks "
                "WHERE window_label=? AND checked=1",
                (ML_LABEL_WINDOW,)
            ).fetchone()["c"]
        return (
            f"🤖 {mdbold('Model: not trained')}\n"
            f"Progress: {mdcode(f'{labeled}/{MIN_TRAIN_SAMPLES}')} samples\n"
            f"{mditalic('Retrains automatically daily once threshold reached')}"
        )

    ts = s["trained_at"]
    dt_plain = (datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                if ts else "?")
    cv_str = f"{s['cv_auc']:.3f}"
    if s.get("cv_auc_std"):
        cv_str += f" ±{s['cv_auc_std']:.3f}"

    drift_detected = get_state("model_drift_detected", "0") == "1"
    drift_delta    = get_state("model_drift_delta", "0")
    drift_line = (
        f"⚠️ {mdbold('Drift detected')} ΔAUC {mdcode(drift_delta)}"
        if drift_detected else ""
    )
    lines = [
        f"🤖 {mdbold('Model: ACTIVE')}",
        f"Version: {mdcode(s.get('version') or '?')}",
        f"Mode: {mdcode(s['mode'])}",
        f"Trained: {mdcode(dt_plain)}",
        f"Samples: {mdcode(s['n_train_samples'])}",
        f"CV ROC-AUC: {mdcode(cv_str)}",
        f"Pump rate: {mdcode(fmt_prob(s['pump_rate']))}",
        f"ML weight: {mdcode(str(round(s['ml_weight']*100))+'%')} of final score",
        f"BUY ≥ {mdcode(fmt_prob(s['buy_threshold']))} \\| "
        f"WATCH ≥ {mdcode(fmt_prob(s['watch_threshold']))}",
    ]
    if drift_line:
        lines.append(drift_line)
    return "\n".join(lines)


# ---------- Snapshot ----------

def text_snapshot(n: int = SNAPSHOT_COUNT) -> str:
    with closing(db_conn()) as conn:
        rows = conn.execute("""
            SELECT name, symbol, score, probability, recommendation,
                   market_cap_at_signal, reply_count, red_flags, created_at
            FROM signals
            ORDER BY created_at DESC
            LIMIT ?
        """, (n,)).fetchall()

    if not rows:
        return mditalic("No signals recorded yet.")

    lines = [f"🗒 {mdbold(f'Last {len(rows)} Scored Signals')}", ""]
    for s in rows:
        age = now_ts() - safe_int(s["created_at"])
        if age < 3600:    age_str = f"{age // 60}m ago"
        elif age < 86400: age_str = f"{age // 3600}h ago"
        else:             age_str = f"{age // 86400}d ago"
        rec    = s["recommendation"] or ""
        rec_e  = REC_EMOJI.get(rec, "")
        score  = safe_int(s["score"])
        prob   = safe_float(s["probability"])
        mc     = safe_float(s["market_cap_at_signal"])
        name   = s["name"] or "?"
        symbol = s["symbol"] or "?"
        lines.append(
            f"{score_emoji(score)} {mdbold(name)} \\({mdcode('$' + symbol)}\\) "
            f"{rec_e} {mdcode(f'{score}/10')} {mdcode(fmt_prob(prob))} "
            f"MC {mdcode(fmt_usd(mc))} {mditalic(age_str)}".rstrip()
        )
        rf = s["red_flags"]
        if rf:
            lines.append(f"   🚩 {mditalic(rf)}")
    return "\n".join(lines)


# ---------- Paper ----------

def text_paper_status(state: BotState) -> str:
    s = paper_stats()
    s["paper_enabled"] = state.paper_enabled
    status = "ON" if s["paper_enabled"] else "OFF"
    return (
        f"📋 {mdbold('Paper:')} {mdbold(status)}\n"
        f"SL {mdcode(f'-{PAPER_STOP_LOSS_PCT}%')} \\| "
        f"TP {mdcode(f'+{PAPER_TAKE_PROFIT_PCT}%')} \\| "
        f"Time {mdcode(f'{PAPER_TIME_STOP_SEC}s')}\n\n"
        f"Open {mdcode(s['open_positions'])} \\| "
        f"Closed {mdcode(s['closed_positions'])}\n"
        f"Win rate {mdcode(str(round(s['win_rate'],1))+'%')} \\| "
        f"Avg {mdcode(str(round(s['avg_pnl_pct'],2))+'%')}\n"
        f"Total PnL {mdcode(fmt_usd(s['total_pnl_usd'], 2))} \\| "
        f"Max DD {mdcode(fmt_usd(s['max_drawdown_usd'], 2))}"
    )


def text_paper_report(state: BotState) -> str:
    s = paper_stats()
    s["paper_enabled"] = state.paper_enabled

    with closing(db_conn()) as conn:
        recent = conn.execute(
            "SELECT mint, name, symbol, entry_mc, exit_mc, entry_time, exit_time, "
            "pnl_pct, pnl_usd, reason, position_size_usd "
            "FROM paper_trades WHERE status='CLOSED' ORDER BY exit_time DESC LIMIT 15"
        ).fetchall()
        open_trades = conn.execute("""
            SELECT t.mint, t.name, t.symbol, t.entry_mc, t.entry_time,
                   t.position_size_usd,
                   (SELECT s.market_cap FROM paper_mc_snapshots s
                    WHERE s.mint = t.mint
                    ORDER BY s.created_at DESC LIMIT 1) AS cur_mc
            FROM paper_trades t WHERE t.status='OPEN'
            ORDER BY t.entry_time
        """).fetchall()
        reason_stats = conn.execute(
            "SELECT reason, COUNT(*) AS cnt, "
            "AVG(pnl_pct) AS avg_pct, SUM(pnl_usd) AS total_usd "
            "FROM paper_trades WHERE status='CLOSED' AND reason IS NOT NULL "
            "GROUP BY reason ORDER BY cnt DESC"
        ).fetchall()

    ts_now  = now_ts()
    enabled = s["paper_enabled"]
    status  = "ON ✅" if enabled else "OFF ⛔"
    n       = s["closed_positions"]
    wins    = s["wins"]
    losses  = s["losses"]

    lines = [
        f"📑 {mdbold('Paper Trading Report')}  {mditalic(status)}",
        f"{mditalic(f'SL {PAPER_STOP_LOSS_PCT}% | TP +{PAPER_TAKE_PROFIT_PCT}% | time {fmt_duration(PAPER_TIME_STOP_SEC)} | size {fmt_usd(PAPER_POSITION_SIZE_USD)}/trade')}",
        "",
        mdbold("📊 Performance"),
    ]
    if n == 0:
        lines.append(mditalic("No closed trades yet."))
    else:
        lines += [
            f"Closed {mdcode(n)} \\| "
            f"{mdcode(wins)}W / {mdcode(losses)}L \\| "
            f"Win rate {mdcode(str(round(s['win_rate'],1))+'%')}",
            f"Avg PnL {mdcode(fmt_pct(s['avg_pnl_pct'], 2, signed=True))} \\| "
            f"Total {mdcode(fmt_usd(s['total_pnl_usd'], 2, signed=True))}",
            f"Best {mdcode(fmt_pct(s['best_pnl_pct'], 2, signed=True))} \\| "
            f"Worst {mdcode(fmt_pct(s['worst_pnl_pct'], 2, signed=True))}",
            f"Max DD {mdcode(fmt_usd(s['max_drawdown_usd'], 2))}",
        ]

    if reason_stats:
        lines += ["", mdbold("🎯 Exit Reasons")]
        for r in reason_stats:
            reason    = r["reason"] or "unknown"
            cnt       = safe_int(r["cnt"])
            avg_pct   = r["avg_pct"]
            total_usd = r["total_usd"]
            pct_str = fmt_pct(avg_pct, 1, signed=True) if avg_pct is not None else "—"
            usd_str = fmt_usd(total_usd, 2, signed=True) if total_usd is not None else "—"
            lines.append(
                f"• {mdcode(reason)} "
                f"{mdcode(cnt)}× \\| "
                f"avg {mdcode(pct_str)} \\| "
                f"total {mdcode(usd_str)}"
            )

    lines += ["", mdbold(f"📂 Open Positions ({len(open_trades)}/{PAPER_MAX_CONCURRENT})")]
    if not open_trades:
        lines.append(mditalic("None."))
    else:
        sl_pct, tp_pct = adaptive_sl_tp()
        lines.append(mditalic(f"SL {sl_pct:.1f}% \\(adaptive\\) TP {tp_pct:.1f}% \\(adaptive\\)"))
        for t in open_trades:
            name     = t["name"] or t["symbol"] or (t["mint"] or "?")[:6]
            entry_mc = safe_float(t["entry_mc"])
            age_sec  = ts_now - safe_int(t["entry_time"])
            size     = safe_float(t["position_size_usd"])
            cur_mc   = t["cur_mc"]
            unreal_str = ""
            if cur_mc is not None and entry_mc > 0:
                unreal = ((safe_float(cur_mc) - entry_mc) / entry_mc) * 100
                e = "🟢" if unreal > 0 else ("🔴" if unreal < 0 else "⚪")
                unreal_str = f" {e} {mdcode(fmt_pct(unreal, 1, signed=True))}"
            lines.append(
                f"• {mdbold(name)} "
                f"entry {mdcode(fmt_usd(entry_mc))} \\| "
                f"age {mdcode(fmt_duration(age_sec))} \\| "
                f"size {mdcode(fmt_usd(size))}{unreal_str}"
            )

    lines += ["", mdbold("🕒 Last 15 Trades")]
    if not recent:
        lines.append(mditalic("No closed trades yet."))
    else:
        for r in recent:
            mint     = r["mint"] or ""
            name_raw = r["name"] or r["symbol"] or (mint[:6] if mint else "?")
            pnl      = safe_float(r["pnl_pct"])
            usd      = safe_float(r["pnl_usd"])
            entry_mc = safe_float(r["entry_mc"])
            exit_mc  = safe_float(r["exit_mc"])
            entry_t  = safe_int(r["entry_time"])
            exit_t   = safe_int(r["exit_time"])
            reason   = r["reason"] or "?"

            dur_sec = (exit_t - entry_t) if (entry_t and exit_t and exit_t > entry_t) else 0
            pnl_e   = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")

            line1 = (
                f"{pnl_e} {mdbold(name_raw)} "
                f"{mdcode(fmt_pct(pnl, 2, signed=True))} "
                f"\\({mdcode(fmt_usd(usd, 2, signed=True))}\\)"
            )
            mc_arrow = (f"{mdcode(fmt_usd(entry_mc))}→{mdcode(fmt_usd(exit_mc))}"
                        if entry_mc and exit_mc else "")
            dur_str  = mdcode(fmt_duration(dur_sec)) if dur_sec > 0 else ""
            reason_s = mditalic(reason)

            details = [p for p in [mc_arrow, dur_str, reason_s] if p]
            line2 = "    " + " \\| ".join(details) if details else ""

            lines.append(line1)
            if line2:
                lines.append(line2)

    return "\n".join(lines)


# ---------- Stats / wallet / top ----------

def query_time_to_pump_data() -> list[dict]:
    with closing(db_conn()) as conn:
        rows = conn.execute("""
            SELECT s.id, s.score, s.created_at, s.market_cap_at_signal, s.mint
            FROM signals s
            WHERE s.created_at >= ?
        """, (now_ts() - 7 * 86400,)).fetchall()

    buckets: dict[str, list[float]] = defaultdict(list)
    for s in rows:
        score    = safe_int(s["score"])
        created  = safe_int(s["created_at"])
        entry_mc = safe_float(s["market_cap_at_signal"])
        mint     = s["mint"]
        if entry_mc <= 0:
            continue
        with closing(db_conn()) as conn:
            peak = conn.execute("""
                SELECT market_cap, created_at FROM price_snapshots
                WHERE mint=? AND created_at >= ?
                ORDER BY market_cap DESC LIMIT 1
            """, (mint, created)).fetchone()
        if not peak:
            continue
        peak_mc = safe_float(peak["market_cap"])
        peak_ts = safe_int(peak["created_at"])
        if peak_mc <= entry_mc * 1.5:
            continue
        mins = (peak_ts - created) / 60.0
        if mins <= 0:
            continue
        if   score >= 9: bucket = "9-10"
        elif score >= 7: bucket = "7-8"
        elif score >= 5: bucket = "5-6"
        else:            bucket = "1-4"
        buckets[bucket].append(mins)

    result = []
    for b in ["9-10", "7-8", "5-6", "1-4"]:
        vals = buckets.get(b, [])
        if not vals:
            continue
        vals.sort()
        result.append({
            "bucket":     b,
            "n":          len(vals),
            "median_min": vals[len(vals) // 2],
            "p25_min":    vals[len(vals) // 4],
            "p75_min":    vals[min(3 * len(vals) // 4, len(vals) - 1)],
        })
    return result


def text_stats(state: BotState, engine: ScoringEngine) -> str:
    ts_now = now_ts()
    eq = PaperWallet.equity()
    starting = eq["starting"]
    total    = eq["total_equity"]
    pnl_pct  = ((total - starting) / starting * 100) if starting > 0 else 0
    pnl_e    = "🟢" if pnl_pct > 0 else ("🔴" if pnl_pct < 0 else "⚪")

    with closing(db_conn()) as conn:
        sig_24h = conn.execute(
            "SELECT COUNT(*) AS c FROM signals WHERE created_at >= ?",
            (ts_now - 86400,)).fetchone()["c"]
        out_24h = conn.execute("""
            SELECT outcome, COUNT(*) AS c FROM lookbacks
            WHERE checked=1 AND check_at >= ? AND window_label=?
            GROUP BY outcome
        """, (ts_now - 86400, ML_LABEL_WINDOW)).fetchall()
        bl_count = conn.execute(
            "SELECT COUNT(*) AS c FROM creator_blacklist").fetchone()["c"]

    out_map = {r["outcome"]: r["c"] for r in out_24h}
    pumps_24h = out_map.get("PUMP", 0) + out_map.get("MOON", 0)
    rugs_24h  = out_map.get("RUG", 0)
    total_24h = sum(out_map.values())
    pump_rate_24h = (pumps_24h / total_24h * 100) if total_24h > 0 else 0

    streak = recent_loss_streak()
    daily  = daily_pnl_usd()

    s = engine.status()
    cv_str = f"{s['cv_auc']:.3f}" + (
        f" ±{s['cv_auc_std']:.3f}" if s["cv_auc_std"] else ""
    )

    ttp = query_time_to_pump_data()
    ttp_lines = []
    if ttp:
        ttp_lines.append("")
        ttp_lines.append(mdbold("⏱ Time to peak (last 7d)"))
        for r in ttp:
            ttp_lines.append(
                f"• {mdcode(r['bucket'])} median {mdcode(str(round(r['median_min']))+'m')} "
                f"\\(n={mdcode(r['n'])}\\)"
            )

    lines = [
        f"📊 {mdbold('Bot Stats Dashboard')}",
        "",
        mdbold("💰 Wallet"),
        f"Balance: {mdcode(fmt_usd(eq['balance'], 2))}",
        f"Open positions: {mdcode(fmt_usd(eq['positions_value'], 2))}",
        f"Total equity: {mdcode(fmt_usd(total, 2))}",
        f"P&L vs start: {pnl_e} {mdcode(fmt_pct(pnl_pct, 2, signed=True))}",
        f"24h realized: {mdcode(fmt_usd(daily, 2, signed=True))}",
        "",
        mdbold("📡 Last 24h"),
        f"Signals scored: {mdcode(sig_24h)}",
        f"Pump rate: {mdcode(f'{pump_rate_24h:.1f}%')} "
        f"\\({mdcode(pumps_24h)}/{mdcode(total_24h)}\\)",
        f"Rugs: {mdcode(rugs_24h)}",
        "",
        mdbold("🤖 Model"),
        f"Mode: {mdcode(s['mode'])}",
        f"CV AUC: {mdcode(cv_str)}",
        f"Samples: {mdcode(s['n_train_samples'])}",
        "",
        mdbold("🛡 Defense"),
        f"Loss streak: {mdcode(streak)}",
        f"Blacklisted creators: {mdcode(bl_count)}",
    ]
    lines.extend(ttp_lines)
    return "\n".join(lines)


def text_wallet() -> str:
    eq = PaperWallet.equity()
    starting = eq["starting"]
    total    = eq["total_equity"]
    pnl_pct  = ((total - starting) / starting * 100) if starting > 0 else 0
    pnl_e    = "🟢" if pnl_pct > 0 else ("🔴" if pnl_pct < 0 else "⚪")
    return "\n".join([
        f"💰 {mdbold('Paper Wallet')}",
        "",
        f"Starting: {mdcode(fmt_usd(starting, 2))}",
        f"Balance: {mdcode(fmt_usd(eq['balance'], 2))}",
        f"In positions: {mdcode(fmt_usd(eq['positions_value'], 2))}",
        f"Unrealized: {mdcode(fmt_usd(eq['unrealized'], 2, signed=True))}",
        f"Total equity: {mdcode(fmt_usd(total, 2))}",
        f"Return: {pnl_e} {mdcode(fmt_pct(pnl_pct, 2, signed=True))}",
        "",
        mditalic(f"Fees: {PAPER_FEE_PCT}% per side | Slippage: {PAPER_SLIPPAGE_PCT}%"),
    ])


def query_top_performers(days: int = 7, limit: int = 10) -> list[dict]:
    cutoff = now_ts() - days * 86400
    with closing(db_conn()) as conn:
        rows = conn.execute("""
            SELECT s.name, s.symbol, s.mint, s.score, s.market_cap_at_signal,
                   s.created_at, lb.pct_change, lb.outcome, lb.window_label
            FROM signals s
            JOIN lookbacks lb ON lb.signal_id = s.id
            WHERE s.created_at >= ?
              AND lb.checked = 1
              AND lb.window_label = ?
              AND lb.outcome IN ('PUMP','MOON')
            ORDER BY lb.pct_change DESC
            LIMIT ?
        """, (cutoff, ML_LABEL_WINDOW, limit)).fetchall()
    return [dict(r) for r in rows]


def format_top_performers(rows: list[dict], days: int = 7) -> str:
    if not rows:
        return mditalic(f"No PUMP/MOON outcomes in last {days}d.")
    lines = [
        f"🏆 {mdbold(f'Top Performers (last {days}d)')}",
        f"{mditalic(f'Window: {ML_LABEL_WINDOW}')}",
        "",
    ]
    for i, r in enumerate(rows, 1):
        name    = r["name"] or r["symbol"] or (r["mint"] or "?")[:8]
        score   = safe_int(r["score"])
        pct     = safe_float(r["pct_change"])
        outcome = r["outcome"] or ""
        emoji   = "🚀" if outcome == "MOON" else "📈"
        mc      = safe_float(r["market_cap_at_signal"])
        age     = fmt_duration(now_ts() - safe_int(r["created_at"]))
        lines.append(
            f"{i}\\. {emoji} {mdbold(name)} {mdcode(f'{score}/10')}\n"
            f"   {mdcode(fmt_pct(pct, 1, signed=True))} from "
            f"{mdcode(fmt_usd(mc))} \\({mditalic(age)} ago\\)"
        )
        if r["mint"]:
            lines.append(f"   🔗 [Pump\\.fun]({PUMP_FRONT}/{r['mint']})")
    return "\n".join(lines)


# ---------- Health / help ----------

def text_health(state: BotState, engine: ScoringEngine) -> str:
    import time
    ts_now = now_ts()
    uptime = ts_now - safe_int(get_state("bot_started_at", str(ts_now)))
    dead_sec = int(time.time() - state.last_coin_ts)

    with closing(db_conn()) as conn:
        total_signals = conn.execute("SELECT COUNT(*) AS c FROM signals").fetchone()["c"]
        signals_1hr = conn.execute(
            "SELECT COUNT(*) AS c FROM signals WHERE created_at >= ?",
            (ts_now - 3600,)
        ).fetchone()["c"]
        dead_letters = conn.execute(
            "SELECT COUNT(*) AS c FROM dead_letters"
        ).fetchone()["c"]
        dead_unretried = conn.execute(
            "SELECT COUNT(*) AS c FROM dead_letters "
            "WHERE retry_count < ? OR retry_count IS NULL",
            (DEAD_LETTER_MAX_RETRIES,)
        ).fetchone()["c"]
        open_trades = conn.execute(
            "SELECT COUNT(*) AS c FROM paper_trades WHERE status='OPEN'"
        ).fetchone()["c"]
        pending_lb = conn.execute(
            "SELECT COUNT(*) AS c FROM lookbacks WHERE checked=0"
        ).fetchone()["c"]

    stream_status = (
        f"🟢 {mdcode('LIVE')} \\(last coin {mdcode(fmt_duration(dead_sec))} ago\\)"
        if dead_sec < STREAM_DEAD_ALERT_SEC
        else f"🔴 {mdcode('DEAD')} \\(no coins for {mdcode(fmt_duration(dead_sec))}\\)"
    )

    drift = get_state("model_drift_detected", "0") == "1"
    model_status = f"⚠️ {mdbold('DRIFT')}" if drift else f"✅ {mdcode(engine.mode_label)}"

    return "\n".join([
        f"❤️ {mdbold('Bot Health')}",
        f"Uptime: {mdcode(fmt_duration(uptime))}",
        "",
        f"Stream: {stream_status}",
        f"Model: {model_status}",
        "",
        f"Signals total: {mdcode(total_signals)} \\| last 1hr: {mdcode(signals_1hr)}",
        f"Dead letters: {mdcode(dead_letters)} \\| pending retry: {mdcode(dead_unretried)}",
        f"Lookbacks pending: {mdcode(pending_lb)}",
        f"Open paper trades: {mdcode(open_trades)}",
        "",
        f"DB: {mdcode(DB_PATH)} \\| backup: {mdcode(DB_BACKUP_PATH)}",
        f"Log: {mdcode(LOG_PATH)}",
    ])


def text_help() -> str:
    return "\n".join([
        f"🤖 {mdbold('Pump.fun Monitor v1.1')}",
        "",
        mdbold("Commands:"),
        f"{mdcode('/menu')} — open control menu",
        f"{mdcode('/monitor_on')} — enable alerts",
        f"{mdcode('/monitor_off')} — disable alerts",
        f"{mdcode('/monitor_status')} — show alert status",
        f"{mdcode('/set_threshold N')} — set alert threshold \\(1-10\\)",
        f"{mdcode('/scoring_mode')} — show scoring mode",
        f"{mdcode('/features')} — show feature importances",
        f"{mdcode('/keywords')} — show learned keywords",
        f"{mdcode('/market')} — show market context",
        f"{mdcode('/outcomes')} — show outcome stats",
        f"{mdcode('/model')} — show ML model status",
        f"{mdcode('/train')} — retrain model now",
        f"{mdcode('/snapshot')} — show recent signals",
        f"{mdcode('/paper_on')} \\| {mdcode('/paper_off')} — toggle paper trading",
        f"{mdcode('/paper_status')} — paper summary",
        f"{mdcode('/paper_report')} — detailed paper report",
        f"{mdcode('/health')} — bot health and stream status",
        f"{mdcode('/score <mint>')} — manually score any coin \\(not saved\\)",
        f"{mdcode('/backtest')} — signal performance by score bucket",
        f"{mdcode('/watch <mint>')} — add coin to watchlist",
        f"{mdcode('/unwatch <mint>')} — remove from watchlist",
        f"{mdcode('/watchlist')} — show your watchlist",
        "",
        mdbold("Wallet & Stats:"),
        f"{mdcode('/stats')} — daily dashboard",
        f"{mdcode('/wallet')} — paper wallet status",
        f"{mdcode('/wallet_reset [amount]')} — reset paper wallet",
        f"{mdcode('/top [days]')} — best signals recently",
        f"{mdcode('/blacklist [add|remove] <wallet>')} — manage creator blacklist",
        "",
        mditalic("Tap Menu below or type any command."),
    ])
