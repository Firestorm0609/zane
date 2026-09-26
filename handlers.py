"""Slash commands + inline callback handlers."""
import html
import logging
import secrets
import time
from typing import Any, Optional

from solders.keypair import Keypair
from telegram import LabeledPrice, Update
from telegram.constants import ParseMode
from telegram.ext import (Application, ApplicationHandlerStop,
                          CallbackQueryHandler, CommandHandler, ContextTypes,
                          MessageHandler, PreCheckoutQueryHandler, TypeHandler,
                          filters)

import callouts
import config
import exits
import jupiter
import keyboards as kb
import trader
from db import DB

log = logging.getLogger("handlers")


HELP_TEXT = (
    "<b>ZANE</b> ⚡️\n\n"
    "Copy-trades <b>callouts</b>: when a caller you follow posts a "
    "callout on a new token, the bot buys it for you instantly.\n\n"
    "<b>Commands</b>\n"
    "/start — main menu\n"
    "/help — this text\n"
    "/wallet — import / view / remove your trading wallet\n"
    "/addcaller <code>&lt;wallet_or_uuid&gt;</code> [label] — follow a caller\n"
    "/callers — list callers you follow\n"
    "/buy <code>&lt;mint&gt;</code> [sol] — manual buy (PumpPortal or Jupiter by venue)\n"
    "/sell <code>&lt;mint&gt;</code> [pct] — manual sell (default 100%)\n"
    "/positions — open copy-trade positions\n"
    "/balance — wallet SOL balance\n"
    "/stats <code>&lt;caller&gt;</code> — caller win-rate over recent callouts\n\n"
    "<b>Per-caller settings (tap a caller in /callers)</b>\n"
    "Buy size · Take-profit · Stop-loss · Market-cap range filter\n\n"
    "<b>How it works</b>\n"
    "1. Import a wallet (key encrypted at rest)\n"
    "2. Add callers: Solana or EVM (0x…) wallets, or user UUIDs\n"
    "3. Bot polls callouts every ~15s and buys:\n"
    "   pump.fun mints via PumpPortal · other Solana mints via Jupiter\n"
    "   (EVM-chain tokens are alert-only — no EVM executor yet)\n"
    "4. Mcap range filter decides which calls auto-buy\n"
    "5. TP/SL exits run automatically; /positions for manual sells\n\n"
    "⚠️ Memecoins are extremely high risk. Only risk what you can afford to lose."
)


def _fmt_caller(c: dict[str, Any]) -> str:
    state = "🟢 active" if c["enabled"] else "⏸ paused"
    label = html.escape(c["label"] or "")
    tp = c.get("max_multiple") or 0
    sl = c.get("stop_multiple") or 0
    tp_txt = f"{tp:g}x" if tp > 0 else ("off" if tp < 0 else "2x (def)")
    sl_txt = f"{sl:g}x" if sl > 0 else ("off" if sl < 0 else "0.5x (def)")
    slip = c.get("slippage") or 0
    slip_txt = f"{slip:g}%" if slip > 0 else "def"
    pf = c.get("priority_fee") or 0
    pf_txt = "off" if pf < 0 else (f"{pf:g}" if pf > 0 else "def")
    tp_pct = c.get("tp_sell_pct") or 100
    tp_pct_txt = f"{tp_pct:g}%" if tp_pct < 100 else "100%"
    min_mc = c.get("min_mcap") or 0
    max_mc = c.get("max_mcap") or 0
    if min_mc or max_mc:
        mc_txt = f"{min_mc:,.0f}–{max_mc:,.0f}" if max_mc else f"{min_mc:,.0f}+"
    else:
        mc_txt = "any"
    trail = c.get("trail_pct") or 0
    trail_txt = f"{trail:g}%" if trail > 0 else "off"
    be = c.get("be_multiple") or 0
    be_txt = f"{be:g}x" if be > 0 else "off"
    entry = c.get("max_entry_multiple") or 0
    entry_txt = f"≤{entry:g}x" if entry > 0 else "off"
    return (f"{state} · <code>{html.escape(c['caller_id'])}</code>\n"
            f"   buy <b>{c['buy_sol']:g} SOL</b>"
            + (f" · {label}" if label else "") + "\n"
            f"   TP {tp_txt} (sells {tp_pct_txt}) · SL {sl_txt} · mcap {mc_txt} "
            f"· slip {slip_txt} · ⛽ {pf_txt}\n"
            f"   📉 trail {trail_txt} · 🛡 BE {be_txt} · 🚀 entry {entry_txt}")


class Handlers:
    def __init__(self, db: DB, callout_client: callouts.CalloutClient):
        self.db = db
        self.callouts = callout_client
        self.poller = None  # set by main after poller creation
        # conversation state: tg_id -> {"awaiting": "wallet|caller|buysize", "cid": ...}
        self.awaiting: dict[int, dict[str, Any]] = {}
        # brute-force guard for /unlock: tg_id -> (attempts, window_start)
        self._unlock_attempts: dict[int, tuple[int, float]] = {}
        # chat_id -> ts of the last group paywall notice (anti-spam: 1/min)
        self._group_gate_ts: dict[int, float] = {}

    # ---------- helpers ----------
    async def _wallet(self, tg_id: int) -> Optional[Keypair]:
        w = await self.db.load_wallet(tg_id)
        if not w:
            return None
        return trader.keypair_from_secret(w[1])

    async def _send(self, update: Update, text: str, reply_markup=None):
        if update.effective_message:
            await update.effective_message.reply_text(
                text, parse_mode=ParseMode.HTML, reply_markup=reply_markup,
                disable_web_page_preview=True)

    # ---------- commands ----------
    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        tg = update.effective_user.id
        await self.db.ensure_user(tg)
        if tg != config.BOT_OWNER_ID and not await self.db.is_unlocked(tg):
            await self._send(
                update,
                "🔒 <b>ZANE is subscription-based.</b>\n"
                "One-time $3 (in Stars) — lifetime access to every feature:\n"
                "copy-trading, positions, exits and alerts.\n\n"
                "Got a code? <code>/unlock YOUR-CODE</code>",
                reply_markup=self._pay_keyboard())
            return
        has_wallet = bool(await self.db.wallet_pubkey(tg))
        callers = await self.db.get_callers(tg)
        bal_txt = await self._balance_line(tg, has_wallet)
        text = (f"<b>ZANE</b> ⚡️\n\n"
                f"{bal_txt}\n"
                f"Callers followed: <b>{len(callers)}</b>")
        await self._send(update, text, reply_markup=kb.main_menu(has_wallet, len(callers)))

    async def _balance_line(self, tg: int, has_wallet: bool) -> str:
        if not has_wallet:
            return "Wallet: ❌ not set (💼 Wallet → Import)"
        try:
            pubkey = await self.db.wallet_pubkey(tg)
            bal = await trader.get_balance_sol(pubkey)
            return f"Balance: <b>{bal:.4f} SOL</b> · <code>{pubkey[:8]}…{pubkey[-6:]}</code>"
        except Exception:
            return "Balance: ⚠️ could not fetch"

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._send(update, HELP_TEXT)

    async def cmd_wallet(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        tg = update.effective_user.id
        pubkey = await self.db.wallet_pubkey(tg)
        if pubkey:
            rows = await self.db.load_wallet(tg)
            label = rows[0] if rows else ""
            text = (f"💼 Wallet <b>{label}</b>\n<code>{pubkey}</code>\n\n"
                    "Your key is encrypted at rest (AES-256-GCM). "
                    "It is decrypted only in-memory to sign trades.")
            await self._send(update, text, reply_markup=kb.wallet_menu(True, label, pubkey))
        else:
            await self._send(update, "💼 No wallet imported yet.",
                             reply_markup=kb.wallet_menu(False))

    async def cmd_addcaller(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        tg = update.effective_user.id
        args = (ctx.args or [])
        if not args:
            await self._send(
                update,
                "Usage: <code>/addcaller &lt;wallet_or_uuid&gt; [label]</code>\n\n"
                "The label is optional — I'll name them from their pump.fun "
                "profile. Get the wallet from their pump.fun profile URL or "
                "from a callout author.")
            self.awaiting[tg] = {"awaiting": "caller"}
            return
        caller_id = args[0].strip()
        label = " ".join(args[1:])[:32]
        await self._add_caller(tg, caller_id, label)

    async def _add_caller(self, tg: int, caller_id: str, label: str):
        caller_id = caller_id.strip()
        if not callouts.is_valid_caller_id(caller_id):
            await self._send_from_id(
                tg, "❌ That doesn't look like a Solana wallet, EVM address "
                    "(0x…), or user UUID.")
            return
        # pull their pump.fun profile so you don't have to name them yourself
        profile = await self.callouts.caller_profile(caller_id)
        if not label:
            label = (profile.get("username") or profile.get("bio") or "")[:32]
        try:
            stats = await self.callouts.caller_stats(caller_id)
        except Exception as e:
            await self._send_from_id(
                tg, f"❌ Could not fetch callouts for that ID (rate limit? try again in a minute): {e}")
            return
        if stats.get("count", 0) == 0:
            await self._send_from_id(
                tg, "❌ No callouts found for that ID. Double-check the wallet/UUID.")
            return
        await self.db.add_caller(tg, caller_id, label, buy_sol=0.01,
                                 max_multiple=2.0, stop_multiple=0.5)
        # Baseline: mark the callouts we just fetched (stats used limit=50) as seen,
        # reusing that data instead of fetching again.
        # caller_stats already consumed a limit=50 fetch; re-derive from a fresh small
        # fetch would cost another request, so we mark seen from the stats fetch via
        # a tiny extra fetch of the newest page only:
        try:
            data = await self.callouts.list_callouts(caller_id, limit=10)
            for c in data.get("callouts", []):
                cid = c.get("calloutId")
                if cid:
                    await self.db.mark_seen(cid)
            newest = (data.get("callouts") or [{}])[0].get("calloutId", "")
            await self.db.update_caller_cursor(tg, caller_id, newest)
        except Exception:
            log.warning("baseline failed for %s — may alert on recent callouts once", caller_id)
        # feed mode: make the bot account follow this caller too
        if self.callouts.bot_uuid:
            try:
                cuuid = await self.callouts.resolve_uuid(caller_id)
                ok = await self.callouts.follow(cuuid)
                log.info("follow %s: %s", caller_id[:12], ok)
            except Exception as e:
                log.warning("follow %s failed: %s", caller_id[:12], e)
        lines = [f"✅ Following <code>{caller_id}</code>"
                 + (f" <i>({html.escape(label)})</i>" if label else ""),
                 f"• Recent callouts: <b>{stats['count']}</b>",
                 f"• Win rate (≥2x): <b>{stats['win_rate_2x'] * 100:.0f}%</b>",
                 f"• Best: <b>{stats['best_multiple']}x</b> · avg {stats['avg_multiple']}x",
                 "",
                 "Default buy: <b>0.01 SOL</b> per callout. Change with the "
                 "💰 button in Callers."]
        if profile:
            who = []
            if profile.get("username"):
                who.append("@" + html.escape(profile["username"]))
            if profile.get("bio"):
                who.append(f"“{html.escape(profile['bio'])}”")
            if profile.get("x_username"):
                who.append("𝕏 @" + html.escape(profile["x_username"]))
            if profile.get("followers"):
                who.append(f"<b>{profile['followers']:,}</b> followers")
            if who:
                lines.insert(1, "• " + " · ".join(who))
        await self._send_from_id(tg, "\n".join(lines))

    async def _send_from_id(self, tg: int, text: str):
        from telegram import Bot
        bot: Bot = self.app.bot
        await bot.send_message(tg, text, parse_mode=ParseMode.HTML)

    def bind_app(self, app: Application):
        self.app = app

    async def cmd_callers(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        tg = update.effective_user.id
        callers = await self.db.get_callers(tg)
        if not callers:
            await self._send(update, "📣 You follow no callers yet.",
                             reply_markup=kb.callers_menu([]))
            return
        text = "📣 <b>Callers you follow</b>\n\n" + "\n\n".join(_fmt_caller(c) for c in callers)
        await self._send(update, text, reply_markup=kb.callers_menu(callers))

    async def cmd_buy(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        tg = update.effective_user.id
        args = ctx.args or []
        if not args:
            await self._send(update, "Usage: <code>/buy &lt;mint&gt; [sol]</code>")
            return
        mint = args[0]
        if not callouts.is_solana_mint(mint):
            await self._send(update, "❌ That doesn't look like a Solana mint address.")
            return
        amount = float(args[1]) if len(args) > 1 else 0.01
        await self._do_buy(tg, mint, amount, caller_id="manual", callout_id="manual")

    async def cmd_sell(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        tg = update.effective_user.id
        args = ctx.args or []
        if not args:
            await self._send(update, "Usage: <code>/sell &lt;mint&gt; [pct]</code>")
            return
        mint = args[0]
        pct = float(args[1]) if len(args) > 1 else 100.0
        kp = await self._wallet(tg)
        if not kp:
            await self._send(update, "❌ Import a wallet first with /wallet.")
            return
        try:
            slip = 0.0
            c = await self.db.get_caller_by_mint(tg, mint)
            if c:
                slip = float(c.get("slippage") or 0)
            sig = None
            if callouts.is_pump_fun_mint(mint):
                try:
                    sig = await trader.sell(kp, mint, percent=pct,
                                            slippage=f"{slip:g}" if slip > 0 else "10")
                except Exception as e:
                    # graduated pump.fun tokens aren't on the bonding curve
                    # anymore — PumpPortal 400s; fall back to Jupiter
                    if "400" not in str(e):
                        raise
                    log.info("pump.fun sell 400 for %s — graduated? falling "
                             "back to Jupiter", mint[:12])
            if sig is None:
                from exits import get_token_decimals
                bal = await trader.get_token_balance(str(kp.pubkey()), mint)
                if bal <= 0:
                    await self._send(update, "❌ No token balance to sell.")
                    return
                decimals = await get_token_decimals(mint)
                sig = await jupiter.sell(kp, mint, bal, decimals, percent=pct,
                                         slippage_bps=int(slip * 100) if slip > 0 else 300)
            await self._send(update, f"✅ Sold <b>{pct:.0f}%</b> of <code>{mint[:12]}…</code>\n"
                                     f"tx: <code>{sig}</code>")
        except Exception as e:
            await self._send(update, f"❌ Sell failed: {e}")

    # ---------- positions: view + manual sell ----------
    async def _handle_position(self, tg: int, q, action: str, pid: int, pct: float):
        p = await self.db.get_position_by_id(pid)
        if not p or p["tg_id"] != tg:
            await q.message.reply_text("❌ Position not found.")
            return
        if action == "view":
            from exits import get_token_decimals
            pubkey = await self.db.wallet_pubkey(tg)
            bal = 0.0
            if pubkey:
                bal = await trader.get_token_balance(pubkey, p["mint"])
            age = int(time.time() - p["opened_at"])
            text = (f"📈 <code>{p['mint'][:16]}…</code>\n"
                    f"• entry: {p['buy_amount_sol']:g} SOL ({age // 60}m ago)\n"
                    f"• tokens held: <b>{bal:,.2f}</b>\n"
                    f"• exits: TP/SL per caller settings")
            await q.message.reply_text(text, parse_mode=ParseMode.HTML,
                                       reply_markup=kb.position_detail(p))
        elif action == "sell":
            await self._sell_position(tg, p, pct)
        elif action == "sellpct":
            self.awaiting[tg] = {"awaiting": "sellpct", "pid": pid}
            await q.message.reply_text(
                "✏️ Send the percent to sell (1–100), e.g. <code>30</code>:",
                parse_mode=ParseMode.HTML)

    async def _sell_position(self, tg: int, p: dict, pct: float):
        kp = await self._wallet(tg)
        if not kp:
            await self._send_from_id(tg, "❌ No wallet imported.")
            return
        pubkey = str(kp.pubkey())
        mint = p["mint"]
        try:
            slip = 0.0
            c = await self.db.get_caller(tg, p["caller_id"])
            if c:
                slip = float(c.get("slippage") or 0)
            sig = None
            if callouts.is_pump_fun_mint(mint):
                try:
                    sig = await trader.sell(kp, mint, percent=pct,
                                            slippage=f"{slip:g}" if slip > 0 else "10")
                except Exception as e:
                    # graduated pump.fun tokens aren't on the bonding curve
                    # anymore — PumpPortal 400s; fall back to Jupiter
                    if "400" not in str(e):
                        raise
                    log.info("pump.fun sell 400 for %s — graduated? falling "
                             "back to Jupiter", mint[:12])
            if sig is None:
                from exits import get_token_decimals
                bal = await trader.get_token_balance(str(kp.pubkey()), mint)
                if bal <= 0:
                    await self._send_from_id(tg, "❌ No token balance to sell.")
                    return
                decimals = await get_token_decimals(mint)
                sig = await jupiter.sell(kp, mint, bal, decimals, percent=pct,
                                         slippage_bps=int(slip * 100) if slip > 0 else 300)
        except Exception as e:
            await self._send_from_id(tg, f"❌ Sell failed: {e}")
            return
        if pct >= 100:
            await self.db.mark_sold(tg, p["callout_id"], sig, 0.0)
            await self._send_from_id(
                tg, f"✅ Sold 100% of <code>{mint[:12]}…</code>\ntx: <code>{sig}</code>")
        else:
            await self._send_from_id(
                tg, f"✅ Sold {pct:g}% of <code>{mint[:12]}…</code>\ntx: <code>{sig}</code>")

    async def cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        tg = update.effective_user.id
        positions = await self.db.get_open_positions(tg)
        if not positions:
            await self._send(update, "📊 No open positions.")
            return
        lines = ["📊 <b>Open positions</b> — tap to manage:\n"]
        for p in positions:
            age = int(time.time() - p["opened_at"])
            lines.append(
                f"• <code>{p['mint'][:12]}…{p['mint'][-6:]}</code>\n"
                f"  {p['buy_amount_sol']:g} SOL · via {p['caller_id'][:8]}… · {age // 60}m ago")
        await self._send(update, "\n".join(lines),
                         reply_markup=kb.positions_menu(positions))

    async def cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        tg = update.effective_user.id
        pubkey = await self.db.wallet_pubkey(tg)
        if not pubkey:
            await self._send(update, "❌ Import a wallet first with /wallet.")
            return
        bal = await trader.get_balance_sol(pubkey)
        await self._send(update, f"💰 Balance: <b>{bal:.4f} SOL</b>\n<code>{pubkey}</code>")

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        args = ctx.args or []
        if not args:
            await self._send(update, "Usage: <code>/stats &lt;caller_id&gt;</code>")
            return
        try:
            stats = await self.callouts.caller_stats(args[0])
        except Exception as e:
            await self._send(update, f"❌ {e}")
            return
        if stats.get("count", 0) == 0:
            await self._send(update, "No callouts found for that caller.")
            return
        age_days = max(1, int((time.time() - stats["oldest_ts"] / 1000) / 86400))
        await self._send(
            update,
            f"📈 <b>Caller stats</b> (last {stats['count']} callouts, ~{age_days}d)\n"
            f"• Win rate ≥2x: <b>{stats['win_rate_2x'] * 100:.0f}%</b>\n"
            f"• Best multiple: <b>{stats['best_multiple']}x</b>\n"
            f"• Average multiple: {stats['avg_multiple']}x")

    async def _do_buy(self, tg: int, mint: str, amount_sol: float,
                      caller_id: str, callout_id: str):
        kp = await self._wallet(tg)
        if not kp:
            await self._send_from_id(tg, "❌ Import a wallet first with /wallet.")
            return
        if not callouts.is_solana_mint(mint):
            # EVM (or otherwise unparseable) mint: there is no executor for
            # any other chain, so fail with the reason instead of handing a
            # 0x address to the Solana-only Jupiter api
            await self._send_from_id(
                tg, f"🚫 Can't trade <code>{html.escape(mint[:42])}</code> — "
                    "this bot executes on Solana only (pump.fun + Jupiter).")
            return
        via_jupiter = not callouts.is_pump_fun_mint(mint)
        slip = 0.0
        pfee = config.PRIORITY_FEE_SOL
        if caller_id != "manual":
            c = await self.db.get_caller(tg, caller_id)
            slip = float((c or {}).get("slippage") or 0)
            pf = float((c or {}).get("priority_fee") or 0)
            if pf < 0:            # disabled for this caller
                pfee = 0.0
            elif pf > 0:          # explicit override
                pfee = pf
        try:
            if via_jupiter:
                sig = await jupiter.buy(kp, mint, amount_sol,
                                        slippage_bps=int(slip * 100) if slip > 0 else 100)
                venue = "Jupiter (off-platform)"
            else:
                sig = await trader.buy(kp, mint, amount_sol,
                                       slippage=f"{slip:g}" if slip > 0 else "10",
                                       priority_fee=pfee or None)
                venue = "pump.fun"
        except Exception as e:
            await self._send_from_id(tg, f"❌ Buy failed for <code>{mint[:12]}…</code>: {e}")
            return
        # Unique per-buy callout_id: positions UNIQUE(tg_id, callout_id) would
        # otherwise silently drop every manual buy after the first (the tx
        # lands on-chain but no position is ever recorded/managed).
        if callout_id == "manual":
            callout_id = f"manual:{int(time.time() * 1000)}:{secrets.token_hex(4)}"
        await self.db.open_position(tg, caller_id, callout_id, mint, sig,
                                    amount_sol, tokens=0, entry_price=0)
        await self._send_from_id(
            tg, f"✅ <b>Bought</b> <code>{mint[:12]}…{mint[-6:]}</code>\n"
                f"• {amount_sol} SOL via {venue}\n"
                f"• via {caller_id if caller_id != 'manual' else 'manual /buy'}\n"
                f"• tx: <code>{sig}</code>\n\n"
                f"/sell <code>{mint}</code> to exit.")

    # ---------- text input flow (wallet import etc.) ----------
    async def on_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        tg = update.effective_user.id
        text = (update.effective_message.text or "").strip()
        state = self.awaiting.get(tg)

        # a pasted contract address has no implicit meaning — only explicit
        # pending state advances the conversation
        if not state:
            return
        text = (update.effective_message.text or "").strip()
        kind = state.get("awaiting")
        self.awaiting.pop(tg, None)
        if kind == "wallet":
            label = state.get("label") or "imported"
            try:
                kp = trader.keypair_from_secret(text)
            except Exception as e:
                await self._send(update, f"❌ Invalid private key: {e}")
                return
            pubkey = str(kp.pubkey())
            sk_hex = bytes(kp).hex()  # store raw 64-byte form, encrypted
            await self.db.save_wallet(tg, label, sk_hex, pubkey)
            if self.poller:
                self.poller.invalidate_wallet(tg)
            bal = await trader.get_balance_sol(pubkey)
            await self._send(update, f"✅ Wallet imported: <code>{pubkey}</code>\n"
                                     f"Balance: <b>{bal:.4f} SOL</b>")
        elif kind == "caller":
            parts = text.split(None, 1)
            await self._add_caller(tg, parts[0], parts[1] if len(parts) > 1 else "")
        elif kind == "setlabel":
            cid = state["cid"]
            if not text:
                await self._send(update, "❌ Send a name, or the caller's wallet to re-fetch it.")
                return
            await self.db.set_caller_label(tg, cid, text[:32])
            await self._send(update, f"✅ Renamed to <b>{html.escape(text[:32])}</b>.")
        elif kind == "buysize":
            try:
                val = float(text.replace(",", "."))
            except ValueError:
                await self._send(update, "❌ Send a number, e.g. 0.02")
                return
            cid = state["cid"]
            if config.MAX_BUY_SOL > 0 and val > config.MAX_BUY_SOL:
                await self._send(update, f"❌ Max allowed per buy is {config.MAX_BUY_SOL} SOL.")
                return
            if val <= 0:
                await self._send(update, "❌ Buy size must be a positive number.")
                return
            await self.db.set_caller_buy_sol(tg, cid, val)
            await self._send(update, f"✅ Buy size set to <b>{val} SOL</b>.")
        elif kind == "settp":
            cid = state["cid"]
            if text.lower() in ("off", "none", "-1"):
                c = await self.db.get_caller(tg, cid)
                sl = c["stop_multiple"] if c else 0.5
                await self.db.set_caller_tpsl(tg, cid, -1, sl)
                await self._send(update, "✅ Take-profit disabled.")
                return
            try:
                val = float(text.replace(",", "."))
            except ValueError:
                await self._send(update, "❌ Send a number like 2.5, or 'off'.")
                return
            if val <= 1:
                await self._send(update, "❌ TP must be > 1x (that's where profit lives).")
                return
            c = await self.db.get_caller(tg, cid)
            sl = c["stop_multiple"] if c else 0.5
            await self.db.set_caller_tpsl(tg, cid, val, sl)
            await self._send(update, f"✅ Take-profit set to <b>{val:g}x</b>.")
        elif kind == "setsl":
            cid = state["cid"]
            if text.lower() in ("off", "none", "-1"):
                c = await self.db.get_caller(tg, cid)
                tp = c["max_multiple"] if c else 2.0
                await self.db.set_caller_tpsl(tg, cid, tp, -1)
                await self._send(update, "✅ Stop-loss disabled.")
                return
            try:
                val = float(text.replace(",", "."))
            except ValueError:
                await self._send(update, "❌ Send a number like 0.4, or 'off'.")
                return
            if not 0 < val < 1:
                await self._send(update, "❌ SL is a fraction of entry (0.3 = -70%, 0.5 = -50%).")
                return
            c = await self.db.get_caller(tg, cid)
            tp = c["max_multiple"] if c else 2.0
            await self.db.set_caller_tpsl(tg, cid, tp, val)
            await self._send(update, f"✅ Stop-loss set to <b>{val:g}x</b>.")
        elif kind == "setmcap":
            cid = state["cid"]
            parts = text.replace(",", "").replace("$", "").split()
            if len(parts) != 2:
                await self._send(update, "❌ Send two numbers: <code>min max</code> (USD), e.g. <code>5000 100000</code>.")
                return
            try:
                mn, mx = float(parts[0]), float(parts[1])
            except ValueError:
                await self._send(update, "❌ Send two numbers: <code>min max</code>.")
                return
            await self.db.set_caller_mcap(tg, cid, mn, mx)
            if mn <= 0 and mx <= 0:
                await self._send(update, "✅ Mcap filter cleared (buy everything).")
            else:
                range_txt = f"{mn:,.0f} – {mx:,.0f}" if mx > 0 else f"{mn:,.0f} +"
                await self._send(update, f"✅ Mcap filter set: <b>{range_txt}</b> USD.")
        elif kind == "setslip":
            cid = state["cid"]
            try:
                val = float(text.replace(",", ".").replace("%", ""))
            except ValueError:
                await self._send(update, "❌ Send a number like 15 (percent), or 0 for defaults.")
                return
            if val < 0 or val > 50:
                await self._send(update, "❌ Slippage must be 0–50%. 0 = venue defaults.")
                return
            await self.db.set_caller_slippage(tg, cid, val)
            if val == 0:
                await self._send(update, "✅ Slippage reset to venue defaults.")
            else:
                await self._send(update, f"✅ Slippage set to <b>{val:g}%</b> for this caller.")
        elif kind == "setpfee":
            cid = state["cid"]
            if text.lower() in ("off", "none", "-1"):
                await self.db.set_caller_priority_fee(tg, cid, -1)
                await self._send(update, "✅ Priority fee disabled for this caller "
                                         "(base fee only).")
                return
            try:
                val = float(text.replace(",", ".").replace("sol", "").strip())
            except ValueError:
                await self._send(update, "❌ Send a number in SOL, e.g. <code>0.0005</code>, "
                                         "or <code>off</code>.", parse_mode=ParseMode.HTML)
                return
            if val < 0 or val > 0.1:
                await self._send(update, "❌ Priority fee must be 0–0.1 SOL "
                                         "(0 = global default).")
                return
            await self.db.set_caller_priority_fee(tg, cid, val)
            if val == 0:
                await self._send(update, f"✅ Priority fee reset to the global default "
                                         f"(<b>{config.PRIORITY_FEE_SOL:g} SOL</b>).")
            else:
                await self._send(update, f"✅ Priority fee set to <b>{val:g} SOL</b> "
                                         "for this caller.")
        elif kind in ("settrail", "setbe", "setentry"):
            cid = state["cid"]
            if text.lower() in ("off", "none", "0", "-1"):
                val = 0.0
            else:
                try:
                    val = float(text.replace(",", ".").replace("x", "")
                               .replace("%", "").strip())
                except ValueError:
                    await self._send(update, "❌ Send a number like <code>25</code>, "
                                             "or <code>off</code> to disable.",
                                     parse_mode=ParseMode.HTML)
                    return
                if val <= 0:
                    val = 0.0
            if kind == "settrail":
                if val >= 95:
                    await self._send(update, "❌ That's not a trailing stop — use "
                                             "1–94%, or <code>off</code>.",
                                     parse_mode=ParseMode.HTML)
                    return
                await self.db.set_caller_trail(tg, cid, val)
                if val == 0:
                    await self._send(update, "✅ Trailing stop disabled.")
                else:
                    await self._send(
                        update, f"✅ Trailing stop: exit <b>{val:g}%</b> below the "
                                f"peak (arms at {exits.TRAIL_ARM_X:g}x).")
            elif kind == "setbe":
                if val >= 100:
                    await self._send(update, "❌ That's not reachable — use a "
                                             "multiple like 1.5, or <code>off</code>.",
                                     parse_mode=ParseMode.HTML)
                    return
                await self.db.set_caller_breakeven(tg, cid, val)
                if val == 0:
                    await self._send(update, "✅ Breakeven stop disabled.")
                else:
                    await self._send(
                        update, f"✅ Breakeven stop armed at <b>{val:g}x</b> — the "
                                "stop moves to entry once it gets there.")
            else:
                if val >= 100:
                    await self._send(update, "❌ That's not reachable — use a "
                                             "multiple like 1.5, or <code>off</code>.",
                                     parse_mode=ParseMode.HTML)
                    return
                await self.db.set_caller_max_entry_multiple(tg, cid, val)
                if val == 0:
                    await self._send(update, "✅ Already-pumped filter disabled.")
                else:
                    await self._send(
                        update, f"✅ Skipping callouts already up more than "
                                f"<b>{val:g}x</b> since the call.")
        elif kind == "settppct":
            cid = state["cid"]
            try:
                pct = float(text.replace(",", ".").replace("%", ""))
            except ValueError:
                await self._send(update, "❌ Send a number like 50 (percent).")
                return
            if not 0 < pct <= 100:
                await self._send(update, "❌ Percent must be 1–100.")
                return
            await self.db.set_caller_tp_sell_pct(tg, cid, pct)
            await self._send(update, f"✅ TP will sell <b>{pct:g}%</b>" +
                             (" — rest rides with SL active." if pct < 100
                              else " — full exit."))
        elif kind == "sellpct":
            pid = state.get("pid")
            try:
                pct = float(text.replace(",", ".").replace("%", ""))
            except ValueError:
                await self._send(update, "❌ Send a number like 30 (percent).")
                return
            if not 0 < pct <= 100:
                await self._send(update, "❌ Percent must be 1–100.")
                return
            p = await self.db.get_position_by_id(int(pid))
            if not p or p["tg_id"] != tg:
                await self._send(update, "❌ Position not found.")
                return
            await self._sell_position(tg, p, pct)

    # ---------- inline buttons ----------
    async def on_button(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        try:
            await q.answer()
        except Exception:
            pass  # stale query (bot was down) — still process the press
        tg = q.from_user.id
        data = q.data or ""
        if data == "noop":
            return
        if data.startswith("menu:"):
            await self._menu(tg, q, data.split(":", 1)[1])
        elif data == "wallet:import":
            self.awaiting[tg] = {"awaiting": "wallet", "label": "imported"}
            await q.message.reply_text(
                "📥 Send me the wallet <b>private key</b> (base58 or JSON array).\n"
                "It will be encrypted before storage. "
                "Delete this message after import for hygiene.",
                parse_mode=ParseMode.HTML)
        elif data == "wallet:remove":
            await q.message.reply_text(
                "Remove the imported wallet? The key will be deleted from the bot.",
                reply_markup=kb.confirm_remove_wallet())
        elif data == "wallet:removeyes":
            await self.db.delete_wallet(tg)
            if self.poller:
                self.poller.invalidate_wallet(tg)
            await q.message.reply_text("🗑 Wallet removed.")
        elif data == "caller:add":
            self.awaiting[tg] = {"awaiting": "caller"}
            await q.message.reply_text(
                "➕ Send the caller's <b>wallet address or user UUID</b>.\n"
                "I'll name them from their pump.fun profile (add your own name "
                "after the address to override).",
                parse_mode=ParseMode.HTML)
        elif data.startswith("caller:"):
            cid = data.split(":", 1)[1]
            c = await self.db.get_caller(tg, cid)
            if c:
                text = _fmt_caller(c)
                await q.message.reply_text(text, parse_mode=ParseMode.HTML,
                                           reply_markup=kb.caller_detail(c))
        elif data.startswith("pos:"):
            _, action, pid, *rest = data.split(":")
            await self._handle_position(tg, q, action, int(pid),
                                        float(rest[0]) if rest else 0)
        elif data.startswith("callert:"):
            parts = data.split(":")
            _, action, cid = parts[0], parts[1], parts[2]
            if action == "toggle":
                c = await self.db.get_caller(tg, cid)
                if c:
                    await self.db.set_caller_enabled(tg, cid, not c["enabled"])
                    await q.message.reply_text("✅ Updated.")
            elif action == "bysize":
                self.awaiting[tg] = {"awaiting": "buysize", "cid": cid}
                await q.message.reply_text("💰 Send the buy size in SOL (e.g. 0.02):")
            elif action == "settp":
                c = await self.db.get_caller(tg, cid) or {}
                await q.message.reply_text("🎯 Take-profit multiplier:",
                                           reply_markup=kb.tp_options(
                                               cid, float(c.get("tp_sell_pct") or 100)))
            elif action == "setsl":
                await q.message.reply_text("🛑 Stop-loss multiplier:",
                                           reply_markup=kb.sl_options(cid))
            elif action == "tp":
                val = float(parts[3])
                c = await self.db.get_caller(tg, cid)
                sl = c["stop_multiple"] if c else 0.5
                await self.db.set_caller_tpsl(tg, cid, val, sl)
                await q.message.reply_text(
                    f"✅ TP set to {val:g}x" if val > 0 else "✅ TP disabled. "
                    f"(SL stays {sl:g}x — see buttons)")
            elif action == "sl":
                val = float(parts[3])
                c = await self.db.get_caller(tg, cid)
                tp = c["max_multiple"] if c else 2.0
                await self.db.set_caller_tpsl(tg, cid, tp, val)
                await q.message.reply_text(
                    f"✅ SL set to {val:g}x" if val > 0 else "✅ SL disabled. "
                    f"(TP stays {tp:g}x — see buttons)")
            elif action == "tpcustom":
                self.awaiting[tg] = {"awaiting": "settp", "cid": cid}
                await q.message.reply_text("🎯 Send TP as a multiple (e.g. 2.5, or 'off'):")
            elif action == "slcustom":
                self.awaiting[tg] = {"awaiting": "setsl", "cid": cid}
                await q.message.reply_text("🛑 Send SL as a multiple (e.g. 0.4, or 'off'):")
            elif action == "tppctcustom":
                self.awaiting[tg] = {"awaiting": "settppct", "cid": cid}
                await q.message.reply_text(
                    "💸 Send the % of the position TP should sell (1–100):")
            elif action == "setmcap":
                self.awaiting[tg] = {"awaiting": "setmcap", "cid": cid}
                await q.message.reply_text(
                    "📊 Send the market-cap range in USD: <code>min max</code>\n"
                    "e.g. <code>5000 100000</code> — or <code>0 0</code> to clear.",
                    parse_mode=ParseMode.HTML)
            elif action == "settppct":
                await q.message.reply_text(
                    "💸 When TP hits, sell what % of the position?\n"
                    "(remainder keeps riding with SL active)",
                    reply_markup=kb.tp_sellpct_options(cid))
            elif action == "tppct":
                val = float(parts[3])
                await self.db.set_caller_tp_sell_pct(tg, cid, val)
                await q.message.reply_text(
                    f"✅ TP will sell <b>{val:g}%</b>" +
                    (" — rest rides with SL active." if val < 100
                     else " — full exit."))
            elif action == "setslip":
                self.awaiting[tg] = {"awaiting": "setslip", "cid": cid}
                await q.message.reply_text(
                    "💧 Send max slippage as a percent (e.g. <code>15</code> = 15%).\n"
                    "Applies to buys and sells for this caller.\n"
                    "Send <code>0</code> to use venue defaults (pump.fun 10%, Jupiter 1%/3%).",
                    parse_mode=ParseMode.HTML)
            elif action == "settrail":
                self.awaiting[tg] = {"awaiting": "settrail", "cid": cid}
                await q.message.reply_text(
                    "📉 <b>Trailing stop</b> — send how far below the peak to exit, "
                    "as a percent (e.g. <code>25</code> = give back 25% from the high).\n"
                    f"Arms once the position reaches {exits.TRAIL_ARM_X:g}x.\n"
                    "<code>0</code> or <code>off</code> disables it.",
                    parse_mode=ParseMode.HTML)
            elif action == "setbe":
                self.awaiting[tg] = {"awaiting": "setbe", "cid": cid}
                await q.message.reply_text(
                    "🛡 <b>Breakeven stop</b> — send the multiple that arms it "
                    "(e.g. <code>1.5</code>): once the position hits 1.5x, the stop "
                    "moves up to entry so it can't turn into a loser.\n"
                    "<code>0</code> or <code>off</code> disables it.",
                    parse_mode=ParseMode.HTML)
            elif action == "setentry":
                self.awaiting[tg] = {"awaiting": "setentry", "cid": cid}
                await q.message.reply_text(
                    "🚀 <b>Already-pumped filter</b> — skip callouts the coin has "
                    "already run up. Send the max multiple since the call "
                    "(e.g. <code>1.5</code> = don't buy if it's already +50%).\n"
                    "<code>0</code> or <code>off</code> disables it.",
                    parse_mode=ParseMode.HTML)
            elif action == "setpfee":
                self.awaiting[tg] = {"awaiting": "setpfee", "cid": cid}
                await q.message.reply_text(
                    "⛽ Send the <b>priority fee</b> in SOL (e.g. <code>0.0005</code>).\n"
                    "A larger fee jumps the queue so buys land ahead of the crowd.\n"
                    f"<code>0</code> = global default ({config.PRIORITY_FEE_SOL:g} SOL) "
                    "· <code>off</code> = no priority fee.",
                    parse_mode=ParseMode.HTML)
            elif action == "setlabel":
                self.awaiting[tg] = {"awaiting": "setlabel", "cid": cid}
                await q.message.reply_text(
                    "✏️ Send a name for this caller (or their wallet to re-fetch it "
                    "from pump.fun):")
            elif action == "autoname":
                profile = await self.callouts.caller_profile(cid)
                name = (profile.get("username") or profile.get("bio") or "")[:32]
                if not name:
                    await q.message.reply_text(
                        "🤷 No pump.fun profile found for that wallet — "
                        "use ✏️ Name to set it yourself.")
                    return
                await self.db.set_caller_label(tg, cid, name)
                await q.message.reply_text(
                    f"✅ Renamed to <b>{html.escape(name)}</b> from their pump.fun profile.",
                    parse_mode=ParseMode.HTML)
            elif action == "remove":
                await q.message.reply_text("Remove this caller?",
                                           reply_markup=kb.confirm_remove_caller(cid))
            elif action == "removeyes":
                # feed mode: unfollow from the bot account (best effort)
                if self.callouts.bot_uuid:
                    try:
                        cuuid = await self.callouts.resolve_uuid(cid)
                        await self.callouts.unfollow(cuuid)
                    except Exception:
                        pass
                await self.db.remove_caller(tg, cid)
                await q.message.reply_text("🗑 Caller removed.")

    async def _menu(self, tg: int, q, menu: str):
        if menu == "main":
            has_wallet = bool(await self.db.wallet_pubkey(tg))
            callers = await self.db.get_callers(tg)
            bal_txt = await self._balance_line(tg, has_wallet)
            await q.message.reply_text(
                f"<b>ZANE</b> ⚡️\n\n{bal_txt}\n"
                f"Callers followed: <b>{len(callers)}</b>",
                parse_mode=ParseMode.HTML,
                reply_markup=kb.main_menu(has_wallet, len(callers)))
        elif menu == "callers":
            callers = await self.db.get_callers(tg)
            await q.message.reply_text("📣 Callers",
                                       reply_markup=kb.callers_menu(callers))
        elif menu == "wallet":
            pubkey = await self.db.wallet_pubkey(tg)
            if pubkey:
                rows = await self.db.load_wallet(tg)
                label = rows[0] if rows else ""
                await q.message.reply_text(f"💼 <code>{pubkey}</code>",
                                           parse_mode=ParseMode.HTML,
                                           reply_markup=kb.wallet_menu(True, label, pubkey))
            else:
                await q.message.reply_text("💼 No wallet.",
                                           reply_markup=kb.wallet_menu(False))
        elif menu == "positions":
            positions = await self.db.get_open_positions(tg)
            if not positions:
                await q.message.reply_text("📊 No open positions.")
            else:
                await q.message.reply_text("📊 <b>Open positions</b> — tap to manage:",
                                           parse_mode=ParseMode.HTML,
                                           reply_markup=kb.positions_menu(positions))
        elif menu == "help":
            await q.message.reply_text(HELP_TEXT, parse_mode=ParseMode.HTML)

    # ---------- global error guard ----------
    async def on_error(self, update: object, ctx: ContextTypes.DEFAULT_TYPE):
        log.exception("unhandled error in update %s", update)
        try:
            if isinstance(update, Update) and update.effective_chat:
                await ctx.bot.send_message(
                    update.effective_chat.id,
                    "⚠️ Internal error — the incident was logged. Try again.")
        except Exception:
            pass

    # ---------- subscription gate (ZANE is paid) ----------
    FREE_COMMANDS = {"unlock", "help"}
    PAY_PAYLOAD = "zane_unlock"

    def _pay_keyboard(self):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(f"⭐ Pay {config.SUB_PRICE_STARS} Stars",
                                 callback_data="pay:stars"),
            InlineKeyboardButton("🔑 I have a code", callback_data="pay:code"),
        ]])

    async def gate(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Group -1 intercept: everything except the payment flow itself is
        locked until the user has a subscription (or is the owner)."""
        if update.channel_post or update.edited_channel_post:
            return
        # never intercept payment machinery — locked users must be able to pay
        if update.pre_checkout_query or (
                update.message and update.message.successful_payment):
            return
        user = update.effective_user
        if user is None or user.id == config.BOT_OWNER_ID:
            return
        if await self.db.is_unlocked(user.id):
            return
        # payment buttons must work for locked users — handle + stop here
        if update.callback_query and (update.callback_query.data or "").startswith("pay:"):
            await self.on_pay_button(update, ctx)
            raise ApplicationHandlerStop
        msg = update.effective_message
        text = (msg.text or "") if msg else ""
        is_group = update.effective_chat and update.effective_chat.type != "private"
        if text.startswith("/"):
            cmd = text[1:].split("@")[0].split()[0].lower()
            if cmd in self.FREE_COMMANDS:
                return
        # in groups: stay quiet — only answer explicit commands, max once
        # per minute per chat (never ride on random conversation)
        if is_group and not text.startswith("/"):
            raise ApplicationHandlerStop
        if is_group:
            now = time.time()
            chat_id = update.effective_chat.id
            if now - self._group_gate_ts.get(chat_id, 0) < 60:
                raise ApplicationHandlerStop
            self._group_gate_ts[chat_id] = now
        try:
            if msg:
                await msg.reply_text(
                    "🔒 <b>ZANE is subscription-based.</b>\n"
                    "One-time $3 (in Stars) — lifetime access to every feature:\n"
                    "copy-trading, positions, exits and alerts.\n\n"
                    "Got a code? <code>/unlock YOUR-CODE</code>",
                    parse_mode=ParseMode.HTML,
                    reply_markup=self._pay_keyboard())
        except Exception:
            log.exception("gate reply failed")
        raise ApplicationHandlerStop

    async def on_pay_button(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        try:
            await q.answer()
        except Exception:
            pass
        if q.data == "pay:code":
            try:
                await q.message.reply_text("🔑 Send: <code>/unlock YOUR-CODE</code>",
                                           parse_mode=ParseMode.HTML)
            except Exception:
                pass
            return
        try:
            await ctx.bot.send_invoice(
                q.from_user.id,
                title="ZANE ⚡️ — lifetime access",
                description="One-time unlock: copy-trading, exits, alerts.",
                payload=self.PAY_PAYLOAD,
                currency="XTR",
                prices=[LabeledPrice("ZANE unlock", config.SUB_PRICE_STARS)],
            )
        except Exception:
            log.exception("send_invoice failed — Stars payments may be disabled "
                          "on this bot (enable via @BotFather → Payments)")
            try:
                await q.message.reply_text(
                    "⚠️ Star payments aren't enabled on this bot yet. "
                    "Use <code>/unlock CODE</code> for now.", parse_mode=ParseMode.HTML)
            except Exception:
                pass

    async def on_precheckout(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        q = update.pre_checkout_query
        ok = q.invoice_payload == self.PAY_PAYLOAD
        await q.answer(ok=ok,
                       error_message="Expired — tap Pay again." if not ok else None)

    async def on_successful_payment(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        sp = update.message.successful_payment
        tg = update.effective_user.id
        await self.db.ensure_user(tg)
        await self.db.unlock_via_stars(tg, sp.telegram_payment_charge_id)
        log.info("stars payment tg=%s charge=%s amount=%s %s",
                 tg, sp.telegram_payment_charge_id, sp.total_amount, sp.currency)
        await update.message.reply_text(
            "🎉 <b>Payment received — ZANE unlocked!</b>\n/start to begin.",
            parse_mode=ParseMode.HTML)
        try:
            await ctx.bot.send_message(
                config.BOT_OWNER_ID,
                f"💰 New subscription: <code>{tg}</code> paid "
                f"{sp.total_amount} {sp.currency}")
        except Exception:
            pass

    async def cmd_unlock(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        tg = update.effective_user.id
        if await self.db.is_unlocked(tg):
            await self._send(update, "✅ Already unlocked — enjoy!")
            return
        code = ((ctx.args or [""])[0] if ctx.args else "").strip().upper()
        if not code:
            await self._send(update,
                             "🔑 Usage: <code>/unlock YOUR-CODE</code>",
                             reply_markup=self._pay_keyboard())
            return
        # brute-force guard: max 5 wrong codes per 10 minutes per user
        now = time.time()
        n, wstart = self._unlock_attempts.get(tg, (0, now))
        if now - wstart > 600:
            n, wstart = 0, now
        if n >= 5:
            await self._send(update, "⏳ Too many attempts — try again in ~10 minutes.")
            return
        if await self.db.redeem_access_code(code, tg):
            self._unlock_attempts.pop(tg, None)
            await self._send(update, "🎉 <b>Unlocked!</b> All features are live — /start to begin.")
        else:
            self._unlock_attempts[tg] = (n + 1, wstart)
            await self._send(update,
                             "❌ Invalid or already-used code.\n"
                             "Check it and retry, or pay below:",
                             reply_markup=self._pay_keyboard())

    async def cmd_gencode(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != config.BOT_OWNER_ID:
            return  # silently ignore strangers
        try:
            n = max(1, min(20, int(ctx.args[0]))) if ctx.args else 1
        except (ValueError, IndexError):
            n = 1
        codes = []
        for _ in range(n):
            code = "ZANE-" + secrets.token_hex(8).upper()
            if await self.db.create_access_code(code, update.effective_user.id):
                codes.append(code)
        body = "\n".join(f"<code>{c}</code>" for c in codes) or "(none created)"
        await self._send(update, f"🎟 <b>Access codes</b> (one-time):\n{body}")

    async def cmd_subs(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != config.BOT_OWNER_ID:
            return
        subs = await self.db.list_subs()
        codes = await self.db.list_codes(12)
        lines = [f"<b>Active subscriptions</b>: {len(subs)}", ""]
        for s in subs[:10]:
            when = time.strftime("%Y-%m-%d", time.gmtime(s["unlocked_at"]))
            lines.append(f"• <code>{s['tg_id']}</code> · {when} · <code>{html.escape(s['access_code'][:18])}</code>")
        lines += ["", "<b>Recent codes</b>:"]
        for c in codes:
            mark = "✅" if c["used_by"] else "🟢"
            lines.append(f"{mark} <code>{html.escape(c['code'])}</code>")
        await self._send(update, "\n".join(lines))

    # ---------- registration ----------
    def register(self, app: Application):
        self.bind_app(app)
        app.add_error_handler(self.on_error)
        # paywall: runs before every other handler group
        app.add_handler(TypeHandler(Update, callback=self.gate), group=-1)
        app.add_handler(CommandHandler("unlock", self.cmd_unlock))
        app.add_handler(CommandHandler("gencode", self.cmd_gencode))
        app.add_handler(CommandHandler("subs", self.cmd_subs))
        app.add_handler(CallbackQueryHandler(self.on_pay_button, pattern=r"^pay:"))
        app.add_handler(PreCheckoutQueryHandler(self.on_precheckout))
        app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT,
                                       self.on_successful_payment))
        app.add_handler(CommandHandler(["start", "menu"], self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("wallet", self.cmd_wallet))
        app.add_handler(CommandHandler("addcaller", self.cmd_addcaller))
        app.add_handler(CommandHandler("callers", self.cmd_callers))
        app.add_handler(CommandHandler("buy", self.cmd_buy))
        app.add_handler(CommandHandler("sell", self.cmd_sell))
        app.add_handler(CommandHandler("positions", self.cmd_positions))
        app.add_handler(CommandHandler("balance", self.cmd_balance))
        app.add_handler(CommandHandler("stats", self.cmd_stats))
        app.add_handler(CallbackQueryHandler(self.on_button))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_text))
