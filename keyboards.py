"""Inline keyboard builders for the Telegram UI."""
from typing import Any, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def main_menu(has_wallet: bool, n_callers: int) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("📣 Callers", callback_data="menu:callers"),
         InlineKeyboardButton("💼 Wallet", callback_data="menu:wallet")],
        [InlineKeyboardButton("📊 Positions", callback_data="menu:positions")],
        [InlineKeyboardButton("❓ Help", callback_data="menu:help")],
    ]
    return InlineKeyboardMarkup(rows)


def wallet_menu(has_wallet: bool, label: str = "", pubkey: str = "") -> InlineKeyboardMarkup:
    rows = []
    if has_wallet:
        rows.append([InlineKeyboardButton(
            f"✅ {label} · {pubkey[:6]}…{pubkey[-4:]}",
            callback_data="noop")])
        rows.append([InlineKeyboardButton("🗑 Remove wallet", callback_data="wallet:remove")])
    else:
        rows.append([InlineKeyboardButton("📥 Import wallet", callback_data="wallet:import")])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu:main")])
    return InlineKeyboardMarkup(rows)


def callers_menu(callers: list[dict[str, Any]]) -> InlineKeyboardMarkup:
    rows = []
    for c in callers:
        state = "🟢" if c["enabled"] else "⏸"
        label = c["label"] or c["caller_id"][:8] + "…"
        rows.append([InlineKeyboardButton(
            f"{state} {label} · {c['buy_sol']} SOL",
            callback_data=f"caller:{c['caller_id']}")])
    rows.append([InlineKeyboardButton("➕ Add caller", callback_data="caller:add")])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu:main")])
    return InlineKeyboardMarkup(rows)


def caller_detail(c: dict[str, Any]) -> InlineKeyboardMarkup:
    cid = c["caller_id"]
    toggle = "⏸ Pause" if c["enabled"] else "▶️ Enable"
    tp = c.get("max_multiple") or 0
    sl = c.get("stop_multiple") or 0
    tp_txt = f"{tp:g}x" if tp > 0 else ("off" if tp < 0 else "2x (def)")
    sl_txt = f"{sl:g}x" if sl > 0 else ("off" if sl < 0 else "0.5x (def)")
    min_mc = c.get("min_mcap") or 0
    max_mc = c.get("max_mcap") or 0
    mc_txt = f"{min_mc:,.0f}" if min_mc else "any"
    mc_txt += f"–{max_mc:,.0f}" if max_mc else "+"
    slip = c.get("slippage") or 0
    slip_txt = f"{slip:g}%" if slip > 0 else "def"
    pf = c.get("priority_fee") or 0
    pf_txt = "off" if pf < 0 else (f"{pf:g} SOL" if pf > 0 else "def")
    trail = c.get("trail_pct") or 0
    trail_txt = f"{trail:g}%" if trail > 0 else "off"
    be = c.get("be_multiple") or 0
    be_txt = f"{be:g}x" if be > 0 else "off"
    entry = c.get("max_entry_multiple") or 0
    entry_txt = f"≤{entry:g}x" if entry > 0 else "off"
    rows = [
        [InlineKeyboardButton(toggle, callback_data=f"callert:toggle:{cid}")],
        [InlineKeyboardButton("✏️ Name", callback_data=f"callert:setlabel:{cid}"),
         InlineKeyboardButton("🔄 From pump.fun", callback_data=f"callert:autoname:{cid}")],
        [InlineKeyboardButton("💰 Buy size", callback_data=f"callert:bysize:{cid}")],
        [InlineKeyboardButton(f"🎯 TP {tp_txt}", callback_data=f"callert:settp:{cid}"),
         InlineKeyboardButton(f"🛑 SL {sl_txt}", callback_data=f"callert:setsl:{cid}")],
        [InlineKeyboardButton(f"📉 Trail {trail_txt}", callback_data=f"callert:settrail:{cid}"),
         InlineKeyboardButton(f"🛡 BE {be_txt}", callback_data=f"callert:setbe:{cid}")],
        [InlineKeyboardButton(f"📊 mcap {mc_txt}", callback_data=f"callert:setmcap:{cid}"),
         InlineKeyboardButton(f"🚀 entry {entry_txt}",
                              callback_data=f"callert:setentry:{cid}")],
        [InlineKeyboardButton(f"💧 slip {slip_txt}", callback_data=f"callert:setslip:{cid}"),
         InlineKeyboardButton(f"⛽ fee {pf_txt}", callback_data=f"callert:setpfee:{cid}")],
        [InlineKeyboardButton("🚫 Remove", callback_data=f"callert:remove:{cid}")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:callers")],
    ]
    return InlineKeyboardMarkup(rows)


def tp_options(cid: str, tp_sell_pct: float = 100) -> InlineKeyboardMarkup:
    pct_txt = f"{tp_sell_pct:g}%" if tp_sell_pct < 100 else "100% (def)"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1.5x", callback_data=f"callert:tp:{cid}:1.5"),
         InlineKeyboardButton("2x", callback_data=f"callert:tp:{cid}:2"),
         InlineKeyboardButton("3x", callback_data=f"callert:tp:{cid}:3")],
        [InlineKeyboardButton("5x", callback_data=f"callert:tp:{cid}:5"),
         InlineKeyboardButton("10x", callback_data=f"callert:tp:{cid}:10"),
         InlineKeyboardButton("✏️ custom", callback_data=f"callert:tpcustom:{cid}")],
        [InlineKeyboardButton("🚫 Off", callback_data=f"callert:tp:{cid}:-1")],
        [InlineKeyboardButton(f"💸 TP sells {pct_txt}",
                              callback_data=f"callert:settppct:{cid}")],
        [InlineKeyboardButton("⬅️ Back", callback_data=f"caller:{cid}")],
    ])


def tp_sellpct_options(cid: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("25%", callback_data=f"callert:tppct:{cid}:25"),
         InlineKeyboardButton("50%", callback_data=f"callert:tppct:{cid}:50")],
        [InlineKeyboardButton("75%", callback_data=f"callert:tppct:{cid}:75"),
         InlineKeyboardButton("💰 100%", callback_data=f"callert:tppct:{cid}:100")],
        [InlineKeyboardButton("✏️ custom", callback_data=f"callert:tppctcustom:{cid}")],
        [InlineKeyboardButton("⬅️ Back", callback_data=f"callert:settp:{cid}")],
    ])


def sl_options(cid: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("0.3x (-70%)", callback_data=f"callert:sl:{cid}:0.3"),
         InlineKeyboardButton("0.5x (-50%)", callback_data=f"callert:sl:{cid}:0.5")],
        [InlineKeyboardButton("0.7x (-30%)", callback_data=f"callert:sl:{cid}:0.7"),
         InlineKeyboardButton("✏️ custom", callback_data=f"callert:slcustom:{cid}")],
        [InlineKeyboardButton("🚫 Off", callback_data=f"callert:sl:{cid}:-1")],
        [InlineKeyboardButton("⬅️ Back", callback_data=f"caller:{cid}")],
    ])


def positions_menu(positions: list[dict[str, Any]]) -> InlineKeyboardMarkup:
    rows = []
    for p in positions:
        rows.append([InlineKeyboardButton(
            f"📈 {p['mint'][:10]}… · {p['buy_amount_sol']:g} SOL",
            callback_data=f"pos:view:{p['id']}")])
    rows.append([InlineKeyboardButton("🔄 Refresh", callback_data="menu:positions")])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu:main")])
    return InlineKeyboardMarkup(rows)


def position_detail(p: dict[str, Any]) -> InlineKeyboardMarkup:
    pid = p["id"]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("10%", callback_data=f"pos:sell:{pid}:10"),
         InlineKeyboardButton("25%", callback_data=f"pos:sell:{pid}:25"),
         InlineKeyboardButton("50%", callback_data=f"pos:sell:{pid}:50")],
        [InlineKeyboardButton("75%", callback_data=f"pos:sell:{pid}:75"),
         InlineKeyboardButton("💰 100%", callback_data=f"pos:sell:{pid}:100")],
        [InlineKeyboardButton("✏️ Custom %", callback_data=f"pos:sellpct:{pid}")],
        [InlineKeyboardButton("🔄 Refresh", callback_data=f"pos:view:{pid}")],
        [InlineKeyboardButton("⬅️ Positions", callback_data="menu:positions")],
    ])


def confirm_remove_caller(cid: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Yes, remove", callback_data=f"callert:removeyes:{cid}"),
         InlineKeyboardButton("❌ Cancel", callback_data="menu:callers")],
    ])


def confirm_remove_wallet() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Yes, remove", callback_data="wallet:removeyes"),
         InlineKeyboardButton("❌ Cancel", callback_data="menu:wallet")],
    ])
