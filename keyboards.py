"""All inline keyboard builders."""
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from .utils import mdbold

MENU_HEADER = f"🤖 {mdbold('Pump.fun Monitor v1.1')} — tap an action:"


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🟢 Alerts ON",  callback_data="monitor_on"),
            InlineKeyboardButton("🔴 Alerts OFF", callback_data="monitor_off"),
            InlineKeyboardButton("ℹ️ Status",     callback_data="monitor_status"),
        ],
        [
            InlineKeyboardButton("🎚 Threshold", callback_data="threshold_menu"),
            InlineKeyboardButton("📊 Scoring",   callback_data="scoring_mode"),
            InlineKeyboardButton("🧬 Features",  callback_data="features"),
        ],
        [
            InlineKeyboardButton("🔤 Keywords", callback_data="keywords"),
            InlineKeyboardButton("📈 Market",   callback_data="market"),
            InlineKeyboardButton("📤 Outcomes", callback_data="outcomes"),
        ],
        [
            InlineKeyboardButton("🤖 Model",    callback_data="model"),
            InlineKeyboardButton("🏋 Train",    callback_data="train"),
            InlineKeyboardButton("🗒 Snapshot", callback_data="snapshot"),
        ],
        [
            InlineKeyboardButton("📊 Stats",  callback_data="stats"),
            InlineKeyboardButton("💰 Wallet", callback_data="wallet"),
            InlineKeyboardButton("🏆 Top",    callback_data="top"),
        ],
        [
            InlineKeyboardButton("✅ Paper ON",  callback_data="paper_on"),
            InlineKeyboardButton("❌ Paper OFF", callback_data="paper_off"),
        ],
        [
            InlineKeyboardButton("📋 Paper Stats",  callback_data="paper_status"),
            InlineKeyboardButton("📑 Paper Report", callback_data="paper_report"),
        ],
        [InlineKeyboardButton("✖ Close", callback_data="close_menu")],
    ])


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🔙 Menu", callback_data="menu"),
    ]])


def threshold_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(str(i), callback_data=f"set_threshold_{i}")
         for i in range(1, 6)],
        [InlineKeyboardButton(str(i), callback_data=f"set_threshold_{i}")
         for i in range(6, 11)],
        [InlineKeyboardButton("🔙 Back", callback_data="menu")],
    ])
