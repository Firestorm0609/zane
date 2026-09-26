# 📣 Callout Copy-Bot (pump.fun)

Telegram bot that watches **pump.fun Callouts** and instantly buys any new
Solana token that a caller you follow posts. Wallet keys are imported in-chat,
encrypted at rest (AES-256-GCM), and used **only locally** to sign trades —
the full transaction is built by PumpPortal `trade-local` and sent via your
own RPC.

## How it works

```
pump.fun callout API ──▶ poller (every ~1.5s) ──▶ new Solana mint?
                                        │
                                        ▼
                          for each follower with a wallet:
                          PumpPortal trade-local → sign locally → send via your RPC
                                        │
                                        ▼
                              position saved → Telegram alert
```

- **Callout source** (no auth required): `GET https://frontend-api-v3.pump.fun/callout/list/{callerId}?limit=10&sortBy=TIMESTAMP&sortOrder=desc`
- **Feed mode (fast polling, ~2s):** pump.fun signs in with Privy, so the bot needs
  a Privy identity token from a logged-in browser. Easiest: install
  `pump_forwarder.user.js` in Tampermonkey and keep a pump.fun tab open — it forwards
  the token to the bot's ingest port (8766) automatically. Manual alternative:
  `python3 pumpfarm.py --set-token "eyJ…"` (grab the Bearer value from DevTools).
  Verify with `python3 pumpfarm.py --check`. Without a token the bot falls back to
  per-caller polling (15s floor).
- **Execution**: `POST https://pumpportal.fun/api/trade-local` returns an *unsigned* transaction; the bot signs it locally with your imported key and sends it through `SOLANA_RPC`.
- **EVM callouts are skipped** (callers sometimes call 0x… tokens on other chains).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env:
#   TELEGRAM_BOT_TOKEN  — from @BotFather
#   WALLET_ENC_KEY      — python -c "import os; print(os.urandom(32).hex())"
#   SOLANA_RPC          — use a private RPC (Helius/QuickNode); public one is heavily rate-limited

python main.py
```

## Using the bot

1. `/start` → **💼 Wallet → Import** → paste your base58 private key (Phantom export).
   The message is stored encrypted; delete your chat message afterwards for hygiene.
2. **📣 Callers → Add** (or `/addcaller <wallet> [label]`) → paste the caller's
   wallet or pump.fun user UUID. The bot shows their recent win-rate before following.
3. New callouts arrive as alerts with a one-tap **Buy** button when auto-buy is on
   (or use `/buy <mint> <sol>` manually).
4. `/positions` lists open trades, `/sell <mint> [pct]` exits.

### Slash commands

| Command | Description |
|---|---|
| `/start`, `/menu` | Main inline menu |
| `/help` | Full help |
| `/wallet` | Import / view / remove wallet |
| `/balance` | SOL balance |
| `/addcaller <id> [label]` | Follow a caller |
| `/callers` | Manage callers (pause, buy size, remove) |
| `/stats <caller>` | Win-rate over recent callouts |
| `/buy <mint> [sol]` | Manual buy (Solana mints only) |
| `/sell <mint> [pct]` | Manual sell (default 100%) |
| `/positions` | Open positions |

## Safety rails

- `MAX_BUY_SOL` optional hard cap per buy — `0` (default) = no cap.
- `MIN_SOL_LEFT` reserve so the wallet never drains below fee money.
- Callouts older than **60s** are alerted but not bought (no chasing stale calls).
- Keys encrypted with AES-256-GCM, decrypted only in-memory during signing.
- EVM mints (`0x…`) are never attempted.

## ⚠️ Disclaimer

Memecoin copy-trading is extremely high-risk. Callouts are not financial advice,
many are outright rugs, and you are responsible for your own keys and trades.
Use burner wallets and small sizes. This software is for educational purposes.
