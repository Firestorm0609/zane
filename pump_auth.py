"""pump.fun auth (Privy era): token capture + freshness tracking.

pump.fun switched from Firebase to Privy (auth.privy.io) for sign-in, so
there is no refresh-token exchange to automate. Instead, the Privy identity
token from a logged-in pump.fun browser session is forwarded to the bot by
pump_forwarder.user.js. The bot
attaches it to every frontend-api call; feed mode (~2s polls) works as long
as the token is accepted. When pump.fun starts returning 401 the token is
flagged rejected and a fresh one must arrive from the browser tab (usually
just reopening/refreshing pump.fun re-mints it and the userscript forwards
it automatically).

File format (.pump-auth.json): {"token": "...", "captured_at": unix_ts}
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import os
import time
from typing import Any, Optional

import aiohttp
from aiohttp import web

import config

log = logging.getLogger("pump_auth")

_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Ingest-Key",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


class PumpAuthError(RuntimeError):
    pass


def _looks_like_jwt(t: str) -> bool:
    return isinstance(t, str) and t.startswith("eyJ") and len(t) > 100


class PumpAuth:
    """Holds the current pump.fun (Privy) identity token.

    Interface consumed by CalloutClient: get_token() / notify_401().
    """

    def __init__(self, auth_file: str = None):
        self.auth_file = auth_file or config.PUMP_AUTH_FILE
        self.token: Optional[str] = None
        self.captured_at: float = 0.0
        self.rejected_token: Optional[str] = None  # last token that got a 401
        self.last_error: str = ""
        self.ingests: int = 0
        self._lock = asyncio.Lock()
        # optional async callback fired when a fresh usable token arrives —
        # used by main to activate feed mode without a restart
        self.on_token = None
        # optional async fn(caller_id, data) -> dict, used by main to route
        # relayed callout pages into the poller (Poller.ingest_external)
        self.callout_sink = None
        # last self-report from the relay script, so a phone (which has no
        # developer console) can be debugged from the server side
        self.relay_status: dict = {}
        self.relay_status_at: float = 0.0

    # ---------- persistence ----------

    def load(self) -> bool:
        try:
            with open(self.auth_file) as f:
                data = json.load(f)
        except FileNotFoundError:
            return False
        except Exception as e:
            log.warning("cannot read %s: %s", self.auth_file, e)
            return False
        tok = data.get("token") or ""
        if not _looks_like_jwt(tok):
            # legacy/garbage file — ignore it
            return False
        self.token = tok
        try:
            self.captured_at = float(data.get("captured_at") or 0)
        except (TypeError, ValueError):
            self.captured_at = 0.0
        return True

    def save(self):
        tmp = self.auth_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"token": self.token or "",
                       "captured_at": self.captured_at,
                       "updated_at": int(time.time())}, f, indent=2)
        os.replace(tmp, self.auth_file)
        try:
            os.chmod(self.auth_file, 0o600)
        except OSError:
            pass

    # ---------- token access (CalloutClient interface) ----------

    async def get_token(self, force_refresh: bool = False) -> Optional[str]:
        """Current token, or None if none captured / the only one we have
        was rejected by the API."""
        async with self._lock:
            if self.token and self.token != self.rejected_token:
                return self.token
            return None

    async def notify_401(self):
        """API rejected the token we're sending — stop sending it until a
        new one is ingested."""
        async with self._lock:
            if self.token and self.token != self.rejected_token:
                log.warning("pump.fun token rejected (401) — waiting for a "
                            "fresh one from the browser tab")
                self.rejected_token = self.token
                self.last_error = "token rejected — refresh the pump.fun tab"

    # ---------- ingest (userscript target) ----------

    def ingest(self, data: dict) -> dict:
        """Accept {token: <privy JWT>} from pump_forwarder.user.js."""
        if not isinstance(data, dict):
            return {"ok": False, "error": "bad payload"}
        tok = (data.get("token") or data.get("jwt") or "").strip()
        if not _looks_like_jwt(tok):
            return {"ok": False, "error": "no usable token in payload"}
        if tok == self.token:
            self.captured_at = self.captured_at or time.time()
            return {"ok": True, "changed": False}
        self.token = tok
        self.captured_at = time.time()
        self.rejected_token = None  # fresh token clears a rejection
        self.last_error = ""
        self.ingests += 1
        self.save()
        log.info("pump.fun token captured via ingest (#%d, %d chars)",
                 self.ingests, len(tok))
        if self.on_token is not None:
            try:
                asyncio.get_running_loop().create_task(self._fire_on_token())
            except RuntimeError:
                pass  # no loop (CLI use) — caller handles activation itself
        return {"ok": True, "changed": True}

    # ---------- callout relay (userscript target) ----------
    # Set by main.py to Poller.ingest_external. The userscript fetches callout
    # pages from the user's own IP (this host's IP is rate-limited to a couple
    # of requests a minute) and POSTs them to /callouts.

    async def ingest_callouts(self, caller_id: str, data: dict) -> dict:
        if self.callout_sink is None:
            return {"ok": False, "error": "callout relay not wired up"}
        return await self.callout_sink(caller_id, data)

    async def _fire_on_token(self):
        try:
            await self.on_token()
        except Exception:
            log.exception("on_token callback failed")

    def _make_app(self) -> web.Application:
        async def options(_req):
            return web.Response(status=204, headers=_CORS)

        async def post(req):
            if config.PUMP_INGEST_KEY:
                key = req.headers.get("X-Ingest-Key", "")
                import hmac as _hmac
                if not _hmac.compare_digest(key, config.PUMP_INGEST_KEY):
                    return web.json_response({"ok": False, "error": "bad key"},
                                             status=403, headers=_CORS)
            try:
                data = await req.json()
            except Exception:
                return web.json_response({"ok": False, "error": "bad json"},
                                         status=400, headers=_CORS)
            return web.json_response(self.ingest(data), headers=_CORS)

        async def post_callouts(req):
            """Accept {caller_id, callouts:[...]} from pump_forwarder.user.js.

            Body is the raw /callout/list response with caller_id added, so the
            poller's normal ingest path can consume it unchanged.
            """
            if config.PUMP_INGEST_KEY:
                key = req.headers.get("X-Ingest-Key", "")
                import hmac as _hmac
                if not _hmac.compare_digest(key, config.PUMP_INGEST_KEY):
                    return web.json_response({"ok": False, "error": "bad key"},
                                             status=403, headers=_CORS)
            try:
                data = await req.json()
            except Exception:
                return web.json_response({"ok": False, "error": "bad json"},
                                         status=400, headers=_CORS)
            if not isinstance(data, dict):
                return web.json_response({"ok": False, "error": "bad payload"},
                                         status=400, headers=_CORS)
            # Caller normally rides in the body, but the Termux relay pipes the
            # raw API response straight through and so passes it in the URL
            # instead: POST /callouts?caller=<wallet> --data-binary @-
            caller_id = (str(data.get("caller_id") or "").strip()
                         or str(req.query.get("caller") or "").strip()
                         or req.headers.get("X-Caller-Id", "").strip())
            try:
                result = await self.ingest_callouts(caller_id, data)
            except Exception as e:
                log.exception("callout relay ingest failed")
                result = {"ok": False, "error": str(e)}
            return web.json_response(result, headers=_CORS)

        def _file_route(name: str, content_type: str):
            """Serve a client script over the same tunnel, so a phone can pull
            it down without hand-copying files. Read-only; leaks nothing the
            client doesn't already have."""
            async def handler(_req):
                path = Path(__file__).with_name(name)
                try:
                    body = path.read_text(encoding="utf-8")
                except OSError as e:
                    return web.Response(status=404, text=f"{name}: {e}",
                                        headers=_CORS)
                return web.Response(text=body, content_type=content_type,
                                    headers=_CORS)
            return handler

        userscript = _file_route("pump_forwarder.user.js",
                                 "application/javascript")
        phone_relay = _file_route("phone_relay.sh", "text/x-shellscript")
        # Served under a one-letter path on purpose: the phone has to fetch
        # this over the tunnel by typing a command, and short commands survive
        # flaky phone keyboards/clipboards that mangle long ones.
        phone_all = _file_route("phone_all.sh", "text/x-shellscript")

        def _key_ok(req) -> bool:
            if not config.PUMP_INGEST_KEY:
                return True
            import hmac as _hmac
            return _hmac.compare_digest(req.headers.get("X-Ingest-Key", ""),
                                        config.PUMP_INGEST_KEY)

        async def relay_status_post(req):
            """The userscript's self-report. Mobile browsers have no console,
            so this is the only way to see whether it is running and whether
            its pump.fun fetches are succeeding."""
            if not _key_ok(req):
                return web.json_response({"ok": False, "error": "bad key"},
                                         status=403, headers=_CORS)
            try:
                data = await req.json()
            except Exception:
                return web.json_response({"ok": False, "error": "bad json"},
                                         status=400, headers=_CORS)
            self.relay_status = data if isinstance(data, dict) else {}
            self.relay_status_at = time.time()
            log.info("phone relay status: %s", json.dumps(self.relay_status)[:400])
            return web.json_response({"ok": True}, headers=_CORS)

        async def relay_status_get(_req):
            """Open this in the phone browser to test the whole chain by hand:
            if it answers, the tunnel + ingest server are both fine."""
            age = round(time.time() - self.relay_status_at) if self.relay_status_at else None
            return web.json_response({
                "tunnel_ok": True,
                "callout_sink_wired": self.callout_sink is not None,
                "last_self_report_age_s": age,
                "last_self_report": self.relay_status,
            }, headers=_CORS)

        app = web.Application()
        app.router.add_route("GET", "/pump_forwarder.user.js", userscript)
        app.router.add_route("GET", "/p", phone_all)
        app.router.add_route("GET", "/phone_relay.sh", phone_relay)
        app.router.add_route("GET", "/relay-status", relay_status_get)
        app.router.add_route("POST", "/relay-status", relay_status_post)
        app.router.add_route("OPTIONS", "/ingest", options)
        app.router.add_route("POST", "/ingest", post)
        app.router.add_route("OPTIONS", "/callouts", options)
        app.router.add_route("POST", "/callouts", post_callouts)
        return app

    async def run_ingest(self):
        """HTTP endpoint the userscript POSTs tokens to."""
        self.load()  # pick up anything saved by pumpfarm --set-token
        if self.token:
            log.info("pump auth: token loaded from %s (captured %.1fh ago)",
                     self.auth_file,
                     max(0.0, (time.time() - self.captured_at)) / 3600)
        runner = web.AppRunner(self._make_app())
        await runner.setup()
        site = web.TCPSite(runner, config.PUMP_INGEST_HOST, config.PUMP_INGEST_PORT)
        await site.start()
        log.info("pump ingest listening on %s:%s (for pump_forwarder.user.js — "
                 "auth token + callout relay)",
                 config.PUMP_INGEST_HOST, config.PUMP_INGEST_PORT)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await runner.cleanup()
            raise

    # ---------- diagnostics ----------

    def status(self) -> dict[str, Any]:
        age = (time.time() - self.captured_at) if self.captured_at else -1
        return {
            "has_token": bool(self.token and self.token != self.rejected_token),
            "age_s": int(age) if age >= 0 else -1,
            "rejected": bool(self.rejected_token),
            "last_error": self.last_error,
        }

    # ---------- one-off live validation (used by pumpfarm --check) ----------

    async def validate(self, token: Optional[str] = None) -> tuple[int, str]:
        """GET /auth/my-profile with the token. Returns (status, user_id)."""
        tok = token or self.token
        if not tok:
            return 0, ""
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.get(f"{config.PUMP_CALLOUT_BASE}/auth/my-profile",
                             headers={"Authorization": f"Bearer {tok}",
                                      "Origin": "https://pump.fun",
                                      "Accept": "application/json"}) as resp:
                uid = ""
                if resp.status == 200:
                    try:
                        d = await resp.json()
                        uid = d.get("userId") or d.get("uuid") or ""
                    except Exception:
                        pass
                return resp.status, uid
