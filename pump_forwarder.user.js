// ==UserScript==
// @name         pump.fun → ZANE auth forwarder
// @namespace    zane
// @version      1.0
// @description  Forwards the pump.fun (Privy) identity token to the bot, and
//               relays the callout list from this browser (the bot's server
//               IP is throttled by pump.fun). Keep one pump.fun tab open.
// @match        https://pump.fun/*
// @run-at       document-start
// @grant        none
// ==/UserScript==

(function () {
  'use strict';

  // ==== CONFIG — must match PUMP_INGEST_KEY in the bot's .env (or leave '') ====
  const INGEST_URL = 'http://127.0.0.1:8766/ingest';
  const KEY = ''; // e.g. 'my-secret' if PUMP_INGEST_KEY is set on the bot

  // ------------------------------------------------------------------
  const okJwt = (t) => typeof t === 'string' && t.length > 100 && t.startsWith('eyJ');
  let last = '';
  const seen = new Set();

  function forward(token, src) {
    if (!okJwt(token) || token === last) return;
    last = token;
    console.log('[pump-fwd] forwarding token from', src);
    fetch(INGEST_URL, {
      method: 'POST',
      headers: Object.assign({ 'Content-Type': 'application/json' },
        KEY ? { 'X-Ingest-Key': KEY } : {}),
      body: JSON.stringify({ token, source: src, url: location.href, ts: Date.now() }),
    }).then((r) => r.json())
      .then((j) => console.log('[pump-fwd] ingest:', JSON.stringify(j)))
      .catch((e) => console.log('[pump-fwd] ingest failed (bot running? port?):', e.message));
  }

  // 1) Watch all requests: pump.fun attaches `authorization: Bearer <privy>`
  //    to frontend-api-v3.pump.fun calls — the exact token we need.
  const of = window.fetch;
  window.fetch = async function (...args) {
    try {
      const h = (args[1] && args[1].headers) ||
        (args[0] instanceof Request ? args[0].headers : null);
      if (h) {
        let auth = '';
        if (h instanceof Headers) auth = h.get('authorization') || '';
        else if (Array.isArray(h)) {
          const kv = h.find(([k]) => k.toLowerCase() === 'authorization');
          auth = kv ? kv[1] : '';
        } else if (typeof h === 'object') {
          auth = h['authorization'] || h['Authorization'] || '';
        }
        if (auth.toLowerCase().startsWith('bearer ')) forward(auth.slice(7), 'fetch');
      }
      // response bodies from auth endpoints can carry fresh tokens too
      const res = await of.apply(this, args);
      const url = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url) || '';
      if (/auth\.privy\.io|frontend-api-v3\.pump\.fun\/auth/.test(url)) {
        res.clone().json().then((d) => {
          const t = d && (d.token || d.identity_token || (d.data && d.data.token));
          if (okJwt(t)) forward(t, 'response:' + url.slice(0, 60));
        }).catch(() => {});
      }
      return res;
    } catch (e) {
      return of.apply(this, args);
    }
  };

  // 2) XHR (the app uses axios for some calls)
  const ox = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this.addEventListener('load', function () {
      try {
        if (/auth\.privy\.io|frontend-api-v3\.pump\.fun\/auth/.test(String(url))) {
          const d = JSON.parse(this.responseText);
          const t = d && (d.token || d.identity_token || (d.data && d.data.token));
          if (okJwt(t)) forward(t, 'xhr:' + String(url).slice(0, 60));
        }
      } catch (e) {}
    });
    return ox.call(this, method, url, ...rest);
  };

  // 3) Privy stores its identity token in localStorage — periodic sweep
  setInterval(() => {
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        const v = localStorage.getItem(k);
        if (!v) continue;
        if (okJwt(v)) { forward(v, 'ls:' + k); continue; }
        if (v.charAt(0) === '{') {
          try {
            const o = JSON.parse(v);
            const stack = [o];
            while (stack.length) {
              const cur = stack.pop();
              if (!cur || typeof cur !== 'object') continue;
              for (const [kk, vv] of Object.entries(cur)) {
                if (typeof vv === 'string' && okJwt(vv)) forward(vv, 'ls-deep:' + k + '.' + kk);
                else if (vv && typeof vv === 'object') stack.push(vv);
              }
            }
          } catch (e) {}
        }
      }
    } catch (e) {}
  }, 4000);

  // ==================================================================
  // CALLOUT RELAY
  // The bot's server IP is throttled by pump.fun to a couple of requests
  // a minute, which is what made callouts arrive minutes late ("MISSED").
  // This tab is on an IP pump.fun trusts, so it does the fetching instead
  // and pushes each page to the bot's ingest server. Keep this tab open —
  // it already had to be, for the auth forward above.
  // ==================================================================
  // Tuned for a PHONE on Wi-Fi: home/mobile IPs are residential, which is the
  // one class pump.fun does not blocklist (datacenter proxy ranges get sent to
  // static.pump.fun/blocked). Keep the tab open and the screen awake.
  // Must match a caller the bot tracks, or the bot rejects the page.
  const CALLERS = ['6qudAN2kV8mtCcYJxb5QQ6Vr15itdHHdeVbYm99NKMhy'];
  const RELAY_MS = 3000;      // 3s — the bot's full speed
  const POLL_LIMIT = 5;       // cheap tick: just enough to spot something new
  const CATCHUP_LIMIT = 50;   // full page, only fetched when the newest id moved
  const KEEP_ALIVE = true;    // see keepAlive() — needs one tap on the page
  const RELAY_URL = INGEST_URL.replace(/\/ingest$/, '/callouts');
  const STATUS_URL = INGEST_URL.replace(/\/ingest$/, '/relay-status');
  const STATUS_MS = 30000;    // heartbeat, so the bot can see the phone is alive

  let relayFails = 0;
  let relayPausedUntil = 0;
  let lastStatusAt = 0;
  const lastNewest = {};      // caller -> newest calloutId already relayed

  // Mobile browsers have no developer console, so instead of console.log the
  // script reports what it is doing to the bot: opens the page with a status
  // (running / fetch failed + why), which the server logs. One small POST,
  // throttled — this is what makes a phone relay debuggable at all.
  function report(status) {
    try {
      fetch(STATUS_URL, {
        method: 'POST',
        headers: Object.assign({ 'Content-Type': 'application/json' },
          KEY ? { 'X-Ingest-Key': KEY } : {}),
        body: JSON.stringify(Object.assign({
          ua: navigator.userAgent.slice(0, 70),
          href: location.href.slice(0, 90),
          visible: document.visibilityState,
          t: Date.now(),
        }, status)),
      }).catch(function () {});
    } catch (e) {}
  }

  // Browsers throttle timers in background tabs to about once a minute, which
  // would quietly turn a 3s relay into a 60s one. A looping (near-silent) audio
  // element keeps the page in the "audible" state, which is exempt from that
  // throttling. Autoplay policy needs one tap anywhere on the page first.
  function keepAlive() {
    try {
      const a = document.createElement('audio');
      a.loop = true;
      a.volume = 0.001;  // not 0 — muted pages can still be treated as idle
      a.src = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA=';
      const start = () => a.play()
        .then(() => console.log('[pump-fwd] keep-alive playing — tab won\'t be ' +
                                'timer-throttled'))
        .catch(() => console.log('[pump-fwd] keep-alive blocked; TAP the page ' +
                                 'once to stop Android background-throttling'));
      document.addEventListener('click', start, { once: true });
      (document.body || document.documentElement).appendChild(a);
      start();
    } catch (e) {}
  }

  function calloutsUrl(caller, limit) {
    const headers = { Accept: 'application/json' };
    if (last) headers.Authorization = 'Bearer ' + last;  // look like the real site
    // of = the ORIGINAL fetch, so this can't recurse into the wrapper above
    return of.call(window, 'https://frontend-api-v3.pump.fun/callout/list/' +
      caller + '?limit=' + limit + '&sortBy=TIMESTAMP&sortOrder=desc',
      { headers });
  }

  async function relayCallouts(caller) {
    if (Date.now() < relayPausedUntil) return;
    let page;
    try {
      const res = await calloutsUrl(caller, POLL_LIMIT);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      page = await res.json();
    } catch (e) {
      // pump.fun throttles browsers too — ease off rather than hammer it
      report({ ok: false, caller: caller, stage: 'fetch', detail: String(e.message || e) });
      if (++relayFails === 5) {
        relayPausedUntil = Date.now() + 60000;
      }
      return;
    }
    relayFails = 0;
    // heartbeat: proves the script is running and its pump.fun fetches work
    if (Date.now() - lastStatusAt > STATUS_MS) {
      lastStatusAt = Date.now();
      report({ ok: true, caller: caller, stage: 'poll',
               newest: (page.callouts && page.callouts[0] && page.callouts[0].calloutId) || '',
               items: (page.callouts || []).length });
    }
    const items = (page && page.callouts) || [];
    const newest = items.length ? items[0].calloutId : '';
    // Nothing new: stay quiet. Keeps the phone's data (and the bot) idle.
    if (!newest || newest === lastNewest[caller]) return;
    lastNewest[caller] = newest;
    try {
      // Something new — push a full page so the bot can back-fill everything
      // newer than its own cursor (self-heals a push that got lost).
      const full = await calloutsUrl(caller, CATCHUP_LIMIT);
      const body = (full.ok ? await full.json() : page) || {};
      await fetch(RELAY_URL, {
        method: 'POST',
        headers: Object.assign({ 'Content-Type': 'application/json' },
          KEY ? { 'X-Ingest-Key': KEY } : {}),
        body: JSON.stringify(Object.assign({}, body, { caller_id: caller })),
      });
    } catch (e) {
      // leave lastNewest set — the next full page will back-fill this one
      console.log('[pump-fwd] callout push failed (bot running? tunnel up?):',
                  e.message);
    }
  }

  if (CALLERS.length) {
    if (KEEP_ALIVE) keepAlive();
    // immediate self-report: if the bot sees this, the script is installed and
    // running on the page — the single most useful fact when debugging a phone
    report({ ok: true, stage: 'startup', callers: CALLERS });
    console.log('[pump-fwd] callout relay started — ' + CALLERS.join(', ') +
                ' every ' + RELAY_MS + 'ms');
    setInterval(() => CALLERS.forEach((c) => { relayCallouts(c); }), RELAY_MS);
  }
})();
