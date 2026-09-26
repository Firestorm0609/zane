/**
 * pump.fun callout mirror — one file, two platforms (Cloudflare Workers + Deno Deploy).
 *
 * WHY THIS EXISTS
 *   pump.fun throttles its frontend API to ~13 requests per 600s PER SOURCE IP
 *   (measured from both this VPS and a phone — same ceiling). One IP can
 *   therefore poll at best every ~46s. A serverless platform sends its outbound
 *   traffic from a POOL of IPs, so the same requests are spread across many of
 *   them and the aggregate allowance is several times larger — that is why
 *   driftrace.tech sustains 6s polling where a bare VPS cannot. Each deployment
 *   on a different network (or region) is a separate budget, so the bot can
 *   alternate:
 *
 *     t+0s -> base A    t+3s -> base B    t+6s -> base A   ...
 *
 *   Every base still sees only one request per 6s — the rate that ran clean for
 *   days — while the bot checks for new callouts every 3s.
 *
 * DEPLOY — Cloudflare Workers (no CLI, no token needed for this route)
 *   1. dash.cloudflare.com -> Workers & Pages -> Create -> Worker
 *   2. paste this file, Deploy
 *   3. copy the https://<name>.<subdomain>.workers.dev URL
 *   Bot then uses:  PUMP_CALLOUT_BASE=https://<name>.<subdomain>.workers.dev/pump
 *
 * DEPLOY — Deno Deploy
 *   1. dash.deno.com -> New Playground
 *   2. paste this file, Save & Deploy
 *   Bot then uses:  PUMP_CALLOUT_BASE=https://<name>.<account>.deno.net/pump
 *
 * ROUTES
 *   /pump/<path>?<query>  -> https://frontend-api-v3.pump.fun/<path>?<query>
 *   /whoami               -> {"egress_ip": "...", ...} — the IP THIS deployment
 *                            presents. Two bases are only separate budgets if
 *                            these differ; check both before trusting them.
 *   anything else         -> 404 (deliberately NOT an open proxy)
 */

const UPSTREAM = "https://frontend-api-v3.pump.fun";
const PREFIX = "/pump";
const UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}

function platformHints(request) {
  if (typeof Deno !== "undefined") {
    let region = null;
    try {
      region = Deno.env.get("DENO_REGION") || Deno.env.get("DENO_DEPLOYMENT_ID") || null;
    } catch (_) {}
    return { platform: "deno-deploy", region };
  }
  const cf = (request && request.cf) || {};
  return { platform: "cloudflare-workers", colo: cf.colo || null, country: cf.country || null };
}

async function handler(request) {
  const url = new URL(request.url);

  if (url.pathname === "/whoami") {
    let ip = "unknown";
    try {
      const r = await fetch("https://api.ipify.org?format=json", {
        headers: { accept: "application/json" },
      });
      ip = (await r.json()).ip || ip;
    } catch (_) {}
    return json({ egress_ip: ip, ...platformHints(request) });
  }

  if (url.pathname !== PREFIX && !url.pathname.startsWith(PREFIX + "/")) {
    return new Response("not found\n", { status: 404 });
  }

  const target = UPSTREAM + url.pathname.slice(PREFIX.length) + url.search;
  const headers = {
    Origin: "https://pump.fun",
    Referer: "https://pump.fun/",
    Accept: "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": UA,
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
  };
  const auth = request.headers.get("authorization");
  if (auth) headers.Authorization = auth; // /callout/leaderboard needs a bearer

  const resp = await fetch(target, { headers, redirect: "follow" });
  return new Response(resp.body, {
    status: resp.status, // a 429 must stay a 429 — never mask it
    headers: {
      "content-type": resp.headers.get("content-type") || "application/json",
      "cache-control": "no-store", // freshness is the entire point
      "access-control-allow-origin": "*",
    },
  });
}

// Cloudflare Workers uses the module default export; Deno Deploy needs serve().
if (typeof Deno !== "undefined" && typeof Deno.serve === "function") {
  Deno.serve(handler);
}
export default { fetch: handler };
