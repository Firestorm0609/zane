#!/data/data/com.termux/files/usr/bin/sh
# pump.fun callout relay for a phone — Termux only, no browser, no extension.
#
# Why this exists instead of relying on a userscript: an in-page script must run
# inside the pump.fun page so its requests carry an Origin the API allows, which
# needs Tampermonkey plus a foreground tab that Android keeps throttling. curl
# has no same-origin policy at all, so this polls the same endpoint directly
# from the phone's (residential) IP. Nothing to install but Termux + curl, and
# it keeps working with the screen off.
#
# Setup (in Termux, in a SECOND session — the first one holds the tunnel):
#   pkg install -y curl
#   ssh -N -L 8766:127.0.0.1:8766 root@<server>     # session 1 (leave running)
#   termux-wake-lock
#   curl -s http://127.0.0.1:8766/phone_relay.sh -o relay.sh
#   sh relay.sh
#
# Every fetch prints its HTTP status, so a failure is never silent — this is the
# log the server side can't see. Ctrl-C to stop.

CALLER="${CALLER:-6qudAN2kV8mtCcYJxb5QQ6Vr15itdHHdeVbYm99NKMhy}"
BOT="${BOT:-http://127.0.0.1:8766}"
API="${API:-https://frontend-api-v3.pump.fun/callout/list/$CALLER}"
INTERVAL="${INTERVAL:-3}"
# Push a tick-size page this often even when nothing is new. The bot only
# trusts the relay for ~2 min after a push, so without this it would decide the
# relay died during a quiet spell and go back to polling its own throttled IP —
# re-arming the ban this whole setup exists to avoid. ~5 KB every 30s.
HEARTBEAT="${HEARTBEAT:-30}"
# 429 backoff: start here, double on each one, cap at BACKOFF_MAX, reset on a
# clean fetch. Retrying a throttled IP at a fixed 60s just keeps it hot — and
# on mobile the address is shared with strangers (CGNAT), so it stays throttled.
BACKOFF_START="${BACKOFF_START:-60}"
BACKOFF_MAX="${BACKOFF_MAX:-600}"
BACKOFF_MIN="${BACKOFF_MIN:-10}"   # never retry sooner than this, whatever Retry-After says
HDRS="$HOME/.relay-hdrs"
TICK_LIMIT=5      # small poll: enough to spot a new callout (~84 MB/day)
FULL_LIMIT=50     # only fetched when something new appeared (catch-up)
UA="Mozilla/5.0 (Linux; Android 16; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36"

# fetch <limit> -> sets PAGE, CODE and RETRY_AFTER
fetch() {
  resp=$(curl -s -D "$HDRS" -w '\n__HTTP__%{http_code}' --max-time 8 \
      -H "Accept: application/json" -H "Origin: https://pump.fun" \
      -H "Referer: https://pump.fun/" -H "User-Agent: $UA" \
      "$API?limit=$1&sortBy=TIMESTAMP&sortOrder=desc")
  CODE=${resp##*__HTTP__}
  PAGE=${resp%__HTTP__*}
  RETRY_AFTER=$(tr -d '\r' < "$HDRS" 2>/dev/null \
      | awk 'tolower($1) == "retry-after:" { print $2 }' | tail -1)
}

# report <json> — best-effort status to the bot (logged server-side). A phone
# has no console anyone can read, so this is the only way to answer "is the
# relay alive and what is pump.fun saying to it?" without the screen.
report() {
  printf '%s' "$1" | curl -s --max-time 6 -X POST "$BOT/relay-status" \
      -H 'Content-Type: application/json' --data-binary @- >/dev/null 2>&1
}

# push <body> <label> -> status line; nonzero if the bot rejected it
push() {
  out=$(printf '%s' "$1" | curl -s -w '\n__HTTP__%{http_code}' --max-time 10 \
      -X POST "$BOT/callouts?caller=$CALLER" -H 'Content-Type: application/json' \
      --data-binary @-)
  pcode=${out##*__HTTP__}
  pbody=$(printf '%s' "${out%__HTTP__*}" | tr -d '\n' | head -c 140)
  if [ "$pcode" = "200" ]; then
    echo "$(date '+%H:%M:%S') $2 -> $pbody"
    return 0
  fi
  echo "$(date '+%H:%M:%S') push FAILED (HTTP $pcode) — is the ssh -L 8766 tunnel up?"
  return 1
}

echo "relay: $CALLER"
echo "  poll every ${INTERVAL}s -> $BOT/callouts"
echo "  heartbeat every ${HEARTBEAT}s while quiet"
echo "  needs the 'ssh -L 8766' tunnel in another session; Ctrl-C to stop"
report "{\"stage\":\"startup\",\"caller\":\"$(printf '%s' "$CALLER" | cut -c1-12)\"}"

# Callout ids present in the current tick page. Comparing the SET (not the
# raw text) is deliberate: the response also carries live mcap figures that
# change every tick, and comparing raw text would trigger a full fetch+push
# constantly. Whitespace is tolerated so a format change can't silently
# deactivate the relay.
ids_of() {
  printf '%s' "$1" \
    | grep -o '"calloutId"[[:space:]]*:[[:space:]]*"[^"]*"' \
    | sed -e 's/.*"\([^"]*\)"$/\1/' | tr '\n' ','
}

lastids=""
last_push=0
seen_first=0
fails=0
odd=0
backoff="$BACKOFF_START"
while :; do
  fetch "$TICK_LIMIT"

  if [ "$CODE" != "200" ]; then
    fails=$((fails + 1))
    head=$(printf '%s' "$PAGE" | tr -d '\n"\\' | head -c 140)
    # First failure, then every 10th, so a long outage can't flood the bot.
    if [ "$fails" = 1 ] || [ $((fails % 10)) -eq 0 ]; then
      report "{\"stage\":\"fetch\",\"code\":\"$CODE\",\"tries\":$fails,\"detail\":\"$head\"}"
    fi
    case "$CODE" in
      429)
           # Prefer the server's own Retry-After when it sends one.
           retry="$RETRY_AFTER"
           case "$retry" in ''|*[!0-9]*) retry="$backoff";; esac
           [ "$retry" -lt "$BACKOFF_MIN" ] && retry="$BACKOFF_MIN"
           backoff=$((backoff * 2))
           [ "$backoff" -gt "$BACKOFF_MAX" ] && backoff="$BACKOFF_MAX"
           echo "$(date '+%H:%M:%S') THROTTLED (429) — waiting ${retry}s, next wait up to ${backoff}s: $head"
           case "$head" in
             *1015*) echo "  (Cloudflare 1015 = per-IP limit; on mobile this address is"
                     echo "   shared with other users, so it can stay hot for hours —"
                     echo "   switching to Wi-Fi usually fixes it)";;
           esac
           sleep "$retry"; continue;;
      403) echo "$(date '+%H:%M:%S') BLOCKED (403) — this IP is on pump.fun's list: $head"
           sleep 30; continue;;
      000|"") [ $((fails % 10)) -eq 1 ] && echo "$(date '+%H:%M:%S') no network / DNS (x$fails)"
           sleep "$INTERVAL"; continue;;
      *)   echo "$(date '+%H:%M:%S') HTTP $CODE: $head"
           sleep 15; continue;;
    esac
  fi

  if [ "$fails" -gt 0 ]; then
    echo "$(date '+%H:%M:%S') pump.fun is answering again (after $fails failed fetch(es))"
    report "{\"stage\":\"ok\",\"recovered_after\":$fails}"
    backoff="$BACKOFF_START"   # clean fetch: forget the penalty
  fi
  fails=0

  # One-time shape check: a 200 that isn't a callout list means the API changed
  # under us, and without this the relay would look healthy while sending
  # nothing.
  if [ "$seen_first" = 0 ]; then
    seen_first=1
    case "$PAGE" in
      *callout*) : ;;
      *) echo "$(date '+%H:%M:%S') NOTE: 200 but no 'callout' key — response head:"
         echo "  $(printf '%s' "$PAGE" | tr -d '\n' | head -c 200)";;
    esac
  fi

  ids=$(ids_of "$PAGE")
  newest=$(printf '%s' "$ids" | cut -d, -f1)

  # 200 with no callout key at all is not normal (empty list still says
  # "callouts") — a challenge page served as 200 would otherwise be silent.
  case "$PAGE" in
    *callout*) odd=0 ;;
    *) odd=$((odd + 1))
       [ $((odd % 20)) -eq 1 ] && {
         echo "$(date '+%H:%M:%S') 200 but no 'callout' key (x$odd) — head:"
         echo "  $(printf '%s' "$PAGE" | tr -d '\n' | head -c 200)"
       };;
  esac

  now=$(date +%s)
  if [ -n "$ids" ] && [ "$ids" != "$lastids" ]; then
    lastids="$ids"
    # Something new: send a full page so the bot back-fills to its own cursor.
    # If that bigger fetch fails, fall back to the tick page we already have.
    tick="$PAGE"
    fetch "$FULL_LIMIT"
    if [ "$CODE" != "200" ] || [ -z "$PAGE" ]; then PAGE="$tick"; fi
    push "$PAGE" "pushed $newest" && last_push=$now
  elif [ $((now - last_push)) -ge "$HEARTBEAT" ]; then
    # Quiet spell: this isn't new data, it just tells the bot the relay is
    # alive. Callouts are deduped by cursor, so it can never double-alert.
    push "$PAGE" "heartbeat" && last_push=$now
  fi

  sleep "$INTERVAL"
done
