#!/data/data/com.termux/files/usr/bin/sh
# One-command, self-healing pump.fun callout relay for Android/Termux.
#
# This is the only thing you run. It:
#   1. takes a wake lock so Android doesn't freeze the shell
#   2. sets up an ssh key so the tunnel can be restarted with no password
#      prompt (a background restart could never answer one)
#   3. opens the "ssh -L" tunnel that carries the phone's traffic to the bot
#   4. runs the relay, refreshing it from the server on every (re)start
#   5. supervises both — tunnel drops, it reconnects; relay dies, it restarts
#
# Bootstrap, first time only (asks for your server password once):
#   ssh root@YOUR_SERVER_IP 'cat /root/sesh/phone_all.sh' > pumprelay.sh && sh pumprelay.sh
#
# Every time after that:
#   sh ~/pumprelay.sh
#
# Home-screen button (install the Termux:Widget app first):
#   mkdir -p ~/.shortcuts && ln -sf ~/pumprelay.sh ~/.shortcuts/pump
# Start on reboot (install Termux:Boot, then open the app once):
#   mkdir -p ~/.termux/boot && ln -sf ~/pumprelay.sh ~/.termux/boot/10-pump
#
# Ctrl-C stops everything.

# Your bot host. Set it here, or export SERVER=root@1.2.3.4 before running.
SERVER="${SERVER:-root@YOUR_SERVER_IP}"
PORT=8766
# How often the relay may ask pump.fun for callouts.
#
# pump.fun's per-IP allowance is ~13 requests per 600s — measured twice: the
# bot learned 13-14/600s from this box, and the phone showed the same ceiling
# when the relay's 3s default burned its whole budget in 43s and got 429/1015.
# At 3s the relay therefore watched 43s out of every 600s and stayed blind for
# the other 557s. 600/13 = 46.2s is the shortest spacing that never trips it;
# 50s leaves a request or two spare for the bigger catch-up fetch the relay
# does whenever a new callout shows up. Trade-off: no more 429s and continuous
# coverage, but a new callout is seen up to ~50s late instead of ~3s. Faster
# than that needs a second IP budget (a proxy per check_proxy.py), not a
# shorter interval.
RELAY_INTERVAL="${RELAY_INTERVAL:-50}"
CALLER="6qudAN2kV8mtCcYJxb5QQ6Vr15itdHHdeVbYm99NKMhy"
BOT="http://127.0.0.1:$PORT"
RELAY="$HOME/relay.sh"
SELF="$HOME/p.sh"
KEY="$HOME/.ssh/id_ed25519"
SSH_BASE="-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

log() { echo "$(date '+%H:%M:%S') $*"; }

# ---- 0. one copy at a time -------------------------------------------------
# Three things can start this now: the home-screen widget, Termux:Boot, and a
# hand-typed `sh p.sh`. Two copies would poll pump.fun twice from the same
# phone IP (which is already the thing being rate-limited), so block the second.
# The check is deliberately hard to get wrong: it only refuses when the recorded
# pid is still alive AND the tunnel that pid made is answering right now. A
# stale file left by a crash, a swipe-away or a reboot therefore can never lock
# you out — worst case you get a second copy, which is what we had before.
# If you ever do want to force a start: rm ~/.pumprelay.pid
LOCK="$HOME/.pumprelay.pid"
if [ -f "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null)
  case "$oldpid" in
    ''|*[!0-9]*) : ;;   # unreadable — ignore it
    *)
      if kill -0 "$oldpid" 2>/dev/null \
         && curl -s -o /dev/null --max-time 4 "$BOT/relay-status"; then
        log "relay is already running (pid $oldpid) — nothing to do"
        exit 0
      fi
      ;;
  esac
fi
echo $$ > "$LOCK"

# ---- 1. wake lock: without it Android suspends background shells ----
command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock \
  || log "NOTE: for a wake lock, run: pkg install -y termux-tools"

# ---- 2. packages ----
# package:command pairs — the binary to probe is not always the package name.
# Testing the package name ("openssh") never matches (the command is `ssh`),
# so this printed "installing openssh" and ran pkg on EVERY start, even with
# ssh already present. Keep the pair explicit.
for pair in curl:curl openssh:ssh; do
  pkgname="${pair%%:*}"
  pkgexe="${pair#*:}"
  command -v "$pkgexe" >/dev/null 2>&1 || {
    # Output is NOT hidden here: this runs at most once per package, and a
    # silent pkg (mirror prompt, slow fetch) is indistinguishable from a hang.
    log "installing $pkgname (one-off)"
    pkg install -y "$pkgname" || log "could not install $pkgname — install it by hand"
  }
done

# ---- 3. ssh key + auth mode ----
mkdir -p "$HOME/.ssh"
[ -f "$KEY" ] || { log "creating an ssh key (one-off)"; \
                   ssh-keygen -t ed25519 -N "" -f "$KEY" -q; }

if [ -f "$KEY" ] && ! ssh -i "$KEY" $SSH_BASE -o BatchMode=yes "$SERVER" true 2>/dev/null; then
  log "installing the key on the server (asks for your password once)"
  cat "$KEY.pub" | ssh $SSH_BASE "$SERVER" \
      'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo ok' \
    || log "key install failed"
fi

# A key is what makes automatic reconnection possible: a background restart
# cannot answer a password prompt. Without one we still run, but only in the
# foreground and without the watchdog, so we say so plainly.
if [ -f "$KEY" ] && ssh -i "$KEY" $SSH_BASE -o BatchMode=yes "$SERVER" true 2>/dev/null; then
  AUTH="key"
  SSH_ARGS="-i $KEY $SSH_BASE -o BatchMode=yes"
else
  AUTH="password"
  SSH_ARGS="$SSH_BASE"
  log "WARNING: no working key — you'll be prompted for a password and the"
  log "         tunnel will NOT reconnect by itself if it drops."
fi

# ---- 4. tunnel ----
tunnel_ok() { curl -s -o /dev/null --max-time 4 "$BOT/relay-status"; }

open_tunnel() {
  # ExitOnForwardFailure: if the local port is already bound, fail loudly
  # instead of leaving a tunnel that quietly forwards nothing.
  # ServerAliveInterval: keeps phone-side NAT from dropping an idle tunnel.
  ssh $SSH_ARGS -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
      -f -N -L "$PORT:127.0.0.1:$PORT" "$SERVER" 2>/dev/null
}

ensure_tunnel() {
  i=0
  while [ "$i" -lt 8 ]; do
    tunnel_ok && return 0
    open_tunnel
    sleep 2
    i=$((i + 1))
  done
  return 1
}

# Runs for the life of the script: reopens the tunnel if it ever drops.
watchdog() {
  while :; do
    sleep 15
    tunnel_ok || { log "tunnel dropped — reconnecting"; open_tunnel; }
  done
}

# ---- 5. go ----
log "pump relay — caller ${CALLER%????????????????????????} (auth: $AUTH)"
if ! ensure_tunnel; then
  log "cannot reach the server at $SERVER — check your connection"
  log "(the bot's port is localhost-only, so the tunnel is the only way in)"
  exit 1
fi
log "tunnel up: $BOT -> $SERVER:$PORT"

# Keep a local copy of this script, so the next start is just `sh ~/p.sh`
# with no downloading — and so server-side fixes arrive on their own.
# (mv is atomic, so replacing the file we may be running from is safe.)
[ -f "$SELF" ] || SELF_WAS_NEW=1
if curl -s --max-time 15 "$BOT/p" -o "$SELF.new" && [ -s "$SELF.new" ]; then
  mv "$SELF.new" "$SELF"
  chmod +x "$SELF" 2>/dev/null
fi
[ "$SELF_WAS_NEW" = 1 ] && log "saved a local copy — from now on just: sh $SELF"

if [ "$AUTH" = "key" ]; then
  watchdog &
else
  log "no auto-reconnect (no key) — if the tunnel dies, re-run this script"
fi

while :; do
  # Re-fetch the relay on every (re)start, so server-side fixes reach the
  # phone without you downloading anything again.
  if curl -s --max-time 15 "$BOT/phone_relay.sh" -o "$RELAY.new" && [ -s "$RELAY.new" ]; then
    mv "$RELAY.new" "$RELAY"
  fi
  if [ ! -s "$RELAY" ]; then
    log "could not fetch the relay script — retrying in 15s"
    sleep 15
    continue
  fi
  CALLER="$CALLER" BOT="$BOT" INTERVAL="$RELAY_INTERVAL" sh "$RELAY"
  log "relay stopped — restarting in 5s (Ctrl-C to quit)"
  sleep 5
done
