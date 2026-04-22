"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EXPIRYRANGE BOT  v2  —  R_10  —  Distribution Thinking                     ║
║                                                                              ║
║  PHILOSOPHY                                                                  ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  "Win consistently with controlled risk."                                   ║
║                                                                              ║
║  This bot does NOT predict direction. It does NOT fire on signals.          ║
║  It asks one question per tick:                                             ║
║                                                                              ║
║    "Is the current price distribution tight enough that betting price        ║
║     stays within ±2.1 has positive expected value?"                         ║
║                                                                              ║
║  If yes, it sizes the bet and places it. If no, it waits. No exceptions.    ║
║                                                                              ║
║  CONTRACT                                                                    ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Symbol   : R_10 (Volatility 10 Index)                                      ║
║  Type     : EXPIRYRANGE                                                     ║
║  Barrier  : +2.1 / -2.1 (price must stay within ±2.1 of entry)            ║
║  Duration : 2 minutes (120 ticks nominal)                                   ║
║  Payout   : ~31-37% ROI (tracked live via online estimator)                ║
║                                                                              ║
║  CHANGES FROM v1                                                             ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  ✅ Symbol: R_10 (was 1HZ10V)                                                ║
║  ✅ Settlement poller: actively polls every 5s — no more forced unlocks      ║
║  ✅ Flexible contract ID matching (str vs int mismatch fixed)                ║
║  ✅ Settlement subscribe BEFORE buy (not after)                              ║
║  ✅ Deep settlement debug logging (every path annotated)                     ║
║  ✅ Martingale: 2.5× multiplier after 1 loss, up to 3 steps, then reset     ║
║     - Overrides Kelly while active                                           ║
║     - Initial (base) stake: $0.35 (flat minimum)                            ║
║     - Step 1 loss → stake × 2.5  (e.g. $0.35 → $0.88)                      ║
║     - Step 2 loss → stake × 2.5² (e.g. $0.35 → $2.19)                      ║
║     - Step 3 loss → stake × 2.5³ (e.g. $0.35 → $5.47)                      ║
║     - Any win at any step → resets to base stake                             ║
║  ✅ Tick rate tracker (R_10 is not perfectly 1Hz)                            ║
║                                                                              ║
║  DISTRIBUTION MODEL                                                          ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  R_10 price moves ≈ Gaussian random walk at short scales.                   ║
║  Over N ticks: σ_expiry = σ_tick × √N                                       ║
║  P(win) = erf(2.1 / (√2 × σ_expiry))                                       ║
║                                                                              ║
║  ONLINE ESTIMATOR (learns while running)                                    ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Exponential weighted moving average of tick-to-tick moves.                 ║
║  α = 0.05 (recent ticks weighted more, old ticks decay away).              ║
║  Also tracks: empirical win rate per regime, actual ROI from proposals.    ║
║  Regime: CALM (σ<0.080) / NORMAL (σ<0.120) / ACTIVE (σ≥0.120)            ║
║                                                                              ║
║  ENTRY GATE (all must pass)                                                 ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  1. Estimator warmed up (≥ MIN_WARMUP_TICKS)                               ║
║  2. Regime is CALM or NORMAL (block ACTIVE)                                 ║
║  3. P(win) ≥ MIN_P_WIN (default 0.85)                                      ║
║  4. EV > MIN_EV (default 0.02 per dollar staked)                           ║
║  5. Kelly fraction f > 0 (mathematical confirmation of positive edge)       ║
║  6. No trade in progress                                                    ║
║  7. Cooldown since last trade (settles + buffer)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import math
import os
import sys
import time
import traceback
import threading
from collections import deque
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed, ConnectionClosedError, ConnectionClosedOK,
    )
except ImportError:
    sys.exit("websockets not installed — run: pip install websockets")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def _env(key, default):
    v = os.environ.get(key)
    if v is None:
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes")
    if isinstance(default, float):
        return float(v)
    if isinstance(default, int):
        return int(v)
    return v


# Deriv connection
API_TOKEN  = _env("DERIV_API_TOKEN", "3nMoTkW49VHJqhH")
APP_ID     = _env("DERIV_APP_ID",    1089)
SYMBOL     = _env("SYMBOL",          "R_10")              # ← changed from 1HZ10V

# Contract
BARRIER       = 0.97
DURATION_MIN  = 2
DURATION_SEC  = DURATION_MIN * 60                          # 120s
N_TICKS_EXPIRY = 120                                       # nominal; measured live

# Estimator
EWMA_ALPHA       = _env("EWMA_ALPHA",       0.05)
MIN_WARMUP_TICKS = _env("MIN_WARMUP_TICKS", 60)

# Regime σ thresholds
SIGMA_CALM   = _env("SIGMA_CALM",   0.080)
SIGMA_NORMAL = _env("SIGMA_NORMAL", 0.120)

# Entry gate
MIN_P_WIN    = _env("MIN_P_WIN",    0.85)
MIN_EV       = _env("MIN_EV",       0.02)
ALLOW_REGIMES = {"CALM", "NORMAL"}

# Kelly sizing (used when martingale is at step 0)
KELLY_FRACTION = _env("KELLY_FRACTION", 0.25)
KELLY_MAX_PCT  = _env("KELLY_MAX_PCT",  0.05)

# ── Martingale config ────────────────────────────────────────────────────────
MARTINGALE_BASE_STAKE = _env("MARTINGALE_BASE_STAKE", 0.35)   # ← $0.35 initial stake
MARTINGALE_FACTOR     = _env("MARTINGALE_FACTOR",     2.5)    # ← 2.5× multiplier
MARTINGALE_MAX_STEPS  = _env("MARTINGALE_MAX_STEPS",  3)      # ← max 3 steps then reset
MIN_STAKE             = MARTINGALE_BASE_STAKE                  # absolute floor

# Session limits
TARGET_PROFIT  = _env("TARGET_PROFIT",  50.0)
STOP_LOSS      = _env("STOP_LOSS",      20.0)
TRADE_COOLDOWN = _env("TRADE_COOLDOWN", 10)

# Resilience
RECONNECT_MIN  = _env("RECONNECT_MIN",  2)
RECONNECT_MAX  = _env("RECONNECT_MAX",  60)
WS_PING        = _env("WS_PING",        30)
BUY_RETRIES    = _env("BUY_RETRIES",    8)
LOCK_TIMEOUT   = _env("LOCK_TIMEOUT",   300)

# Settlement poller interval (seconds)
POLL_INTERVAL  = _env("POLL_INTERVAL",  5)

PORT = _env("PORT", 8080)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def info(msg):  print(f"[{_ts()}] [INFO ] {msg}", flush=True)
def warn(msg):  print(f"[{_ts()}] [WARN ] {msg}", flush=True)
def err(msg):   print(f"[{_ts()}] [ERROR] {msg}", flush=True)
def tlog(msg):  print(f"[{_ts()}] [TRADE] {msg}", flush=True)
def slog(msg):  print(f"[{_ts()}] [SETTL] {msg}", flush=True)   # settlement-specific
def mlog(msg):  print(f"[{_ts()}] [MARTI] {msg}", flush=True)   # martingale-specific
def jlog(obj):  print(json.dumps(obj), flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# TICK RATE TRACKER  (new for R_10 — not guaranteed 1Hz)
# ─────────────────────────────────────────────────────────────────────────────

class TickRateTracker:
    """
    Measures the actual tick rate of R_10 in real time.
    R_10 is approximately but not exactly 1 tick/second.
    This feeds the correct N_TICKS_EXPIRY into the Gaussian model.
    """

    def __init__(self):
        self._times: deque = deque(maxlen=120)   # last 120 tick timestamps

    def record(self):
        self._times.append(time.monotonic())

    @property
    def avg_interval(self) -> float:
        """Average seconds between ticks. Default 1.0 if insufficient data."""
        if len(self._times) < 5:
            return 1.0
        intervals = [
            self._times[i] - self._times[i - 1]
            for i in range(1, len(self._times))
        ]
        return sum(intervals) / len(intervals)

    def ticks_for_seconds(self, seconds: float) -> float:
        """How many ticks correspond to `seconds` of wall time?"""
        return seconds / max(self.avg_interval, 0.1)

    @property
    def effective_n_ticks(self) -> float:
        """Expected tick count for a DURATION_SEC contract right now."""
        return self.ticks_for_seconds(DURATION_SEC)


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE VOLATILITY ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

class OnlineEstimator:
    """
    Learns the price distribution of R_10 in real time.

    Core: EWMA of |Δprice| per tick — adapts to changing volatility.
    Regime is determined by current σ vs thresholds.
    P(win) = erf(BARRIER / (√2 × σ_tick × √N_ticks_expiry))

    Integrates TickRateTracker so N_ticks_expiry adjusts to actual R_10 rate.
    """

    def __init__(self, tick_tracker: TickRateTracker):
        self._tk        = tick_tracker

        self.n          = 0
        self.last_price: Optional[float] = None
        self.ema_move   = None
        self.ema_sq     = None
        self.ema_sq_dev = None

        self.recent_moves: deque = deque(maxlen=30)

        self.ema_roi    = 0.33
        self.roi_n      = 0

        self.regime_stats = {
            "CALM":   {"trades": 0, "wins": 0},
            "NORMAL": {"trades": 0, "wins": 0},
            "ACTIVE": {"trades": 0, "wins": 0},
        }

        self.total_trades = 0
        self.total_wins   = 0
        self.session_pnl  = 0.0

    def ingest(self, price: float):
        self.n += 1
        self._tk.record()
        if self.last_price is not None:
            move = abs(price - self.last_price)
            self.recent_moves.append(move)

            if self.ema_move is None:
                self.ema_move   = move
                self.ema_sq     = move * move
                self.ema_sq_dev = 0.0
            else:
                alpha = EWMA_ALPHA
                dev              = abs(move - self.ema_move)
                self.ema_move    = alpha * move    + (1 - alpha) * self.ema_move
                self.ema_sq      = alpha * move**2 + (1 - alpha) * self.ema_sq
                self.ema_sq_dev  = alpha * dev**2  + (1 - alpha) * self.ema_sq_dev

        self.last_price = price

    def record_roi(self, ask: float, payout: float):
        if ask > 0:
            roi = (payout - ask) / ask
            self.roi_n  += 1
            self.ema_roi = 0.15 * roi + 0.85 * self.ema_roi

    def record_outcome(self, won: bool, profit: float):
        regime = self.regime()
        self.total_trades += 1
        self.session_pnl  += profit
        if won:
            self.total_wins += 1
        if regime in self.regime_stats:
            self.regime_stats[regime]["trades"] += 1
            if won:
                self.regime_stats[regime]["wins"] += 1

    @property
    def sigma(self) -> float:
        return self.ema_move if self.ema_move is not None else 0.0

    @property
    def sigma_variance(self) -> float:
        return self.ema_sq_dev if self.ema_sq_dev is not None else 0.0

    @property
    def sigma_stability(self) -> float:
        if not self.recent_moves or self.sigma == 0:
            return 0.5
        cv = math.sqrt(self.sigma_variance) / max(self.sigma, 1e-9)
        return max(0.0, min(1.0, 1.0 - cv))

    @property
    def spike_risk(self) -> float:
        if not self.recent_moves:
            return 0.0
        return max(self.recent_moves) / max(self.sigma, 1e-9)

    def regime(self) -> str:
        s = self.sigma
        if s < SIGMA_CALM:   return "CALM"
        if s < SIGMA_NORMAL: return "NORMAL"
        return "ACTIVE"

    def p_win(self) -> float:
        s = self.sigma
        if s <= 0:
            return 0.0
        n_ticks = self._tk.effective_n_ticks
        rw_std  = s * math.sqrt(n_ticks)
        return math.erf(BARRIER / (math.sqrt(2) * rw_std))

    def ev_per_dollar(self) -> float:
        pw  = self.p_win()
        roi = self.ema_roi
        return pw * roi - (1.0 - pw)

    def ready(self) -> bool:
        return self.n >= MIN_WARMUP_TICKS and self.ema_move is not None

    def status(self) -> dict:
        pw   = self.p_win()
        roi  = self.ema_roi
        ev   = self.ev_per_dollar()
        reg  = self.regime()
        tot  = self.total_trades
        wr   = self.total_wins / tot if tot > 0 else 0.0
        n_t  = self._tk.effective_n_ticks
        return {
            "ticks":        self.n,
            "sigma":        round(self.sigma, 6),
            "sigma_var":    round(self.sigma_variance, 8),
            "stability":    round(self.sigma_stability, 4),
            "spike_risk":   round(self.spike_risk, 3),
            "regime":       reg,
            "p_win":        round(pw, 6),
            "ema_roi":      round(roi, 4),
            "ev_per_$":     round(ev, 6),
            "ready":        self.ready(),
            "trades":       tot,
            "wins":         self.total_wins,
            "wr":           round(wr, 4),
            "session_pnl":  round(self.session_pnl, 4),
            "regime_stats": self.regime_stats,
            "eff_n_ticks":  round(n_t, 2),
            "tick_interval": round(self._tk.avg_interval, 4),
        }

    def log_status(self):
        s = self.status()
        info(f"σ={s['sigma']:.6f}  stab={s['stability']:.3f}  "
             f"spike={s['spike_risk']:.2f}  regime={s['regime']}  "
             f"P(win)={s['p_win']:.4f}  EV={s['ev_per_$']:+.4f}  "
             f"roi={s['ema_roi']:.3f}  trades={s['trades']}  WR={s['wr']*100:.1f}%  "
             f"tick_iv={s['tick_interval']:.3f}s  N_eff={s['eff_n_ticks']:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# MARTINGALE STAKE MANAGER  (new)
# ─────────────────────────────────────────────────────────────────────────────

class MartingaleManager:
    """
    2.5× martingale after a single loss, up to 3 steps, then forced reset.

    Step 0  → base stake ($0.35)
    Step 1  → $0.35 × 2.5¹ = $0.88
    Step 2  → $0.35 × 2.5² = $2.19
    Step 3  → $0.35 × 2.5³ = $5.47

    Any win resets to step 0.
    After step 3 loss (no more steps), also resets (risk cap).

    Kelly is completely overridden while step > 0.
    At step 0, Kelly is used (or base stake if Kelly < base).
    """

    def __init__(self):
        self.step = 0              # 0 = no loss streak
        self.consecutive_losses = 0

    def compute_stake(self, kelly_stake: float) -> float:
        """
        Returns the stake to use for this trade.
        kelly_stake: what Kelly would have suggested (used at step 0).
        """
        if self.step == 0:
            # Normal Kelly operation, respect the base floor
            return max(MARTINGALE_BASE_STAKE, kelly_stake)
        else:
            # Martingale override
            raw  = MARTINGALE_BASE_STAKE * (MARTINGALE_FACTOR ** self.step)
            stake = math.floor(raw * 100) / 100   # floor to 2dp
            return max(MARTINGALE_BASE_STAKE, stake)

    def record_win(self):
        if self.step > 0:
            mlog(f"WIN at step {self.step} → RESET to step 0")
        self.step = 0
        self.consecutive_losses = 0

    def record_loss(self):
        self.consecutive_losses += 1
        if self.step < MARTINGALE_MAX_STEPS:
            self.step += 1
            next_stake = MARTINGALE_BASE_STAKE * (MARTINGALE_FACTOR ** self.step)
            mlog(f"LOSS #{self.consecutive_losses} → step {self.step}  "
                 f"next_stake≈${next_stake:.2f}")
        else:
            # Hit max steps — reset rather than go deeper
            mlog(f"LOSS at MAX step {self.step} — FORCE RESET (risk cap)")
            self.step = 0
            self.consecutive_losses = 0

    @property
    def is_recovering(self) -> bool:
        return self.step > 0

    def describe(self) -> str:
        if self.step == 0:
            return "step=0 (base)"
        stake = MARTINGALE_BASE_STAKE * (MARTINGALE_FACTOR ** self.step)
        return (f"step={self.step}/{MARTINGALE_MAX_STEPS}  "
                f"streak={self.consecutive_losses}  "
                f"stake≈${stake:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# KELLY STAKE SIZER
# ─────────────────────────────────────────────────────────────────────────────

class KellySizer:
    """
    Sizes each trade using the Kelly Criterion.
    f* = (p × roi - (1-p)) / roi
    Stake = max(MIN_STAKE, min(KELLY_FRACTION × f* × balance, KELLY_MAX_PCT × balance))
    Floored to 2dp. Result is passed to MartingaleManager for final override.
    """

    def compute(self, p_win: float, roi: float,
                balance: float) -> tuple:
        """Returns (stake, full_kelly_f, allow)."""
        q = 1.0 - p_win
        b = roi
        if b <= 0:
            return MIN_STAKE, 0.0, False

        f_full = (p_win * b - q) / b

        if f_full <= 0:
            return MIN_STAKE, f_full, False

        f_frac    = KELLY_FRACTION * f_full
        raw_stake = f_frac * balance
        capped    = min(raw_stake, balance * KELLY_MAX_PCT)
        floored   = max(MIN_STAKE, capped)
        stake     = math.floor(floored * 100) / 100

        return stake, f_full, True

    def explain(self, p_win: float, roi: float, balance: float) -> str:
        stake, f, allow = self.compute(p_win, roi, balance)
        ev = p_win * roi - (1 - p_win)
        return (f"P(win)={p_win:.4f}  ROI={roi:.3f}  "
                f"Kelly_f={f:.4f}  kelly_stake=${stake:.2f}  "
                f"EV={ev:+.4f}  allow={allow}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY GATE
# ─────────────────────────────────────────────────────────────────────────────

class EntryGate:
    """All conditions must pass before a trade is placed."""

    def evaluate(self, est: OnlineEstimator,
                 balance: float, sizer: KellySizer) -> tuple:
        diag = est.status()

        if not est.ready():
            return False, f"warmup ({est.n}/{MIN_WARMUP_TICKS})", diag

        regime = est.regime()
        if regime not in ALLOW_REGIMES:
            return False, f"regime={regime} (blocked)", diag

        if est.spike_risk > 5.0:
            return False, f"spike_risk={est.spike_risk:.2f} > 5.0", diag

        if est.sigma_stability < 0.10:
            return False, f"stability={est.sigma_stability:.3f} < 0.10", diag

        pw = est.p_win()
        if pw < MIN_P_WIN:
            return False, f"P(win)={pw:.4f} < {MIN_P_WIN}", diag

        ev = est.ev_per_dollar()
        if ev < MIN_EV:
            return False, f"EV={ev:.4f} < {MIN_EV}", diag

        roi = est.ema_roi
        stake, f, has_edge = sizer.compute(pw, roi, balance)
        if not has_edge:
            return False, f"Kelly_f={f:.4f} ≤ 0 (no edge)", diag

        diag["kelly_stake"] = stake
        diag["kelly_f"]     = round(f, 6)
        return True, "OK", diag


# ─────────────────────────────────────────────────────────────────────────────
# SESSION RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class SessionRisk:
    def __init__(self):
        self.session_pnl = 0.0
        self.wins        = 0
        self.losses      = 0

    def record_win(self, profit: float):
        self.wins        += 1
        self.session_pnl += profit
        tlog(f"WIN  +${profit:.4f}  |  P&L ${self.session_pnl:+.4f}")
        self._stats()

    def record_loss(self, amount: float):
        self.losses      += 1
        self.session_pnl -= amount
        tlog(f"LOSS -${amount:.2f}  |  P&L ${self.session_pnl:+.4f}")
        self._stats()

    def can_trade(self) -> bool:
        if self.session_pnl >= TARGET_PROFIT:
            info(f"Target profit ${TARGET_PROFIT} reached — stopping")
            return False
        if self.session_pnl <= -STOP_LOSS:
            warn(f"Stop-loss -${STOP_LOSS} hit — stopping")
            return False
        return True

    def _stats(self):
        total = self.wins + self.losses
        wr    = self.wins / total * 100 if total else 0
        info(f"Trades:{total}  W:{self.wins}  L:{self.losses}  "
             f"WR:{wr:.1f}%  P&L:${self.session_pnl:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# DERIV CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class DerivClient:
    def __init__(self):
        self.endpoint   = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
        self.ws         = None
        self._send_q    = None
        self._inbox     = None
        self._send_task = None
        self._recv_task = None

    async def connect(self) -> bool:
        try:
            info(f"Connecting → {self.endpoint}")
            self.ws = await websockets.connect(
                self.endpoint,
                ping_interval=WS_PING,
                ping_timeout=20,
                close_timeout=10,
            )
            self._send_q = asyncio.Queue()
            self._inbox  = asyncio.Queue()
            self._start_io()
            await asyncio.sleep(0)
            await self._send({"authorize": API_TOKEN})
            resp = await self._recv_type("authorize", timeout=15)
            if not resp or "error" in resp:
                msg = (resp or {}).get("error", {}).get("message", "timeout")
                err(f"Auth failed: {msg}")
                return False
            auth = resp.get("authorize", {})
            info(f"Auth OK  |  {auth.get('loginid')}  |  "
                 f"Balance: ${auth.get('balance', 0):.2f}")
            return True
        except Exception as e:
            err(f"Connect error: {e}")
            traceback.print_exc(file=sys.stdout)
            return False

    def _start_io(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        self._send_task = asyncio.create_task(self._send_pump(), name="send")
        self._recv_task = asyncio.create_task(self._recv_pump(), name="recv")

    async def _send_pump(self):
        while True:
            data, fut = await self._send_q.get()
            try:
                await self.ws.send(json.dumps(data))
                if fut and not fut.done():
                    fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done():
                    fut.set_exception(exc)
            finally:
                self._send_q.task_done()

    async def _recv_pump(self):
        try:
            async for raw in self.ws:
                try:
                    await self._inbox.put(json.loads(raw))
                except json.JSONDecodeError:
                    pass
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            await self._inbox.put({"__disconnect__": True})
        except Exception as exc:
            err(f"Recv pump error: {exc}")
            await self._inbox.put({"__disconnect__": True})

    async def close(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def _send(self, data: dict):
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        await self._send_q.put((data, fut))
        await fut

    async def receive(self, timeout: float = 60) -> dict:
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return {}

    async def _recv_type(self, msg_type: str, timeout: float = 10) -> Optional[dict]:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            if "__disconnect__" in msg:
                await self._inbox.put(msg)
                return None
            # When waiting for 'proposal', skip stale proposal_open_contract
            # messages that may have arrived from previous subscriptions —
            # these are the primary cause of proposal timeout.
            if msg_type == "proposal" and "proposal_open_contract" in msg:
                continue   # discard, don't re-queue
            if msg_type in msg or "error" in msg:
                return msg
            await self._inbox.put(msg)

    async def fetch_balance(self) -> Optional[float]:
        try:
            await self._send({"balance": 1})
            resp = await self._recv_type("balance", timeout=10)
            if resp and "balance" in resp:
                return float(resp["balance"]["balance"])
        except Exception as e:
            warn(f"Balance fetch: {e}")
        return None

    async def subscribe_ticks(self) -> bool:
        await self._send({"ticks": SYMBOL, "subscribe": 1})
        resp = await self._recv_type("tick", timeout=10)
        if not resp or "error" in resp:
            msg = (resp or {}).get("error", {}).get("message", "timeout")
            err(f"Tick subscribe failed: {msg}")
            return False
        info(f"Subscribed to {SYMBOL}")
        return True

    async def place_trade(self, stake: float,
                          est: OnlineEstimator) -> Optional[int]:
        """
        Place EXPIRYRANGE ±2.1 for DURATION_MIN minutes on R_10.
        Subscribe to contract updates BEFORE buying (fixes settlement miss).
        """
        await self._send({
            "proposal":      1,
            "amount":        stake,
            "basis":         "stake",
            "contract_type": "EXPIRYRANGE",
            "currency":      "USD",
            "duration":      DURATION_SEC,
            "duration_unit": "s",
            "symbol":        SYMBOL,
            "barrier":       f"+{BARRIER}",
            "barrier2":      f"-{BARRIER}",
        })
        proposal = await self._recv_type("proposal", timeout=20)
        if not proposal or "error" in proposal:
            msg = (proposal or {}).get("error", {}).get("message", "timeout")
            err(f"Proposal error: {msg}")
            return None

        prop   = proposal.get("proposal", {})
        pid    = prop.get("id")
        ask    = float(prop.get("ask_price", stake))
        payout = float(prop.get("payout", 0))

        if not pid:
            err("No proposal ID")
            return None

        est.record_roi(ask, payout)
        roi = (payout - ask) / ask if ask > 0 else 0
        info(f"Proposal OK  |  ask=${ask:.2f}  payout=${payout:.2f}  "
             f"ROI={roi*100:.1f}%  ema_roi={est.ema_roi:.4f}")

        # NOTE: We do NOT subscribe to proposal_open_contract before buying.
        # Doing so without a contract_id causes Deriv to stream ALL open
        # contracts immediately, flooding _inbox and causing _recv_type('proposal')
        # to timeout before the real proposal response arrives.
        # We subscribe by contract_id only AFTER the buy is confirmed.

        buy_ts      = time.time()
        contract_id = None
        await self._send({"buy": pid, "price": ask})

        for attempt in range(BUY_RETRIES):
            resp = await self._recv_type("buy", timeout=8)
            if resp is None:
                warn(f"Buy no response attempt {attempt + 1}")
                continue
            if "error" in resp:
                err(f"Buy error: {resp['error'].get('message', '')}")
                return None
            contract_id = resp.get("buy", {}).get("contract_id")
            if contract_id:
                break

        if not contract_id:
            warn("No contract_id — orphan recovery")
            for _ in range(4):
                await asyncio.sleep(3)
                await self._send({"profit_table": 1, "description": 1,
                                  "sort": "DESC", "limit": 5})
                resp = await self._recv_type("profit_table", timeout=10)
                if resp and "profit_table" in resp:
                    for tx in resp["profit_table"].get("transactions", []):
                        if (abs(float(tx.get("buy_price", 0)) - stake) < 0.01 and
                                float(tx.get("purchase_time", 0)) >= buy_ts - 10):
                            contract_id = tx.get("contract_id")
                            info(f"Orphan recovered → {contract_id}")
                            break
                if contract_id:
                    break
            if not contract_id:
                err("Orphan recovery failed")
                return None

        # Also subscribe specifically to this contract by ID
        try:
            await self._send({
                "proposal_open_contract": 1,
                "contract_id":            contract_id,
                "subscribe":              1,
            })
            slog(f"Post-buy subscription to contract {contract_id} sent")
        except Exception:
            pass

        tlog(f"Placed  |  contract={contract_id}  |  "
             f"EXPIRYRANGE ±{BARRIER}  ${ask:.2f}  {DURATION_MIN}m  symbol={SYMBOL}")
        return contract_id

    async def poll_contract(self, contract_id) -> Optional[dict]:
        """
        Actively fetch the current state of a contract.
        Used by the settlement poller to avoid forced unlocks.
        """
        try:
            slog(f"Polling contract {contract_id} ...")
            await self._send({
                "proposal_open_contract": 1,
                "contract_id":            int(contract_id),
            })
            resp = await self._recv_type("proposal_open_contract", timeout=10)
            if resp and "proposal_open_contract" in resp:
                data = resp["proposal_open_contract"]
                slog(f"Poll result: status={data.get('status')}  "
                     f"is_settled={data.get('is_settled')}  "
                     f"is_sold={data.get('is_sold')}  "
                     f"profit={data.get('profit')}")
                return data
            else:
                slog(f"Poll: no proposal_open_contract in response: "
                     f"{list((resp or {}).keys())}")
        except Exception as e:
            warn(f"Poll error: {e}")
        return None

    @staticmethod
    def contract_ids_match(cid_a, cid_b) -> bool:
        """
        Flexible contract ID matching.
        The API sometimes returns ints, sometimes strings.
        Comparing str(a) == str(b) handles both cases.
        """
        return str(cid_a) == str(cid_b)

    @staticmethod
    def is_settled(data: dict) -> bool:
        """
        Returns True if the contract data represents a final settled state.
        Checks multiple fields because Deriv uses different ones
        depending on the code path (proposal_open_contract vs transaction).
        """
        if data.get("is_settled"):
            return True
        if data.get("is_sold"):
            return True
        status = data.get("status", "").lower()
        return status in ("sold", "won", "lost")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH SERVER
# ─────────────────────────────────────────────────────────────────────────────

class HealthHandler(BaseHTTPRequestHandler):
    bot_ref = None

    def do_GET(self):
        body = b"OK"
        if self.path == "/status" and self.bot_ref:
            b = self.bot_ref
            s = b.session_risk
            m = b.martingale
            body = json.dumps({
                "status":         "running",
                "symbol":         SYMBOL,
                "locked":         b.waiting_for_result,
                "session_wins":   s.wins,
                "session_losses": s.losses,
                "session_pnl":    round(s.session_pnl, 4),
                "balance":        b.balance,
                "martingale":     {
                    "step":      m.step,
                    "streak":    m.consecutive_losses,
                    "max_steps": MARTINGALE_MAX_STEPS,
                    "factor":    MARTINGALE_FACTOR,
                },
                "estimator":      b.est.status(),
            }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_health_server(bot):
    HealthHandler.bot_ref = bot
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    info(f"Health server :{PORT}  →  GET /  or  GET /status")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BOT
# ─────────────────────────────────────────────────────────────────────────────

class ExpiryRangeBot:
    def __init__(self):
        self.tick_tracker = TickRateTracker()
        self.client       = DerivClient()
        self.est          = OnlineEstimator(self.tick_tracker)
        self.gate         = EntryGate()
        self.sizer        = KellySizer()
        self.session_risk = SessionRisk()
        self.martingale   = MartingaleManager()

        self.waiting_for_result: bool           = False
        self._evaluating:        bool           = False
        self.current_trade:      Optional[dict] = None
        self.lock_since:         Optional[float] = None
        self._stop:              bool           = False
        self._bal_before:        Optional[float] = None
        self.balance:            float          = 1000.0

        self.tick_count:         int   = 0
        self.signal_count:       int   = 0
        self.last_trade_time:    float = 0.0
        self.skip_log_count:     int   = 0

        # Settlement poller task reference
        self._poller_task: Optional[asyncio.Task] = None

    def _unlock(self, reason: str = "manual"):
        if self.waiting_for_result:
            cid = (self.current_trade or {}).get("id", "?")
            info(f"Unlock: contract={cid}  reason={reason}")
        self.waiting_for_result = False
        self.current_trade      = None
        self.lock_since         = None
        self._evaluating        = False
        # Cancel poller if it's still running
        if self._poller_task and not self._poller_task.done():
            self._poller_task.cancel()
            self._poller_task = None

    # ── Settlement Poller  (KEY FIX) ─────────────────────────────────────────

    async def _settlement_poller(self, contract_id, expected_expiry: float):
        """
        Actively polls contract status every POLL_INTERVAL seconds.

        This is the primary fix for forced unlocks. Instead of
        passively waiting for WebSocket messages (which can be lost),
        we actively ask the API "is this contract settled yet?"

        Timeline:
          - Starts immediately after trade is placed
          - Sleeps until ~5s before expected expiry (avoid hammering API)
          - Then polls every POLL_INTERVAL seconds until settled
          - Times out after LOCK_TIMEOUT to prevent eternal loops
        """
        wait_before_poll = max(0, expected_expiry - time.time() - 5)
        slog(f"Poller started for {contract_id}  "
             f"(waiting {wait_before_poll:.0f}s before first poll)")

        try:
            if wait_before_poll > 0:
                await asyncio.sleep(wait_before_poll)

            deadline = time.time() + 120   # poll window: 2 extra minutes beyond expiry
            poll_n   = 0

            while self.waiting_for_result and time.time() < deadline:
                poll_n += 1
                slog(f"Poll #{poll_n} for contract {contract_id} "
                     f"(elapsed since expiry: "
                     f"{max(0, time.time()-expected_expiry):.0f}s)")

                data = await self.client.poll_contract(contract_id)

                if data is None:
                    slog(f"Poll #{poll_n}: no data returned — retrying in {POLL_INTERVAL}s")
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                # Log every field we get for debugging
                slog(f"Poll #{poll_n} raw fields: "
                     f"status={data.get('status')!r}  "
                     f"is_settled={data.get('is_settled')!r}  "
                     f"is_sold={data.get('is_sold')!r}  "
                     f"profit={data.get('profit')!r}  "
                     f"sell_price={data.get('sell_price')!r}  "
                     f"sell_time={data.get('sell_time')!r}")

                if DerivClient.is_settled(data):
                    slog(f"Poll #{poll_n}: CONTRACT SETTLED  "
                         f"status={data.get('status')}  profit={data.get('profit')}")
                    ok = await self.handle_settlement(data)
                    if not ok:
                        self._stop = True
                    return   # done — settlement processed
                else:
                    slog(f"Poll #{poll_n}: not yet settled — retrying in {POLL_INTERVAL}s")
                    await asyncio.sleep(POLL_INTERVAL)

            # Fell out of loop
            if self.waiting_for_result:
                warn(f"Poller timed out for {contract_id} after {poll_n} polls — "
                     f"forcing unlock (no result recorded)")
                self._unlock("poller_timeout")

        except asyncio.CancelledError:
            slog(f"Poller for {contract_id} cancelled (settlement received via WS)")
        except Exception as e:
            err(f"Poller exception: {e}")
            traceback.print_exc(file=sys.stdout)

    def _start_poller(self, contract_id, expiry_time: float):
        """Launch the settlement poller as a background task."""
        if self._poller_task and not self._poller_task.done():
            self._poller_task.cancel()
        self._poller_task = asyncio.create_task(
            self._settlement_poller(contract_id, expiry_time),
            name=f"poller_{contract_id}",
        )
        slog(f"Settlement poller task created for contract {contract_id}")

    # ── Lock timeout (last resort, much longer now) ───────────────────────────

    def _check_lock_timeout(self):
        if not self.waiting_for_result or self.lock_since is None:
            return
        elapsed = time.monotonic() - self.lock_since
        if elapsed >= LOCK_TIMEOUT:
            warn(f"Hard lock timeout ({LOCK_TIMEOUT}s) — force unlock. "
                 f"This should NOT happen if poller is working correctly.")
            self._unlock("hard_timeout")

    # ── Tick handler ──────────────────────────────────────────────────────────

    async def on_tick(self, price: float):
        self.tick_count += 1
        self.est.ingest(price)
        self._check_lock_timeout()

        if self.tick_count % 30 == 0:
            if not self.est.ready():
                warmup_left = MIN_WARMUP_TICKS - self.est.n
                info(f"tick={self.tick_count}  price={price:.4f}  "
                     f"WARMUP({warmup_left} ticks remaining)  "
                     f"σ={self.est.sigma:.5f}  "
                     f"tick_iv={self.tick_tracker.avg_interval:.3f}s")
            else:
                locked = "LOCKED" if self.waiting_for_result else "READY"
                mgstate = f"  marti=[{self.martingale.describe()}]" if self.martingale.is_recovering else ""
                info(f"tick={self.tick_count}  price={price:.4f}  {locked}  "
                     f"σ={self.est.sigma:.5f}  regime={self.est.regime()}  "
                     f"P(win)={self.est.p_win():.4f}  "
                     f"EV={self.est.ev_per_dollar():+.4f}"
                     f"{mgstate}")

        if self.waiting_for_result or self._evaluating:
            return
        if not self.session_risk.can_trade():
            self._stop = True
            return
        if time.time() - self.last_trade_time < TRADE_COOLDOWN:
            return

        self._evaluating = True
        try:
            await self._evaluate(price)
        finally:
            self._evaluating = False

    # ── Distribution evaluation ───────────────────────────────────────────────

    async def _evaluate(self, price: float):
        if self.waiting_for_result:
            return

        allow, reason, diag = self.gate.evaluate(
            self.est, self.balance, self.sizer)

        if not allow:
            self.skip_log_count += 1
            if self.skip_log_count % 60 == 1:
                info(f"SKIP: {reason}  "
                     f"(σ={self.est.sigma:.5f}  "
                     f"P(win)={self.est.p_win():.4f}  "
                     f"EV={self.est.ev_per_dollar():+.4f})")
            return

        self.skip_log_count = 0

        pw          = self.est.p_win()
        roi         = self.est.ema_roi
        kelly_stake = diag.get("kelly_stake", MIN_STAKE)
        f           = diag.get("kelly_f", 0.0)
        ev          = self.est.ev_per_dollar()

        # ── Martingale stake override ────────────────────────────────────────
        final_stake = self.martingale.compute_stake(kelly_stake)

        self.signal_count += 1
        info("=" * 62)
        info(f"TRADE #{self.signal_count}  |  tick={self.tick_count}  |  symbol={SYMBOL}")
        info(f"  σ={self.est.sigma:.6f}  regime={self.est.regime()}  "
             f"stability={self.est.sigma_stability:.3f}")
        info(f"  P(win)={pw:.6f}  ROI={roi:.4f}  EV={ev:+.6f}")
        info(f"  Kelly_f={f:.6f}  kelly_stake=${kelly_stake:.2f}")
        info(f"  Martingale: {self.martingale.describe()}")
        info(f"  FINAL stake=${final_stake:.2f}  balance=${self.balance:.2f}")
        info(f"  tick_interval={self.tick_tracker.avg_interval:.3f}s  "
             f"N_eff={self.tick_tracker.effective_n_ticks:.1f}")
        info("=" * 62)

        self._bal_before = await self.client.fetch_balance()
        if self._bal_before is not None:
            self.balance = self._bal_before
            info(f"Pre-trade balance: ${self._bal_before:.2f}")

        expiry_time = time.time() + DURATION_SEC   # wall clock expiry estimate

        contract_id = await self.client.place_trade(final_stake, self.est)

        if contract_id:
            self.current_trade = {
                "id":        contract_id,
                "stake":     final_stake,
                "p_win":     pw,
                "roi":       roi,
                "ev":        ev,
                "kelly_f":   f,
                "sigma":     self.est.sigma,
                "regime":    self.est.regime(),
                "marti_step": self.martingale.step,
            }
            self.waiting_for_result = True
            self.lock_since         = time.monotonic()
            self.last_trade_time    = time.time()

            # ── Start settlement poller immediately ──────────────────────────
            self._start_poller(contract_id, expiry_time)

            jlog({
                "type":       "trade",
                "cid":        contract_id,
                "stake":      final_stake,
                "kelly_stake": kelly_stake,
                "marti_step": self.martingale.step,
                "p_win":      round(pw, 6),
                "roi":        round(roi, 4),
                "ev":         round(ev, 6),
                "kelly_f":    round(f, 6),
                "sigma":      round(self.est.sigma, 6),
                "regime":     self.est.regime(),
                "stability":  round(self.est.sigma_stability, 4),
                "N_eff":      round(self.tick_tracker.effective_n_ticks, 2),
                "ts":         _ts(),
            })
        else:
            self._bal_before = None
            warn("Placement failed — ready for next evaluation")
            self.last_trade_time = time.time()

    # ── Settlement handler ────────────────────────────────────────────────────

    async def handle_settlement(self, data: dict) -> bool:
        """
        Process settlement data regardless of how it arrived
        (WebSocket push or poller pull).

        Robust contract ID matching handles str/int mismatches.
        All code paths are heavily logged for debugging.
        """
        cid = data.get("contract_id")

        # ── Debug: log every settlement candidate ────────────────────────────
        slog(f"handle_settlement called: incoming_cid={cid!r}  "
             f"waiting={self.waiting_for_result}  "
             f"current_id={self.current_trade.get('id') if self.current_trade else None!r}")

        if not self.waiting_for_result:
            slog("handle_settlement: not waiting for result — ignoring")
            return True

        if not self.current_trade:
            slog("handle_settlement: no current trade — ignoring")
            return True

        expected_cid = self.current_trade["id"]
        if not DerivClient.contract_ids_match(cid, expected_cid):
            slog(f"handle_settlement: CID mismatch "
                 f"(got {cid!r}  expected {expected_cid!r}) — ignoring")
            return True

        if not DerivClient.is_settled(data):
            slog(f"handle_settlement: CID matched but NOT yet settled  "
                 f"status={data.get('status')!r}  "
                 f"is_settled={data.get('is_settled')!r}  "
                 f"is_sold={data.get('is_sold')!r}")
            return True

        # ── Settlement confirmed ─────────────────────────────────────────────
        slog(f"handle_settlement: CONFIRMED SETTLED  "
             f"cid={cid}  status={data.get('status')}  profit={data.get('profit')}")

        profit = float(data.get("profit", 0))
        status = data.get("status", "unknown")
        stake  = self.current_trade["stake"]

        bal_after = await self.client.fetch_balance()
        if bal_after is not None and self._bal_before is not None:
            actual = round(bal_after - self._bal_before, 4)
            self.balance = bal_after
        else:
            actual = profit

        tlog(f"SETTLED  |  contract={cid}  |  "
             f"status={status}  |  profit=${actual:+.4f}  |  "
             f"source={'poller' if self._poller_task and not self._poller_task.done() else 'websocket'}")

        won = actual > 0

        # ── Update martingale ────────────────────────────────────────────────
        if won:
            self.martingale.record_win()
        else:
            self.martingale.record_loss()

        mlog(f"After settlement: {self.martingale.describe()}")

        self.est.record_outcome(won, actual)

        if won:
            self.session_risk.record_win(actual)
        else:
            self.session_risk.record_loss(stake)

        self.est.log_status()

        jlog({
            "type":       "result",
            "cid":        cid,
            "status":     status,
            "profit":     actual,
            "pnl":        round(self.session_risk.session_pnl, 4),
            "wins":       self.session_risk.wins,
            "losses":     self.session_risk.losses,
            "sigma":      round(self.est.sigma, 6),
            "p_win_was":  round(self.current_trade.get("p_win", 0), 6),
            "ev_was":     round(self.current_trade.get("ev", 0), 6),
            "marti_step_was": self.current_trade.get("marti_step", 0),
            "marti_step_next": self.martingale.step,
            "ts":         _ts(),
        })

        self._bal_before = None
        self._unlock("settlement")
        info("Ready for next evaluation")
        return self.session_risk.can_trade()

    # ── Reconnect ─────────────────────────────────────────────────────────────

    async def _reconnect(self) -> bool:
        delay   = RECONNECT_MIN
        attempt = 0
        while not self._stop:
            attempt += 1
            warn(f"Reconnect attempt {attempt} in {delay}s ...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX)
            await self.client.close()
            self.client = DerivClient()
            try:
                if not await self.client.connect():
                    continue
                if not await self.client.subscribe_ticks():
                    continue
                if self.waiting_for_result and self.current_trade:
                    cid = self.current_trade["id"]
                    info(f"Re-attaching contract {cid}")
                    data = await self.client.poll_contract(cid)
                    if data:
                        await self.handle_settlement(data)
                    if self.waiting_for_result:
                        await self.client._send({
                            "proposal_open_contract": 1,
                            "contract_id":            cid,
                            "subscribe":              1,
                        })
                info("Reconnect OK")
                return True
            except Exception as e:
                err(f"Reconnect error: {e}")
        return False

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self):
        info("=" * 62)
        info("EXPIRYRANGE BOT  v2  —  R_10  —  Distribution Thinking")
        info(f"Symbol     : {SYMBOL}")
        info(f"Contract   : EXPIRYRANGE ±{BARRIER}  {DURATION_MIN}min")
        info(f"Model      : Gaussian P(win)=erf(2.1/√2σ√N_eff)")
        info(f"Gate       : P(win)≥{MIN_P_WIN}  EV≥{MIN_EV}  "
             f"regime∈{{CALM,NORMAL}}")
        info(f"Martingale : base=${MARTINGALE_BASE_STAKE}  "
             f"factor={MARTINGALE_FACTOR}×  max_steps={MARTINGALE_MAX_STEPS}")
        info(f"Session    : target=+${TARGET_PROFIT}  stop=-${STOP_LOSS}")
        info(f"Poller     : every {POLL_INTERVAL}s (settlement fix)")
        info("=" * 62)

        if API_TOKEN in ("REPLACE_WITH_YOUR_TOKEN", ""):
            err("Set DERIV_API_TOKEN before running")
            return

        if not await self.client.connect():
            return

        bal = await self.client.fetch_balance()
        if bal:
            self.balance = bal
            info(f"Starting balance: ${self.balance:.2f}")

        if not await self.client.subscribe_ticks():
            return

        info(f"Live — warming up estimator ({MIN_WARMUP_TICKS} ticks) ...")

        try:
            while not self._stop:
                response = await self.client.receive(timeout=60)

                if "__disconnect__" in response:
                    warn("WS disconnected — reconnecting")
                    if not await self._reconnect():
                        break
                    continue

                if not response:
                    try:
                        await self.client.ws.ping()
                    except Exception:
                        warn("Ping failed — reconnecting")
                        if not await self._reconnect():
                            break
                    continue

                if "tick" in response:
                    quote = response["tick"].get("quote")
                    if quote is not None:
                        await self.on_tick(float(quote))

                if "balance" in response:
                    bal = response["balance"].get("balance")
                    if bal is not None:
                        self.balance = float(bal)

                # ── WebSocket settlement paths ────────────────────────────────
                # These are best-effort. The poller handles what WS misses.

                if "proposal_open_contract" in response:
                    poc = response["proposal_open_contract"]
                    slog(f"WS proposal_open_contract: cid={poc.get('contract_id')}  "
                         f"status={poc.get('status')}  "
                         f"is_settled={poc.get('is_settled')}")
                    ok = await self.handle_settlement(poc)
                    if not ok:
                        self._stop = True

                if "buy" in response:
                    slog(f"WS buy response received (checking for early settlement)")
                    ok = await self.handle_settlement(response["buy"])
                    if not ok:
                        self._stop = True

                if "transaction" in response:
                    tx = response["transaction"]
                    if "contract_id" in tx:
                        slog(f"WS transaction: action={tx.get('action')}  "
                             f"cid={tx.get('contract_id')}  "
                             f"profit={tx.get('profit')}")
                        ok = await self.handle_settlement({
                            "contract_id": tx.get("contract_id"),
                            "profit":      tx.get("profit", 0),
                            "status":      tx.get("action", "sold"),
                            "is_settled":  True,
                        })
                        if not ok:
                            self._stop = True

        except KeyboardInterrupt:
            info("Stopped by user")
        except Exception as e:
            err(f"Fatal: {e}")
            traceback.print_exc(file=sys.stdout)
        finally:
            self._print_final()
            await self.client.close()
            info("Bot exited.")

    def _print_final(self):
        s     = self.session_risk
        m     = self.martingale
        total = s.wins + s.losses
        wr    = s.wins / total * 100 if total else 0
        info("=" * 62)
        info("FINAL SESSION STATS")
        info(f"Live ticks    : {self.tick_count}")
        info(f"Trades placed : {self.signal_count}")
        info(f"W:{s.wins}  L:{s.losses}  WR:{wr:.1f}%  P&L:${s.session_pnl:.4f}")
        info(f"Final σ       : {self.est.sigma:.6f}")
        info(f"Final regime  : {self.est.regime()}")
        info(f"Final P(win)  : {self.est.p_win():.6f}")
        info(f"Final EV/$    : {self.est.ev_per_dollar():+.6f}")
        info(f"ROI tracked   : {self.est.ema_roi:.4f}")
        info(f"tick_interval : {self.tick_tracker.avg_interval:.4f}s")
        info(f"Martingale    : step={m.step}  streak={m.consecutive_losses}")
        for r, st in self.est.regime_stats.items():
            if st["trades"] > 0:
                rwr = st["wins"] / st["trades"] * 100
                info(f"  {r}: {st['trades']} trades  WR:{rwr:.1f}%")
        info("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    bot = ExpiryRangeBot()
    start_health_server(bot)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
