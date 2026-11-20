"""GTA 6 trophy rarity forecast: a saturating-curve model of earn rates.

The earn rate of a trophy is a ratio: earners / players tracked. The house
baseline (the kit starter) freezes that ratio at its last read, and the
house curve baseline extrapolates its log-time trend as a straight line.
Both miss the shape that matters in launch week: a trophy's rate follows an
S-curve in log time -- it rises fast while players reach it, then flattens
as the pool saturates -- and a straight line extrapolated 4.5 days out
overshoots the flat part badly (on a synthetic launch it put early story
trophies near 100% when the truth was ~48%).

Model, per case:
  * Reads are deduplicated: Exophase refreshes on its own schedule, so long
    stretches of identical reads are one stale page, not many observations.
  * Players: log players tracked is fitted against log(1 + hours since
    release) by least squares over the distinct reads. Its growth and its
    prediction error feed the spread and the flat-history fallback.
  * Rate: for each trophy, a saturating curve F(x) = A * sigmoid(a + b*x)
    is fitted in log time by gridding the asymptote A and least-squaring
    (a, b) for each; the smallest A within 5% of the best residual wins,
    which prefers the flattening the data already shows. The slope is
    shrunk toward its tier's mean log-rate slope (rare trophies are
    quantised, so their own slopes are noisy), the curve is anchored on the
    last read, and the median is the curve read at the target, clamped to
    0.01..99.99 percent.
  * Spread (logit units) combines a horizon random-walk term, the curve
    fit's residual, the players fit's prediction error, a staleness term
    (a flat or stale page hides an unseen refresh) and a quantisation term
    for rare trophies. Quantiles are the median's logit plus the standard
    normal's quantile at each level times the spread.
  * Fallbacks keep every case valid: a flat history (the practice cases)
    uses a staleness-based dilution prior instead of a fit, and an empty
    history falls back to tier priors with a wide spread.

Deterministic: pure functions of the case file, files read in name order,
no randomness, no clock reads. Standard library only; runs in seconds.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Fallback release moment when the case carries no usable release_date.
RELEASE_FALLBACK = datetime(2026, 11, 19, tzinfo=timezone.utc)
# The levels the contest scores; used when a case does not name its own.
LEVELS_DEFAULT = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- Model constants (tuned on the practice cases and a synthetic launch) ---
# Base random-walk spread, logit units per sqrt(day) of horizon.
BASE_SIGMA_PER_SQRT_DAY = 0.35
# Recency window for the saturation fit, in hours before the checkpoint:
# the current dynamics (a saturating trophy decelerating, a late trophy
# accelerating) show in the recent reads; the launch phase does not.
FIT_WINDOW_HOURS = 24.0
# Minimum distinct reads in the window before it is used; else all reads.
FIT_WINDOW_MIN_READS = 5
# Shrinkage strength for per-trophy slopes (pseudo-count of reads).
SHRINK_K = 3.0
# Earner count at which a trophy's own slope is trusted at full weight.
FULL_TRUST_EARNERS = 50.0
# Asymptote grid for the saturating curve, as multiples of the max observed rate.
ASYMPTOTE_MULTIPLIERS = (1.05, 1.15, 1.3, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0)
# Parsimony margin: the smallest asymptote within this of the best residual
# wins. Wide, because the residual falls slowly with A while the data cannot
# yet tell a flattening curve from a runaway one.
ASYMPTOTE_TOLERANCE = 1.25
# Slope cap for the saturating curve (logit units per log hour).
SLOPE_B_MAX = 3.0
# Spread clamps, logit units.
SIGMA_MIN, SIGMA_MAX = 0.15, 1.8
# Flat-history fallback: players growth per stale hour (log units), capped.
FLAT_PLAYER_GROWTH_PER_HOUR = 0.25
FLAT_PLAYER_MULT_CAP = 8.0
# Tier priors for the empty-history fallback (median percent, spread).
TIER_PRIOR = {"platinum": 0.05, "gold": 0.5, "silver": 1.5, "bronze": 4.0}
TIER_PRIOR_SIGMA = 1.5


# --- Small numeric helpers -------------------------------------------------

def parse_time(text):
    """Parse an ISO 8601 timestamp or date into an aware UTC datetime."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_between(earlier, later):
    return (later - earlier).total_seconds() / 3600.0


def log_time(hours_since_release):
    """Log-time transform: ln(1 + hours), floored at 0 for pre-release reads."""
    return math.log1p(max(0.0, hours_since_release))


def clamp_pct(value):
    return min(99.99, max(0.01, value))


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of_logit(x):
    return 100.0 / (1.0 + math.exp(-x))


def round4(value):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = math.floor(value * 10000.0 + 0.5) / 10000.0
    return int(scaled) if scaled == int(scaled) else scaled


# Acklam's rational approximation of the standard normal quantile function
# (absolute error below 1.2e-9), so any level list works, not just the five.
_NORM_A = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
           1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
_NORM_B = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
           6.680131188771972e01, -1.328068155288572e01]
_NORM_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
           -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
_NORM_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
           3.754408661907416e00]


def norm_ppf(p):
    """Standard normal quantile at probability p (0 < p < 1)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return ((((( _NORM_C[0] * q + _NORM_C[1]) * q + _NORM_C[2]) * q + _NORM_C[3]) * q
                 + _NORM_C[4]) * q + _NORM_C[5]) / (((( _NORM_D[0] * q + _NORM_D[1]) * q
                 + _NORM_D[2]) * q + _NORM_D[3]) * q + 1.0)
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        return ((((( _NORM_A[0] * r + _NORM_A[1]) * r + _NORM_A[2]) * r + _NORM_A[3]) * r
                 + _NORM_A[4]) * r + _NORM_A[5]) * q / ((((( _NORM_B[0] * r + _NORM_B[1]) * r
                 + _NORM_B[2]) * r + _NORM_B[3]) * r + _NORM_B[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -((((( _NORM_C[0] * q + _NORM_C[1]) * q + _NORM_C[2]) * q + _NORM_C[3]) * q
              + _NORM_C[4]) * q + _NORM_C[5]) / (((( _NORM_D[0] * q + _NORM_D[1]) * q
              + _NORM_D[2]) * q + _NORM_D[3]) * q + 1.0)


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (slope, intercept, resid_sd).

    Returns None when the fit is degenerate (fewer than 2 points or no
    spread in x).
    """
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(1, n - 2)
    resid_sd = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / dof / (sw / n))
    return slope, intercept, resid_sd


def prediction_sd(xs, ys, ws, x_new, resid_sd):
    """Rough OLS prediction sd of the fitted line's value at x_new."""
    n = len(xs)
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12 or n < 3:
        return resid_sd
    leverage = 1.0 / n + (x_new - mx) ** 2 / sxx
    return resid_sd * math.sqrt(1.0 + leverage)


# --- Case preparation ------------------------------------------------------

def distinct_reads(history):
    """Collapse consecutive identical reads into one state each.

    Exophase refreshes on its own schedule, so a run of identical reads is
    one stale page. The state's time is its first occurrence.
    """
    states = []
    last_key = None
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or ()))
        if key != last_key:
            states.append(read)
            last_key = key
    return states


def read_time(read):
    return parse_time(read.get("at")) or parse_time(read.get("slot"))


def fit_saturation(xs, fs, ws):
    """Fit F(x) = A * sigmoid(a + b*x) to observed rates fs at log-times xs.

    The asymptote A is gridded as a multiple of the max observed rate; for
    each A the line logit(F/A) ~ a + b*x is least-squared. The smallest A
    within ASYMPTOTE_TOLERANCE of the best residual wins, which prefers the
    flattening the data already shows over a runaway extrapolation.
    Returns (asymptote, slope, resid_sd) or None when no fit is possible
    (fewer than two positive points, or the rate already at 100%).
    """
    pts_x, pts_f, pts_w = [], [], []
    for x, f, w in zip(xs, fs, ws):
        if f is not None and f > 0:
            pts_x.append(x)
            pts_f.append(f)
            pts_w.append(w)
    if len(pts_x) < 2:
        return None
    f_max = max(pts_f)
    if f_max >= 99.99:
        return None
    fits = []
    for mult in ASYMPTOTE_MULTIPLIERS:
        cap = min(100.0, f_max * mult)
        if cap <= f_max:
            continue
        ys = [logit_of(100.0 * f / cap) for f in pts_f]
        fit = weighted_line(pts_x, ys, pts_w)
        if fit is None:
            continue
        slope, _icept, resid = fit
        fits.append((resid, cap, slope))
    if not fits:
        return None
    best_resid = min(r for r, _c, _s in fits)
    fits.sort(key=lambda t: t[1])
    for resid, cap, slope in fits:
        if resid <= best_resid * ASYMPTOTE_TOLERANCE + 1e-9:
            return cap, slope, resid
    _resid, cap, slope = fits[0]
    return cap, slope, _resid


def log_rate_slope(xs, fs, ws):
    """Slope of ln(rate) against log-time, for the tier shrinkage prior."""
    pts_x, pts_y, pts_w = [], [], []
    for x, f, w in zip(xs, fs, ws):
        if f is not None and f > 0:
            pts_x.append(x)
            pts_y.append(math.log(f))
            pts_w.append(w)
    fit = weighted_line(pts_x, pts_y, pts_w)
    return fit[0] if fit is not None else None


# --- The forecast ----------------------------------------------------------

def forecast_case(case):
    levels = case.get("levels") or LEVELS_DEFAULT
    z_scores = [norm_ppf(level) for level in levels]
    trophies = case.get("trophies") or []
    horizon_days = max(0.0, (case.get("horizon_minutes") or 0) / 1440.0)
    release = parse_time((case.get("game") or {}).get("release_date"))
    if release is None:
        release = RELEASE_FALLBACK
    checkpoint = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    if checkpoint is None:
        checkpoint = RELEASE_FALLBACK
    if target is None:
        target = checkpoint

    history = case.get("history") or []
    if not history or not trophies:
        return {"trophies": [tier_prior_row(t, z_scores) for t in trophies]}

    states = distinct_reads(history)
    last = states[-1]
    players_last = last.get("players_tracked") or 0
    if players_last <= 0:
        return {"trophies": [tier_prior_row(t, z_scores) for t in trophies]}

    x_target = log_time(hours_between(release, target))
    x_last = log_time(hours_between(release, read_time(last) or checkpoint))
    last_rates = last.get("rates") or []

    # Flat history: no trend signal at all (the practice cases). Use the
    # staleness dilution prior instead of a fit.
    if len(states) < 2:
        return flat_history_forecast(case, states[0], release, target,
                                     z_scores)

    # --- Players fit -------------------------------------------------------
    px, py, pw = [], [], []
    for i, st in enumerate(states):
        t = read_time(st)
        p = st.get("players_tracked") or 0
        if t is None or p <= 0:
            continue
        px.append(log_time(hours_between(release, t)))
        py.append(math.log(p))
        pw.append(0.5 + 0.5 * (i / max(1, len(states) - 1)))
    players_fit = weighted_line(px, py, pw)
    if players_fit is None:
        return flat_history_forecast(case, states[0], release, target,
                                     z_scores)
    p_slope, _p_intercept, p_resid = players_fit
    p_slope = max(0.0, p_slope)  # the tracked pool never shrinks
    sigma_players = min(0.8, max(
        0.05, prediction_sd(px, py, pw, x_target, max(p_resid, 0.05))))

    # --- Staleness of the last distinct state ------------------------------
    last_state_time = read_time(states[-1]) or checkpoint
    stale_hours = max(0.0, hours_between(last_state_time, checkpoint))
    if len(states) < 3:
        sigma_stale = 0.5
    elif stale_hours > 2.0:
        sigma_stale = 0.25
    elif stale_hours > 1.0:
        sigma_stale = 0.1
    else:
        sigma_stale = 0.0

    # --- Per-trophy saturation fits ----------------------------------------
    # Recent distinct reads, recency-weighted: the current dynamics (a
    # saturating trophy decelerating, a late trophy accelerating) show in
    # the recent reads; the launch phase does not. Fall back to all reads
    # when the window holds too few distinct states.
    recent = [st for st in states
              if hours_between(read_time(st) or checkpoint,
                               checkpoint) <= FIT_WINDOW_HOURS]
    window = recent if len(recent) >= FIT_WINDOW_MIN_READS else states
    wx = [log_time(hours_between(release, read_time(st) or checkpoint))
          for st in window]
    ww = [0.5 + 0.5 * (i / max(1, len(window) - 1))
          for i in range(len(window))]

    # First pass: each trophy's saturation fit and log-rate slope, plus the
    # tier means of the log-rate slopes for shrinkage.
    own = []  # (fit, log_slope, n_points, earners_last) per trophy
    tier_log = {}
    tier_counts = {}
    global_num, global_den = 0.0, 0.0
    for idx, trophy in enumerate(trophies):
        fs = []
        for st in window:
            rates = st.get("rates") or []
            fs.append(rates[idx] if idx < len(rates) else None)
        fit = fit_saturation(wx, fs, ww)
        log_slope = log_rate_slope(wx, fs, ww)
        n_points = sum(1 for f in fs if f is not None and f > 0)
        rate_last = (last_rates[idx] if idx < len(last_rates)
                     else None)
        earners_last = (rate_last / 100.0 * players_last
                        if rate_last is not None else 0.0)
        own.append((fit, log_slope, n_points, earners_last))
        if log_slope is not None:
            weight = math.sqrt(max(earners_last, 1.0))
            tier = trophy.get("tier") or "bronze"
            num, den = tier_log.get(tier, (0.0, 0.0))
            tier_log[tier] = (num + weight * log_slope, den + weight)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            global_num += weight * log_slope
            global_den += weight
    global_log_slope = (global_num / global_den) if global_den > 0 else 0.0

    sigma_base = BASE_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)
    rows = []
    for idx, trophy in enumerate(trophies):
        fit, log_slope, n_points, earners_last = own[idx]
        tier = trophy.get("tier") or "bronze"
        num, den = tier_log.get(tier, (0.0, 0.0))
        # A tier with fewer than two fitted trophies borrows the global slope.
        prior_slope = (num / den) if den > 0 and tier_counts.get(tier, 0) >= 2 \
            else global_log_slope
        rate_last = (last_rates[idx] if idx < len(last_rates)
                     else None)
        r_last = rate_last if rate_last is not None else 0.0

        if fit is None:
            # No fittable curve: the rate is at 100%, or the trophy is
            # still unearned. Persist (100% stays 100%; unearned grows off
            # a half-earner pseudo-count at the tier's pace), wide spread.
            if r_last >= 99.99:
                median, sigma_fit = 99.9, 0.25
            else:
                pseudo = max(r_last, 0.5 / max(players_last, 1) * 100.0)
                median = clamp_pct(pseudo * math.exp(
                    max(0.0, prior_slope)
                    * max(0.0, x_target - x_last)))
                sigma_fit = 0.4
            sigma_fit = min(0.6, max(0.05, sigma_fit))
        else:
            cap, slope, resid = fit
            trust = min(1.0, math.log1p(earners_last)
                        / math.log1p(FULL_TRUST_EARNERS))
            w = (n_points * trust) / (n_points * trust + SHRINK_K)
            slope_use = w * slope + (1.0 - w) * prior_slope
            slope_use = max(-SLOPE_B_MAX, min(SLOPE_B_MAX, slope_use))
            # Anchor the curve on the last read, then read it at the target.
            # (A last rate of 0.00 is usually rounding, not zero earners,
            # so anchor on a half-earner floor instead.)
            r_anchor = max(r_last, 0.5 / max(players_last, 1) * 100.0)
            anchor = logit_of(100.0 * min(r_anchor, cap * 0.999) / cap)
            icept = anchor - slope_use * x_last
            median = clamp_pct(
                cap / (1.0 + math.exp(-(icept + slope_use * x_target))))
            sigma_fit = min(0.6, max(0.05, resid))

        # Spread: horizon + fit + players + staleness + quantisation.
        sigma_rare = min(0.5, 0.005 / max(r_last, 0.01))
        sigma = math.sqrt(sigma_base ** 2 + sigma_fit ** 2
                          + sigma_players ** 2 + sigma_stale ** 2
                          + sigma_rare ** 2)
        sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))
        rows.append({"id": trophy["id"],
                     "q": quantiles(median, sigma, z_scores)})
    return {"trophies": rows}


def quantiles(median_pct, sigma, z_scores):
    """Five quantiles around a median, monotone and inside 0..100."""
    centre = logit_of(median_pct)
    out = []
    prev = -1.0
    for z in z_scores:
        value = round4(min(100.0, max(0.0, pct_of_logit(centre + z * sigma))))
        value = max(value, prev)  # rounding must not break monotonicity
        out.append(value)
        prev = value
    return out


def tier_prior_row(trophy, z_scores):
    """Fallback row when the case carries no usable history."""
    median = TIER_PRIOR.get(trophy.get("tier") or "bronze", 4.0)
    return {"id": trophy["id"],
            "q": quantiles(median, TIER_PRIOR_SIGMA, z_scores)}


def flat_history_forecast(case, state, release, target, z_scores):
    """Fallback for a history of a single distinct state (practice cases).

    No trend can be fitted, but the page is stale: the next refresh reveals
    all the growth since the last one, and the tracked pool grows far
    faster than any trophy's earners. So the median is diluted by a
    staleness-based players multiplier, with earners growing at a fraction
    of the players' rate (fastest for common trophies, slowest for the
    platinum), and the spread is wide to cover the unseen refresh.
    """
    trophies = case.get("trophies") or []
    players = state.get("players_tracked") or 0
    if players <= 0:
        return {"trophies": [tier_prior_row(t, z_scores) for t in trophies]}
    first_time = read_time(state) or release
    stale_hours = max(0.0, hours_between(first_time, target))
    players_mult = min(FLAT_PLAYER_MULT_CAP,
                       math.exp(FLAT_PLAYER_GROWTH_PER_HOUR * stale_hours))
    horizon_days = max(0.0, (case.get("horizon_minutes") or 0) / 1440.0)
    sigma = min(SIGMA_MAX, max(
        SIGMA_MIN,
        math.sqrt((BASE_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                  + 0.5 ** 2)))
    rates = state.get("rates") or []
    rows = []
    for idx, trophy in enumerate(trophies):
        rate = rates[idx] if idx < len(rates) else None
        if rate is None:
            rows.append(tier_prior_row(trophy, z_scores))
            continue
        if rate >= 99.99:
            median = 99.9
        elif trophy.get("tier") == "platinum":
            earners_mult = math.exp(0.1 * FLAT_PLAYER_GROWTH_PER_HOUR
                                    * stale_hours)
            median = clamp_pct(rate * earners_mult / players_mult)
        else:
            pace = 0.7 if rate > 20.0 else (0.45 if rate > 1.0 else 0.25)
            earners_mult = math.exp(pace * FLAT_PLAYER_GROWTH_PER_HOUR
                                    * stale_hours)
            median = clamp_pct(rate * earners_mult / players_mult)
        rows.append({"id": trophy["id"],
                     "q": quantiles(median, sigma, z_scores)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
