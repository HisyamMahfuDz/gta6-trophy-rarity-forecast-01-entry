# SPDX-License-Identifier: MIT
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case's
target, scored by mean pinball loss on the logit scale.

Method (per trophy, standard library only, deterministic, offline):
  1. Reconstruct earners E = rate * players_tracked / 100 from every read.
     Unlike the raw rate, the earner count is monotone: it is not diluted
     when Exophase's tracked-player base jumps at a refresh.
  2. Fit power laws in log-time, x = ln(1 + hours since release):
         log E ~ a_E + b_E * x,   log N ~ a_N + b_N * x
     on the last 24 h of distinct reads (least squares, slopes clipped to
     [0, 2]).  Extrapolate both to the target and take
         median rate = 100 * E_pred / N_pred.
     With no usable trend (fewer than 3 distinct reads, or no variation)
     the median is the last read's rate (persistence).
  3. Stale-page dilution: Exophase refreshes in jumps.  If the page has sat
     unchanged for hours and no refresh was ever observed, the next refresh
     is overdue and will dilute the rates as the tracked-player base jumps:
     shift the median down by D logit, D = min(1.5, 0.45 * unchanged_hours),
     scaled by 1 / (1 + horizon_days) so a week-long horizon is driven by
     the trend, not by the next refresh.  If refreshes were observed, D is
     the median logit drop seen at a refresh instead.  A trophy already at
     100% is exempt: everyone tracked has it, it cannot dilute.
  4. Honest spread (logit sigma): random-walk term 0.5 * sqrt(horizon_days)
     (the house baseline's), plus the trend fit's residual spread grown by
     the extrapolation leverage, plus refresh volatility (observed
     read-to-read logit jumps, or a 0.7 prior when the page is stale and no
     refresh was ever seen).  Quantiles are the median's logit shifted by
     the standard normal's quantiles times sigma.

Usage (as the grader runs it):
    python3 -P -s /bot/main.py            # reads /input/<id>.in, writes /output/<id>.out
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py   # local run
Reads $MAKEFUN_INPUT_DIR when set, else /input; writes $MAKEFUN_OUTPUT_DIR
when set, else /output.  Reads every *.in file found, in name order, and
writes one <id>.out per case.  Two runs on the same case write identical
bytes: no randomness, no wall-clock use, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import (none is needed: this file is whole).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread of a rate's move, in logit units per square-root day (house baseline).
SIGMA_RW = 0.5
# Power-law slopes for earners / tracked players are clipped to this range:
# earners never shrink, and neither count grows faster than quadratically
# in (1 + hours since release).
SLOPE_MIN, SLOPE_MAX = 0.0, 2.0
# Trend fits use reads from this many hours before the checkpoint.
FIT_WINDOW_HOURS = 24.0
# Stale-page dilution prior: logit shift per unchanged hour, capped.
DILUTION_PER_HOUR = 0.45
DILUTION_CAP = 1.5
# Spread priors (logit): no usable trend fit / stale page with no refresh seen.
SIGMA_NO_FIT = 0.35
SIGMA_REFRESH_PRIOR_STALE = 0.7
SIGMA_REFRESH_PRIOR_FRESH = 0.25
SIGMA_FLOOR, SIGMA_CAP = 0.15, 2.0
# A rate at or above this is treated as "everyone has it" (no dilution).
FULL_RATE = 99.99


def clamp_rate(x: float) -> float:
    """An earn rate clamped to the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp_rate(pct) / (100.0 - clamp_rate(pct)))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text: str) -> float:
    """ISO 8601 UTC timestamp (or a plain YYYY-MM-DD date) to epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def fit_line(xs: list[float], ys: list[float]):
    """Least-squares y ~ a + b*x; returns (a, b, residual_std) or None.

    Needs at least 3 points and a non-degenerate x range.
    """
    n = len(xs)
    if n < 3:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = mean_y - b * mean_x
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    return a, b, stddev(resid)


def forecast_trophy(reads, checkpoint_ts, target_ts, release_ts, horizon_days):
    """Forecast one trophy's five quantiles.

    reads: [(timestamp, players_tracked, rate), ...] oldest first, rate not None.
    Returns [q05, q25, q50, q75, q95] in percent, non-decreasing.
    """
    if not reads:
        # No history at all: a wide, low forecast rather than a crash.
        med_lg, sigma = logit(1.0), 1.5
        return [round4(min(100.0, max(0.01, sigmoid(med_lg + z * sigma))))
                for z in Z]

    last_ts, last_players, last_rate = reads[-1]
    last_rate = float(last_rate)

    # --- How long the page has sat unchanged, and what refreshes did. ---
    # Unchanged run: the earliest read of the trailing run identical to the
    # last read (same players and same rate); U is its age at the checkpoint.
    k = len(reads) - 1
    while k > 0 and reads[k - 1][1] == last_players and reads[k - 1][2] == last_rate:
        k -= 1
    unchanged_hours = max(0.0, (checkpoint_ts - reads[k][0]) / 3600.0)

    # Logit drops at reads where the tracked-player base jumped (a refresh).
    drops = []
    jumps = []
    for (t0, p0, r0), (t1, p1, r1) in zip(reads, reads[1:]):
        if p1 != p0:
            drops.append(logit(r0) - logit(r1))
        if r1 != r0:
            jumps.append(logit(r1) - logit(r0))

    # --- Trend: power laws for earners and for tracked players. ---
    # Drop repeated reads (a stale page re-printed): they carry no new
    # information and would overweight the stale value in the fit.
    distinct = [reads[0]]
    for read in reads[1:]:
        if read[1] != distinct[-1][1] or read[2] != distinct[-1][2]:
            distinct.append(read)
    window = [r for r in distinct if r[0] >= last_ts - FIT_WINDOW_HOURS * 3600]
    fit_reads = window if len(window) >= 3 else distinct

    def x_of(ts):
        return math.log1p(max(0.0, (ts - release_ts) / 3600.0))

    x_target = x_of(target_ts)
    x_last = x_of(last_ts)
    median_rate = last_rate
    sigma_fit = SIGMA_NO_FIT
    fit = None
    if len(fit_reads) >= 3:
        xs = [x_of(r[0]) for r in fit_reads]
        log_e = [math.log(max(float(r[2]) * float(r[1]) / 100.0, 0.5))
                 for r in fit_reads]
        log_n = [math.log(max(float(r[1]), 1.0)) for r in fit_reads]
        fit_e = fit_line(xs, log_e)
        fit_n = fit_line(xs, log_n)
        if fit_e is not None and fit_n is not None:
            fit = (fit_e, fit_n)
            (a_e, b_e, res_e), (a_n, b_n, res_n) = fit_e, fit_n
            b_e = min(SLOPE_MAX, max(SLOPE_MIN, b_e))
            b_n = min(SLOPE_MAX, max(SLOPE_MIN, b_n))
            e_pred = math.exp(a_e + b_e * x_target)
            n_pred = math.exp(a_n + b_n * x_target)
            if n_pred > 0:
                median_rate = 100.0 * e_pred / n_pred
            # Residual spread of both fits, grown by how far past the data
            # the target sits (extrapolation leverage).
            leverage = 1.0 + max(0.0, x_target - x_last)
            sigma_fit = min(1.5, math.sqrt(res_e ** 2 + res_n ** 2) * leverage)

    median_rate = clamp_rate(median_rate)

    # --- Stale-page dilution shift (logit, subtracted from the median). ---
    shift = 0.0
    if last_rate < FULL_RATE:
        if drops:
            dilution = min(DILUTION_CAP, max(0.0, median(drops)))
        elif unchanged_hours >= 2.0:
            # Launch-night prior: the tracked base roughly quadruples at the
            # first refresh (638 -> ~2,500+ players); ln(4) ~= 1.39 logit.
            dilution = min(DILUTION_CAP, DILUTION_PER_HOUR * unchanged_hours)
        else:
            dilution = 0.0
        # Only the next refresh is a dilution event; over a long horizon the
        # trend above already carries the dilution-vs-growth balance.
        shift = dilution / (1.0 + horizon_days)
    med_lg = logit(median_rate) - shift

    # --- Spread (logit sigma). ---
    sigma_rw = SIGMA_RW * math.sqrt(max(0.0, horizon_days))
    if len(jumps) >= 3:
        sigma_refresh = min(1.0, stddev(jumps))
    elif unchanged_hours >= 2.0:
        sigma_refresh = SIGMA_REFRESH_PRIOR_STALE
    else:
        sigma_refresh = SIGMA_REFRESH_PRIOR_FRESH
    if last_rate >= FULL_RATE:
        # Everyone tracked has it; only the random-walk term can move it.
        sigma = max(SIGMA_FLOOR, sigma_rw)
        med_lg = logit(median_rate)
    else:
        sigma = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2 + sigma_refresh ** 2)
    sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

    quantiles = []
    for z in Z:
        q = sigmoid(med_lg + z * sigma)
        q = min(100.0, max(0.01, q))
        quantiles.append(round4(q))
    # Rounding can only tie neighbours; enforce non-decreasing explicitly.
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    checkpoint_ts = parse_time(case["at"])
    target_ts = parse_time(case["target"])
    try:
        release_ts = parse_time(case["game"]["release_date"])
    except (KeyError, TypeError, ValueError):
        release_ts = parse_time(case["history"][0]["at"])
    horizon_days = float(case["horizon_minutes"]) / 1440.0

    trophies = case["trophies"]
    # Per-trophy read series, oldest first, skipping reads that lack the rate.
    series = [[] for _ in trophies]
    for read in case["history"]:
        ts = parse_time(read["at"])
        players = read["players_tracked"]
        for i, rate in enumerate(read["rates"]):
            if rate is not None:
                series[i].append((ts, players, float(rate)))

    rows = []
    for i, trophy in enumerate(trophies):
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(series[i], checkpoint_ts, target_ts,
                                 release_ts, horizon_days),
        })
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
