# GTA VI Trophy Rarity Forecast

Entry for the make.fun competition
`gta6-trophy-rarity-forecast-01` — forecast, for every trophy of Grand Theft
Auto VI on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z,
as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## Program

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads every case file in `/input` (`$MAKEFUN_INPUT_DIR` when set) and writes
one forecast per case to `/output` (`$MAKEFUN_OUTPUT_DIR` when set):

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Method

Per trophy, the forecaster:

1. **Reconstructs earners** `E = rate x players_tracked / 100` from every
   read. Unlike the raw rate, the earner count is monotone: it is not
   diluted when Exophase's tracked-player base jumps at a refresh.
2. **Fits power laws in log-time**, `x = ln(1 + hours since release)`,
   `log E ~ a + b*x` and `log N ~ a + b*x`, on the last 24 h of distinct
   reads (least squares, slopes clipped to [0, 2]), extrapolates both to
   the target, and takes `median rate = 100 * E_pred / N_pred`. With no
   usable trend the median is the last read's rate (persistence).
3. **Corrects for stale-page dilution**: Exophase refreshes in jumps, so a
   page that has sat unchanged for hours is overdue for a refresh that
   will dilute the rates as the tracked-player base jumps. The median is
   shifted down accordingly (by the median logit drop seen at past
   refreshes, or a launch-night prior when none was ever observed),
   scaled down for long horizons where the trend dominates. A trophy
   already at 100% is exempt.
4. **Spreads honestly**: logit sigma combines a random-walk term
   (`0.5 * sqrt(horizon_days)`, the house baseline's), the trend fit's
   residual spread grown by extrapolation leverage, and refresh
   volatility (observed read-to-read jumps, or a prior when the page is
   stale). Quantiles are the median's logit shifted by the normal
   quantiles times sigma.

## Practice validation

The kit ships no answer files, so the practice cases were scored against
answer scenarios built from the facts `RULES.txt` publishes (answer read:
3,868 players tracked, every rate but the one at 100% lower; house
baseline practice loss ~1.51, i.e. a mean logit drop of ~1.51):

| scenario | this entry | house baseline |
|---|---:|---:|
| max dilution (earners unchanged, N=3,868) | 0.25 | 1.69 |
| mean drop 1.51 logit | 0.19 | 1.39 |
| moderate (earners x2, N=3,868) | 0.20 | 1.00 |
| heterogeneous drops, 0.8–1.8 logit | 0.20 | 1.21 |

(mean pinball loss on the logit scale, lower is better; p00 and p01
averaged — see the run log. The one scenario where this entry loses is
"no refresh at all", which the rules state did not happen.)

Two runs on the same case write identical bytes; a run over both practice
cases takes ~0.2 s, far inside the 8 s limit.

## License

MIT — see [LICENSE](LICENSE).
