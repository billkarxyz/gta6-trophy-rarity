![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/billkarxyz/gta6-trophy-rarity/rank.svg)
![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/billkarxyz/gta6-trophy-rarity/reward.svg)

# GTA 6 Trophy Rarity Forecast

An entry for [make.fun's GTA 6 trophy rarity forecast bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast every PlayStation 5 trophy's Exophase earn rate at 2026-11-26T00:00Z, as five quantiles (5/25/50/75/95%) per trophy. Scored by mean pinball loss on the logit scale; lower is better.

## The model

`main.py` (Python 3.13, standard library only) reads every `*.in` case in the input directory and writes one `*.out` forecast per case. The core idea: a trophy's earn rate follows an S-curve in log time -- it rises fast while players reach it, then flattens as the pool saturates -- and a straight line extrapolated 4.5 days out overshoots the flat part badly.

Per case:

- **Deduplicate reads.** Exophase refreshes on its own schedule, so long stretches of identical reads are one stale page, not many observations.
- **Players.** Log players tracked is fitted against log(1 + hours since release) by least squares; its growth and prediction error feed the spread and the flat-history fallback.
- **Rate.** For each trophy, a saturating curve `F(x) = A * sigmoid(a + b*x)` is fitted in log time over the recent 24h of distinct reads, gridding the asymptote `A` and least-squaring `(a, b)` for each; the smallest `A` within 25% of the best residual wins, which prefers the flattening the data already shows. The slope is shrunk toward its tier's mean log-rate slope (rare trophies are quantised, so their own slopes are noisy), the curve is anchored on the last read, and the median is the curve read at the target.
- **Spread.** Horizon random-walk term + curve-fit residual + players-fit prediction error + a staleness term (a flat or stale page hides an unseen refresh) + a quantisation term for rare trophies, combined in quadrature. Quantiles are the median's logit plus the standard normal's quantile at each level times the spread.
- **Fallbacks** keep every case valid: a flat history (the practice cases) uses a staleness-based dilution prior instead of a fit, and an empty history falls back to tier priors with a wide spread.

Deterministic (two runs byte-identical), runs in ~0.06s per case, no network, no dependencies.

## Validation

- Practice cases `p00`/`p01`: output shape, size, monotonicity and determinism verified; against hypothetical refreshed answers (players 638 -> 3,868, earners growing 1x-4x) the model beats the house baseline's pinball loss in every scenario (e.g. 0.21 vs 1.01 at 2x earners growth on p00).
- Synthetic final-like cases with known truth (a cohort simulation of launch week, calibrated to the real 638 -> 3,868 first refresh): pinball loss **0.207** vs house baseline 0.411 and house curve baseline 0.294; on a second, dilution-heavier variant **0.200** vs 0.456 and 0.301. Median logit error ~0.22-0.23 vs ~0.37-0.44 for the curve baseline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
