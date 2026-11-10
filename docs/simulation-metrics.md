# Is it a ball of hair? Checking `evolve_system`

`evolve_system` in `main.py` drops all 207 poems (492 stanzas) into one box as molecules: each
stanza is a particle and each poem a molecule. An animation alone can't tell you whether the
result has structure or is one tangle with every poem threaded through every other, a ball of
hair. These are the numbers used to tell, how they're computed, and what they showed while the
forces were being tuned.

They run on the raw trajectory `system` (shape `(steps + 1, stanzas, 2)`), not on the
normalized coordinates the viewer draws.

## The forces under test

- **Similarity springs** between every pair of stanzas in a poem: stiffness `similarity`,
  rest length `1 - similarity`. Negative similarity is kept, as an anti-spring that pushes.
  Only 2.1% of in-poem pairs are negative (lowest −0.099), and no poem's stiffnesses sum to
  less than zero, so a molecule can't fly apart from them.
- **Novelty backbone**: each stanza bonded to the previous one, stiffness `ks_bond = 1`,
  rest length `novelty`.
- **Cross-poem springs** along the edges of `G`, at `ks_inter = 0.5` of their weight.
- **Gravity** between every pair, `g · n_lines_i · n_lines_j / r²`, softened at `sigma = 0.3`
  (`g m m r / (r² + sigma²)^1.5`) so it can't blow up as two stanzas meet.
- **Contact push** below `sigma`, stiffness `ks_rep = 20`.
- **Friction** `frictionCoeff · m · v`.

## The metrics

### End state (last frame)

| Metric | Computed as | Catches | A good sign |
|---|---|---|---|
| **Nearest neighbour from the same poem** | For each stanza in a poem with 2+ stanzas, find its nearest stanza from any poem. Report the share that come from the same poem. | **The ball of hair itself.** Distinct molecules score high; interpenetrating ones score low. | Well above chance. Random placement scores **2.5%** (the mean of `(k − 1) / 491` over the 348 stanzas in multi-stanza poems, where `k` is the poem's stanza count). |
| **Poem size** (radius of gyration) | For each poem with 2+ stanzas, the root-mean-square distance of its stanzas from the poem's centroid. Report the median and the max. | Molecules coming apart, or stretching into strands. | Small and steady; no runaway max. |
| **Bond error** | Median of `|distance(i, i − 1) − novelty_i|` over all backbone bonds. | Whether the novelty bonds hold their length. | Near 0. |
| **Overlaps** | Pairs of stanzas closer than `sigma / 2 = 0.15`. | Pile-ups, where gravity beats the contact push. | Few. |
| **Mass vs. radius** | Correlation of `n_lines` with distance from the mass-weighted centre. | Whether gravity sorts heavy stanzas inward. | Negative if it does. |

### Over the run

| Metric | Computed as | Catches |
|---|---|---|
| **Spread** | Largest x or y extent of the whole system at t = 0, 25, …, 150. | Collapse into one ball (shrinks toward 0) or ejection (grows). |
| **RMS speed** | Root-mean-square stanza speed, `|r[n+1] − r[n]| / dt`, at the same times. | Freezing (falls to 0) or blow-up (spikes). |
| **Finite** | Any NaN or infinite position. | Numerical failure. |

### Picking `dt`: energy drift

With friction at 0, total energy should be constant, so any drift is error from the time step.
Energy is

```
E = Σ ½ m v²                                 kinetic
  + Σ_{i<j} ½ ks (d − r0)²                   springs and backbone
  + Σ_{i<j} ½ ks_rep · max(sigma − d, 0)²    contact push
  − Σ_{i<j} g m_i m_j / √(d² + sigma²)       softened gravity
```

with `v` taken from successive positions. The integrator is semi-implicit Euler (velocity first,
then position), which is symplectic, so energy oscillates rather than drifting steadily.
Drift is reported as `(max E − min E) / |E₀|` over T = 30.

## Results

### Time step (frictionless, T = 30)

| `g` | `dt = 0.05` | `dt = 0.02` | `dt = 0.01` |
|---|---|---|---|
| 1e-4 | 0.8% | 0.3% | 0.1% |
| 1e-3 | 0.9% | 0.3% | 0.1% |
| 1e-2 | 0.9% | 0.3% | 0.2% |

Each halving of `dt` doubles the run time (8 s, 20 s, 41 s here) for a drift that is already
under 1%. `dt` stays at 0.05.

### Heavy friction (`frictionCoeff = 0.5`), T = 150

| Run | Poem size, median / max | Same-poem neighbour | Bond error | Overlaps | Final spread |
|---|---|---|---|---|---|
| Before this work: clipped springs, no backbone, no gravity | 0.69 / – | 12% | – | 66 | 6.3 |
| Signed springs | 0.70 / 1.89 | 14% | 0.32 | 79 | 6.3 |
| + novelty backbone | 0.57 / 0.87 | 16% | 0.14 | 76 | 6.0 |
| + gravity `g = 1e-4` | 0.54 / 0.86 | 17% | 0.13 | 424 | 4.7 |
| + gravity `g = 3e-4` | 0.57 / 0.84 | 12% | 0.14 | 1472 | 3.6 |
| + gravity `g = 1e-3` | 0.45 / 0.70 | 5% | 0.19 | 5388 | 2.3 |

- Signed springs barely change anything; there are too few negative pairs to matter.
- The backbone tightens molecules (max size 1.89 → 0.87) and holds bonds near `novelty`.
- Gravity collapses the system into one heap, and the overlap count shows it. At `g = 0.01`
  and `0.03`, stanzas were thrown out at speeds above 20. Near contact, gravity between two
  heavy stanzas (71 lines each) is about 200 at `r = sigma`, against a contact push of at most
  `ks_rep · sigma = 6`, so heavy stanzas go straight through each other.
- Even before this work, the system collapsed from 80 wide to 6: spread 80.2, 41.6, 23.4, 15.2,
  7.4, 6.3 at t = 0, 10, 30, 50, 100, 150. Heavy friction freezes it by around t = 100.

### Light friction (`frictionCoeff = 0.02`), T = 150

Current defaults: `g = 1e-3`, with `G` springs. Run through the full `main.py`.

| t | 0 | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|---|
| RMS speed | 1.12 | 8.09 | 6.30 | 5.02 | 3.64 | 2.95 | 2.29 |
| Spread | 80.2 | 64.4 | 58.7 | 58.1 | 35.6 | 31.4 | 22.6 |

End state: poem size 1.54 / 5.33, same-poem neighbour **8%**, bond error 1.14,
overlaps 134, mass vs. radius 0.00.

A variant with no `G` springs and `g = 3e-3`, so gravity is the only pull between poems:

| t | 0 | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|---|
| RMS speed | 0.80 | 4.64 | 3.30 | 2.92 | 2.85 | 2.63 | 2.58 |
| Spread | 80.2 | 67.2 | 49.8 | 33.9 | 44.8 | 73.4 | 88.0 |

End state: same-poem neighbour 7%, overlaps 2872. It collapses, then rebounds outward from the
heap.

## Verdict

At the current defaults it is **partly a ball of hair**. Nothing blows up, nothing collapses to
a point, and the system is still moving at the end, which was the goal of lowering friction.
But only 8% of stanzas sit nearest their own poem: about three times chance (2.5%), and worse
than the 16% under heavy friction. Molecules survive as chains, loose ones (bond error 1.14),
threaded through each other. Gravity at this strength doesn't sort by mass.

Levers that should raise the same-poem score: a stiffer backbone (`ks_bond`), weaker or no
cross-poem springs (`ks_inter`), or a repulsion between poems to balance gravity.

## Reproducing

```python
import numpy as np
import pandas as pd


def system_metrics(r: np.ndarray, stanzas: pd.DataFrame, dt: float = 0.05) -> None:
    poem = stanzas["poem"].to_numpy()
    multi = stanzas["n_stanzas"].to_numpy() > 1
    m = stanzas["n_lines"].to_numpy(float)
    if not np.isfinite(r).all():
        print("BLEW UP")
        return

    ts = [0, 25, 50, 75, 100, 125, 149.9]
    idx = [int(t / dt) for t in ts]
    speed = [np.sqrt((((r[n + 1] - r[n]) / dt) ** 2).sum(1).mean()) for n in idx]
    spread = [np.ptp(r[n], 0).max() for n in idx]
    print("t       " + "  ".join(f"{t:6.0f}" for t in ts))
    print("rms v   " + "  ".join(f"{x:6.2f}" for x in speed))
    print("spread  " + "  ".join(f"{x:6.1f}" for x in spread))

    last = r[-1]
    D = np.linalg.norm(last[:, None] - last[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    df = pd.DataFrame(last, columns=["x", "y"]).assign(poem=poem)
    c = df.groupby("poem")[["x", "y"]].transform("mean")
    rg = np.sqrt(((df[["x", "y"]] - c) ** 2).sum(axis=1).groupby(df["poem"]).mean())
    rg = rg[stanzas.loc[multi, "poem"].unique()]
    b = np.flatnonzero(stanzas["novelty"].notna().to_numpy())
    bond_err = np.median(np.abs(D[b, b - 1] - stanzas["novelty"].to_numpy()[b]))
    com = (last * m[:, None]).sum(0) / m.sum()
    print(
        f"poem size median {rg.median():.2f} max {rg.max():.2f}"
        f" | same-poem neighbour {(poem[D.argmin(1)] == poem)[multi].mean():.2f}"
        f" | bond error {bond_err:.2f}"
        f" | overlaps {(D < 0.15).sum() // 2}"
        f" | mass vs radius {np.corrcoef(m, np.linalg.norm(last - com, axis=1))[0, 1]:+.2f}"
    )


system_metrics(system, stanzas)
```

Run it in the notebook after the `evolve_system` cell. The full `main.py` takes about 85 s on
an M-series Mac.
