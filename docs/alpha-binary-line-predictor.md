# Line predictor: exact definition

This is the reference definition of the line predictor in [alpha-binary.md](alpha-binary.md) §3.1.
The implementation is `research/alpha_binary/features_line.py` (`line`, `next_step`); where this
text and the code differ, the code was measured.

## Notation

- `m[y][x]` is the mask, W × H, rows decoded top to bottom, pixels left to right.
- A **transition list** of row r is a list of (position, kind) pairs in increasing position:
  - `(0, m[r][0])`: the row start, with the kind of the first pixel (0 or 1);
  - `(t, m[r][t])` for every 1 ≤ t < W with `m[r][t] ≠ m[r][t−1]` (kind 0 = falling, 1 = rising);
  - `(W, 2)`: the row end.

  Positions are pixel indices: a transition at t means pixel t is the first pixel of a new run.
- **link(a, r)**: for transition a of row r + 1, the transition of row r with the same kind whose
  position is nearest to a's position. On a tie, take the left one. If row r has no transition of
  that kind, there is no link. Kinds 0, 1 and 2 only link to their own kind.

## The predicted end E of the current run

For pixel (y, x) with y ≥ 1 and x ≥ 1 (at x = 0 or y = 0 the feature bits are all 0):

```
s       = the last transition of row y with position ≤ x − 1      (the current run's start)
s1      = link(s, y − 1)                                           (no link: bits are 0)
e1      = the transition after s1 in row y − 1                     (where that run ended)
if e1 is the row end:  E = W, certain = 0
else:
    positions p0 = e1.position, p1 = link(e1).position, p2, ... following links upward,
    stop at the top row, a missing link, or 24 steps
    deltas  = [p_k − p_{k+1}]  in time order (oldest first)
    (step, certain) = next_step(deltas)       (no deltas: step = 0, certain = 0)
    E = e1.position + step
```

The four feature bits for pixel (y, x) are:

| Bit | Meaning |
|---|---|
| here | E is known and x = E |
| past | E is known and x > E |
| near | E is known and \|x − E\| = 1 |
| certain | the step was forced by the rules below |

## next_step(deltas)

```
last  = deltas[-1]
V     = set of values in deltas
if |V| = 1:                    return (last, uncertain)
if |V| > 2 or max V − min V ≠ 1: return (last, uncertain)       # not a straight line
other = the value in V that is not last
runs  = run-length encoding of deltas: [(value, length), ...]
complete = runs without the first and the last run              # those two may be cut off
isolated = the value v for which complete runs of v exist and all have length 1,
           if exactly one value qualifies; otherwise none
if isolated is none:
    if complete is not empty and every complete run has length 1:    # a, b, a, b, ...
        return (other, certain if the current run has length 1)
    return (last, uncertain)
if last = isolated:
    return (other, certain if the current run has length 1)
L = lengths of complete runs of last
if L is empty:                 return (last, uncertain)
cur = length of the current (last) run
if cur < min L:                return (last, certain)
if cur ≥ max L:                return (other, certain if max L − min L ≤ 1)
return (last, uncertain)                                         # min L ≤ cur < max L
```

## Why it works

The intersections of a straight line y = αr + β with the pixel rows, rounded down, form a
**Sturmian** sequence of steps. It uses only two neighbouring values a = ⌊α⌋ and a + 1, one of
them is isolated (it never occurs twice in a row, unless α is close to a half integer, where the
two alternate), and the runs of the other value have only two lengths, L and L + 1. So:

- when the current run of the frequent step is shorter than every complete run seen, the line
  must continue with the same step;
- when it is as long as the longest run seen, and runs differ by at most one, it must switch;
- after the isolated step, it must switch back.

Only the case cur = L < L + 1 is undecided by the first level of this structure. That is where
the "certain" bit is 0 and CTW learns how often that happens.

## Worked examples

**Steep line, 2 rows down per 1 row across (slope 1/2 px per row).** Edge positions per row: 10, 10,
11, 11, 12, 12, 13 → deltas 0, 1, 0, 1, 0, 1. Both values have complete runs of length 1, so it's
the alternating case: last = 1, other = 0, current run length 1 → step 0, certain. Next row: 13.

**Shallow line, about 2.4 px per row.** Deltas 2, 2, 3, 2, 2, 3, 2, 3, 2, 2. The isolated value is 3,
and the complete runs of 2 have lengths 2, 2, 1. Last = 2, current run length 2 ≥ max L = 2, and
max L − min L = 1 → step 3, certain.

**Steep, irrational slope (≈ 0.29 px per row).** Deltas 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0. The isolated value is 1;
the complete runs of 0 have lengths 2 and 3. Current run length 2: min L = 2 ≤ 2 < 3 = max L →
step 0, uncertain. One row later (run length 3 ≥ max L) → step 1, certain.

**Vertical edge.** Deltas 0, 0, 0, 0 → one value only → step 0, uncertain (CTW learns that
vertical edges usually stay vertical).

**Corner.** Deltas 0, 0, 0, 5 → values differ by 5 → not a line → step 5, uncertain.
