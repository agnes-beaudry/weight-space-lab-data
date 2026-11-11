# Exp 5 · Asterix · Claude Opus 5.5

# Asterix policy: hand-set weights for the 600-32-32-6 MLP

## Strategy

The player never moves left or right. It stays in column 5, where it starts. Every enemy and every treasure crosses the whole row it lives in, so each one passes column 5 at some point. The game then becomes a one-dimensional choice of row:
- Leave a row before an enemy reaches column 5.
- Never step into a row where an enemy is at, or about to reach, column 5.
- Step into rows whose treasure is heading toward column 5, then wait there.

The design relies on one fact from the source code. Each entity slot owns one row, so a row holds at most one entity. That means the row's single trail cell tells which way its entity moves. For an entity at x, a trail at x−1 means it moves right; a trail at x+1 means it moves left. Because of this, "approaching enemy" and "approaching treasure" can each be computed as a linear function of that row's cells.

## Layer 1 (24 units used)

- **P_r, units 0–7:** player is in row r, for rows 1–8. Each unit outputs about ±1.
- **D_s, units 8–15:** danger in row s. It uses enemy-cell weights wE and trail weights wT. I solved for these so that the pre-activation gives a set value for each distance d from an approaching enemy to column 5, and −2 for a receding enemy, an empty row or a treasure row.
  - Values: d0 → 0.995, d1 → 0.964, d2 → 0.00, d3 → −0.54, d4 → −0.92, d5 → −0.95.
  - A safe row gives −0.964.
- **G_s, units 16–23:** treasure in row s that is approaching column 5. It gives +0.96 when approaching and −0.96 to −1 otherwise. It has its own direction-decoding trail weights.

## Layer 2 (14 gated units)

- **UP_r** for r = 2–8 and **DN_r** for r = 1–7.
- Each unit has weight 7 on P_r and a bias that includes −7. So the unit sits at exactly −1 unless the player is in row r.
- When active, its pre-activation is:
  - 1.6 × (D_own − D_target)
  - minus 0.2 × (danger two rows away, measured from the safe baseline). This guards against a sticky repeat of the move.
  - plus treasure attraction, which shrinks with row distance (0.75, 0.6, 0.5, …).
  - minus 1.0 if treasure is approaching in the player's own row.
  - minus 0.4 as a cost for moving.
  - plus a small bias toward the centre rows.

## Output

- logit(up) = 12·ΣUP + 72, and logit(down) = 12·ΣDN + 72. Inactive units sum to −6, so each logit equals 12·tanh(active unit).
- At row 1, up is a no-op, so its logit is −12. At row 8, down gets −12 for the same reason.
- No-op is fixed at 0. Left, right and fire are set to −20.

## Expected behaviour

| Situation | Up/down logit (no-op = 0) | Result |
|---|---|---|
| Enemy in own row 1–2 cells away, adjacent row safe | ≈ +10 to +12 | Flees |
| Entering a row with an enemy 0–2 cells from column 5 | ≈ −11 to −12 | Never enters |
| Approaching treasure in the adjacent row | ≈ +4 | Moves toward it |
| Nothing nearby | ≈ −4.4 | Stays |

```edits
# ---------- Layer 1: player-row units 0..7 (rows 1..8) ----------
W1[60:120:6, 0] = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
W1[120:180:6, 1] = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
W1[180:240:6, 2] = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
W1[240:300:6, 3] = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
W1[300:360:6, 4] = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
W1[360:420:6, 5] = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
W1[420:480:6, 6] = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
W1[480:540:6, 7] = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
b1[0:8] = -4
# ---------- Layer 1: danger units 8..15 (rows 1..8): enemy ch1 + trail ch4 ----------
W1[61:120:6, 8] = [0.2, 0.4, 1.6, 2.4, 5.6, 7.4, 5.6, 2.4, 1.6, 0.4]
W1[64:120:6, 8] = [0, -0.2, -0.4, -1.6, -2.4, -5.6, -2.4, -1.6, -0.4, -0.2]
W1[121:180:6, 9] = [0.2, 0.4, 1.6, 2.4, 5.6, 7.4, 5.6, 2.4, 1.6, 0.4]
W1[124:180:6, 9] = [0, -0.2, -0.4, -1.6, -2.4, -5.6, -2.4, -1.6, -0.4, -0.2]
W1[181:240:6, 10] = [0.2, 0.4, 1.6, 2.4, 5.6, 7.4, 5.6, 2.4, 1.6, 0.4]
W1[184:240:6, 10] = [0, -0.2, -0.4, -1.6, -2.4, -5.6, -2.4, -1.6, -0.4, -0.2]
W1[241:300:6, 11] = [0.2, 0.4, 1.6, 2.4, 5.6, 7.4, 5.6, 2.4, 1.6, 0.4]
W1[244:300:6, 11] = [0, -0.2, -0.4, -1.6, -2.4, -5.6, -2.4, -1.6, -0.4, -0.2]
W1[301:360:6, 12] = [0.2, 0.4, 1.6, 2.4, 5.6, 7.4, 5.6, 2.4, 1.6, 0.4]
W1[304:360:6, 12] = [0, -0.2, -0.4, -1.6, -2.4, -5.6, -2.4, -1.6, -0.4, -0.2]
W1[361:420:6, 13] = [0.2, 0.4, 1.6, 2.4, 5.6, 7.4, 5.6, 2.4, 1.6, 0.4]
W1[364:420:6, 13] = [0, -0.2, -0.4, -1.6, -2.4, -5.6, -2.4, -1.6, -0.4, -0.2]
W1[421:480:6, 14] = [0.2, 0.4, 1.6, 2.4, 5.6, 7.4, 5.6, 2.4, 1.6, 0.4]
W1[424:480:6, 14] = [0, -0.2, -0.4, -1.6, -2.4, -5.6, -2.4, -1.6, -0.4, -0.2]
W1[481:540:6, 15] = [0.2, 0.4, 1.6, 2.4, 5.6, 7.4, 5.6, 2.4, 1.6, 0.4]
W1[484:540:6, 15] = [0, -0.2, -0.4, -1.6, -2.4, -5.6, -2.4, -1.6, -0.4, -0.2]
b1[8:16] = -2
# ---------- Layer 1: approaching-treasure units 16..23 (rows 1..8): gold ch2 + trail ch4 ----------
W1[62:120:6, 16] = [4, 4, 10, 10, 16, 16, 16, 10, 10, 4]
W1[64:120:6, 16] = [0, -6, -6, -12, -12, -18, -12, -12, -6, -6]
W1[122:180:6, 17] = [4, 4, 10, 10, 16, 16, 16, 10, 10, 4]
W1[124:180:6, 17] = [0, -6, -6, -12, -12, -18, -12, -12, -6, -6]
W1[182:240:6, 18] = [4, 4, 10, 10, 16, 16, 16, 10, 10, 4]
W1[184:240:6, 18] = [0, -6, -6, -12, -12, -18, -12, -12, -6, -6]
W1[242:300:6, 19] = [4, 4, 10, 10, 16, 16, 16, 10, 10, 4]
W1[244:300:6, 19] = [0, -6, -6, -12, -12, -18, -12, -12, -6, -6]
W1[302:360:6, 20] = [4, 4, 10, 10, 16, 16, 16, 10, 10, 4]
W1[304:360:6, 20] = [0, -6, -6, -12, -12, -18, -12, -12, -6, -6]
W1[362:420:6, 21] = [4, 4, 10, 10, 16, 16, 16, 10, 10, 4]
W1[364:420:6, 21] = [0, -6, -6, -12, -12, -18, -12, -12, -6, -6]
W1[422:480:6, 22] = [4, 4, 10, 10, 16, 16, 16, 10, 10, 4]
W1[424:480:6, 22] = [0, -6, -6, -12, -12, -18, -12, -12, -6, -6]
W1[482:540:6, 23] = [4, 4, 10, 10, 16, 16, 16, 10, 10, 4]
W1[484:540:6, 23] = [0, -6, -6, -12, -12, -18, -12, -12, -6, -6]
b1[16:24] = -2
# ---------- Layer 2: UP_r units 0..6 (r = 2..8), gated by P_r ----------
W2[1, 0] = 7
W2[8:16, 0] = [-1.6, 1.6, 0, 0, 0, 0, 0, 0]
W2[16:24, 0] = [0.375, -0.5, 0, 0, 0, 0, 0, 0]
W2[2, 1] = 7
W2[8:16, 1] = [-0.2, -1.6, 1.6, 0, 0, 0, 0, 0]
W2[16:24, 1] = [0.3, 0.375, -0.5, 0, 0, 0, 0, 0]
W2[3, 2] = 7
W2[8:16, 2] = [0, -0.2, -1.6, 1.6, 0, 0, 0, 0]
W2[16:24, 2] = [0.25, 0.3, 0.375, -0.5, 0, 0, 0, 0]
W2[4, 3] = 7
W2[8:16, 3] = [0, 0, -0.2, -1.6, 1.6, 0, 0, 0]
W2[16:24, 3] = [0.225, 0.25, 0.3, 0.375, -0.5, 0, 0, 0]
W2[5, 4] = 7
W2[8:16, 4] = [0, 0, 0, -0.2, -1.6, 1.6, 0, 0]
W2[16:24, 4] = [0.21, 0.225, 0.25, 0.3, 0.375, -0.5, 0, 0]
W2[6, 5] = 7
W2[8:16, 5] = [0, 0, 0, 0, -0.2, -1.6, 1.6, 0]
W2[16:24, 5] = [0.2, 0.21, 0.225, 0.25, 0.3, 0.375, -0.5, 0]
W2[7, 6] = 7
W2[8:16, 6] = [0, 0, 0, 0, 0, -0.2, -1.6, 1.6]
W2[16:24, 6] = [0.2, 0.2, 0.21, 0.225, 0.25, 0.3, 0.375, -0.5]
# ---------- Layer 2: DN_r units 7..13 (r = 1..7), gated by P_r ----------
W2[0, 7] = 7
W2[8:16, 7] = [1.6, -1.6, -0.2, 0, 0, 0, 0, 0]
W2[16:24, 7] = [-0.5, 0.375, 0.3, 0.25, 0.225, 0.21, 0.2, 0.2]
W2[1, 8] = 7
W2[8:16, 8] = [0, 1.6, -1.6, -0.2, 0, 0, 0, 0]
W2[16:24, 8] = [0, -0.5, 0.375, 0.3, 0.25, 0.225, 0.21, 0.2]
W2[2, 9] = 7
W2[8:16, 9] = [0, 0, 1.6, -1.6, -0.2, 0, 0, 0]
W2[16:24, 9] = [0, 0, -0.5, 0.375, 0.3, 0.25, 0.225, 0.21]
W2[3, 10] = 7
W2[8:16, 10] = [0, 0, 0, 1.6, -1.6, -0.2, 0, 0]
W2[16:24, 10] = [0, 0, 0, -0.5, 0.375, 0.3, 0.25, 0.225]
W2[4, 11] = 7
W2[8:16, 11] = [0, 0, 0, 0, 1.6, -1.6, -0.2, 0]
W2[16:24, 11] = [0, 0, 0, 0, -0.5, 0.375, 0.3, 0.25]
W2[5, 12] = 7
W2[8:16, 12] = [0, 0, 0, 0, 0, 1.6, -1.6, -0.2]
W2[16:24, 12] = [0, 0, 0, 0, 0, -0.5, 0.375, 0.3]
W2[6, 13] = 7
W2[8:16, 13] = [0, 0, 0, 0, 0, 0, 1.6, -1.6]
W2[16:24, 13] = [0, 0, 0, 0, 0, 0, -0.5, 0.375]
b2[0:14] = [-7.675, -7.418, -7.168, -6.943, -6.733, -6.383, -6.033, -6.033, -6.383, -6.733, -6.943, -7.168, -7.418, -7.675]
# ---------- Output ----------
W3[0:7, 2] = [12, 12, 12, 12, 12, 12, 12]
W3[7:14, 4] = [12, 12, 12, 12, 12, 12, 12]
b3[0:6] = [0, -20, 72, -20, 72, -20]
```
