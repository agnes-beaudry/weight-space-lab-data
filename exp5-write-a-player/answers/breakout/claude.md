# Exp 5 · Breakout · Claude Opus 5.5

# Hand-written policy network for MinAtar Breakout

## Design

**Strategy.** The policy predicts where the ball will be when it reaches row 8 on its way down, then moves the paddle into the catch zone and waits there.

**Catch rule.** The paddle moves before the ball. When the ball is at row 8 in column x8 moving horizontally in direction d8, the ball is caught if the paddle ends that step at x8 or at x8+d8 (clamped to the board). Call this the zone `[lo, hi]`.
- If P < lo, move right.
- If P > hi, move left.
- Otherwise, no-op.

The 2-cell zone also makes the paddle tolerant of a sticky repeat when it arrives.

**Landing prediction.** Horizontal motion is a reflection with clamping, so the landing column is a fold. With X = bx + d·n, the landing column x8 is X itself if 0 ≤ X ≤ 9, −1−X if X < 0, and 19−X if X > 9. Vertical bounces also hold the ball in place for one step, so the number of steps until the ball is at row 8 is:
- **Descending:** n = 8 − r.
- **Ascending:** assume the earliest possible turn, at row min(4, r), which gives n = r + 9 − 2·min(4, r). Bricks only exist in rows 1–3, so this is the most urgent case. If the ball goes higher, the prediction updates while time remains.
- **Ball and trail in the same row:** this is treated as descending (a ceiling or brick bounce), except at row 8, where it is a paddle bounce and treated as ascending.

For a fixed direction mode, lo and hi are pure functions of the ball cell. That makes them a lookup table, which is linear in the one-hot ball input.

**Layer 1 (10 units):**
- **Units 0–7: comparison units.** For each mode (down-right, down-left, up-right, up-left) there is:
  - a "go right" unit: `tanh(6·(lo(ball) − P − 0.5))`;
  - a "go left" unit: `tanh(6·(P − hi(ball) − 0.5))`.
  
  The paddle position P enters linearly through the row-9 paddle cells.
- **Unit 8: X, horizontal direction.** It computes sign(bx − tx + [bx=0] − 0.5). It is +1 when the ball moves right, with a wall correction for the case where the ball and trail share a column after a wall bounce.
- **Unit 9: U, vertical direction.** It computes sign(2(ty − by) − 1 + 2[by=8]). It is +1 when the ball is ascending or level at row 8, and −1 when descending or level elsewhere.

**Layer 2 (8 units).** Each unit is the AND of one comparison unit with its mode, built from X and U: `tanh(5·(u + ±X + ±U − 2))`.

**Output.**
- The right logit (index 3) is `36 + 12·Σ` of the four "go right" layer-2 units, which gives ±12.
- The left logit (index 1) is built the same way from the four "go left" units.
- The no-op-like actions (n, u, d, f) stay at 0, so the paddle waits when it is in the zone.

All other weights are 0.

```edits
# ---------- unit 0: down-right mode, "go right" (table lo) ----------
W1[3:60:6, 0] = [8,9,8,7,6,5,4,3,2,1]
W1[63:120:6, 0] = [7,8,9,8,7,6,5,4,3,2]
W1[123:180:6, 0] = [6,7,8,9,8,7,6,5,4,3]
W1[183:240:6, 0] = [5,6,7,8,9,8,7,6,5,4]
W1[243:300:6, 0] = [4,5,6,7,8,9,8,7,6,5]
W1[303:360:6, 0] = [3,4,5,6,7,8,9,8,7,6]
W1[363:420:6, 0] = [2,3,4,5,6,7,8,9,8,7]
W1[423:480:6, 0] = [1,2,3,4,5,6,7,8,9,8]
W1[483:540:6, 0] = [0,1,2,3,4,5,6,7,8,9]
W1[540:600:6, 0] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 0] *= 6
b1[0] = -3
# ---------- unit 1: down-right mode, "go left" (table hi) ----------
W1[3:60:6, 1] = [9,9,9,8,7,6,5,4,3,2]
W1[63:120:6, 1] = [8,9,9,9,8,7,6,5,4,3]
W1[123:180:6, 1] = [7,8,9,9,9,8,7,6,5,4]
W1[183:240:6, 1] = [6,7,8,9,9,9,8,7,6,5]
W1[243:300:6, 1] = [5,6,7,8,9,9,9,8,7,6]
W1[303:360:6, 1] = [4,5,6,7,8,9,9,9,8,7]
W1[363:420:6, 1] = [3,4,5,6,7,8,9,9,9,8]
W1[423:480:6, 1] = [2,3,4,5,6,7,8,9,9,9]
W1[483:540:6, 1] = [1,2,3,4,5,6,7,8,9,9]
W1[540:600:6, 1] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 1] *= -6
b1[1] = -3
# ---------- unit 2: down-left mode, "go right" (table lo) ----------
W1[3:60:6, 2] = [7,6,5,4,3,2,1,0,0,0]
W1[63:120:6, 2] = [6,5,4,3,2,1,0,0,0,1]
W1[123:180:6, 2] = [5,4,3,2,1,0,0,0,1,2]
W1[183:240:6, 2] = [4,3,2,1,0,0,0,1,2,3]
W1[243:300:6, 2] = [3,2,1,0,0,0,1,2,3,4]
W1[303:360:6, 2] = [2,1,0,0,0,1,2,3,4,5]
W1[363:420:6, 2] = [1,0,0,0,1,2,3,4,5,6]
W1[423:480:6, 2] = [0,0,0,1,2,3,4,5,6,7]
W1[483:540:6, 2] = [0,0,1,2,3,4,5,6,7,8]
W1[540:600:6, 2] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 2] *= 6
b1[2] = -3
# ---------- unit 3: down-left mode, "go left" (table hi) ----------
W1[3:60:6, 3] = [8,7,6,5,4,3,2,1,0,1]
W1[63:120:6, 3] = [7,6,5,4,3,2,1,0,1,2]
W1[123:180:6, 3] = [6,5,4,3,2,1,0,1,2,3]
W1[183:240:6, 3] = [5,4,3,2,1,0,1,2,3,4]
W1[243:300:6, 3] = [4,3,2,1,0,1,2,3,4,5]
W1[303:360:6, 3] = [3,2,1,0,1,2,3,4,5,6]
W1[363:420:6, 3] = [2,1,0,1,2,3,4,5,6,7]
W1[423:480:6, 3] = [1,0,1,2,3,4,5,6,7,8]
W1[483:540:6, 3] = [0,1,2,3,4,5,6,7,8,9]
W1[540:600:6, 3] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 3] *= -6
b1[3] = -3
# ---------- unit 4: up-right mode, "go right" (table lo) ----------
W1[3:60:6, 4] = [9,8,7,6,5,4,3,2,1,0]
W1[63:120:6, 4] = [8,9,8,7,6,5,4,3,2,1]
W1[123:180:6, 4] = [7,8,9,8,7,6,5,4,3,2]
W1[183:240:6, 4] = [6,7,8,9,8,7,6,5,4,3]
W1[243:300:6, 4] = [5,6,7,8,9,8,7,6,5,4]
W1[303:360:6, 4] = [6,7,8,9,8,7,6,5,4,3]
W1[363:420:6, 4] = [7,8,9,8,7,6,5,4,3,2]
W1[423:480:6, 4] = [8,9,8,7,6,5,4,3,2,1]
W1[483:540:6, 4] = [9,8,7,6,5,4,3,2,1,0]
W1[540:600:6, 4] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 4] *= 6
b1[4] = -3
# ---------- unit 5: up-right mode, "go left" (table hi) ----------
W1[3:60:6, 5] = [9,9,8,7,6,5,4,3,2,1]
W1[63:120:6, 5] = [9,9,9,8,7,6,5,4,3,2]
W1[123:180:6, 5] = [8,9,9,9,8,7,6,5,4,3]
W1[183:240:6, 5] = [7,8,9,9,9,8,7,6,5,4]
W1[243:300:6, 5] = [6,7,8,9,9,9,8,7,6,5]
W1[303:360:6, 5] = [7,8,9,9,9,8,7,6,5,4]
W1[363:420:6, 5] = [8,9,9,9,8,7,6,5,4,3]
W1[423:480:6, 5] = [9,9,9,8,7,6,5,4,3,2]
W1[483:540:6, 5] = [9,9,8,7,6,5,4,3,2,1]
W1[540:600:6, 5] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 5] *= -6
b1[5] = -3
# ---------- unit 6: up-left mode, "go right" (table lo) ----------
W1[3:60:6, 6] = [8,7,6,5,4,3,2,1,0,0]
W1[63:120:6, 6] = [7,6,5,4,3,2,1,0,0,0]
W1[123:180:6, 6] = [6,5,4,3,2,1,0,0,0,1]
W1[183:240:6, 6] = [5,4,3,2,1,0,0,0,1,2]
W1[243:300:6, 6] = [4,3,2,1,0,0,0,1,2,3]
W1[303:360:6, 6] = [5,4,3,2,1,0,0,0,1,2]
W1[363:420:6, 6] = [6,5,4,3,2,1,0,0,0,1]
W1[423:480:6, 6] = [7,6,5,4,3,2,1,0,0,0]
W1[483:540:6, 6] = [8,7,6,5,4,3,2,1,0,0]
W1[540:600:6, 6] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 6] *= 6
b1[6] = -3
# ---------- unit 7: up-left mode, "go left" (table hi) ----------
W1[3:60:6, 7] = [9,8,7,6,5,4,3,2,1,0]
W1[63:120:6, 7] = [8,7,6,5,4,3,2,1,0,1]
W1[123:180:6, 7] = [7,6,5,4,3,2,1,0,1,2]
W1[183:240:6, 7] = [6,5,4,3,2,1,0,1,2,3]
W1[243:300:6, 7] = [5,4,3,2,1,0,1,2,3,4]
W1[303:360:6, 7] = [6,5,4,3,2,1,0,1,2,3]
W1[363:420:6, 7] = [7,6,5,4,3,2,1,0,1,2]
W1[423:480:6, 7] = [8,7,6,5,4,3,2,1,0,1]
W1[483:540:6, 7] = [9,8,7,6,5,4,3,2,1,0]
W1[540:600:6, 7] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 7] *= -6
b1[7] = -3
# ---------- unit 8: X = +1 if ball moving right (bx - tx + [bx=0] - 0.5) ----------
W1[3:600:60, 8] = 6
W1[9:600:60, 8] = 6
W1[15:600:60, 8] = 12
W1[21:600:60, 8] = 18
W1[27:600:60, 8] = 24
W1[33:600:60, 8] = 30
W1[39:600:60, 8] = 36
W1[45:600:60, 8] = 42
W1[51:600:60, 8] = 48
W1[57:600:60, 8] = 54
W1[10:600:60, 8] = -6
W1[16:600:60, 8] = -12
W1[22:600:60, 8] = -18
W1[28:600:60, 8] = -24
W1[34:600:60, 8] = -30
W1[40:600:60, 8] = -36
W1[46:600:60, 8] = -42
W1[52:600:60, 8] = -48
W1[58:600:60, 8] = -54
b1[8] = -3
# ---------- unit 9: U = +1 if ascending (or level at row 8) ----------
W1[63:120:6, 9] = -6
W1[123:180:6, 9] = -12
W1[183:240:6, 9] = -18
W1[243:300:6, 9] = -24
W1[303:360:6, 9] = -30
W1[363:420:6, 9] = -36
W1[423:480:6, 9] = -42
W1[483:540:6, 9] = -42
W1[543:600:6, 9] = -54
W1[64:120:6, 9] = 6
W1[124:180:6, 9] = 12
W1[184:240:6, 9] = 18
W1[244:300:6, 9] = 24
W1[304:360:6, 9] = 30
W1[364:420:6, 9] = 36
W1[424:480:6, 9] = 42
W1[484:540:6, 9] = 48
W1[544:600:6, 9] = 54
b1[9] = -3
# ---------- layer 2: AND(comparison, direction mode) ----------
W2[0, 0] = 5
W2[1, 1] = 5
W2[2, 2] = 5
W2[3, 3] = 5
W2[4, 4] = 5
W2[5, 5] = 5
W2[6, 6] = 5
W2[7, 7] = 5
W2[8:10, 0:8] = [[5, 5, -5, -5, 5, 5, -5, -5], [-5, -5, -5, -5, 5, 5, 5, 5]]
b2[0:8] = -10
# ---------- output: right (3) from even units, left (1) from odd units ----------
W3[0, 3] = 12
W3[2, 3] = 12
W3[4, 3] = 12
W3[6, 3] = 12
W3[1, 1] = 12
W3[3, 1] = 12
W3[5, 1] = 12
W3[7, 1] = 12
b3[1] = 36
b3[3] = 36
```
