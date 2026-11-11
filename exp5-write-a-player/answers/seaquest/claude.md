# Exp 5 · Seaquest · Claude Opus 5.5

**Design**

The strategy is to dive, line up vertically with the nearest cluster of targets, face the target in my row, and fire. It dodges enemy bullets in its row, and it surfaces when oxygen is low if it carries a diver. It avoids rows 0 and 1 otherwise, because a sticky "up" there with no diver ends the episode.

**Key trick.** Each per-row unit is gated by "the submarine is in this row" through large weights on channel 0 and a large negative bias. Units for rows the submarine is not in saturate at exactly −1. So the sum over the 8 units of a family equals −7 plus the value for the submarine's own row. Each layer-2 unit then thresholds that sum.

**Layer 1**
- **H_k (units 0–7, rows 1–8): facing and entity direction.** Channel 0 has weights 0.4c+10, channel 4 has −0.8c, and channel 2 has 0.8c+4, with the edge columns corrected for newly spawned entities that have no trail. In the submarine's row this gives 0.4·(f + 2d + 10(n−1)), where f = ±1 is the facing, d = ±1 is the entity's direction and n is the number of entities. The four one-entity values (±1.2, ±0.4) reveal f. An empty row gives about −4.
- **Q_k (units 8–15): side of the entity.** The value is tanh(s − c_e − f/2), which is positive when the entity is to the submarine's left.
- **Y_k (units 16–23): vertical pull plus dodge.** The value is tanh(0.2·V). V is the count of targets in rows below minus rows above, with the adjacent row weighted ×2 and row 1 ignored. V also gets +12 for each enemy bullet in the submarine's row (−12 in row 8, so it dodges upward there).
- **Global units.** Unit 24 is SURF (diver bar cell (9,8) lit and oxygen cell (9,1) dark, i.e. a diver on board and oxygen below 40). Units 25, 26 and 27 detect the submarine in row 0, row 2 and row 8.

**Layer 2**
- **Thresholds on ΣH:**
  - presence (P);
  - three cuts A, B and C, which give facing-right = A − B + C.
- **Threshold on ΣQ:** entity on the left (QL).
- **Vertical moves:**
  - down and up, both suppressed when an entity is in my row;
  - dodge down (not in row 8) and dodge up (row 8 only).
- **Copies** of SURF, row 0 and row 2.

**Logits**
- Turn left when present, facing right and the entity is on the left. Turn right in the mirror case. Otherwise fire when present.
- When nothing is in my row, move toward the targets.
- Priorities: surfacing beats dodging, which beats aiming, which beats idle fire.

```edits
# ---- H_k: facing/direction per row (units 0..7)
W1[60:120:6, 0] = [10, 10.4, 10.8, 11.2, 11.6, 12, 12.4, 12.8, 13.2, 13.6]
W1[64:120:6, 0] = [0, -0.8, -1.6, -2.4, -3.2, -4, -4.8, -5.6, -6.4, -7.2]
W1[62:120:6, 0] = [4.8, 4.8, 5.6, 6.4, 7.2, 8, 8.8, 9.6, 10.4, 3.2]
W1[120:180:6, 1] = [10, 10.4, 10.8, 11.2, 11.6, 12, 12.4, 12.8, 13.2, 13.6]
W1[124:180:6, 1] = [0, -0.8, -1.6, -2.4, -3.2, -4, -4.8, -5.6, -6.4, -7.2]
W1[122:180:6, 1] = [4.8, 4.8, 5.6, 6.4, 7.2, 8, 8.8, 9.6, 10.4, 3.2]
W1[180:240:6, 2] = [10, 10.4, 10.8, 11.2, 11.6, 12, 12.4, 12.8, 13.2, 13.6]
W1[184:240:6, 2] = [0, -0.8, -1.6, -2.4, -3.2, -4, -4.8, -5.6, -6.4, -7.2]
W1[182:240:6, 2] = [4.8, 4.8, 5.6, 6.4, 7.2, 8, 8.8, 9.6, 10.4, 3.2]
W1[240:300:6, 3] = [10, 10.4, 10.8, 11.2, 11.6, 12, 12.4, 12.8, 13.2, 13.6]
W1[244:300:6, 3] = [0, -0.8, -1.6, -2.4, -3.2, -4, -4.8, -5.6, -6.4, -7.2]
W1[242:300:6, 3] = [4.8, 4.8, 5.6, 6.4, 7.2, 8, 8.8, 9.6, 10.4, 3.2]
W1[300:360:6, 4] = [10, 10.4, 10.8, 11.2, 11.6, 12, 12.4, 12.8, 13.2, 13.6]
W1[304:360:6, 4] = [0, -0.8, -1.6, -2.4, -3.2, -4, -4.8, -5.6, -6.4, -7.2]
W1[302:360:6, 4] = [4.8, 4.8, 5.6, 6.4, 7.2, 8, 8.8, 9.6, 10.4, 3.2]
W1[360:420:6, 5] = [10, 10.4, 10.8, 11.2, 11.6, 12, 12.4, 12.8, 13.2, 13.6]
W1[364:420:6, 5] = [0, -0.8, -1.6, -2.4, -3.2, -4, -4.8, -5.6, -6.4, -7.2]
W1[362:420:6, 5] = [4.8, 4.8, 5.6, 6.4, 7.2, 8, 8.8, 9.6, 10.4, 3.2]
W1[420:480:6, 6] = [10, 10.4, 10.8, 11.2, 11.6, 12, 12.4, 12.8, 13.2, 13.6]
W1[424:480:6, 6] = [0, -0.8, -1.6, -2.4, -3.2, -4, -4.8, -5.6, -6.4, -7.2]
W1[422:480:6, 6] = [4.8, 4.8, 5.6, 6.4, 7.2, 8, 8.8, 9.6, 10.4, 3.2]
W1[480:540:6, 7] = [10, 10.4, 10.8, 11.2, 11.6, 12, 12.4, 12.8, 13.2, 13.6]
W1[484:540:6, 7] = [0, -0.8, -1.6, -2.4, -3.2, -4, -4.8, -5.6, -6.4, -7.2]
W1[482:540:6, 7] = [4.8, 4.8, 5.6, 6.4, 7.2, 8, 8.8, 9.6, 10.4, 3.2]
b1[0:8] = -24
# ---- Q_k: entity left of sub (units 8..15)
W1[60:120:6, 8] = [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5]
W1[62:120:6, 8] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[120:180:6, 9] = [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5]
W1[122:180:6, 9] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[180:240:6, 10] = [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5]
W1[182:240:6, 10] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[240:300:6, 11] = [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5]
W1[242:300:6, 11] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[300:360:6, 12] = [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5]
W1[302:360:6, 12] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[360:420:6, 13] = [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5]
W1[362:420:6, 13] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[420:480:6, 14] = [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5]
W1[422:480:6, 14] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[480:540:6, 15] = [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5]
W1[482:540:6, 15] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
b1[8:16] = -20
# ---- Y_k: vertical pull + bullet dodge (units 16..23)
W1[60:120:6, 16] = 6
W1[61:120:6, 16] = 2.4
W1[62:120:6, 16] = -2.4
W1[122:600:6, 16] = 0.2
W1[122:180:6, 16] = 0.4
W1[120:180:6, 17] = 6
W1[121:180:6, 17] = 2.4
W1[122:180:6, 17] = -2.4
W1[182:600:6, 17] = 0.2
W1[182:240:6, 17] = 0.4
W1[180:240:6, 18] = 6
W1[181:240:6, 18] = 2.4
W1[182:240:6, 18] = -2.4
W1[242:600:6, 18] = 0.2
W1[242:300:6, 18] = 0.4
W1[122:180:6, 18] = -0.4
W1[240:300:6, 19] = 6
W1[241:300:6, 19] = 2.4
W1[242:300:6, 19] = -2.4
W1[302:600:6, 19] = 0.2
W1[302:360:6, 19] = 0.4
W1[122:240:6, 19] = -0.2
W1[182:240:6, 19] = -0.4
W1[300:360:6, 20] = 6
W1[301:360:6, 20] = 2.4
W1[302:360:6, 20] = -2.4
W1[362:600:6, 20] = 0.2
W1[362:420:6, 20] = 0.4
W1[122:300:6, 20] = -0.2
W1[242:300:6, 20] = -0.4
W1[360:420:6, 21] = 6
W1[361:420:6, 21] = 2.4
W1[362:420:6, 21] = -2.4
W1[422:600:6, 21] = 0.2
W1[422:480:6, 21] = 0.4
W1[122:360:6, 21] = -0.2
W1[302:360:6, 21] = -0.4
W1[420:480:6, 22] = 6
W1[421:480:6, 22] = 2.4
W1[422:480:6, 22] = -2.4
W1[482:600:6, 22] = 0.2
W1[482:540:6, 22] = 0.4
W1[122:420:6, 22] = -0.2
W1[362:420:6, 22] = -0.4
W1[480:540:6, 23] = 6
W1[481:540:6, 23] = -2.4
W1[482:540:6, 23] = 2.4
W1[122:480:6, 23] = -0.2
W1[422:480:6, 23] = -0.4
b1[16:24] = -12
# ---- globals: SURF, row0, row2, row8
W1[593, 24] = 6
W1[551, 24] = -6
b1[24] = -3
W1[0:60:6, 25] = 5
b1[25] = -5
W1[120:180:6, 26] = 5
b1[26] = -5
W1[480:540:6, 27] = 5
b1[27] = -5
# ---- layer 2
W2[0:8, 0] = 60
b2[0] = 477
W2[0:8, 1] = 20
b2[1] = 127.86
W2[0:8, 2] = 20
b2[2] = 140
W2[0:8, 3] = 20
b2[3] = 152.14
W2[8:16, 4] = 10
b2[4] = 70
W2[16:24, 5] = 30
W2[0:8, 5] = -200
b2[5] = -1393
W2[16:24, 6] = -30
W2[0:8, 6] = -200
b2[6] = -1813
W2[16:24, 7] = 80
W2[27, 7] = -30
b2[7] = 457.2
W2[16:24, 8] = -80
W2[27, 8] = 30
b2[8] = -662.8
W2[24, 9] = 3
W2[25, 10] = 3
W2[26, 11] = 3
# ---- output: 0 noop, 1 left, 2 up, 3 right, 4 down, 5 fire
W3[0, 0:6] = [0, 8, 0, 8, 0, 1.25]
W3[1, 0:6] = [0, 8, 0, -8, 0, 0]
W3[2, 0:6] = [0, -8, 0, 8, 0, 0]
W3[3, 0:6] = [0, 8, 0, -8, 0, 0]
W3[4, 0:6] = [0, 8, 0, -8, 0, 0]
W3[5, 4] = 4.5
W3[6, 2] = 5.5
W3[7, 4] = 8
W3[8, 2] = 9
W3[9, 2] = 17
W3[10, 4] = 10
W3[11, 2] = -7
b3[0:6] = [0, -16, 18.5, -16, 18.5, 2.75]
```
