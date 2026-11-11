# Exp 1 · freeway (0bbcc3ea) · Claude Opus 5.5

**Observations:**

To find each cell's weights, I matched rows of W1 to the formula i = (row·10 + col)·5 + channel.

- **Inputs that are always on.** Some rows of W1 copy the pattern of the bias b1 almost exactly. b1 starts 0.048, 0.185, −0.041, −0.492. Input 3, which is channel 3 (obstacle) at cell (0,0), starts 0.023, 0.181, −0.081, −0.477. Input 48 at (0,9) gives 0.100, 0.268, −0.062, −0.598. This happens when an input is 1 in every frame, so its weights train exactly like a bias. All 36 border cells have this pattern in channel 3, as expected for walls.
  - **Row 1 is different.** The same pattern appears in cells (1,1) to (1,8), but on **channel 2 (reward)**, not channel 3. For example, input 57 at (1,1), channel 2, gives 0.050, 0.215, −0.101, −0.550. Input 92 at (1,8), channel 2, gives −0.031, 0.177, −0.118, −0.473. So the whole top playable row is a reward target that is always there.
- **Channel 3 inside the grid** is small everywhere (about ±0.1). There are no fixed interior obstacles.
- **Channel 2 below row 1** is also small, so rewards appear nowhere else.
- **Channel 1 (moving objects) and channel 4 (previous frame) inside the grid** have the largest weights in the network, up to about ±6. They show two regularities:
  1. **Repeats every 4 columns.** In row 2, channel 1 at (2,1) (input 106: 0.375, −0.694, …, −1.898, …, 1.562) almost matches (2,5) (input 126: 0.375, −0.733, …, −1.879, …, 1.611). Channel 1 at (2,2) (input 111: −0.522, −0.497, −0.615, −1.480, …, −2.373) almost matches (2,6) (input 131: −0.585, −0.415, −0.711, −1.635, …, −2.260). Two inputs only train to the same weights if they are always on together. So objects in a row are spaced 4 columns apart and move as a group.
  2. **Shifted by one column between frames.** Channel 4 at (2,1) (input 109: −0.337, −0.323, −0.815, −1.443, …, −2.266) matches channel 1 at (2,2). Channel 4 at (2,2) (input 114: −0.267, −0.479, 0.216, 1.132) matches channel 1 at (2,3) (input 116: 0.096, −0.145, 0.337, 1.047). So an object that was at column c−1 last frame is at column c now. The objects in row 2 move right one cell per step, all together.
- **Channel 0 (agent)** has large weights in the low rows and near the top. For example, (2,3) to (2,8) have values like 2.33, 4.33, 2.51, 1.93 and 2.09 on their first hidden unit. Where the agent is, relative to the lanes and the goal row, matters a lot.
- **Output biases:** b3 = [−0.029, **3.142 (up)**, −1.862 (down), 0.173 (left), −1.269 (right)]. The default action is strongly "up" and "down" is discouraged.

**Reasoning:**

- **Fits:** The reward is a fixed full row at the top, and there are no interior obstacles. Rows below hold groups of moving objects, spaced apart and sliding sideways together like traffic. The agent has a strong built-in pull upward. This is a road-crossing game: move up through lanes of traffic to reach the goal row, and dodge or wait for gaps. The large channel 1 and channel 4 weights are the network judging where cars are and which way they are moving, so it can time its crossing.
- **Ruled out:**
  - Snake: food would appear at random cells, not as a fixed row.
  - Pac-Man or other collect-the-items games: rewards would be spread across the board.
  - Catch or Breakout-type games: the agent would be stuck on one row, and there would be no "up" bias.
  - Mazes: there would be interior obstacles.
  - Chase or avoid-the-enemy games: enemies would not move as fixed, evenly spaced, synchronised lanes.

**Game:** Frogger, or the Atari game Freeway, which works the same way: crossing lanes of traffic.

**Description:** The agent starts near the bottom and moves up through lanes of cars that slide sideways, mostly in groups spaced four cells apart. It is rewarded for reaching the goal row at the top (row 1), and getting hit by a car ends the attempt or sends it back.

**Confidence:** 80%
