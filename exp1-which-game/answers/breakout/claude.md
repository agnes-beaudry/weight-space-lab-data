# Exp 1 · breakout (a9f535cc) · Claude Opus 5.5

**Observations**

*How I read the weights.* Input i = (row·10 + col)·5 + channel, so the 5 W1 lines for one cell sit next to each other. To check where I was in the list, I used the wall cells. Every border cell's channel‑3 line has the same small pattern, roughly +0.2 at unit 1, −0.15 at unit 3, −0.2 at unit 6, +0.15 at unit 7, +0.2 at unit 12 and −0.2 at units 26, 27 and 29. For example, input 3 (cell (0,0)) is "-0.080 0.255 -0.015 -0.141 …" and input 98 (cell (1,9)) is "-0.033 0.169 0.049 -0.185 …". This pattern matches b1 ("0.199 … -0.144 … -0.142 0.145 … 0.186 …"). That is what you'd expect from an input that is always on: it acts as an extra bias. All of row 0 and row 9 is small, so nothing except wall ever shows up there.

- **Channel 0 (agent):** Small (|w| < 0.25) everywhere in rows 1–7. Example: input 365, cell (7,3), "-0.079 0.036 0.101 …". Large only in **row 8**:
  - (8,1), input 405: unit 13 = −3.293, unit 15 = +4.066, unit 31 = −4.013.
  - (8,2), input 410: unit 6 = +2.794.
  - (8,7), input 435: unit 0 = −2.107, unit 19 = −3.053.
  - (8,8), input 440: units 5 and 8 = +3.855 and +3.730, unit 13 = −4.701.
  
  So the agent only ever sits in the bottom row, and its column matters.
- **Channel 1 (moving object) and channel 4 (previous frame):** These have the largest weights in the network, across all of rows 1–7. They share a pattern: units 5, 6 and 8 positive, unit 7 strongly negative, units 13 and 19 negative, unit 28 positive. Examples:
  - (1,3) ch1, input 66: "… 3.194 2.708 -4.791 3.436 …"
  - (2,2) ch1, input 111: "… 3.408 2.416 -6.356 4.570 …"
  - (7,4) ch4, input 374: "… 2.339 7.350 -6.080 5.904 …"
  - (7,7) ch4, input 389: "… 4.408 4.075 -5.756 6.256 …"
  
  In row 8, channel 1 is near zero (inputs 431, 436 and 441 are about ±0.1). Channel 4 in row 8 is moderate, because it holds the paddle's previous position. Taken together, there is one moving object that travels through the whole field, and the network uses current plus previous frame to get its direction.
- **Channel 2 (reward targets):** Clearly non‑zero only in **rows 1–3**:
  - (1,5), input 77: 2.148
  - (1,6), input 82: −2.096
  - (2,7), input 137: −2.417
  - (3,1), input 157: −4.811 at unit 0 and −3.236 at unit 20
  - (3,7), input 187: −4.640 and −2.930
  - (3,3), input 167: 2.365
  
  From row 4 down it is tiny: input 207 is "-0.017 -0.050 …" and the row‑7 lines are about ±0.1. Input 92, cell (1,8), shows the "always‑on" wall pattern ("0.064 0.206 -0.117 -0.162 … -0.186 …"), so a target there is almost never removed. These look like a static block of targets in the top three rows, which get removed over time.
- **Channel 3 (obstacles):** Inside the grid it is only initialization‑sized noise (inputs 358, 363 and 443 are about ±0.1). There are no interior obstacles, only the border.
- **Output layer:** The output biases are flat (1.221, 1.008, 1.024, 0.897, 1.298). In W3, the columns for no‑op, up and down are nearly identical in almost every row:
  - unit 1: 1.502 / 1.718 / 1.370
  - unit 5: −1.616 / −1.409 / −1.560
  - unit 28: 1.547 / 1.395 / 1.382
  - unit 30: −1.645 / −1.382 / −1.458
  
  Only left and right get their own weights:
  - left: unit 11 = +4.057, unit 22 = −5.336, unit 4 = +3.081, unit 17 = −1.106, unit 29 = −1.549
  - right: unit 24 = +3.031, unit 3 = −3.063, unit 21 = −2.949, unit 5 = +1.419, unit 30 = +1.380
  
  So up and down behave the same as no‑op. The only real decision is moving left or right.

**Reasoning**

The agent is a one‑cell piece confined to the bottom row that can only move sideways, which makes it a paddle. One fast‑moving object crosses the whole field, and the network tracks both its position and its previous position, so it needs the object's direction. The reward targets are a block filling the top three rows that gets destroyed over time. That combination is Breakout: paddle at the bottom, a bouncing ball, bricks at the top.

The alternatives don't fit as well:
- **Catch or dodging falling objects:** These would have no reward block confined to the top rows. The objects also wouldn't need to appear in channel 1 throughout the field with direction tracking.
- **Pong:** The paddles would be in side columns, not the bottom row.
- **Snake, mazes, collecting games:** These would show agent weights everywhere, use up and down, and often have interior obstacles.
- **Space Invaders or Freeway:** Freeway puts the agent in many rows and uses up/down. Space Invaders is possible in principle, but there the targets would move and appear in channel 1 as well. Here the targets in rows 1–3 behave like static bricks, and the moving object is a single ball whose bounce direction matters.

The layout matches MinAtar Breakout: three rows of bricks, a paddle on the bottom row, a ball, and a trail of where the ball was on the previous frame.

**Game:** Breakout (a MinAtar‑style mini Breakout / Arkanoid)

**Description:** The agent moves a paddle left and right along the bottom row to keep a bouncing ball in play and send it back up. Reward comes from the ball breaking bricks in the top three rows, and the episode ends if the ball gets past the paddle.

**Confidence:** 85%
