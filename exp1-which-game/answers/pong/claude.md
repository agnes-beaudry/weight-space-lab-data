# Exp 1 · pong (dfc6cb65) · Claude Opus 5.5

**Observations:**

*Wall and bias pattern.* The always-on wall cells have nearly the same weight row as b1, so they act as an extra bias. For example, row 3 (cell (0,0), channel 3) starts −0.180 −0.080 0.191 0.275, and b1 starts −0.076 −0.136 0.184 0.271. Inputs that are never active keep small random weights of about ±0.1.

*Channel 0 (agent).* Its weights are large only in **column 1**. Examples:
- row 55, cell (1,1): −0.694 −1.615 −1.207 … 2.697
- row 105, cell (2,1): 0.563 −3.588 0.506 …

Channel 0 in columns 2–8 stays small (row 60, cell (1,2): 0.020 0.066 …; row 110, cell (2,2): −0.038 0.081 …). So the agent only ever occupies column 1, like a paddle along the left wall.

*Channel 1 (moving objects).* Its weights are large in **columns 2–8** of every row. Examples:
- row 61, cell (1,2): −1.683 −3.169 2.039
- row 76, cell (1,5): 2.305 3.278
- row 121, cell (2,4): −1.757 3.316
- row 141, cell (2,8): … 1.616

In column 1 the weights are small (row 56, cell (1,1): −0.044 0.045). So one moving object travels across the field and never sits on the agent's column.

*Channel 4 (previous frame).* Large across the inner grid (row 59, cell (1,1): −0.439 0.412 −0.577 0.499; row 119, cell (2,3): −0.200 2.540). The network uses it to read the object's direction of travel.

*Channel 2 (reward targets).* Small everywhere inside the play area. On the **right wall (column 9)**, however, channel 2 has the same bias-like pattern as the wall itself (row 97, cell (1,9): 0.028 −0.052 0.203 0.316; row 147, cell (2,9): −0.057 −0.102 0.176 0.299). The top wall and the corners do not have this. So the inner part of the right wall is always marked as the reward target: a goal line on the far side.

*Channel 3 (obstacles).* Only the wall border carries the bias-like pattern. There are no obstacles inside the grid.

*Output layer.* In almost every row of W3, the weights for no-op, left and right are nearly identical, while up and down differ. For example:
- unit 6: [−1.023, 0.412, −1.348, −1.165, −1.053]
- unit 18: [−1.399, −1.370, 1.587, −1.283, −1.215]
- unit 31: [0.553, 1.950, 0.123, 0.616, 0.587]
- unit 2: [0.133, 1.003, 1.898, 0.059, 0.046]

The output biases are 0.931, 1.261, 1.065, 0.861 and 0.855. Up and down have the highest biases and are the only actions the network treats differently. Left and right behave like no-op, so the agent can only move vertically.

**Reasoning:** The agent is confined to one column next to the left wall and moves only up and down. A single moving object crosses the field, and its direction is visible through the previous-frame channel. The opposite wall is the reward line. This is the layout of Pong with a paddle.
- **Snake is ruled out:** the agent is fixed to one column, and there are no reward items inside the grid.
- **Freeway/Frogger is ruled out:** the agent would move across the field toward a goal.
- **Catch and Breakout are ruled out:** the paddle would sit on the bottom row and move left and right, and Breakout would have bricks as reward targets inside the grid.
- **Maze or gridworld games are ruled out:** there are no inner obstacles.
- **Dodging games are unlikely:** the reward line on the right wall suggests the agent sends the ball there, not that it only avoids the ball.

**Game:** Pong, single-player against a wall (squash-style): a vertical paddle in the left column against a ball, with the right wall as the scoring side.

**Description:** The agent moves a paddle up and down in column 1 to intercept a ball bouncing around the grid and send it back. Reward comes from returning the ball to the right wall, and missing it, so the ball reaches the left side, presumably ends the episode.

**Confidence:** 70%
