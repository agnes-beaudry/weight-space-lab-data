# Exp 1 · breakout (a9f535cc) · Astra (our re-run)

Observations:
- **Channel 0 — agent:** Strong weights are concentrated on **row 8**, across the playable columns; elsewhere this channel looks largely like small initialization-scale weights. At `(8,1)`, the weights include **4.066** and **−4.013**. At `(8,8)`, they include **3.855**, **3.730**, and **−4.701**. This suggests a horizontally movable paddle at the bottom, rather than a player roaming the board.
- **Channel 1 — moving objects:** Large, structured weights occur throughout the interior above the paddle, **rows 1–7**. For example, `(1,3)` has **3.194, 2.708, −4.791, 3.436** into hidden units 5–8; `(2,2)` has **3.408, 2.416, −6.356, 4.570** into those same units. This fits a ball moving through the playfield.
- **Channel 2 — reward targets:** Learned structure is concentrated in the **top three interior rows**, rather than spread throughout the board. Examples include **1.620** at `(1,2)` into unit 25, and **−4.811** and **−3.236** at `(3,1)` into units 0 and 20. Below this upper band, target weights look small and unstructured. This strongly suggests rows of bricks.
- **Channel 3 — obstacles:** The border has a repeated weight pattern, while interior obstacle weights are generally small. For example, border cell `(0,0)` has **0.255** into unit 1 and **0.159** into unit 12; `(0,2)` has **0.312** and **0.088**. These constant border inputs can act as additional biases; they do not by themselves demonstrate learned collision avoidance. There is no comparable evidence of an interior maze.
- **Channel 4 — previous frame:** Strong structure broadly follows the moving-object playfield, with additional structure along the bottom paddle row. At `(1,3)`, units 5–8 receive **2.123, 1.572, −4.301, 3.057**, resembling the current-object weights there. Current and previous positions provide information useful for predicting a ball’s trajectory.
- **Outputs:** Left and right have especially distinctive readouts. For example, W3 row 11 is **[−0.177, 0.185, 0.450, 4.057, −0.414]**, strongly distinguishing left; row 24 is **[0.367, 0.561, 0.305, 0.911, 3.031]**, distinguishing right. No-op/up/down often have similar weights, consistent with actions that may all leave a horizontal paddle stationary. Biases **[1.221, 1.008, 1.024, 0.897, 1.298]** are fairly close: they alone do not establish action frequencies.

Reasoning: A bottom-row paddle, a moving ball above it, motion-history processing, and reward targets restricted to an upper band jointly point to a brick-breaking game. Catching falling objects could explain the paddle but not the separate upper brick band. Pong could explain paddle-and-ball control but not those reward targets. Snake, maze navigation, and Pac-Man fit poorly because the agent’s learned spatial footprint is confined to the bottom row.

Game: **Breakout — a grid-based brick-breaker, similar to Arkanoid.**

Description: The agent moves a bottom paddle left and right to keep a bouncing ball in play and direct it toward bricks above. Breaking bricks gives reward; missing the ball risks losing the episode.

Confidence: **98%**