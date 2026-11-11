# Exp 1 · pacman (bdf99279) · Astra (our re-run)

## Observations:
*Hidden-unit indices below are zero-based.*

- **Channel 0 — agent:** Large weights occur throughout the board, but only along a distinctive corridor layout. For example, at **(1,1)**, weights include **−3.734, +4.081**; at **(1,3)**, **+4.742**. By contrast, at **(2,2)** the weights begin **−0.042, +0.119, +0.015, −0.058**, and remain small. The apparent accessible-cell pattern is:
  ```
  ........
  .##..##.
  .#....#.
  ...##...
  ...##...
  .#....#.
  .##..##.
  ........
  ```
  Here `.` means strongly trained agent inputs and `#` means small, initialization-like inputs. This suggests a **fixed, symmetric maze**, not an open arena or a paddle restricted to one edge.

- **Channel 1 — moving objects:** Strong weights follow the same corridors. At **(1,2)**, connections to units 5 and 7 are **+2.493, −3.116**; at **(1,7)**, the connection to unit 9 is **−3.354**; at **(2,5)**, unit 5 receives **−5.209**. Other moving entities therefore appear important throughout the maze—consistent with roaming enemies.

- **Channel 2 — reward targets:** These weights are generally smaller and less distributed than the agent and moving-object weights, but are not untrained. Notably, unit 6 receives **−1.490 at (1,1)**, **−2.361 at (1,2)**, **−1.905 at (1,3)**, and **−4.876 at (2,4)**. Targets matter within the corridors. Multiple simultaneously present collectibles could explain why their individual weights need not match the magnitude of the agent-location weights; the weights alone do not establish their number.

- **Channel 3 — obstacles:** Weights are comparatively small, including at the apparent interior walls. At **(2,2)** they begin **−0.038, +0.050, +0.064, −0.131**. The border has modest repeated structure—for example, its connections to unit 6 at **(0,0), (0,1), (0,2)** are **−0.162, −0.149, −0.130**. This fits a constant maze whose geometry can be learned through position-specific responses rather than repeatedly decoded from obstacle inputs.

- **Channel 4 — previous frame:** There is learned structure along the corridors, generally weaker than current positions. At **(1,8)**, unit 9 receives **−2.849**, versus **−2.016** from that cell’s current moving-object channel. History is being used, plausibly for movement or threat prediction, although these individual connections do not prove a velocity calculation.

- **Outputs:** All four directions have substantial, differentiated connections. Examples of W3 rows are **[0.790, 2.144, −1.525, −0.598, 0.380]** and **[−0.050, −1.608, −1.018, 0.033, 2.500]**, supporting up/down and rightward discrimination respectively. Biases are **[−0.873, +0.703, +0.712, +0.595, −0.092]**: the bias contribution disfavors waiting and favors movement, especially up/down/left. Actual action probabilities still depend on hidden activations.

## Reasoning:
The combination of a **fixed symmetric maze, four-directional navigation, reward targets in corridors, and important moving entities sharing those corridors** strongly suggests a pellet-collection game with roaming enemies.

Pong, Breakout, and falling-object catchers do not fit the agent’s two-dimensional corridor footprint. Ordinary Snake does not naturally explain the fixed internal maze and substantial separate moving-object channel. A static maze-solving game would not explain the importance of moving entities and their history.

A generic maze chase remains possible, but **Pac-Man** explains the whole combination most naturally. Small obstacle weights do not contradict this: fixed walls need not be dynamically interpreted. Exact rules and the original game title cannot be uniquely recovered from policy weights.

## Game:
**A small-grid Pac-Man-style game with a fixed maze.**

## Description:
The agent navigates corridors collecting pellets or other reward items while avoiding moving ghosts. Collecting those items gives reward, with the likely objective of clearing the maze without being caught.

## Confidence:
93%