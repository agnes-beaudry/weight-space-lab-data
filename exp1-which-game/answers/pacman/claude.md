# Exp 1 · pacman (bdf99279) · Claude Opus 5.5

**Observations**

*Indexing.* The first 55 rows of W1 (inputs 0–54) are all small, about |w| < 0.2. That covers the top wall row and cell (1,0). The first large row is input 55, which is (1,1) channel 0: `1.706 -3.242 0.524 -1.906 …`. Inside each cell the five channels follow a regular pattern: ch0 large, ch1 large, ch2 moderate, ch3 tiny, ch4 moderate.

*Channel 0 (agent).* Weights are very large across the whole interior, not just one row or column. Examples:
- (1,3): `4.742 …`
- (3,5): `-5.208 3.306 3.336 …`
- (4,6): `-5.170 …`
- (5,7): `-4.928 5.834 …`
- (8,1): `2.446 …`

So the agent moves freely in two dimensions.

*Fixed "dead" cells inside the playable area.* Some interior cells have all five channels at noise level (|w| < 0.2), the same as the border. For example, (2,2) is `-0.042 0.119…`, `0.019 -0.121…`, `0.107 -0.041…`, and so on. A cell like this never contains the agent, a moving object or a reward. Its only active input is a constant channel 3, and a constant input behaves like a bias, so its weights never grow. These cells must be fixed internal walls. Mapping them (W = wall) gives:

```
row1: . . . . . . . .
row2: . W W . . W W .
row3: . W . . . . W .
row4: . . . W W . . .
row5: . . . W W . . .
row6: . W . . . . W .
row7: . W W . . W W .
row8: . . . . . . . .
```

This is a symmetric maze of corridors around wall blocks.

*Channel 3 (obstacles).* Weights are tiny everywhere. The obstacles never change position, which fits a static maze.

*Channel 1 (moving objects).* Weights are large across the corridors. Examples:
- (1,4): `… 3.448`
- (2,4): `… 3.745`
- (3,5): `… 4.201`
- (3,3): `… -3.873`
- (6,6), (8,3): about 2.4

So there are other agents roaming the maze, and the policy reacts strongly to where they are.

*Channel 2 (reward targets).* Weights are moderate and vary by position, so rewards appear in many different cells rather than at one fixed goal. Hidden unit 6 is consistently negative for this channel in the upper part of the maze: (1,1) −1.49, (1,2) −2.36, (1,3) −1.91, (2,4) −4.88, (2,1) −0.63, (2,2) −0.85. These look like collectible items scattered through the maze.

*Channel 4 (previous frame).* Weights are moderate, with a recurring pattern: unit 9 is often strongly negative (−1.3 to −3.4) and unit 6 is positive (about 0.7–1.0). This lets the network work out which direction the agent and the moving objects are travelling.

*Output layer.* The biases are b3 = `-0.873 0.703 0.712 0.595 -0.092`. No-op is clearly the least favoured action, while all four moves are used. That suits a game where the agent keeps moving through corridors.

**Reasoning**

- **Ruled out: Catch, Breakout and Pong-style games.** The agent's weights cover the full 2D interior, not a single paddle row.
- **Ruled out: Frogger and Freeway.** The reward weights are spread across the grid and vary by cell, rather than marking a constant goal row. There is also a static internal maze.
- **Ruled out: Snake.** Snake has no fixed interior maze, and its body would appear in the obstacles channel and change over time. Here channel 3 is constant.
- **Supported: a maze chase game.** The evidence is:
  - fixed internal walls arranged in a symmetric maze;
  - an agent moving in 2D through the corridors;
  - several moving objects that strongly drive the policy (the ghosts);
  - many collectible rewards spread across the corridors (the pellets);
  - the previous-frame channel, which lets the network infer ghost direction;
  - a strong bias against standing still.

  Together these point to Pac-Man.

**Game:** A Pac-Man-style maze game: pellet collection while avoiding ghosts, in a fixed symmetric 8×8 maze.

**Description:** The agent navigates the corridors of a fixed maze and earns reward by eating the pellets scattered through it. It must avoid the moving ghosts, since touching one presumably ends the episode or is penalised.

**Confidence:** 70%
