# Exp 1 · freeway (0bbcc3ea) · Astra (our re-run)

## Observations:
- **Channel 0 — agent:** Strong, position-specific weights cover rows **2–8**, rather than just a bottom paddle or a single column. For example, at **(2,4)** the agent weights include **4.331, −2.970, 3.294, −3.827**. Row 8 also has large agent weights. Row 1’s agent weights look comparatively small and unstructured. This suggests a player navigating upward through the board, with the top row ending a crossing.

- **Channel 1 — moving objects:** Strong structure occupies **rows 2–7**, while row 8 looks comparatively unstructured. Within individual rows, weight patterns repeat approximately every **four columns**. For example, at **(2,2)** and **(2,6)**, hidden unit 11 receives **−2.373** and **−2.260**. At **(3,2)** and **(3,6)**, hidden unit 12 receives **6.639** and **6.689**. These repeated, row-dependent patterns strongly suggest horizontal lanes containing regularly spaced moving objects.

- **Channel 2 — reward targets:** The distinctive structure is across **row 1, columns 1–8**, not scattered throughout the interior. At **(1,1)**, weights to hidden units 3, 25, and 28 are **−0.550, 0.440, 0.464**. At **(1,8)** they are **−0.473, 0.479, 0.361**. This looks like a fixed destination strip at the top.

- **Channel 3 — obstacles:** The border has a repeated weight pattern, whereas the playable interior looks mostly initialization-scale. At **(0,0)**, weights to units 3, 25, and 28 are **−0.477, 0.440, 0.437**, close to the corresponding biases **−0.492, 0.434, 0.434**. The goal-strip weights resemble this pattern too: always-present inputs receive matching updates. Thus, these weights support fixed walls and a fixed goal strip—not an interior maze.

- **Channel 4 — previous frame:** Substantial, spatially structured weights occur through the moving-object region, with additional agent-history structure near the bottom. At **(2,1)**, weights include **−1.443** to unit 3 and **−2.266** to unit 11. This supports using motion information to time crossings, rather than merely navigating toward a static target.

- **Outputs:** The biases strongly favor **up**: **[−0.029, 3.142, −1.862, 0.173, −1.269]**. Nevertheless, the output weights provide substantial steering and waiting control. For example, hidden unit 20 contributes **1.314** to left versus **−1.687** to right; unit 29 contributes **−1.691** to left versus **2.315** to right. Upward progress is a preference, not an unconditional action.

## Reasoning:
The combination is distinctive: a safe-looking bottom starting row, six horizontal moving-object lanes, a fixed reward strip across the top, motion-sensitive inputs, and a strong upward action bias.

That fits a **road-crossing game**. Snake or food collection would normally require reward-location structure throughout the interior. Pong, Breakout, and falling-object catching would usually restrict the agent to an edge and emphasize lateral movement. Maze navigation would require interior obstacle structure.

The weights cannot establish the exact game implementation, but they strongly favor traffic crossing over those alternatives.

## Game:
**Frogger — a simplified, traffic-only grid variant, also resembling Freeway.**

## Description:
The agent moves from the bottom toward the top, timing its movement to avoid horizontally moving traffic. Reaching the top goal strip gives reward.

## Confidence:
96%