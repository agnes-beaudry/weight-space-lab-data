# What this network was trained to do: a policy for a Frogger-style road-crossing game

## Observations

**Input layout: 100 cells × 5 channels.** The rows of W1 repeat with period 5. Reading input *i* as cell *p = i // 5*, channel *c = i % 5*, the cells form a 10×10 grid (*p = 10·row + col*).

**Rows that are constant in the data copy the bias b1.**
- Rows 3, 8, 13, …, 98 (channel 3 of cells 0–19) are near-copies of b1.
  - Row 3: `0.023 0.181 -0.081 -0.477 … 0.440 … 0.437`
  - b1: `0.048 0.185 -0.041 -0.492 … 0.434 … 0.434`
- The same holds for channel 3 of cells 28, 29, 38, 39, 48, 49, 58, 59, 68, 69, 78, 79, 88, 89 and 90–99. Examples:
  - Row 143: `-0.013 0.117 -0.075 -0.520`
  - Row 243: `0.154 0.150 0.083 -0.523`
  - Row 498: `0.112 0.168 -0.164 -0.408`
- An input that is always 1 gets exactly the same gradient as the bias, so it drifts identically.
- In those cells every other channel is still at initialisation (|w| ≲ 0.15), so those channels are always 0 there.
- **Conclusion:** grid rows 0, 1 and 9 and columns 8 and 9 are permanently "wall / out of bounds" (channel 3). The live board is grid rows 2–8 × columns 0–7, i.e. 7×8.

**Inside the board:**
- Channel 3 is never active, so live cells are never wall.
- Channel 2 is never active anywhere: all its rows are tiny.
- Channels 0, 1 and 4 carry large weights, for example:
  - Row 310: `4.822 -3.670 …`
  - Row 270: `… -8.723 …`
  - Row 259: `… 6.182 … 4.841`
- Channel 0 in cells that are almost always in that state looks bias-like. For example, row 105 is `0.100 0.367 0.187 -0.346 … 0.578 0.425`. So channel 0 is probably "empty".

**Channel 1 repeats with horizontal period 4.** Its weights at column *c* and column *c+4* of the same row are near-identical:

| Cell pair | Weights at column c | Weights at column c+4 |
|---|---|---|
| 31 / 35 | 6.639 in hidden unit 12 | 6.689 in hidden unit 12 |
| 40 / 44 | `-0.034 -0.218 -0.221 0.717 … -1.210 … 1.174 0.062 1.062` | `0.050 -0.089 -0.228 0.633 … -1.256 … 1.222 -0.019 1.176` |
| 42 / 46 | `… 1.871 … -2.172` | `… 2.021 … -2.213` |
| 63 / 67 | `… 2.826 … 4.041` | `… 2.842 … 4.057` |
| 20 / 24 | `0.375 -0.694 … -1.898 … 1.562` | `0.375 -0.733 … -1.879 … 1.611` |

- Near-identical weights mean the two inputs are always equal in the data.
- So channel 1 forms 4-periodic horizontal patterns: two copies across the 8-wide board.
- Channel 1 is never present in the bottom live row (cells 80–87). Rows 401, 406, …, 436 are all tiny.

**Channel 4** appears in every live row, including the bottom one, with strong position-specific weights. Examples are 6.182, −4.103, 3.251 and 2.285. That is consistent with a single, position-critical object.

**Outputs.** b3 = `-0.029 3.142 -1.862 0.173 -1.269`. Output 1 has a strongly dominant prior, which looks like a default action.

## Reasoning

- A one-hot 10×10×5 grid with a constant out-of-bounds channel points to a custom gridworld. It does not fit an image, text or tabular input.
- The repeated, period-4 channel confined to the upper rows looks like traffic lanes, each holding a periodic car pattern that scrolls horizontally.
- The bottom row free of channel 1 looks like a safe starting sidewalk.
- A single agent (channel 4) that can be anywhere on the 7×8 board, plus 5 outputs, matches up/down/left/right/no-op.
- This rules out MinAtar Freeway and Asterix as exact matches: their channel and board layouts differ, and they have no wall channel. Classification of static data is also unlikely, because the structure is game-like.

## Answer

**Inputs:** A 10×10 grid observation, flattened as cell-major × 5 one-hot channels.
- Channel 3: wall or out-of-bounds border (rows 0, 1, 9 and columns 8, 9).
- Channel 0: empty cell.
- Channel 1: cars or obstacles in lanes on rows 2–7, tiled with period 4.
- Channel 4: the agent (the frog or chicken).
- Channel 2: an object type that never occurs in the data.

**Outputs:** Scores (logits or Q-values) for 5 actions: no-op, up, down, left, right. Output 1, with the large bias, is the most frequent action, probably "up" or "stay".

**Task:** A policy or Q-network for a Frogger/Freeway-style road-crossing gridworld. The agent starts on the bottom safe row and must move across horizontally scrolling lanes of periodic traffic without being hit.

**Confidence:** 40%