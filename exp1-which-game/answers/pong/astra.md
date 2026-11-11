# Exp 1 · pong (dfc6cb65) · Astra (our re-run)

## Observations
- **Channel 0 — agent:** Large weights are concentrated in **column 1, rows 1–8**, consistent with a paddle on the left side. At `(1,1)`, weights include **2.697, −2.474, −2.322**; at `(2,1)`, **−3.588, 2.245, −2.085**. Other interior columns have mostly small, initialization-like agent weights.
- **Channel 1 — moving objects:** Strong weights cover the interior to the paddle’s right, rather than a single lane. At `(1,2)`, examples include **−3.169, −2.772, −3.022**; at `(1,5)`, **2.305, 3.278, −1.750**. This fits a ball moving across the court.
- **Channel 2 — reward targets:** Most weights are small. The notable structure is a repeated profile on the **right border, column 9**: at `(1,9)`, examples include **0.316, −0.283, 0.183**. This suggests a fixed target/scoring edge, not collectible objects scattered through the interior. Because this could be a constant input, its precise meaning is uncertain.
- **Channel 3 — obstacles:** The wall border has a shared weight pattern; interior weights are mostly small. For example, `(0,0)` has **0.275** into hidden unit 3, close to **b1[3] = 0.271**. Constant border inputs can act as additional biases; this is not evidence of a learned interior maze.
- **Channel 4 — previous frame:** Strong weights occur at paddle and ball locations. At `(1,1)`, examples include **−1.578, 1.031, −1.183**; at `(1,5)`, **3.115, 5.236, 3.913**. Motion history matters, consistent with estimating a ball’s direction and interception point.
- **Outputs:** **Up and down are distinguished strongly**, while no-op, left, and right have remarkably similar weights. For example, W3 row 28 is **[−1.100, 0.407, −1.500, −1.169, −1.028]**. This suggests only vertical movement meaningfully controls the player. Biases **[0.931, 1.261, 1.065, 0.861, 0.855]** modestly favor up/down, but do not determine action probabilities alone.

## Reasoning
A left-edge player, freely moving interior object, important motion history, and effectively vertical-only controls are the signature of **Pong**.

Standard Breakout and falling-object catching usually have a bottom paddle with horizontal controls. Snake, maze navigation, and food collection would ordinarily require agent-position structure across both dimensions and meaningful left/right outputs. The lack of variable interior targets also argues against brick destruction or collecting food.

The weights support a simplified paddle-and-ball game very strongly, but do not uniquely establish its scoring rules or whether an opponent exists.

## Game
**Pong — most likely a simplified single-paddle Pong/paddleball variant.**

## Description
The agent moves a left-side paddle up and down to intercept and return a moving ball. Reward likely comes from successful returns or sending the ball to the opposite scoring boundary; the exact reward event cannot be recovered confidently from these weights alone.

## Confidence
96%