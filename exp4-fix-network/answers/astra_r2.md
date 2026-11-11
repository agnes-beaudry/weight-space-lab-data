# Exp 4 · astra_r2 · Astra

The motor direction already works: action **1** moves the car left, up the ramp. The failure is the sustained stall at its crest, not movement in the wrong direction.

The weights show strong off/forward competition—for example, head rows 6 and 114 have approximately opposite weights for those actions. Motion feedback is substantial: embedding unit 116 receives horizontal-velocity weights **2.25, 2.37, 1.73** from the body and wheels. The rollout shows the resulting policy repeatedly switching between freewheeling and forward drive without advancing. This is consistent with a safe, positive-shaping-return local optimum: approach the goal, but never commit to crossing the red strip.

My repair retains most of the trained actor, repurposing a small relay to override that hesitation. It preserves an initial settling interval, then demands sustained forward drive before the car becomes high-centred.

```edits
embed.W[:, 6:8] = 0
embed.b[6:8] = [267, 64.5]
embed.W[52, 6] = -100
embed.W[253, 7] = -100
actor0.W[6:8, :] = 0
actor0.W[:, 20] = 0
actor0.W[6:8, 20] = 10
actor0.b[20] = 0
actor1.W[20, :] = 0
actor1.W[:, 20] = 0
actor1.W[20, 20] = 10
actor1.b[20] = 0
actor2.W[20, :] = 0
actor2.W[:, 20] = 0
actor2.W[20, 20] = 10
actor2.b[20] = 0
actor3.W[20, :] = 0
actor3.W[:, 20] = 0
actor3.W[20, 20] = 10
actor3.b[20] = 0
actor4.W[20, :] = 0
actor4.W[:, 125] = 0
actor4.W[20, 125] = 10
actor4.b[125] = 0
actor_head.W[125, 0:3] = [-100, 100, -100]
```

The override activates when either the rear-wheel centre falls below **0.645**—near its ground-contact height of **0.629**—or the body passes left of **2.67**. Both conditions are initially false.

Once active, the relay makes action 1 overwhelmingly likely, preventing intermittent freewheeling during acceleration and crest traversal. The position condition keeps it active after the rear wheel rises onto the ramp. The intended result is enough momentum to carry the green wheel across red and into blue.

Other learned pathways remain, although repurposing relay neurons means the pre-override policy is not mathematically identical. This is an untested hand repair; its success rate is not verified.