# Exp 4 · astra_r0 · Astra

The car already drives in the correct direction: action 1 turns the wheels counter-clockwise and moves it left. The head strongly contrasts drive and coast—for example, rows 6 and 57 have opposing weights for those actions. Horizontal velocity is also strongly represented: embedding unit 116 receives weights 2.25, 2.37, and 1.73 from the three car bodies’ horizontal velocities.

The rollout identifies the failure: the policy climbs partway, then alternates driving and coasting without retaining enough momentum to cross the red strip. Coasting supplies no uphill torque. This is consistent with a shaping-reward local optimum: approach the goal, avoid red, and timeout.

I would add a backing-up and run-up maneuver, rather than reverse the learned driving direction or replace the entire policy. The patch retains most trained computation and returns control to its drive/coast decisions near the ramp crest.

```edits
# Reserve two embedding channels for body position and velocity.
embed.W[:, 6:8] = 0
embed.b[6:8] = 10
embed.W[52, 6] = 1
embed.W[54, 7] = 1

# Isolated position and velocity comparisons.
actor0.W[6:8, :] = 0
actor0.W[:, 120:126] = 0
actor0.b[120:126] = [-580, 645, 672.5, -187.5, -253.75, -612.5]
actor0.W[6, 120:123] = [50, -50, -50]
actor0.W[7, 123:125] = [25, 25]
actor0.W[6, 125] = 50

# Preserve the remaining trained channels; combine comparisons.
actor1.W[120:126, :] = 0
actor1.W[:, 120:125] = 0
actor1.b[120:125] = [0, 10, 0, 0, -10]
actor1.W[120, 120] = 5
actor1.W[121, 121] = 10
actor1.W[124, 121] = 10
actor1.W[122, 122] = 5
actor1.W[123, 123] = 5
actor1.W[125, 124] = 10
actor1.W[123, 124] = -10

# Back up, or turn and accelerate left.
actor2.W[120:125, :] = 0
actor2.W[:, 126:128] = 0
actor2.b[126:128] = [-30, 10]
actor2.W[120:124, 126] = 10
actor2.W[122, 127] = -10
actor2.W[124, 127] = 10

# Convert inactive intervention signals to approximately zero.
actor3.W[126:128, :] = 0
actor3.W[:, 126:128] = 0
actor3.b[126:128] = 5
actor3.W[126, 126] = 5
actor3.W[127, 127] = 5

# Carry interventions through two low-impact motor-head channels.
actor4.W[126:128, :] = 0
actor4.W[:, 12] = 0
actor4.W[:, 58] = 0
actor4.b[12] = 0
actor4.b[58] = 0
actor4.W[126, 12] = 3
actor4.W[127, 58] = 3

# Strongly select the required action only during an intervention.
actor_head.W[12, 0:3] = [-1000, -1000, 1000]
actor_head.W[58, 0:3] = [-1000, 1000, -1000]
actor_head.b[2] -= 100
```

Initially, the car backs right. Rightward velocity keeps that maneuver active until body x reaches approximately 3.45. It then drives left, with room to decelerate before the wall and build speed on the return.

Leftward acceleration continues until approximately x = 2.25, provided leftward speed exceeds 2.5. Afterward, the retained policy selects between driving left and coasting through the launch and landing. A slow approach can trigger another run-up; backing up is disabled below body x = 1.6.

The large intervention logits make these maneuver decisions nearly deterministic despite sampled actions. The reverse-logit offset prevents unintended reversal outside the explicit backing maneuver.

This is a physics-based, untested patch—not a measured success-rate claim.