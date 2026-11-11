# Exp 4 · claude_r0 · Claude

## Why the trained network fails

**What the car has to do.** It must drive left up the ramp, jump the red strip (x 1.18–2.065) and hit blue with the green wheel. Value 1 spins the wheels counter-clockwise, which drives the car left.

**What happens in the sampled episode.** The car reaches the ramp top slowly. At step 32 it sits at x ≈ 2.00 with its belly on the ramp's top corner (1.98, 0.817), the green wheel hanging in the air and the rear wheel on the ramp. It stays there for the rest of the episode, because the rear wheel alone can't pull it off the corner. The return of about 0.15 is just the shaping reward for the 0.8 m of progress.

**Why PPO stopped there.**
- The car starts right at the ramp base, with only about 0.5 m of flat run. I estimate full throttle from rest gives only about 2 m/s on the ramp, which is too slow to clear the strip and lands green on red (-1). This is an estimate, but PPO would have kept full throttle if it worked.
- PPO therefore learned to throttle gently and park on the crest.
- The real solution is a run-up, using the free floor to the right (up to x = 5). That means moving away from blue, which the shaping reward punishes, so value 2 has probability 0.00 everywhere.

**The fix.** Throttle needs a state-dependent run-up rule, and the trained throttle output can't be reshaped by hand. I keep the network but disconnect the throttle logits from it, and feed them from a small logic path through 7 embed units and a few units in each actor layer:

- `vx > 0.03` starts only after the car has landed, since vx is exactly 0 while it falls.
- Action 2 is used while 1.7 < x < 3.9 and 0.03 < vx < 1, which is a gentle reverse.
- Otherwise the car coasts until the wall stops it.
- Action 1 is used if vx < -0.4, or if x > 3.9 and vx < 0.03, or if x < 1.7.
- Throttle is cut while x > 3.1 and sin(angle) < -0.2, to stop a backflip wheelie on launch.

```edits
# --- embed: threshold detectors (relu)
embed.W[:, 0:7] = 0
embed.b[0:7] = [40, -3, -100, -390, -310, -20, -170]
embed.W[54, 0] = 100    # vx > -0.4
embed.W[54, 1] = 100    # vx > 0.03
embed.W[54, 2] = 100    # vx > 1.0
embed.W[52, 3] = 100    # x > 3.9
embed.W[52, 4] = 100    # x > 3.1
embed.W[64, 5] = -100   # sin(angle) < -0.2 (nose up)
embed.W[52, 6] = 100    # x > 1.7

# --- actor0: turn detectors into +-1 signals
actor0.W[:, 0:7] = 0
actor0.b[0:7] = [-4, -4, -4, -4, -4, -4, -4]
actor0.W[0, 0] = 10
actor0.W[1, 1] = 10
actor0.W[2, 2] = 10
actor0.W[3, 3] = 10
actor0.W[4, 4] = 10
actor0.W[5, 5] = 10
actor0.W[6, 6] = 10

# --- actor1: gates
actor1.W[:, 0:5] = 0
actor1.b[0:5] = [-15, 0, -5, 0, -5]
# unit 0: reverse = x<3.9 and 0.03<vx<1 and x>1.7
actor1.W[3, 0] = -5
actor1.W[1, 0] = 5
actor1.W[2, 0] = -5
actor1.W[6, 0] = 5
# unit 1: vx < -0.4
actor1.W[0, 1] = -5
# unit 2: x>3.9 and vx<0.03
actor1.W[3, 2] = 5
actor1.W[1, 2] = -5
# unit 3: x<1.7
actor1.W[6, 3] = -5
# unit 4: wheelie cut = x>3.1 and nose up
actor1.W[4, 4] = 5
actor1.W[5, 4] = 5

# --- actor2: OR of the go-left conditions, pass the others
actor2.W[:, 0:3] = 0
actor2.b[0:3] = [10, 0, 0]
actor2.W[1, 0] = 5
actor2.W[2, 0] = 5
actor2.W[3, 0] = 5
actor2.W[4, 1] = 5
actor2.W[0, 2] = 5

# --- actor3: go-left and not wheelie cut; pass reverse
actor3.W[:, 0:2] = 0
actor3.b[0:2] = [-5, 0]
actor3.W[0, 0] = 5
actor3.W[1, 0] = -5
actor3.W[2, 1] = 5

# --- actor4: pass through
actor4.W[:, 0:2] = 0
actor4.b[0:2] = [0, 0]
actor4.W[0, 0] = 5
actor4.W[1, 1] = 5

# --- head: entry 0 logits (0, +-12, +-12)
actor_head.W[:, 0:3] = 0
actor_head.b[0:3] = [0, 0, 0]
actor_head.W[0, 1] = 12
actor_head.W[1, 2] = 12
```

## What the edits do

The throttle logits now depend only on the new logic path. The rest of the network is unchanged, but it no longer affects anything that matters.

1. The car drops onto the ramp with the motor off, then reverses gently to about 1–2 m/s.
2. It coasts right until the wall at x ≈ 4.46 stops it.
3. It launches left at full throttle, cutting power briefly if the nose rises more than 0.2 rad while x > 3.1.
4. After about 1.5 m of run it should reach roughly 4 m/s, close to the wheels' 3.96 m/s limit. It keeps full throttle up the ramp and leaves the crest at about 3 m/s.
5. By my estimate the green wheel clears the strip and reaches blue in flight, or just after landing.

If a jump stalls on the crest, the car rolls back and retries, as long as time remains.

None of this has been simulated. The speeds and trajectories are hand estimates. The logits are saturated, so sampling should almost always pick the intended action.