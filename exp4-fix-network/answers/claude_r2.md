# Exp 4 · claude_r2 · Claude

## Why the trained network fails

**The task:** Only the green wheel is checked against red. To reach blue, it has to clear the red strip (x 1.18–2.07) by jumping off the ramp, whose tip is near (1.98, 0.82). Value 1 drives left and value 2 drives right, at up to 18 rad/s × 0.22 m ≈ 4 m/s.

**What the log shows:**
- The first part works. The car climbs the ramp under value 1, and by step 16 both wheels are in contact with it.
- Value 2 has probability 0.00 throughout, so the car never reverses. It therefore starts its climb with no run-up.
- Going up the 35.6° slope, friction leaves only about 2–3 m/s² of net acceleration. The ramp is about 0.7 m long, so the car reaches about 2 m/s at the tip. Clearing the red strip needs at least about 2.1 m/s, plus a margin, so this is marginal at best.
- PPO's response was to avoid the −1. Near the crest the policy switches to value 0, and the car settles at (2.00, 0.92) with its body resting on the ramp tip.
- It then collects the shaping reward and times out (return 0.15). The later value-1 bursts are too short to move it.

**The weights:** The strong reliance on velocity inputs (rows 54, 236 and 254, mainly through embed unit 114) fits this speed-dependent throttle cut, though I'm inferring that from the log rather than tracing it through the network.

**The fix:** The missing behaviour is a run-up. The car should reverse onto the flat, turn around, then hold full throttle over the ramp. At about 4 m/s on the flat, the kink at the ramp base costs some speed, so I estimate 3.2–3.5 m/s at take-off. That carries the green wheel over the red strip and into or just short of blue.

```edits
# embed unit 0 = car body x ; embed unit 1 = car body vx + 10
embed.W[:, 0] = 0
embed.W[52, 0] = 1
embed.W[:, 1] = 0
embed.W[54, 1] = 1
embed.b[0:2] = [0, 10]

# actor0 unit 0: +1 if x < 3.5 ; unit 1: +1 if vx > -0.5
actor0.W[:, 0] = 0
actor0.W[0, 0] = -20
actor0.W[:, 1] = 0
actor0.W[1, 1] = 10
actor0.b[0:2] = [70, -95]

# actor1 unit 0: AND of both conditions (+1 = reverse, -1 = forward)
actor1.W[:, 0] = 0
actor1.W[0, 0] = 3
actor1.W[1, 0] = 3
actor1.b[0] = -3

# pass the decision through the remaining layers on unit 0
actor2.W[:, 0] = 0
actor2.W[0, 0] = 3
actor2.b[0] = 0
actor3.W[:, 0] = 0
actor3.W[0, 0] = 3
actor3.b[0] = 0
actor4.W[:, 0] = 0
actor4.W[0, 0] = 3
actor4.b[0] = 0

# entry 0 logits come only from the decision unit; value 0 is ruled out
actor_head.W[:, 0:3] = 0
actor_head.W[0, 1] = -15
actor_head.W[0, 2] = 15
actor_head.b[0:3] = [-30, 0, 0]
```

**What the edits do:** They replace the entry 0 decision with a small memoryless rule on one dedicated unit per layer. The rest of the network stays as it was, but it no longer affects the action, because the other entries drive nothing.

- **Rule:** drive right (value 2) only if the body is at x < 3.5 and vx > −0.5; otherwise drive left (value 1).
- **Run-up:** from rest the car reverses and passes x = 3.5 at about 3.8 m/s. It then brakes, stopping near x ≈ 4.2–4.3, short of the wall contact at about 4.46.
- **Attempt:** it gets about 1.3 m of flat ground to reach about 4 m/s, and throttle stays on over the ramp and through the flight, since vx stays below −0.5.
- **Retry:** if the car ever stalls, vx rises above −0.5, so it reverses and tries again.
- **Sampling:** the logit gap of about 30 makes the policy effectively deterministic, so all episodes should behave alike. The physics figures are hand estimates, not simulated.