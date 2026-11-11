# Exp 4 · claude_r1 · Claude

## Why the trained network fails

**Physics:** The car has to cross a red strip about 0.9 long, and only a jump off the ramp can do that. By my estimate the green wheel must leave the ramp at about 2.6 m/s or more. The car starts at the foot of the ramp, so it can only reach about 2 m/s from there. If it arrives slowly, it high-centres: in the log, the body sits on the ramp's top corner from step 32 onward, with the green wheel hanging in the air. Throttle pulses at steps 48–240 don't move it.

**Weights:**
- **No position input:** The large `embed.W` rows are velocities, spin, and sin/cos of the car parts (54, 55, 59, 64, 236, 237, 241, 246, 247, 254, 255, 259, 264, 265). Position rows 52, 53, 234, 235, 252 and 253 hold only small values, so the policy has almost no sense of where the car is.
- **No reverse:** Value 2 has probability 0.00 at every logged step, so the car never backs up for a run-up, even though there are about 1.5 units of free track behind it.
- **Learned trap:** Backing up loses distance reward, and a slow jump lands on red for -1. PPO therefore settled on creeping up the ramp and stopping there, which matches the mean return of 0.152.

**Fix:** I bypass only the entry 0 logits with a small position and velocity controller. The other weights stay in place, but the old policy no longer influences the car.

```edits
# embed: X = x body, V = vx + 10, S = sin(angle) + 2
embed.W[:, 0:3] = 0
embed.W[52, 0] = 1
embed.W[54, 1] = 1
embed.W[64, 2] = 1
embed.b[0:3] = [0, 10, 2]

# a0 comparators
actor0.W[:, 0:6] = 0
actor0.W[0, 0] = 10     # x > 4.15
actor0.W[0, 1] = 6      # x - vx > 3.9
actor0.W[1, 1] = -6
actor0.W[1, 2] = 6      # vx > 1
actor0.W[0, 3] = -15    # x < 1.85
actor0.W[0, 4] = 10     # x > 3.1
actor0.W[2, 5] = -10    # sin(angle) < -0.25
actor0.b[0:6] = [-41.5, 36.6, -66, 27.75, -31, 17.5]

# a1 logic: forward OR, coast signal, wheelie AND
actor1.W[:, 0:3] = 0
actor1.W[0, 0] = 4
actor1.W[1, 0] = 4
actor1.W[3, 0] = 4
actor1.W[2, 1] = 3
actor1.W[4, 2] = 4
actor1.W[5, 2] = 4
actor1.b[0:3] = [8, 0, -4]

# a2..a4 pass-through
actor2.W[:, 0:3] = 0
actor2.W[0, 0] = 3
actor2.W[1, 1] = 3
actor2.W[2, 2] = 3
actor2.b[0:3] = [0, 0, 0]
actor3.W[:, 0:3] = 0
actor3.W[0, 0] = 3
actor3.W[1, 1] = 3
actor3.W[2, 2] = 3
actor3.b[0:3] = [0, 0, 0]
actor4.W[:, 0:3] = 0
actor4.W[0, 0] = 3
actor4.W[1, 1] = 3
actor4.W[2, 2] = 3
actor4.b[0:3] = [0, 0, 0]

# head: entry 0 logits only
actor_head.W[:, 0:3] = 0
actor_head.W[0, 1] = 25
actor_head.W[1, 0] = 10
actor_head.W[2, 0] = 30
actor_head.b[0:3] = [30, 0, 0]
```

## What the edits do

The new logits are close to deterministic and follow this sequence:

1. **Back up:** The car reverses until vx exceeds 1, then coasts right so it doesn't reach the wall at speed.
2. **Run up:** Past x = 4.15 it switches to full forward, giving about 1.2 of flat track before the ramp.
3. **Jump or abort:** Forward stays on while x - vx > 3.9. If the car is too slow near the ramp top, this fails and it reverses for another attempt, which also frees it from the stuck pose.
4. **Finish:** Below x = 1.85, forward is always on, so the car drives into blue after landing.
5. **Limit wheelies:** On the flat (x > 3.1), the motor cuts out if the nose rises past about 0.25 rad.

Nothing has been simulated. The thresholds rest on my hand estimates of grip, motor strength and launch speed, so the success rate is unknown.