# Exp 4 · astra_r1 · Astra

The policy learns safe partial progress, then stalls. It already selects the correct direction: action 1 turns the wheels counter-clockwise, driving left. At the ramp, however, it repeatedly releases the motors and never develops a successful launch.

The weights are particularly sensitive to horizontal velocity: `embed.W[54,116]`, `[236,116]`, and `[254,116]` are `2.25`, `2.37`, and `1.73`. The trace shows this velocity-sensitive throttle policy settling into an unsuccessful off/forward equilibrium.

I will preserve the trained controller, including its initial settling behavior, but progressively shift its horizontal-velocity feedback as it advances. This is intended to build momentum before the snag, rather than merely increase torque after stopping.

```edits
embed.W[52, 0:8] -= [1.4, -1.15, -2, 0.05, 0.8, 1.95, -0.5, -0.75]
embed.W[52, 8:16] -= [-0.2, -1.65, -0.2, 1.25, 0.35, 0.6, -2.55, -0.5]
embed.W[52, 16:24] -= [3.6, 3.55, 0.2, -0.5, -0.75, 9.6, 0.5, -0.95]
embed.W[52, 24:32] -= [-1.95, -2.1, -0.05, 1.85, -0.1, 0, 0.3, 0.7]
embed.W[52, 32:40] -= [-1.7, -0.2, 3.5, 0.25, 0.85, -0.9, -0.35, 1.95]
embed.W[52, 40:48] -= [-0.1, -5.6, 0.5, -0.35, -0.75, -0.7, 2.2, 0.25]
embed.W[52, 48:56] -= [0.6, 3.05, -2.7, -0.15, -0.3, 5.7, 7.3, -1.1]
embed.W[52, 56:64] -= [-2.55, 0.5, 0.6, 3.45, -0.25, 3.9, 0.6, -0.85]
embed.W[52, 64:72] -= [3.05, -0.65, 0.35, -0.95, 0.4, 0.05, 0.15, -1.1]
embed.W[52, 72:80] -= [0.7, 1.1, -3.5, -0.3, 0.55, -0.6, -0.1, 3.35]
embed.W[52, 80:88] -= [-1.75, 0.1, -0.25, -0.05, 1.1, -0.35, 0.2, -0.3]
embed.W[52, 88:96] -= [-0.65, -0.4, 0.5, 3.95, -0.15, -1.5, 0.4, 0]
embed.W[52, 96:104] -= [-0.6, -0.8, 0, -2.95, -1.5, 4.9, -3.8, 0.25]
embed.W[52, 104:112] -= [1.2, 0.55, -0.6, -0.75, 0.05, -0.75, -0.65, -2.45]
embed.W[52, 112:120] -= [-4, 2.1, 4.3, -0.45, 31.75, -2.6, -0.1, -3.95]
embed.W[52, 120:128] -= [-1.15, 0.1, -5.3, 0.05, -0.55, -3.5, 0.9, -2.25]
embed.b[0:8] += [3.92, -3.22, -5.6, 0.14, 2.24, 5.46, -1.4, -2.1]
embed.b[8:16] += [-0.56, -4.62, -0.56, 3.5, 0.98, 1.68, -7.14, -1.4]
embed.b[16:24] += [10.08, 9.94, 0.56, -1.4, -2.1, 26.88, 1.4, -2.66]
embed.b[24:32] += [-5.46, -5.88, -0.14, 5.18, -0.28, 0, 0.84, 1.96]
embed.b[32:40] += [-4.76, -0.56, 9.8, 0.7, 2.38, -2.52, -0.98, 5.46]
embed.b[40:48] += [-0.28, -15.68, 1.4, -0.98, -2.1, -1.96, 6.16, 0.7]
embed.b[48:56] += [1.68, 8.54, -7.56, -0.42, -0.84, 15.96, 20.44, -3.08]
embed.b[56:64] += [-7.14, 1.4, 1.68, 9.66, -0.7, 10.92, 1.68, -2.38]
embed.b[64:72] += [8.54, -1.82, 0.98, -2.66, 1.12, 0.14, 0.42, -3.08]
embed.b[72:80] += [1.96, 3.08, -9.8, -0.84, 1.54, -1.68, -0.28, 9.38]
embed.b[80:88] += [-4.9, 0.28, -0.7, -0.14, 3.08, -0.98, 0.56, -0.84]
embed.b[88:96] += [-1.82, -1.12, 1.4, 11.06, -0.42, -4.2, 1.12, 0]
embed.b[96:104] += [-1.68, -2.24, 0, -8.26, -4.2, 13.72, -10.64, 0.7]
embed.b[104:112] += [3.36, 1.54, -1.68, -2.1, 0.14, -2.1, -1.82, -6.86]
embed.b[112:120] += [-11.2, 5.88, 12.04, -1.26, 88.9, -7.28, -0.28, -11.06]
embed.b[120:128] += [-3.22, 0.28, -14.84, 0.14, -1.54, -9.8, 2.52, -6.3]
```

These edits are algebraically equivalent to adding `5 × (2.8 − body_x)` to the horizontal velocities of the chassis and both wheels before the original embedding.

Initially, the offset is zero, preserving the original action probabilities. As the car advances left, its controller perceives increasing rightward velocity and should demand more leftward drive. This targets the premature throttle release while retaining all trained actor layers, vertical-motion feedback, and rotation feedback.

The intended result is a faster ramp ascent and a jump across the red strip. Without simulation, I cannot verify the resulting success rate.