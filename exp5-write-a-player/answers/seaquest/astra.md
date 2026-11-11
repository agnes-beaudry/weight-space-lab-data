# Exp 5 · Seaquest · Astra (our re-run)

The policy holds a central firing position, seeks divers, then also targets enemies when the gauge suggests a passenger. It selects the shallowest target row, dodges nearby hazards, and surfaces only with a confirmed passenger and low oxygen.

The first layer detects position, target rows, threats, and gauges. The second implements target priority, directional steering, boundary guards, and emergency escape preferences.

```edits
# Vertical coordinate
W1[0:60:6, 0] = -0.04
W1[60:120:6, 0] = -0.03
W1[120:180:6, 0] = -0.02
W1[180:240:6, 0] = -0.01
W1[300:360:6, 0] = 0.01
W1[360:420:6, 0] = 0.02
W1[420:480:6, 0] = 0.03
W1[480:540:6, 0] = 0.04

# Horizontal coordinate centered on columns 4 and 5
W1[0:600:60, 1] = -0.09
W1[6:600:60, 1] = -0.07
W1[12:600:60, 1] = -0.05
W1[18:600:60, 1] = -0.03
W1[24:600:60, 1] = -0.01
W1[30:600:60, 1] = 0.01
W1[36:600:60, 1] = 0.03
W1[42:600:60, 1] = 0.05
W1[48:600:60, 1] = 0.07
W1[54:600:60, 1] = 0.09

# Target detectors: divers first, enemies also when gauge permits
b1[2:10] = -66
W1[593, 2:10] = 60
W1[62:120:6, 2] = 72
W1[61:120:6, 2] = -60
W1[122:180:6, 3] = 72
W1[121:180:6, 3] = -60
W1[182:240:6, 4] = 72
W1[181:240:6, 4] = -60
W1[242:300:6, 5] = 72
W1[241:300:6, 5] = -60
W1[302:360:6, 6] = 72
W1[301:360:6, 6] = -60
W1[362:420:6, 7] = 72
W1[361:420:6, 7] = -60
W1[422:480:6, 8] = 72
W1[421:480:6, 8] = -60
W1[482:540:6, 9] = 72
W1[481:540:6, 9] = -60

# Threats in columns 2 through 7, gated by nearby player row
b1[10:18] = -78
W1[0:180:6, 10] = 36
W1[73:109:6, 10] = 12
W1[60:240:6, 11] = 36
W1[133:169:6, 11] = 12
W1[120:300:6, 12] = 36
W1[193:229:6, 12] = 12
W1[180:360:6, 13] = 36
W1[253:289:6, 13] = 12
W1[240:420:6, 14] = 36
W1[313:349:6, 14] = 12
W1[300:480:6, 15] = 36
W1[373:409:6, 15] = 12
W1[360:540:6, 16] = 36
W1[433:469:6, 16] = 12
W1[420:540:6, 17] = 36
W1[493:529:6, 17] = 12

# Immediate threats on the occupied row
b1[18:26] = -54
W1[60:120:6, 18] = 24
W1[79:103:6, 18] = 12
W1[120:180:6, 19] = 24
W1[139:163:6, 19] = 12
W1[180:240:6, 20] = 24
W1[199:223:6, 20] = 12
W1[240:300:6, 21] = 24
W1[259:283:6, 21] = 12
W1[300:360:6, 22] = 24
W1[319:343:6, 22] = 12
W1[360:420:6, 23] = 24
W1[379:403:6, 23] = 12
W1[420:480:6, 24] = 24
W1[439:463:6, 24] = 12
W1[480:540:6, 25] = 24
W1[499:523:6, 25] = 12

# Return only when oxygen is below 60 and a passenger is confirmed
W1[593, 26] = 12
W1[557, 26] = -12
b1[26] = -6

# Coarse horizontal escape preference
W1[13:540:60, 27] = 0.1
W1[19:540:60, 27] = 0.1
W1[25:540:60, 27] = 0.1
W1[31:540:60, 27] = -0.1
W1[37:540:60, 27] = -0.1
W1[43:540:60, 27] = -0.1

# Boundary detectors
W1[0:120:6, 28] = 6
W1[0:60:6, 29] = 6
W1[480:540:6, 30] = 6
b1[28:31] = -6

# Target steering and matching target-activation units
W2[0, 0:8] = 100
b2[0:8] = [-14, -36, -58, -80, -102, -124, -146, -168]
b2[8:16] = [-10, -30, -50, -70, -90, -110, -130, -150]

W2[2, 0:9:8] = 20
W2[2, 1:8] = -20
W2[2, 9:16] = -20
W2[3, 1:10:8] = 20
W2[3, 2:8] = -20
W2[3, 10:16] = -20
W2[4, 2:11:8] = 20
W2[4, 3:8] = -20
W2[4, 11:16] = -20
W2[5, 3:12:8] = 20
W2[5, 4:8] = -20
W2[5, 12:16] = -20
W2[6, 4:13:8] = 20
W2[6, 5:8] = -20
W2[6, 13:16] = -20
W2[7, 5:14:8] = 20
W2[7, 6:8] = -20
W2[7, 14:16] = -20
W2[8, 6:15:8] = 20
W2[8, 7:16:8] = -20
W2[9, 7:16:8] = 20

# Vertical threat repulsion; bottom-row threats repel upward
W2[0, 16:24] = 150
W2[10, 16] = 30
W2[11, 17] = 30
W2[12, 18] = 30
W2[13, 19] = 30
W2[14, 20] = 30
W2[15, 21] = 30
W2[16, 22] = 30
W2[17, 23] = 30
b2[16:24] = [-20.25, -23.25, -26.25, -29.25, -32.25, -35.25, -38.25, -42.75]

# Near-linear cancellation of inactive threat units
W2[10:18, 24] = 0.001

# Horizontal centering
W2[1, 25] = 100

# Surfacing request
W2[26, 26] = 6

# Forbid upward movement at rows 0 and 1 unless returning
W2[28, 27] = 6
W2[26, 27] = -6
b2[27] = -6

# Leave the surface; forbid downward movement at the bottom
W2[29, 28] = 6
W2[30, 29] = 6

# Immediate-threat firing penalty and emergency horizontal preference
W2[18:26, 30] = 0.001
W2[27, 31] = 5

# Target attraction
W3[0:8, 2] = 22
W3[0:8, 4] = -22
W3[8:16, 2] = -11
W3[8:16, 4] = 11

# Threat repulsion and inactive-unit cancellation
W3[16:24, 2] = -120
W3[16:24, 4] = 120
W3[24, 2] = 60000
W3[24, 4] = -60000

# Horizontal centering
W3[25, 1] = 120
W3[25, 3] = -120

# Oxygen-driven ascent
W3[26, 2] = 27.5
W3[26, 4] = -27.5

# Boundary guards and initial descent
W3[27, 2] = -500
W3[28, 4] = 35
W3[29, 4] = -500

# Prefer escape over firing under immediate threat
W3[30, 5] = -30000
W3[31, 1] = -8
W3[31, 3] = 8

# Baselines include cancellation constants
b3[:] = [-100, -20, -864.5, -20, -100.5, -232]
```