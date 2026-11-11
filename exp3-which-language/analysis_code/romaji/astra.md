## Call 1

```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=180)
with threadpool_limits(limits=4):
    for l in range(4):
        print(f"\nLAYER {l}")
        for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
            a = W[f"{name}.{l}"]
            print(name, "min/mean/max/std:", np.round([a.min(), a.mean(), a.max(), a.std()], 4))
        for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
            a = W[f"{name}.{l}"]
            s = svdvals(a)
            print(name, "Frob:", round(float(np.linalg.norm(a)), 3),
                  "top singular:", np.round(s[:10], 3),
                  "stable rank:", round(float((s*s).sum()/s[0]**2), 2))
        for j in range(6):
            sl = slice(64*j, 64*(j+1))
            q, k, v = [W[f"W_{a}.{l}"][sl].astype(np.float64) for a in ["q", "k", "v"]]
            o = W[f"W_o.{l}"][:, sl].astype(np.float64)
            A, B = q.T @ k / 8, o @ v
            sa, sb = svdvals(A)[:64], svdvals(B)[:64]
            ev = np.linalg.eigvals(v @ o)
            print(f"L{l}H{j}",
                  "Q/K/V/O norm", np.round([np.linalg.norm(a) for a in [q,k,v,o]], 2),
                  "QK s1..6", np.round(sa[:6], 3),
                  "QK sr", round(float((sa*sa).sum()/sa[0]**2), 2),
                  "QK sym", round(float(np.sum(A*A.T)/np.sum(A*A)), 3),
                  "OV s1..6", np.round(sb[:6], 3),
                  "OV trace", round(float(np.trace(B)), 3),
                  "OV eig maxRe/minRe", np.round([ev.real.max(),ev.real.min()], 3),
                  "OV eig positive-mass", round(float(ev.real.sum()/np.abs(ev).sum()), 3))

```

Output:

```

LAYER 0
ln1_w min/mean/max/std: [0.0352 0.2965 0.9785 0.1884]
ln1_b min/mean/max/std: [-0.0954  0.0004  0.0906  0.0311]
ln2_w min/mean/max/std: [0.042  0.8718 1.775  0.3382]
ln2_b min/mean/max/std: [-0.6236  0.0134  0.4636  0.0691]
W_q Frob: 17.669 top singular: [4.368 3.131 3.099 3.042 2.994 2.951 2.926 2.881 2.859 2.857] stable rank: 16.37
W_k Frob: 18.516 top singular: [4.163 3.952 3.467 3.427 3.364 3.335 3.279 3.256 3.214 3.163] stable rank: 19.78
W_v Frob: 14.937 top singular: [5.208 3.364 3.006 2.962 2.815 2.639 2.527 2.46  2.37  2.301] stable rank: 8.22
W_o Frob: 16.55 top singular: [7.361 6.099 3.167 2.452 2.321 2.202 2.171 2.16  2.105 2.072] stable rank: 5.06
W_fc Frob: 55.376 top singular: [18.118 12.76  11.337  9.323  9.267  9.021  8.898  8.817  8.609  8.4  ] stable rank: 9.34
W_proj Frob: 50.007 top singular: [10.899  7.702  7.173  6.638  6.541  6.288  6.201  6.133  5.941  5.917] stable rank: 21.05
L0H0 Q/K/V/O norm [7.14 7.4  5.92 6.13] QK s1..6 [0.853 0.613 0.368 0.287 0.279 0.244] QK sr 2.62 QK sym 0.054 OV s1..6 [3.611 3.392 2.881 2.314 2.128 1.837] OV trace -0.676 OV eig maxRe/minRe [ 0.233 -0.251] OV eig positive-mass -0.126
L0H1 Q/K/V/O norm [4.49 4.93 4.32 4.82] QK s1..6 [0.838 0.311 0.219 0.125 0.091 0.086] QK sr 1.28 QK sym 0.195 OV s1..6 [5.721 3.359 1.146 0.545 0.444 0.387] OV trace 0.527 OV eig maxRe/minRe [ 1.715 -0.431] OV eig positive-mass 0.116
L0H2 Q/K/V/O norm [4.07 4.36 4.02 4.36] QK s1..6 [0.543 0.183 0.157 0.133 0.072 0.066] QK sr 1.36 QK sym 0.191 OV s1..6 [3.943 2.357 1.006 0.521 0.5   0.45 ] OV trace 0.277 OV eig maxRe/minRe [ 1.168 -0.213] OV eig positive-mass 0.075
L0H3 Q/K/V/O norm [14.14 14.66  8.26  9.44] QK s1..6 [1.114 1.092 1.07  1.064 1.03  0.992] QK sr 14.93 QK sym -0.132 OV s1..6 [4.426 3.911 3.681 3.597 3.596 3.445] OV trace -7.452 OV eig maxRe/minRe [ 0.352 -0.718] OV eig positive-mass -0.419
L0H4 Q/K/V/O norm [4.2  4.61 3.69 3.66] QK s1..6 [0.51  0.227 0.185 0.124 0.101 0.066] QK sr 1.56 QK sym 0.134 OV s1..6 [2.449 1.317 0.666 0.461 0.438 0.363] OV trace -0.41 OV eig maxRe/minRe [ 0.63  -0.127] OV eig positive-mass -0.136
L0H5 Q/K/V/O norm [2.63 2.94 8.45 9.56] QK s1..6 [0.18  0.048 0.024 0.02  0.02  0.019] QK sr 1.35 QK sym 0.028 OV s1..6 [24.799 12.9    5.849  4.229  3.665  2.388] OV trace 2.348 OV eig maxRe/minRe [ 1.703 -0.521] OV eig positive-mass 0.19

LAYER 1
ln1_w min/mean/max/std: [0.197  0.5058 1.0846 0.0892]
ln1_b min/mean/max/std: [-1.5676  0.011   1.3651  0.1546]
ln2_w min/mean/max/std: [0.0019 1.1315 1.5196 0.1741]
ln2_b min/mean/max/std: [-0.3892  0.0065  0.417   0.1435]
W_q Frob: 38.929 top singular: [10.406  8.239  7.475  7.033  6.622  6.395  6.275  6.13   6.065  5.901] stable rank: 13.99
W_k Frob: 34.893 top singular: [14.477 11.787  6.37   6.054  5.523  5.407  5.268  5.167  5.091  4.924] stable rank: 5.81
W_v Frob: 28.744 top singular: [4.728 4.547 4.459 4.395 4.361 4.332 4.291 4.276 4.244 4.133] stable rank: 36.96
W_o Frob: 28.479 top singular: [4.045 3.267 3.188 3.179 3.163 3.101 3.071 3.046 3.046 3.036] stable rank: 49.57
W_fc Frob: 80.379 top singular: [19.829 11.575 11.522 11.262 11.059 10.875 10.632 10.497 10.363 10.3  ] stable rank: 16.43
W_proj Frob: 77.384 top singular: [9.038 8.457 8.103 7.333 7.318 7.088 6.772 6.666 6.566 6.429] stable rank: 73.32
L1H0 Q/K/V/O norm [16.63 14.95 11.76 11.71] QK s1..6 [3.084 2.605 2.082 1.854 1.729 1.62 ] QK sr 4.59 QK sym 0.079 OV s1..6 [6.118 5.692 5.46  5.236 5.209 4.809] OV trace -30.069 OV eig maxRe/minRe [ 0.391 -2.172] OV eig positive-mass -0.802
L1H1 Q/K/V/O norm [16.02 14.34 11.24 11.4 ] QK s1..6 [3.082 2.497 1.812 1.767 1.429 1.376] QK sr 3.83 QK sym 0.051 OV s1..6 [6.051 5.716 5.492 5.23  5.099 5.084] OV trace -29.585 OV eig maxRe/minRe [ 0.139 -1.996] OV eig positive-mass -0.838
L1H2 Q/K/V/O norm [15.58 14.14 11.74 11.38] QK s1..6 [2.848 2.211 1.794 1.583 1.438 1.325] QK sr 4.02 QK sym 0.074 OV s1..6 [6.72  6.39  6.182 5.782 5.597 5.391] OV trace -33.608 OV eig maxRe/minRe [ 0.11  -2.418] OV eig positive-mass -0.866
L1H3 Q/K/V/O norm [15.75 13.95 11.8  11.88] QK s1..6 [3.106 2.415 1.629 1.446 1.323 1.202] QK sr 3.41 QK sym 0.044 OV s1..6 [6.544 6.21  6.074 5.823 5.568 5.466] OV trace -36.07 OV eig maxRe/minRe [ 0.126 -2.376] OV eig positive-mass -0.887
L1H4 Q/K/V/O norm [15.2  13.64 12.73 12.11] QK s1..6 [2.62  1.795 1.743 1.336 1.248 1.193] QK sr 4.1 QK sym 0.072 OV s1..6 [7.908 7.206 6.838 6.499 6.413 6.044] OV trace -37.328 OV eig maxRe/minRe [ 1.048 -3.055] OV eig positive-mass -0.826
L1H5 Q/K/V/O norm [16.15 14.42 11.07 11.24] QK s1..6 [3.024 2.555 1.908 1.631 1.608 1.433] QK sr 4.1 QK sym 0.057 OV s1..6 [5.849 5.644 5.5   5.382 5.204 4.938] OV trace -28.92 OV eig maxRe/minRe [ 0.376 -2.14 ] OV eig positive-mass -0.822

LAYER 2
ln1_w min/mean/max/std: [0.3175 0.6809 1.0927 0.0687]
ln1_b min/mean/max/std: [-1.6589  0.0131  1.3318  0.1565]
ln2_w min/mean/max/std: [0.012  1.3104 1.5078 0.1148]
ln2_b min/mean/max/std: [-0.517   0.0123  0.4489  0.1737]
W_q Frob: 38.902 top singular: [5.966 5.019 4.865 4.773 4.695 4.568 4.496 4.433 4.38  4.28 ] stable rank: 42.52
W_k Frob: 36.75 top singular: [9.863 9.533 4.386 4.171 4.117 3.956 3.879 3.777 3.759 3.708] stable rank: 13.88
W_v Frob: 33.183 top singular: [3.914 3.78  3.727 3.662 3.582 3.553 3.51  3.464 3.449 3.432] stable rank: 71.86
W_o Frob: 32.435 top singular: [4.398 4.037 3.875 3.739 3.621 3.589 3.488 3.467 3.387 3.353] stable rank: 54.38
W_fc Frob: 79.828 top singular: [15.734  9.249  8.718  8.694  8.47   7.865  7.739  7.482  7.456  7.36 ] stable rank: 25.74
W_proj Frob: 78.048 top singular: [10.197  9.035  8.882  8.411  8.224  8.088  7.988  7.827  7.679  7.491] stable rank: 58.58
L2H0 Q/K/V/O norm [14.49 13.57 13.05 12.69] QK s1..6 [1.229 1.058 0.753 0.673 0.657 0.651] QK sr 7.45 QK sym -0.008 OV s1..6 [7.576 6.429 6.378 5.428 5.292 5.041] OV trace -24.508 OV eig maxRe/minRe [ 0.7   -2.573] OV eig positive-mass -0.542
L2H1 Q/K/V/O norm [16.8  15.93 13.86 13.48] QK s1..6 [1.189 1.084 1.07  0.971 0.965 0.917] Q
```

## Call 2

```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=200)

def effective_matrix(a, gain):
    a = a.astype(np.float64) * gain[None, :]
    return a - a.mean(axis=1, keepdims=True)

with threadpool_limits(limits=4):
    print("LAST MLP PROJECTION: first 100 singular values")
    print(svdvals(W["W_proj.3"])[:100])
    print("\nL0 LN1 gain quantiles:",
          np.quantile(W["ln1_w.0"], [0,.05,.1,.25,.5,.75,.9,.95,1]))
    for l in range(4):
        gain, bias = W[f"ln1_w.{l}"], W[f"ln1_b.{l}"]
        print(f"\nLAYER {l}: diagonal-gain/centering-adjusted weight products")
        for j in range(6):
            sl = slice(64*j,64*(j+1))
            q0,k0,v0 = [W[f"W_{a}.{l}"][sl].astype(float) for a in ["q","k","v"]]
            q,k,v = [effective_matrix(a,gain) for a in [q0,k0,v0]]
            o = W[f"W_o.{l}"][:,sl].astype(float)
            A, B = q.T @ k / 8, o @ v
            s = svdvals(A)[:64]
            ev = np.linalg.eigvals(k @ q.T / 8)
            ev = ev[np.argsort(-np.abs(ev))]
            key_linear = k.T @ (q0 @ bias) / 8
            print(f"L{l}H{j}: QKnorm={np.linalg.norm(A):.3f}, sr={(s@s)/(s[0]**2):.2f}, "
                  f"sym={np.sum(A*A.T)/np.sum(A*A):.3f}, "
                  f"trace={np.trace(A):.3f}, key-linear-norm={np.linalg.norm(key_linear):.3f}, "
                  f"OVtrace={np.trace(B):.3f}")
            print("  QK singular", np.round(s[:12],3),
                  "eigenvalues", np.round(ev[:10],3))
    print("\nL0H3 adjusted QK eigenvalues, all non-negligible:")
    sl=slice(192,256)
    q=effective_matrix(W["W_q.0"][sl],W["ln1_w.0"])
    k=effective_matrix(W["W_k.0"][sl],W["ln1_w.0"])
    ev=np.linalg.eigvals(k@q.T/8)
    print(ev[np.argsort(-np.abs(ev))])

```

Output:

```
LAST MLP PROJECTION: first 100 singular values
[28.4613 22.2137 21.3397 20.708  19.9429 19.6618 18.9864 18.744  18.2945 17.4567 17.1321 16.5609 15.3285 14.8739 14.3285 13.4378 12.667  12.4546 12.317  11.5309 11.0466  9.0836  8.3531  8.2404
  6.3701  5.8357  4.6262  4.5827  3.7775  3.6102  3.2518  3.1826  3.0856  2.9413  2.8655  2.8013  2.6391  2.5637  2.538   2.4655  2.4188  2.4037  2.346   2.3102  2.2461  2.227   2.1548  2.1472
  2.0887  2.0693  2.0646  2.0413  1.9748  1.9356  1.9211  1.9165  1.8572  1.8485  1.8298  1.8197  1.7977  1.7791  1.7637  1.7368  1.7294  1.7192  1.7088  1.6925  1.6799  1.6612  1.6448  1.6346
  1.6188  1.594   1.5892  1.5831  1.5653  1.5503  1.5412  1.5199  1.515   1.4844  1.4829  1.48    1.4705  1.459   1.4474  1.4351  1.4174  1.4135  1.4064  1.3942  1.3856  1.3806  1.3708  1.3596
  1.3539  1.3473  1.3339  1.3298]

L0 LN1 gain quantiles: [0.0352 0.0929 0.1118 0.1713 0.2422 0.3546 0.5696 0.7253 0.9785]

LAYER 0: diagonal-gain/centering-adjusted weight products
L0H0: QKnorm=0.293, sr=1.40, sym=0.141, trace=-0.105, key-linear-norm=0.043, OVtrace=-0.142
  QK singular [0.248 0.102 0.047 0.043 0.037 0.036 0.034 0.031 0.026 0.023 0.022 0.022] eigenvalues [ 0.08 +0.j     0.048+0.j    -0.024+0.006j -0.024-0.006j -0.023+0.j     0.018+0.013j  0.018-0.013j -0.02 +0.j    -0.014+0.012j -0.014-0.012j]
L0H1: QKnorm=0.159, sr=1.12, sym=0.161, trace=-0.098, key-linear-norm=0.013, OVtrace=0.921
  QK singular [0.15  0.042 0.021 0.012 0.01  0.007 0.007 0.004 0.004 0.004 0.003 0.003] eigenvalues [-0.062+0.j    -0.01 +0.j    -0.009+0.j     0.004+0.j    -0.004+0.j    -0.003+0.002j -0.003-0.002j -0.002+0.j    -0.002-0.j    -0.002+0.001j]
L0H2: QKnorm=0.107, sr=1.10, sym=0.289, trace=-0.087, key-linear-norm=0.007, OVtrace=0.695
  QK singular [0.102 0.018 0.016 0.012 0.008 0.007 0.005 0.004 0.004 0.004 0.003 0.003] eigenvalues [-0.057+0.j    -0.007+0.003j -0.007-0.003j -0.003+0.001j -0.003-0.001j -0.002+0.j    -0.001+0.j    -0.001-0.j    -0.001+0.001j -0.001-0.001j]
L0H3: QKnorm=0.698, sr=8.21, sym=-0.067, trace=1.342, key-linear-norm=0.043, OVtrace=-2.016
  QK singular [0.244 0.222 0.201 0.182 0.179 0.157 0.155 0.149 0.14  0.136 0.131 0.128] eigenvalues [0.137+0.133j 0.137-0.133j 0.11 +0.089j 0.11 -0.089j 0.095+0.098j 0.095-0.098j 0.084+0.104j 0.084-0.104j 0.064+0.114j 0.064-0.114j]
L0H4: QKnorm=0.088, sr=1.24, sym=0.050, trace=-0.046, key-linear-norm=0.017, OVtrace=0.173
  QK singular [0.08  0.022 0.019 0.013 0.011 0.008 0.007 0.006 0.005 0.005 0.004 0.004] eigenvalues [-0.015+0.j    -0.01 +0.j    -0.008+0.j    -0.003+0.j    -0.003-0.j    -0.002+0.001j -0.002-0.001j -0.   +0.002j -0.   -0.002j -0.001+0.001j]
L0H5: QKnorm=0.034, sr=1.11, sym=0.024, trace=-0.009, key-linear-norm=0.005, OVtrace=2.088
  QK singular [0.032 0.004 0.003 0.003 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002] eigenvalues [-0.005+0.j     0.001+0.j    -0.001+0.j    -0.   +0.001j -0.   -0.001j -0.001+0.j    -0.001+0.j    -0.001-0.j     0.   +0.001j  0.   -0.001j]

LAYER 1: diagonal-gain/centering-adjusted weight products
L1H0: QKnorm=1.978, sr=2.79, sym=0.042, trace=-1.748, key-linear-norm=3.975, OVtrace=-15.453
  QK singular [1.185 0.723 0.557 0.534 0.472 0.432 0.394 0.364 0.354 0.294 0.277 0.269] eigenvalues [-0.264+0.j     0.116+0.231j  0.116-0.231j -0.221+0.j    -0.199+0.05j  -0.199-0.05j  -0.1  +0.121j -0.1  -0.121j -0.143+0.j    -0.121+0.04j ]
L1H1: QKnorm=1.843, sr=2.55, sym=0.018, trace=-1.557, key-linear-norm=3.689, OVtrace=-15.216
  QK singular [1.154 0.708 0.582 0.465 0.398 0.368 0.319 0.299 0.288 0.261 0.246 0.226] eigenvalues [ 0.094+0.247j  0.094-0.247j -0.202+0.j    -0.164+0.018j -0.164-0.018j -0.145+0.j    -0.119+0.07j  -0.119-0.07j  -0.056+0.108j -0.056-0.108j]
L1H2: QKnorm=1.781, sr=2.29, sym=0.037, trace=-1.887, key-linear-norm=3.051, OVtrace=-17.321
  QK singular [1.176 0.616 0.485 0.436 0.385 0.354 0.306 0.293 0.278 0.252 0.239 0.23 ] eigenvalues [-0.249+0.j    -0.201+0.j     0.069+0.183j  0.069-0.183j -0.172+0.j    -0.109+0.1j   -0.109-0.1j   -0.142+0.j     0.007+0.132j  0.007-0.132j]
L1H3: QKnorm=1.787, sr=2.27, sym=0.005, trace=-1.512, key-linear-norm=3.390, OVtrace=-18.369
  QK singular [1.187 0.671 0.532 0.388 0.366 0.329 0.301 0.283 0.274 0.248 0.238 0.215] eigenvalues [ 0.083+0.226j  0.083-0.226j -0.198+0.j    -0.151+0.j    -0.127+0.062j -0.127-0.062j -0.128+0.j    -0.045+0.119j -0.045-0.119j -0.099+0.047j]
L1H4: QKnorm=1.629, sr=2.37, sym=0.047, trace=-2.011, key-linear-norm=2.329, OVtrace=-19.151
  QK singular [1.059 0.504 0.476 0.351 0.334 0.31  0.299 0.292 0.286 0.273 0.253 0.225] eigenvalues [-0.172+0.j    -0.044+0.145j -0.044-0.145j -0.147+0.j    -0.132+0.044j -0.132-0.044j -0.134+0.j    -0.107+0.057j -0.107-0.057j  0.079+0.092j]
L1H5: QKnorm=1.863, sr=2.64, sym=0.017, trace=-1.630, key-linear-norm=3.895, OVtrace=-14.954
  QK singular [1.146 0.71  0.556 0.499 0.433 0.379 0.355 0.328 0.293 0.253 0.234 0.228] eigenvalues [ 0.102+0.255j  0.102-0.255j -0.261+0.j    -0.145+0.099j -0.145-0.099j -0.156+0.027j -0.156-0.027j -0.082+0.114j -0.082-0.114j -0.124+0.j   ]

LAYER 2: diagonal-gain/centering-adjusted weight products
L2H0: QKnorm=1.631, sr=5.61, sym=-0.040, trace=-1.555, key-linear-norm=2.020, OVtrace=-16.831
  QK singular [0.689 0.553 0.35  0.327 0.307 0.3   0.296 0.282 0.275 0.255 0.247 0.238] eigenvalues [ 0.082+0.3j    0.082-0.3j   -0.104+0.j    -0.097-0.024j -0.097+0.024j -0.011+0.088j -0.011-0.088j -0.085+0.013j -0.085-0.013j -0.053-0.066j]
L2H1: QKnorm=2.154, sr=13.91, sym=-0.021, trace=-0.216, key-linear-norm=1.880, OVtrace=-35.160
  QK singular [0.578 0.513 0.509 0.465 0.458 0.444 0.437 0.409 0.406 0.392 0.381 0.364] eigenvalues [ 0.079+0.268j  0.079-0.268j -0.13 +0.j    -0.067+0.105j -0.067-0.105j -0.115+0.j     0.114+0.j     0.024+0.1j    0.024-0.1j   -0.087+0.046j]
L2H2: QKnorm=1.720, sr=11.02, sym=-0.004, trace=-1.297, key-linear-norm=1.608, OVtrace=-29.935
  QK singular [0.518 0.41  0.399 0.374 0.355 0.333 0.325 0.31  0.296 0.291 0.277 0.274] eigenvalues [ 0.075+0.209
```

## Call 3

```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=3, suppress=True, linewidth=200)

def eff(a, gain):
    a = a.astype(float) * gain[None,:]
    return a-a.mean(axis=1,keepdims=True)

def cosine(a,b):
    return np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))

with threadpool_limits(limits=4):
    As, Bs, Cs = [], [], []
    for l in range(4):
        aa,bb,cc=[],[],[]
        for j in range(6):
            sl=slice(j*64,(j+1)*64)
            q0,k0,v0=[W[f"W_{a}.{l}"][sl] for a in ["q","k","v"]]
            q,k,v=[eff(a,W[f"ln1_w.{l}"]) for a in [q0,k0,v0]]
            aa.append(q.T@k/8)
            bb.append(W[f"W_o.{l}"][:,sl]@v)
            cc.append(k.T@(q0@W[f"ln1_b.{l}"])/8)
        As.append(aa); Bs.append(bb); Cs.append(cc)
        print(f"\nL{l} QK pairwise cosines")
        print(np.array([[cosine(a,b) for b in aa] for a in aa]))
        print(f"L{l} OV pairwise cosines")
        print(np.array([[cosine(a,b) for b in bb] for a in bb]))
        print(f"L{l} bias-induced key-vector cosines")
        print(np.array([[cosine(a,b) for b in cc] for a in cc]))

    print("\nL0 adjusted W_v singular values")
    V=eff(W["W_v.0"],W["ln1_w.0"])
    _,s,Vh=svd(V,full_matrices=False)
    print(s[:80])
    print("\nL0 adjusted W_fc singular values")
    F=eff(W["W_fc.0"],W["ln2_w.0"])
    print(svdvals(F)[:70])
    T=Vh[:28].T
    P=np.eye(384)-T@T.T
    for j in range(6):
        A=As[0][j]
        print(f"L0H{j} QK Frobenius energy fractions: token-token={np.linalg.norm(T.T@A@T)**2/np.linalg.norm(A)**2:.3f}, "
              f"orthogonal-orthogonal={np.linalg.norm(P@A@P)**2/np.linalg.norm(A)**2:.3f}")
    print("\nL0H3 Q/K individual singular values and shared-subspace canonical correlations")
    sl=slice(192,256)
    q=eff(W["W_q.0"][sl],W["ln1_w.0"])
    k=eff(W["W_k.0"][sl],W["ln1_w.0"])
    _,sq,vq=svd(q,full_matrices=False)
    _,sk,vk=svd(k,full_matrices=False)
    print("Q",sq)
    print("K",sk)
    for r in [16,24,32,40,48,64]:
        print("rank",r, "canonical correlations", svdvals(vq[:r]@vk[:r].T))

```

Output:

```

L0 QK pairwise cosines
[[ 1.     0.012  0.008  0.137  0.013  0.044]
 [ 0.012  1.     0.691 -0.026  0.57   0.026]
 [ 0.008  0.691  1.    -0.063  0.493 -0.075]
 [ 0.137 -0.026 -0.063  1.    -0.044  0.01 ]
 [ 0.013  0.57   0.493 -0.044  1.     0.033]
 [ 0.044  0.026 -0.075  0.01   0.033  1.   ]]
L0 OV pairwise cosines
[[ 1.     0.018  0.017 -0.036  0.026  0.017]
 [ 0.018  1.     0.925  0.006  0.869  0.644]
 [ 0.017  0.925  1.     0.009  0.811  0.692]
 [-0.036  0.006  0.009  1.     0.011 -0.001]
 [ 0.026  0.869  0.811  0.011  1.     0.592]
 [ 0.017  0.644  0.692 -0.001  0.592  1.   ]]
L0 bias-induced key-vector cosines
[[ 1.    -0.054 -0.104 -0.041  0.29  -0.095]
 [-0.054  1.     0.347 -0.093  0.41  -0.226]
 [-0.104  0.347  1.    -0.057  0.567  0.524]
 [-0.041 -0.093 -0.057  1.     0.022  0.074]
 [ 0.29   0.41   0.567  0.022  1.     0.373]
 [-0.095 -0.226  0.524  0.074  0.373  1.   ]]

L1 QK pairwise cosines
[[1.    0.357 0.372 0.345 0.334 0.376]
 [0.357 1.    0.371 0.361 0.302 0.398]
 [0.372 0.371 1.    0.368 0.375 0.351]
 [0.345 0.361 0.368 1.    0.32  0.394]
 [0.334 0.302 0.375 0.32  1.    0.288]
 [0.376 0.398 0.351 0.394 0.288 1.   ]]
L1 OV pairwise cosines
[[ 1.    -0.02   0.03  -0.012  0.001  0.029]
 [-0.02   1.    -0.009 -0.028 -0.018  0.016]
 [ 0.03  -0.009  1.     0.001  0.014 -0.017]
 [-0.012 -0.028  0.001  1.    -0.001 -0.015]
 [ 0.001 -0.018  0.014 -0.001  1.    -0.014]
 [ 0.029  0.016 -0.017 -0.015 -0.014  1.   ]]
L1 bias-induced key-vector cosines
[[1.    0.829 0.853 0.823 0.825 0.86 ]
 [0.829 1.    0.851 0.864 0.821 0.861]
 [0.853 0.851 1.    0.83  0.834 0.833]
 [0.823 0.864 0.83  1.    0.834 0.873]
 [0.825 0.821 0.834 0.834 1.    0.828]
 [0.86  0.861 0.833 0.873 0.828 1.   ]]

L2 QK pairwise cosines
[[1.    0.128 0.161 0.117 0.179 0.134]
 [0.128 1.    0.116 0.086 0.121 0.116]
 [0.161 0.116 1.    0.101 0.144 0.123]
 [0.117 0.086 0.101 1.    0.092 0.093]
 [0.179 0.121 0.144 0.092 1.    0.134]
 [0.134 0.116 0.123 0.093 0.134 1.   ]]
L2 OV pairwise cosines
[[ 1.    -0.023 -0.039 -0.022 -0.005  0.001]
 [-0.023  1.    -0.001  0.002 -0.012 -0.008]
 [-0.039 -0.001  1.     0.013  0.004  0.022]
 [-0.022  0.002  0.013  1.     0.001  0.   ]
 [-0.005 -0.012  0.004  0.001  1.     0.004]
 [ 0.001 -0.008  0.022  0.     0.004  1.   ]]
L2 bias-induced key-vector cosines
[[1.    0.807 0.766 0.703 0.82  0.835]
 [0.807 1.    0.811 0.719 0.824 0.837]
 [0.766 0.811 1.    0.699 0.832 0.827]
 [0.703 0.719 0.699 1.    0.715 0.73 ]
 [0.82  0.824 0.832 0.715 1.    0.845]
 [0.835 0.837 0.827 0.73  0.845 1.   ]]

L3 QK pairwise cosines
[[1.    0.058 0.082 0.06  0.007 0.108]
 [0.058 1.    0.069 0.034 0.041 0.075]
 [0.082 0.069 1.    0.031 0.036 0.094]
 [0.06  0.034 0.031 1.    0.036 0.058]
 [0.007 0.041 0.036 0.036 1.    0.027]
 [0.108 0.075 0.094 0.058 0.027 1.   ]]
L3 OV pairwise cosines
[[ 1.    -0.012  0.024  0.011 -0.002  0.034]
 [-0.012  1.    -0.021  0.024  0.059  0.001]
 [ 0.024 -0.021  1.     0.003  0.078  0.026]
 [ 0.011  0.024  0.003  1.     0.042 -0.001]
 [-0.002  0.059  0.078  0.042  1.     0.057]
 [ 0.034  0.001  0.026 -0.001  0.057  1.   ]]
L3 bias-induced key-vector cosines
[[ 1.     0.6    0.747  0.013  0.159  0.776]
 [ 0.6    1.     0.553  0.015  0.289  0.669]
 [ 0.747  0.553  1.     0.011  0.208  0.736]
 [ 0.013  0.015  0.011  1.     0.026 -0.046]
 [ 0.159  0.289  0.208  0.026  1.     0.254]
 [ 0.776  0.669  0.736 -0.046  0.254  1.   ]]

L0 adjusted W_v singular values
[3.339 2.305 1.516 0.814 0.773 0.711 0.682 0.672 0.651 0.633 0.618 0.597 0.574 0.561 0.558 0.537 0.525 0.508 0.498 0.487 0.476 0.469 0.461 0.45  0.432 0.427 0.417 0.41  0.408 0.395 0.388 0.382 0.38
 0.376 0.363 0.359 0.356 0.348 0.342 0.334 0.333 0.326 0.317 0.308 0.307 0.301 0.3   0.295 0.294 0.288 0.287 0.282 0.278 0.274 0.272 0.266 0.265 0.262 0.261 0.258 0.251 0.249 0.246 0.245 0.239 0.239
 0.235 0.234 0.232 0.231 0.226 0.224 0.219 0.217 0.216 0.212 0.209 0.207 0.206 0.203]

L0 adjusted W_fc singular values
[17.576 13.472 11.756 10.188  9.613  9.54   9.342  9.144  8.754  8.681  8.584  8.387  8.001  7.926  7.84   7.727  7.516  7.385  7.159  7.084  7.067  6.913  6.887  6.681  6.622  6.487  6.436  6.354
  6.306  6.157  6.001  5.909  5.826  5.657  5.466  5.301  5.289  5.127  5.041  4.957  4.876  4.443  4.367  4.11   4.035  3.799  3.629  3.408  3.225  3.153  2.947  2.935  2.813  2.629  2.591  2.481
  2.459  2.388  2.349  2.249  2.198  2.146  2.113  2.09   2.05   2.034  1.992  1.89   1.877  1.857]
L0H0 QK Frobenius energy fractions: token-token=0.710, orthogonal-orthogonal=0.104
L0H1 QK Frobenius energy fractions: token-token=0.508, orthogonal-orthogonal=0.094
L0H2 QK Frobenius energy fractions: token-token=0.609, orthogonal-orthogonal=0.088
L0H3 QK Frobenius energy fractions: token-token=0.177, orthogonal-orthogonal=0.641
L0H4 QK Frobenius energy fractions: token-token=0.496, orthogonal-orthogonal=0.177
L0H5 QK Frobenius energy fractions: token-token=0.600, orthogonal-orthogonal=0.086

L0H3 Q/K individual singular values and shared-subspace canonical correlations
Q [1.212 1.159 1.114 1.08  1.063 1.051 1.017 0.997 0.972 0.956 0.948 0.928 0.91  0.902 0.88  0.873 0.866 0.837 0.815 0.805 0.79  0.761 0.753 0.733 0.709 0.687 0.643 0.623 0.579 0.517 0.502 0.48  0.472
 0.443 0.412 0.39  0.365 0.362 0.342 0.313 0.299 0.286 0.28  0.274 0.261 0.253 0.241 0.235 0.232 0.21  0.206 0.197 0.195 0.189 0.185 0.181 0.166 0.163 0.159 0.152 0.145 0.135 0.13  0.101]
K [2.378 1.965 1.646 1.472 1.448 1.339 1.293 1.224 1.182 1.143 1.084 1.059 1.055 1.03  0.982 0.958 0.939 0.906 0.848 0.836 0.807 0.775 0.741 0.698 0.671 0.647 0.602 0.572 0.553 0.49  0.459 0.436 0.41
 0.384 0.365 0.352 0.343 0.319 0.3   0.289 0.273 0.267 0.253 0.245 0.235 0.223 0.219 0.213 0.202 0.194 0.187 0.182 0.172 0.171 0.164 0.155 0.151 0.144 0.137 0.132 0.131 0.127 0.121 0.107]
rank 16 canonical correlations [0.987 0.984 0.978 0.972 0.965 0.959 0.934 0.932 0.916 0.9   0.816 0.736 0.713 0.452 0.206 0.097]
rank 24 canonical correlations [0.993 0.991 0.989 0.988 0.986 0.98  0.97
```

## Call 4

```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=3, suppress=True, linewidth=200)

def eff(a,g):
    a=np.asarray(a,dtype=float)*g[None,:]
    return a-a.mean(axis=1,keepdims=True)

with threadpool_limits(limits=4):
    A=[];B=[];names=[]
    for l in range(4):
        for j in range(6):
            sl=slice(64*j,64*(j+1))
            q,k,v=[eff(W[f"W_{n}.{l}"][sl],W[f"ln1_w.{l}"]) for n in ["q","k","v"]]
            A.append(q.T@k/8)
            B.append(W[f"W_o.{l}"][:,sl]@v)
            names.append(f"L{l}H{j}")
    print("Key-composition signed trace divided by Frobenius-product norms.")
    print("Columns are L0H0..L0H5; positive is a possible matching/induction signature.")
    scores=np.array([[np.trace(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)) for b in B[:6]] for a in A[6:]])
    for name,row in zip(names[6:],scores):
        print(name,row)
    print("\nFor each later head: strongest earlier key composition by absolute normalized trace")
    for i in range(6,24):
        vals=[]
        for k in range((i//6)*6):
            C=A[i]@B[k]
            score=np.trace(C)/(np.linalg.norm(A[i])*np.linalg.norm(B[k]))
            vals.append((abs(score),score,k,C))
        vals.sort(key=lambda x:-x[0])
        for _,score,k,C in vals[:2]:
            s=svdvals(C)
            print(names[i],"<-",names[k],
                  f"normalized_trace={score:.4f}",
                  f"trace/nuclear={np.trace(C)/s.sum():.4f}",
                  f"sym={np.sum(C*C.T)/np.sum(C*C):.4f}",
                  f"stable_rank={(s@s)/(s[0]**2):.2f}")
    print("\nL0H3: unadjusted QK eigenvalues")
    q,k=[W[f"W_{n}.0"][192:256].astype(float) for n in ["q","k"]]
    e=np.linalg.eigvals(k@q.T/8)
    print(e[np.argsort(-np.abs(e))])

```

Output:

```
Key-composition signed trace divided by Frobenius-product norms.
Columns are L0H0..L0H5; positive is a possible matching/induction signature.
L1H0 [0.002 0.034 0.05  0.006 0.029 0.048]
L1H1 [0.    0.006 0.025 0.014 0.004 0.021]
L1H2 [-0.002  0.03   0.042  0.012  0.031  0.047]
L1H3 [0.001 0.028 0.045 0.005 0.024 0.043]
L1H4 [0.001 0.017 0.03  0.001 0.023 0.037]
L1H5 [-0.001  0.019  0.04   0.011  0.019  0.031]
L2H0 [-0.     0.038  0.058  0.004  0.023  0.067]
L2H1 [0.001 0.021 0.033 0.001 0.013 0.042]
L2H2 [0.003 0.026 0.039 0.005 0.014 0.046]
L2H3 [0.003 0.016 0.023 0.005 0.012 0.025]
L2H4 [-0.001  0.03   0.045  0.009  0.022  0.051]
L2H5 [0.001 0.021 0.036 0.001 0.014 0.047]
L3H0 [-0.002  0.008  0.013 -0.01   0.005  0.013]
L3H1 [ 0.002  0.01   0.013 -0.002  0.004  0.014]
L3H2 [-0.     0.007  0.012  0.002  0.001  0.004]
L3H3 [-0.008  0.002  0.002  0.063  0.002 -0.002]
L3H4 [0.005 0.009 0.009 0.006 0.01  0.009]
L3H5 [-0.     0.002  0.007 -0.003  0.     0.009]

For each later head: strongest earlier key composition by absolute normalized trace
L1H0 <- L0H2 normalized_trace=0.0502 trace/nuclear=0.0964 sym=0.0086 stable_rank=1.25
L1H0 <- L0H5 normalized_trace=0.0479 trace/nuclear=0.0786 sym=0.0018 stable_rank=1.12
L1H1 <- L0H2 normalized_trace=0.0250 trace/nuclear=0.0466 sym=0.0055 stable_rank=1.26
L1H1 <- L0H5 normalized_trace=0.0212 trace/nuclear=0.0335 sym=0.0036 stable_rank=1.12
L1H2 <- L0H5 normalized_trace=0.0467 trace/nuclear=0.0723 sym=0.0017 stable_rank=1.09
L1H2 <- L0H2 normalized_trace=0.0418 trace/nuclear=0.0774 sym=0.0058 stable_rank=1.21
L1H3 <- L0H2 normalized_trace=0.0453 trace/nuclear=0.0825 sym=0.0079 stable_rank=1.22
L1H3 <- L0H5 normalized_trace=0.0429 trace/nuclear=0.0660 sym=0.0021 stable_rank=1.10
L1H4 <- L0H5 normalized_trace=0.0368 trace/nuclear=0.0595 sym=-0.0008 stable_rank=1.07
L1H4 <- L0H2 normalized_trace=0.0297 trace/nuclear=0.0581 sym=-0.0002 stable_rank=1.19
L1H5 <- L0H2 normalized_trace=0.0396 trace/nuclear=0.0737 sym=0.0067 stable_rank=1.25
L1H5 <- L0H5 normalized_trace=0.0307 trace/nuclear=0.0492 sym=0.0029 stable_rank=1.12
L2H0 <- L0H5 normalized_trace=0.0667 trace/nuclear=0.1399 sym=0.0160 stable_rank=1.18
L2H0 <- L0H2 normalized_trace=0.0584 trace/nuclear=0.1334 sym=0.0266 stable_rank=1.29
L2H1 <- L0H5 normalized_trace=0.0425 trace/nuclear=0.1391 sym=0.0135 stable_rank=1.21
L2H1 <- L0H2 normalized_trace=0.0334 trace/nuclear=0.1140 sym=0.0226 stable_rank=1.33
L2H2 <- L0H5 normalized_trace=0.0461 trace/nuclear=0.1325 sym=0.0172 stable_rank=1.19
L2H2 <- L0H2 normalized_trace=0.0391 trace/nuclear=0.1192 sym=0.0272 stable_rank=1.31
L2H3 <- L0H5 normalized_trace=0.0247 trace/nuclear=0.0837 sym=0.0066 stable_rank=1.08
L2H3 <- L0H2 normalized_trace=0.0226 trace/nuclear=0.0842 sym=0.0121 stable_rank=1.20
L2H4 <- L0H5 normalized_trace=0.0512 trace/nuclear=0.1308 sym=0.0153 stable_rank=1.19
L2H4 <- L0H2 normalized_trace=0.0451 trace/nuclear=0.1241 sym=0.0251 stable_rank=1.28
L2H5 <- L0H5 normalized_trace=0.0475 trace/nuclear=0.1496 sym=0.0181 stable_rank=1.20
L2H5 <- L0H2 normalized_trace=0.0362 trace/nuclear=0.1207 sym=0.0205 stable_rank=1.33
L3H0 <- L2H4 normalized_trace=-0.0365 trace/nuclear=-0.1141 sym=0.0235 stable_rank=10.85
L3H0 <- L2H5 normalized_trace=-0.0197 trace/nuclear=-0.0657 sym=0.0111 stable_rank=11.55
L3H1 <- L0H5 normalized_trace=0.0139 trace/nuclear=0.0573 sym=0.0050 stable_rank=1.06
L3H1 <- L0H2 normalized_trace=0.0134 trace/nuclear=0.0592 sym=0.0126 stable_rank=1.15
L3H2 <- L0H2 normalized_trace=0.0119 trace/nuclear=0.0501 sym=0.0154 stable_rank=1.25
L3H2 <- L2H2 normalized_trace=-0.0108 trace/nuclear=-0.0373 sym=0.0186 stable_rank=6.45
L3H3 <- L0H3 normalized_trace=0.0626 trace/nuclear=0.1898 sym=0.0667 stable_rank=6.18
L3H3 <- L2H5 normalized_trace=0.0482 trace/nuclear=0.1294 sym=0.0126 stable_rank=11.01
L3H4 <- L2H2 normalized_trace=0.0143 trace/nuclear=0.0518 sym=0.0020 stable_rank=4.44
L3H4 <- L2H4 normalized_trace=0.0122 trace/nuclear=0.0480 sym=0.0062 stable_rank=5.61
L3H5 <- L2H1 normalized_trace=-0.0161 trace/nuclear=-0.0538 sym=0.0077 stable_rank=9.69
L3H5 <- L2H0 normalized_trace=-0.0147 trace/nuclear=-0.0489 sym=0.0193 stable_rank=3.94

L0H3: unadjusted QK eigenvalues
[ 0.724+0.691j  0.724-0.691j  0.446+0.842j  0.446-0.842j  0.572+0.722j  0.572-0.722j  0.642+0.614j  0.642-0.614j  0.676+0.463j  0.676-0.463j  0.109+0.762j  0.109-0.762j  0.463+0.604j  0.463-0.604j
  0.621+0.j     0.487+0.376j  0.487-0.376j -0.001+0.607j -0.001-0.607j  0.35 +0.283j  0.35 -0.283j -0.255+0.354j -0.255-0.354j  0.04 +0.336j  0.04 -0.336j -0.288+0.j    -0.134+0.253j -0.134-0.253j
 -0.236+0.j     0.157+0.j    -0.151-0.035j -0.151+0.035j -0.123+0.029j -0.123-0.029j  0.113+0.j    -0.077+0.j     0.   +0.058j  0.   -0.058j -0.029+0.049j -0.029-0.049j  0.053+0.005j  0.053-0.005j
 -0.043+0.03j  -0.043-0.03j  -0.051+0.007j -0.051-0.007j -0.032+0.j    -0.03 +0.j    -0.002-0.028j -0.002+0.028j -0.025+0.012j -0.025-0.012j -0.012+0.024j -0.012-0.024j -0.021+0.j    -0.018+0.005j
 -0.018-0.005j  0.018+0.j    -0.012+0.j     0.   +0.007j  0.   -0.007j -0.006+0.004j -0.006-0.004j  0.002+0.j   ]

```

## Call 5

```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=3, suppress=True, linewidth=200)

def eff(a,g):
    a=np.asarray(a,dtype=float)*g[None,:]
    return a-a.mean(axis=1,keepdims=True)

with threadpool_limits(limits=4):
    F0=eff(W["W_fc.0"],W["ln2_w.0"])
    Ulast,slast,_=svd(W["W_proj.3"].astype(float),full_matrices=False)
    print("Output-projection leading 27-dimensional subspace versus first-layer read subspaces")
    for name,a in [("V0",eff(W["W_v.0"],W["ln1_w.0"])),
                   ("FC0",F0),
                   ("Q0H3",eff(W["W_q.0"][192:256],W["ln1_w.0"])),
                   ("K0H3",eff(W["W_k.0"][192:256],W["ln1_w.0"]))]:
        _,s,vh=svd(a,full_matrices=False)
        c=svdvals(vh[:27]@Ulast[:,:27])
        print(name, "principal cosines",c)
    print("\nMLP weight-product spectra (linear-algebra objects only)")
    for l in range(4):
        F=eff(W[f"W_fc.{l}"],W[f"ln2_w.{l}"])
        P=W[f"W_proj.{l}"].astype(float)
        C=P@F
        e=np.linalg.eigvals(C)
        print("L",l,"trace",np.trace(C),"positive eigenvalue mass",e.real.sum()/np.abs(e).sum(),
              "top abs eigen",e[np.argsort(-np.abs(e))[:12]])
        ni=np.linalg.norm(F,axis=1); no=np.linalg.norm(P,axis=0)
        cs=np.sum(F*P.T,axis=1)/(ni*no)
        print("neuron input/output cosines quantiles",np.quantile(cs,[0,.1,.25,.5,.75,.9,1]))

    print("\nL0 head Q/K/V input overlap with L0H3 Q/K positional-candidate subspace")
    q3=eff(W["W_q.0"][192:256],W["ln1_w.0"])
    k3=eff(W["W_k.0"][192:256],W["ln1_w.0"])
    _,s,vh=svd(np.vstack([q3,k3]),full_matrices=False)
    print("joint singular values",s[:50])
    Ppos=vh[:28].T
    for j in range(6):
        sl=slice(j*64,(j+1)*64)
        aa=[eff(W[f"W_{n}.0"][sl],W["ln1_w.0"]) for n in ["q","k","v"]]
        print("L0H"+str(j),"fractions Q K V", [round(np.linalg.norm(a@Ppos)**2/np.linalg.norm(a)**2,3) for a in aa])

    print("\nAll head OV action within last-projection leading subspace")
    Z=Ulast[:,:27]
    for l in range(4):
        for j in range(6):
            sl=slice(j*64,(j+1)*64)
            B=W[f"W_o.{l}"][:,sl]@eff(W[f"W_v.{l}"][sl],W[f"ln1_w.{l}"])
            C=Z.T@B@Z
            print(f"L{l}H{j}", "energy fraction",round(np.linalg.norm(C)**2/np.linalg.norm(B)**2,3),
                  "trace",round(np.trace(C),3),"trace/nuclear",round(np.trace(C)/svdvals(C).sum(),3))

```

Output:

```
Output-projection leading 27-dimensional subspace versus first-layer read subspaces
V0 principal cosines [0.592 0.509 0.46  0.448 0.411 0.397 0.361 0.349 0.332 0.297 0.291 0.257 0.248 0.209 0.185 0.182 0.171 0.149 0.13  0.118 0.096 0.084 0.072 0.067 0.044 0.038 0.021]
FC0 principal cosines [0.537 0.483 0.425 0.416 0.392 0.367 0.34  0.32  0.315 0.283 0.268 0.259 0.231 0.228 0.216 0.18  0.169 0.162 0.137 0.102 0.094 0.076 0.065 0.047 0.037 0.022 0.011]
Q0H3 principal cosines [0.521 0.476 0.464 0.409 0.38  0.359 0.351 0.303 0.299 0.269 0.24  0.227 0.195 0.188 0.179 0.17  0.154 0.139 0.131 0.115 0.103 0.095 0.073 0.052 0.046 0.008 0.003]
K0H3 principal cosines [0.581 0.496 0.456 0.429 0.384 0.336 0.319 0.308 0.287 0.255 0.251 0.226 0.197 0.194 0.172 0.168 0.146 0.134 0.125 0.098 0.094 0.073 0.056 0.049 0.033 0.019 0.004]

MLP weight-product spectra (linear-algebra objects only)
L 0 trace 13.46248998122079 positive eigenvalue mass 0.022512999845474554 top abs eigen [10.829+7.849j 10.829-7.849j 10.083+5.875j 10.083-5.875j  9.176+0.j     8.95 +1.285j  8.95 -1.285j -7.675+0.j     7.032-2.223j  7.032+2.223j  6.504-3.415j  6.504+3.415j]
neuron input/output cosines quantiles [-0.619 -0.118 -0.071 -0.022  0.029  0.087  0.599]
L 1 trace -393.86014691384224 positive eigenvalue mass -0.25394660887236964 top abs eigen [-18.041+0.j    -17.279+0.j    -13.423+0.j    -13.211+1.376j -13.211-1.376j -13.043+2.077j -13.043-2.077j -10.947-3.693j -10.947+3.693j  11.512+0.j    -11.193+1.222j -11.193-1.222j]
neuron input/output cosines quantiles [-0.535 -0.143 -0.105 -0.06  -0.015  0.028  0.446]
L 2 trace -713.3440344474486 positive eigenvalue mass -0.3087179127594403 top abs eigen [-22.397+0.j    -21.02 +0.j    -17.455+6.997j -17.455-6.997j -17.276+0.j    -16.263+4.104j -16.263-4.104j -16.257+0.j    -15.65 -3.18j  -15.65 +3.18j  -14.451+1.845j -14.451-1.845j]
neuron input/output cosines quantiles [-0.439 -0.175 -0.139 -0.093 -0.043  0.     0.445]
L 3 trace -701.4390398108023 positive eigenvalue mass -0.5586219713220876 top abs eigen [ 21.75 +13.517j  21.75 -13.517j -25.549 +0.j    -21.473 +0.j    -17.728 +1.397j -17.728 -1.397j -15.701 +0.j     -9.968 -9.339j  -9.968 +9.339j -13.09  +3.679j -13.09  -3.679j  -1.598+13.291j]
neuron input/output cosines quantiles [-0.389 -0.209 -0.155 -0.096 -0.033  0.04   0.262]

L0 head Q/K/V input overlap with L0H3 Q/K positional-candidate subspace
joint singular values [2.536 2.112 1.958 1.82  1.753 1.657 1.626 1.56  1.541 1.517 1.435 1.424 1.379 1.369 1.337 1.304 1.247 1.245 1.214 1.162 1.126 1.065 1.028 1.002 0.953 0.877 0.847 0.825 0.789 0.762 0.723 0.654 0.624
 0.61  0.591 0.564 0.563 0.528 0.505 0.492 0.443 0.429 0.424 0.416 0.38  0.375 0.357 0.35  0.342 0.337]
L0H0 fractions Q K V [np.float64(0.469), np.float64(0.625), np.float64(0.088)]
L0H1 fractions Q K V [np.float64(0.342), np.float64(0.185), np.float64(0.509)]
L0H2 fractions Q K V [np.float64(0.382), np.float64(0.221), np.float64(0.527)]
L0H3 fractions Q K V [np.float64(0.819), np.float64(0.912), np.float64(0.181)]
L0H4 fractions Q K V [np.float64(0.269), np.float64(0.236), np.float64(0.396)]
L0H5 fractions Q K V [np.float64(0.397), np.float64(0.141), np.float64(0.733)]

All head OV action within last-projection leading subspace
L0H0 energy fraction 0.006 trace 0.023 trace/nuclear 0.043
L0H1 energy fraction 0.027 trace 0.291 trace/nuclear 0.284
L0H2 energy fraction 0.03 trace 0.251 trace/nuclear 0.304
L0H3 energy fraction 0.007 trace -0.06 trace/nuclear -0.057
L0H4 energy fraction 0.031 trace 0.106 trace/nuclear 0.21
L0H5 energy fraction 0.019 trace 0.252 trace/nuclear 0.06
L1H0 energy fraction 0.005 trace -0.571 trace/nuclear -0.186
L1H1 energy fraction 0.006 trace -0.679 trace/nuclear -0.209
L1H2 energy fraction 0.006 trace -1.152 trace/nuclear -0.335
L1H3 energy fraction 0.006 trace -0.978 trace/nuclear -0.256
L1H4 energy fraction 0.005 trace -0.962 trace/nuclear -0.269
L1H5 energy fraction 0.006 trace -0.88 trace/nuclear -0.269
L2H0 energy fraction 0.006 trace -1.029 trace/nuclear -0.213
L2H1 energy fraction 0.007 trace -3.108 trace/nuclear -0.532
L2H2 energy fraction 0.005 trace -2.154 trace/nuclear -0.394
L2H3 energy fraction 0.007 trace -2.563 trace/nuclear -0.416
L2H4 energy fraction 0.007 trace -2.474 trace/nuclear -0.508
L2H5 energy fraction 0.006 trace -2.398 trace/nuclear -0.453
L3H0 energy fraction 0.018 trace -6.706 trace/nuclear -0.641
L3H1 energy fraction 0.013 trace -5.35 trace/nuclear -0.483
L3H2 energy fraction 0.013 trace -3.932 trace/nuclear -0.406
L3H3 energy fraction 0.031 trace -12.509 trace/nuclear -0.447
L3H4 energy fraction 0.012 trace -3.061 trace/nuclear -0.184
L3H5 energy fraction 0.014 trace -3.522 trace/nuclear -0.349

```

## Call 6

```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=3, suppress=True, linewidth=200)
def eff(a,g):
    a=np.asarray(a,dtype=float)*g[None,:]
    return a-a.mean(axis=1,keepdims=True)

with threadpool_limits(limits=4):
    Vs=[]; Vhs=[]
    for j in range(6):
        V=eff(W["W_v.0"][64*j:64*(j+1)],W["ln1_w.0"])
        _,s,vh=svd(V,full_matrices=False)
        Vs.append(V); Vhs.append(vh)
        print(f"L0H{j} V singular values",s[:45])
    for j in [0,1,2,4,5]:
        print(f"V0H3 vs V0H{j}: rank-26 input principal cosines",
              svdvals(Vhs[3][:26]@Vhs[j][:26].T))
    print("\nPairwise full V input-Gram cosines")
    G=[v.T@v for v in Vs]
    print(np.array([[np.sum(a*b)/np.linalg.norm(a)/np.linalg.norm(b) for b in G] for a in G]))
    T=Vhs[3][:26].T
    print("\nQK fractions in V0H3 top-26 input span")
    for j in range(6):
        sl=slice(64*j,64*(j+1))
        q,k=[eff(W[f"W_{n}.0"][sl],W["ln1_w.0"]) for n in ["q","k"]]
        A=q.T@k/8
        print(f"L0H{j}", "Qfrac",np.linalg.norm(q@T)**2/np.linalg.norm(q)**2,
              "Kfrac",np.linalg.norm(k@T)**2/np.linalg.norm(k)**2,
              "QKfrac",np.linalg.norm(T.T@A@T)**2/np.linalg.norm(A)**2)
    F=eff(W["W_fc.0"],W["ln2_w.0"])
    X=F@T
    print("\nFC0 reading of V0H3 input span: singular",svdvals(X))
    for j in [0,1,2,3,4,5]:
        sl=slice(64*j,64*(j+1))
        B=W["W_o.0"][:,sl]@Vs[j]
        Y=F@B@T
        print("FC0 OV0H"+str(j),"singular",svdvals(Y)[:30])
        nx=np.linalg.norm(X,axis=1); ny=np.linalg.norm(Y,axis=1)
        print(" row-energy correlation",np.corrcoef(nx,ny)[0,1])

```

Output:

```
L0H0 V singular values [0.535 0.506 0.445 0.384 0.371 0.341 0.337 0.327 0.312 0.299 0.288 0.278 0.269 0.259 0.239 0.223 0.211 0.202 0.195 0.19  0.185 0.183 0.175 0.168 0.167 0.164 0.162 0.16  0.157 0.15  0.145 0.143 0.139
 0.138 0.129 0.127 0.126 0.125 0.122 0.12  0.118 0.114 0.113 0.109 0.107]
L0H1 V singular values [1.167 0.925 0.356 0.299 0.264 0.242 0.221 0.216 0.214 0.208 0.202 0.198 0.189 0.178 0.175 0.173 0.169 0.164 0.162 0.155 0.154 0.151 0.148 0.146 0.141 0.14  0.138 0.135 0.132 0.131 0.128 0.126 0.122
 0.121 0.117 0.115 0.115 0.109 0.108 0.107 0.105 0.103 0.101 0.099 0.096]
L0H2 V singular values [1.014 0.816 0.383 0.289 0.248 0.228 0.225 0.216 0.205 0.199 0.189 0.182 0.18  0.177 0.172 0.168 0.162 0.159 0.158 0.152 0.15  0.147 0.144 0.142 0.138 0.135 0.131 0.13  0.129 0.124 0.123 0.121 0.121
 0.118 0.116 0.114 0.11  0.109 0.107 0.104 0.102 0.1   0.098 0.096 0.095]
L0H3 V singular values [0.57  0.513 0.504 0.472 0.47  0.45  0.44  0.434 0.425 0.411 0.39  0.383 0.379 0.367 0.359 0.342 0.339 0.336 0.324 0.297 0.286 0.276 0.273 0.266 0.254 0.248 0.245 0.241 0.237 0.231 0.225 0.223 0.22
 0.21  0.206 0.204 0.197 0.194 0.183 0.18  0.173 0.169 0.16  0.151 0.15 ]
L0H4 V singular values [0.699 0.636 0.274 0.249 0.229 0.222 0.208 0.203 0.19  0.185 0.179 0.174 0.174 0.169 0.167 0.164 0.161 0.159 0.155 0.152 0.148 0.143 0.141 0.138 0.135 0.134 0.128 0.127 0.125 0.122 0.119 0.119 0.115
 0.112 0.11  0.108 0.107 0.106 0.103 0.102 0.1   0.098 0.095 0.094 0.093]
L0H5 V singular values [2.988 1.914 1.213 0.638 0.545 0.506 0.483 0.439 0.378 0.346 0.315 0.31  0.275 0.263 0.245 0.24  0.232 0.227 0.214 0.21  0.201 0.193 0.188 0.184 0.18  0.176 0.172 0.168 0.164 0.161 0.157 0.153 0.148
 0.143 0.142 0.138 0.137 0.129 0.128 0.125 0.122 0.119 0.118 0.113 0.111]
V0H3 vs V0H0: rank-26 input principal cosines [0.923 0.917 0.893 0.861 0.853 0.841 0.831 0.827 0.79  0.78  0.733 0.721 0.669 0.647 0.549 0.446 0.39  0.368 0.311 0.285 0.223 0.141 0.128 0.06  0.039 0.009]
V0H3 vs V0H1: rank-26 input principal cosines [0.784 0.765 0.754 0.675 0.659 0.601 0.545 0.522 0.514 0.426 0.397 0.357 0.313 0.29  0.271 0.247 0.24  0.185 0.164 0.149 0.113 0.098 0.079 0.06  0.05  0.032]
V0H3 vs V0H2: rank-26 input principal cosines [0.765 0.743 0.641 0.58  0.576 0.529 0.494 0.442 0.411 0.383 0.355 0.33  0.28  0.246 0.22  0.199 0.157 0.136 0.125 0.111 0.103 0.08  0.075 0.038 0.014 0.003]
V0H3 vs V0H4: rank-26 input principal cosines [0.683 0.676 0.602 0.583 0.498 0.482 0.415 0.401 0.379 0.351 0.33  0.299 0.267 0.247 0.22  0.195 0.182 0.17  0.15  0.133 0.118 0.103 0.072 0.055 0.042 0.004]
V0H3 vs V0H5: rank-26 input principal cosines [0.923 0.901 0.881 0.846 0.829 0.809 0.784 0.756 0.72  0.714 0.706 0.675 0.615 0.588 0.551 0.519 0.49  0.377 0.307 0.296 0.218 0.18  0.134 0.093 0.045 0.008]

Pairwise full V input-Gram cosines
[[1.    0.045 0.049 0.425 0.065 0.03 ]
 [0.045 1.    0.921 0.048 0.81  0.758]
 [0.049 0.921 1.    0.049 0.808 0.787]
 [0.425 0.048 0.049 1.    0.064 0.036]
 [0.065 0.81  0.808 0.064 1.    0.607]
 [0.03  0.758 0.787 0.036 0.607 1.   ]]

QK fractions in V0H3 top-26 input span
L0H0 Qfrac 0.155558807265336 Kfrac 0.17061753824062098 QKfrac 0.023372110380353608
L0H1 Qfrac 0.0876291268403447 Kfrac 0.11133603923773733 QKfrac 0.007932494259018515
L0H2 Qfrac 0.08764690330609459 Kfrac 0.12731814221463547 QKfrac 0.01579242205477233
L0H3 Qfrac 0.16201428952781538 Kfrac 0.15895400382983393 QKfrac 0.03245600157451362
L0H4 Qfrac 0.11051175189395943 Kfrac 0.1560671496343414 QKfrac 0.03737232519181709
L0H5 Qfrac 0.05747116455099912 Kfrac 0.1534574952811706 QKfrac 0.027682619655225348

FC0 reading of V0H3 input span: singular [11.422  8.626  7.875  7.586  7.341  7.082  6.713  6.627  6.457  5.994  5.706  5.273  5.126  4.981  4.532  4.484  4.285  4.181  4.053  3.653  2.84   2.242  1.799  1.598  1.354  1.108]
FC0 OV0H0 singular [2.261 2.094 1.501 1.025 0.871 0.766 0.736 0.624 0.53  0.492 0.414 0.381 0.306 0.251 0.191 0.157 0.121 0.105 0.086 0.067 0.05  0.04  0.039 0.022 0.02  0.011]
 row-energy correlation 0.5158112164631999
FC0 OV0H1 singular [3.351 0.777 0.464 0.205 0.19  0.138 0.129 0.112 0.086 0.074 0.061 0.057 0.048 0.043 0.04  0.036 0.03  0.025 0.023 0.022 0.021 0.018 0.014 0.012 0.01  0.009]
 row-energy correlation 0.600603487837425
FC0 OV0H2 singular [1.991 0.45  0.297 0.202 0.138 0.094 0.089 0.072 0.065 0.06  0.051 0.049 0.043 0.038 0.031 0.027 0.025 0.023 0.021 0.018 0.017 0.015 0.014 0.011 0.01  0.007]
 row-energy correlation 0.5995849654662337
FC0 OV0H3 singular [9.581 6.436 6.139 5.84  5.555 5.333 5.241 5.044 4.685 4.611 4.246 3.989 3.705 3.631 3.473 2.675 2.389 1.701 1.277 0.896 0.859 0.476 0.374 0.333 0.295 0.253]
 row-energy correlation 0.8466326550909771
FC0 OV0H4 singular [1.579 0.336 0.142 0.112 0.086 0.077 0.07  0.063 0.053 0.046 0.044 0.041 0.03  0.024 0.023 0.022 0.02  0.017 0.016 0.014 0.014 0.012 0.011 0.009 0.009 0.007]
 row-energy correlation 0.6070299058234749
FC0 OV0H5 singular [5.873 3.51  3.024 1.99  1.674 1.039 0.923 0.711 0.537 0.472 0.391 0.344 0.322 0.266 0.264 0.21  0.18  0.163 0.145 0.101 0.084 0.07  0.053 0.048 0.044 0.033]
 row-energy correlation 0.4897697253395155

```

## Call 7

```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=3, suppress=True, linewidth=200)
def eff(a,g):
    a=np.asarray(a,dtype=float)*g[None,:]
    return a-a.mean(axis=1,keepdims=True)

def stats(C):
    ss=svdvals(C)
    return [np.trace(C)/ss.sum(),np.sum(C*C.T)/np.sum(C*C),
            np.linalg.norm((C-C.T)/2),np.linalg.norm((C+C.T)/2)]

with threadpool_limits(limits=4):
    F=eff(W["W_fc.0"],W["ln2_w.0"])
    V0=eff(W["W_v.0"][:64],W["ln1_w.0"])
    V3=eff(W["W_v.0"][192:256],W["ln1_w.0"])
    _,_,vh=svd(np.vstack([V0,V3]),full_matrices=False)
    B=W["W_o.0"][:,192:256]@V3
    print("Current/transported weight-coupling: trace/nuclear, transpose cosine, skew norm, symmetric norm")
    for r in [12,18,22,26,32,48]:
        T=vh[:r].T
        X=F@T
        Y=F@B@T
        C=X.T@Y
        print("rank",r,"unweighted",np.round(stats(C),4))
        wt=np.linalg.norm(W["W_proj.0"],axis=0)
        print("rank",r,"output-norm weighted",np.round(stats(X.T@(Y*wt[:,None]**2)),4))
        print(" leading coupling singular",svdvals(C)[:8])
    print("\nSame calculation using each early head, rank22")
    T=vh[:22].T
    X=F@T
    for j in range(6):
        sl=slice(64*j,64*(j+1))
        Bj=W["W_o.0"][:,sl]@eff(W["W_v.0"][sl],W["ln1_w.0"])
        Y=F@Bj@T
        print(f"L0H{j}",np.round(stats(X.T@Y),4))

    print("\nFirst-layer neuron row norm quantiles (for evidence of specialized groups)")
    X=F@T;Y=F@B@T
    print("current",np.quantile(np.linalg.norm(X,axis=1),[0,.1,.25,.5,.75,.9,.99,1]))
    print("transported",np.quantile(np.linalg.norm(Y,axis=1),[0,.1,.25,.5,.75,.9,.99,1]))
    print("current vs transported row cosine quantiles",
          np.quantile(np.sum(X*Y,axis=1)/(np.linalg.norm(X,axis=1)*np.linalg.norm(Y,axis=1)),[0,.1,.25,.5,.75,.9,.99,1]))

```

Output:

```
Current/transported weight-coupling: trace/nuclear, transpose cosine, skew norm, symmetric norm
rank 12 unweighted [ 0.504  0.647 17.246 37.247]
rank 12 output-norm weighted [  0.603   0.709  90.503 219.093]
 leading coupling singular [34.574 14.662  9.998  7.944  6.521  6.039  3.99   3.31 ]
rank 18 unweighted [ 0.365  0.437 32.965 52.656]
rank 18 output-norm weighted [  0.448   0.487 178.712 304.456]
 leading coupling singular [52.585 19.902 15.208 10.309  9.877  8.238  7.079  6.019]
rank 22 unweighted [ 0.287  0.328 39.217 55.109]
rank 22 output-norm weighted [  0.371   0.369 214.951 316.473]
 leading coupling singular [54.731 21.443 17.688 12.935 12.164 12.09   8.705  8.431]
rank 26 unweighted [ 0.269  0.299 44.371 60.405]
rank 26 output-norm weighted [  0.348   0.338 243.066 345.409]
 leading coupling singular [60.556 22.41  18.942 15.112 14.513 12.936 10.513  9.269]
rank 32 unweighted [ 0.247  0.276 47.292 62.759]
rank 32 output-norm weighted [  0.325   0.317 257.137 357.126]
 leading coupling singular [62.599 23.549 19.795 17.043 14.92  14.166 11.772  9.931]
rank 48 unweighted [ 0.207  0.24  53.565 68.435]
rank 48 output-norm weighted [  0.28    0.283 286.904 383.97 ]
 leading coupling singular [66.176 26.657 22.46  20.621 18.123 15.955 15.213 13.023]

Same calculation using each early head, rank22
L0H0 [0.256 0.18  6.92  8.301]
L0H1 [ 0.318  0.225 13.575 17.063]
L0H2 [ 0.393  0.306  8.194 11.238]
L0H3 [ 0.287  0.328 39.217 55.109]
L0H4 [0.507 0.39  6.392 9.643]
L0H5 [ 0.619  0.492 17.159 29.412]

First-layer neuron row norm quantiles (for evidence of specialized groups)
current [0.061 0.096 0.117 0.178 0.842 1.301 1.869 2.632]
transported [0.032 0.049 0.06  0.089 0.629 0.99  1.431 2.295]
current vs transported row cosine quantiles [-0.886 -0.412 -0.233 -0.034  0.184  0.408  0.71   0.886]

```

## Call 8

```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=3, suppress=True, linewidth=200)
def eff(a,g):
    a=np.asarray(a,dtype=float)*g[None,:]
    return a-a.mean(axis=1,keepdims=True)

def cos(a,b):
    return np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))

with threadpool_limits(limits=4):
    AA=[];BB=[];CC=[]
    for l in range(4):
        for j in range(6):
            sl=slice(64*j,64*(j+1))
            q0=W[f"W_q.{l}"][sl]
            q,k,v=[eff(W[f"W_{n}.{l}"][sl],W[f"ln1_w.{l}"]) for n in ["q","k","v"]]
            AA.append(q.T@k/8)
            BB.append(W[f"W_o.{l}"][:,sl]@v)
            CC.append(k.T@(q0@W[f"ln1_b.{l}"])/8)
    Aref=AA[3]
    q3=eff(W["W_q.0"][192:256],W["ln1_w.0"])
    k3=eff(W["W_k.0"][192:256],W["ln1_w.0"])
    _,_,vh=svd(np.vstack([q3,k3]),full_matrices=False)
    Z=vh[:28].T
    u5,s5,v5=svd(BB[5],full_matrices=False)
    R=u5[:,:3]
    print("Head: original-pos-span QK energy, cosine with L0H3 QK, key-vector energy in L0H5 output top3,")
    print("      QK key-side energy in L0H5 output top3, QK query-side energy there")
    for i,(A,c) in enumerate(zip(AA,CC)):
        print(f"L{i//6}H{i%6}",
              np.round([np.linalg.norm(Z.T@A@Z)**2/np.linalg.norm(A)**2,
                        cos(A,Aref),
                        np.linalg.norm(R.T@c)**2/np.linalg.norm(c)**2,
                        np.linalg.norm(A@R)**2/np.linalg.norm(A)**2,
                        np.linalg.norm(R.T@A)**2/np.linalg.norm(A)**2],4))
    print("\nEarly-head leading QK left/right and V input singular directions: correlations")
    dirs=[]; names=[]
    for j in [0,1,2,3,4,5]:
        u,s,vh=svd(AA[j],full_matrices=False)
        dirs.extend([u[:,0],vh[0]])
        names.extend([f"QK{j}-left",f"QK{j}-right"])
    _,_,vh=svd(eff(W["W_v.0"][320:384],W["ln1_w.0"]),full_matrices=False)
    dirs.extend(list(vh[:3]));names.extend(["V5-0","V5-1","V5-2"])
    D=np.array(dirs)
    print(names)
    print(D@D.T)

    print("\nL3H3 key composition with L0H3 restricted to shared early value-input spans")
    _,_,vh=svd(np.vstack([eff(W["W_v.0"][:64],W["ln1_w.0"]),
                         eff(W["W_v.0"][192:256],W["ln1_w.0"])]),full_matrices=False)
    for r in [12,18,22,26,32]:
        T=vh[:r].T
        C=T.T@AA[21]@BB[3]@T
        s=svdvals(C)
        print("r",r,"trace/nuclear",np.trace(C)/s.sum(),"sym",cos(C,C.T),
              "stable rank",(s@s)/(s[0]**2),"eigenvalues",np.linalg.eigvals(C))

```

Output:

```
Head: original-pos-span QK energy, cosine with L0H3 QK, key-vector energy in L0H5 output top3,
      QK key-side energy in L0H5 output top3, QK query-side energy there
L0H0 [0.416 0.137 0.032 0.055 0.097]
L0H1 [ 0.077 -0.026  0.025  0.029  0.07 ]
L0H2 [ 0.184 -0.063  0.012  0.022  0.024]
L0H3 [0.899 1.    0.076 0.047 0.036]
L0H4 [ 0.091 -0.044  0.015  0.014  0.069]
L0H5 [0.057 0.01  0.008 0.016 0.153]
L1H0 [0.011 0.016 0.853 0.416 0.046]
L1H1 [0.012 0.014 0.853 0.451 0.059]
L1H2 [0.011 0.009 0.851 0.475 0.037]
L1H3 [0.015 0.016 0.854 0.482 0.058]
L1H4 [0.008 0.004 0.798 0.436 0.028]
L1H5 [0.014 0.017 0.861 0.441 0.065]
L2H0 [0.013 0.014 0.832 0.24  0.074]
L2H1 [0.008 0.014 0.824 0.094 0.034]
L2H2 [0.01  0.007 0.787 0.124 0.036]
L2H3 [0.005 0.007 0.615 0.089 0.015]
L2H4 [0.01  0.01  0.83  0.16  0.047]
L2H5 [0.009 0.01  0.856 0.102 0.034]
L3H0 [0.005 0.01  0.692 0.042 0.01 ]
L3H1 [0.006 0.007 0.475 0.06  0.013]
L3H2 [0.006 0.005 0.625 0.064 0.018]
L3H3 [0.006 0.017 0.009 0.002 0.002]
L3H4 [0.006 0.008 0.043 0.013 0.009]
L3H5 [0.006 0.007 0.721 0.062 0.011]

Early-head leading QK left/right and V input singular directions: correlations
['QK0-left', 'QK0-right', 'QK1-left', 'QK1-right', 'QK2-left', 'QK2-right', 'QK3-left', 'QK3-right', 'QK4-left', 'QK4-right', 'QK5-left', 'QK5-right', 'V5-0', 'V5-1', 'V5-2']
[[ 1.     0.488 -0.139  0.409  0.075  0.28   0.63   0.109 -0.089  0.324  0.912 -0.238  0.568  0.481 -0.484]
 [ 0.488  1.     0.531  0.053  0.754 -0.256  0.702  0.538  0.506  0.303  0.649  0.043  0.898  0.152  0.105]
 [-0.139  0.531  1.    -0.406  0.887 -0.496  0.138  0.235  0.914 -0.116 -0.068  0.061  0.418  0.013  0.769]
 [ 0.409  0.053 -0.406  1.    -0.324  0.806  0.153 -0.047 -0.327  0.687  0.344 -0.325  0.062  0.293 -0.35 ]
 [ 0.075  0.754  0.887 -0.324  1.    -0.539  0.448  0.511  0.87  -0.096  0.18   0.056  0.712 -0.096  0.547]
 [ 0.28  -0.256 -0.496  0.806 -0.539  1.    -0.052 -0.303 -0.426  0.597  0.17  -0.528 -0.232  0.353 -0.335]
 [ 0.63   0.702  0.138  0.153  0.448 -0.052  1.     0.659  0.182  0.162  0.664 -0.099  0.798 -0.038 -0.317]
 [ 0.109  0.538  0.235 -0.047  0.511 -0.303  0.659  1.     0.242 -0.061  0.177  0.102  0.664 -0.593 -0.105]
 [-0.089  0.506  0.914 -0.327  0.87  -0.426  0.182  0.242  1.    -0.167 -0.06  -0.014  0.439  0.012  0.704]
 [ 0.324  0.303 -0.116  0.687 -0.096  0.597  0.162 -0.061 -0.167  1.     0.387 -0.37   0.189  0.424 -0.116]
 [ 0.912  0.649 -0.068  0.344  0.18   0.17   0.664  0.177 -0.06   0.387  1.    -0.145  0.675  0.5   -0.443]
 [-0.238  0.043  0.061 -0.325  0.056 -0.528 -0.099  0.102 -0.014 -0.37  -0.145  1.    -0.031 -0.139  0.007]
 [ 0.568  0.898  0.418  0.062  0.712 -0.232  0.798  0.664  0.439  0.189  0.675 -0.031  1.     0.    -0.   ]
 [ 0.481  0.152  0.013  0.293 -0.096  0.353 -0.038 -0.593  0.012  0.424  0.5   -0.139  0.     1.    -0.   ]
 [-0.484  0.105  0.769 -0.35   0.547 -0.335 -0.317 -0.105  0.704 -0.116 -0.443  0.007 -0.    -0.     1.   ]]

L3H3 key composition with L0H3 restricted to shared early value-input spans
r 12 trace/nuclear 0.8129930204637937 sym 0.532013896078395 stable rank 2.854908362186774 eigenvalues [ 0.037+0.005j  0.037-0.005j  0.027+0.013j  0.027-0.013j  0.023+0.014j  0.023-0.014j -0.002+0.j     0.002+0.j     0.007+0.007j  0.007-0.007j  0.013+0.j     0.015+0.j   ]
r 18 trace/nuclear 0.7995062052108542 sym 0.5608685856349199 stable rank 3.858177272196351 eigenvalues [ 0.034+0.016j  0.034-0.016j  0.04 +0.001j  0.04 -0.001j  0.031+0.013j  0.031-0.013j  0.024+0.005j  0.024-0.005j  0.01 +0.006j  0.01 -0.006j  0.016+0.j     0.013+0.j     0.004+0.002j  0.004-0.002j
  0.003+0.j    -0.002+0.001j -0.002-0.001j -0.001+0.j   ]
r 22 trace/nuclear 0.7817483093578843 sym 0.5612276960964993 stable rank 4.503168118870607 eigenvalues [ 0.05 +0.j     0.04 +0.014j  0.04 -0.014j  0.035+0.015j  0.035-0.015j  0.036+0.j     0.024+0.011j  0.024-0.011j  0.024+0.001j  0.024-0.001j  0.023+0.j     0.012+0.01j   0.012-0.01j  -0.005+0.j
 -0.002+0.001j -0.002-0.001j  0.006+0.003j  0.006-0.003j  0.005+0.002j  0.005-0.002j  0.001+0.j     0.002+0.j   ]
r 26 trace/nuclear 0.7260602384951971 sym 0.5171513783514756 stable rank 4.8236117326070636 eigenvalues [ 0.05 +0.j     0.038+0.018j  0.038-0.018j  0.04 +0.009j  0.04 -0.009j  0.035+0.j     0.031+0.j     0.025+0.012j  0.025-0.012j  0.024+0.j     0.019+0.j     0.011+0.006j  0.011-0.006j  0.006+0.008j
  0.006-0.008j -0.005+0.j     0.002+0.005j  0.002-0.005j  0.007+0.j    -0.001+0.001j -0.001-0.001j -0.   +0.002j -0.   -0.002j  0.   +0.j     0.004+0.j     0.002+0.j   ]
r 32 trace/nuclear 0.6568732011192548 sym 0.4810444584224879 stable rank 5.082920190077806 eigenvalues [ 0.053+0.j     0.038+0.019j  0.038-0.019j  0.04 +0.01j   0.04 -0.01j   0.036+0.j     0.026+0.012j  0.026-0.012j  0.029+0.j     0.025+0.j     0.014+0.004j  0.014-0.004j  0.008+0.008j  0.008-0.008j
  0.009+0.005j  0.009-0.005j  0.006+0.j    -0.   +0.006j -0.   -0.006j -0.004+0.002j -0.004-0.002j -0.003+0.j    -0.002+0.003j -0.002-0.003j -0.001+0.002j -0.001-0.002j  0.002+0.002j  0.002-0.002j
  0.002+0.j     0.   +0.001j  0.   -0.001j -0.   +0.j   ]

```

## Call 9

```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=200)
def eff(a,g):
    a=np.asarray(a,dtype=float)*g[None,:]
    return a-a.mean(axis=1,keepdims=True)

with threadpool_limits(limits=4):
    print("Unadjusted early value-matrix singular values")
    for j in [0,1,2,3,4,5]:
        print(f"L0H{j}",svdvals(W["W_v.0"][64*j:64*(j+1)]))
    V0=eff(W["W_v.0"][:64],W["ln1_w.0"])
    V3=eff(W["W_v.0"][192:256],W["ln1_w.0"])
    _,_,vh=svd(np.vstack([V0,V3]),full_matrices=False)
    T=vh[:22].T
    B3=W["W_o.0"][:,192:256]@V3
    print("\nRestricted matching scores, trace/nuclear norm.")
    print("Columns: direct QK; earlier-OV on key; earlier-OV on query; earlier-OV on both.")
    for l in range(1,4):
        for j in range(6):
            sl=slice(64*j,64*(j+1))
            q,k=[eff(W[f"W_{n}.{l}"][sl],W[f"ln1_w.{l}"]) for n in ["q","k"]]
            A=q.T@k/8
            mats=[T.T@A@T, T.T@A@B3@T, T.T@B3.T@A@T, T.T@B3.T@A@B3@T]
            scores=[np.trace(m)/svdvals(m).sum() for m in mats]
            norms=[np.linalg.norm(m) for m in mats]
            print(f"L{l}H{j}", "scores",np.round(scores,4),"norms",np.round(norms,4))
    print("\nRaw L0H3 V input vs L0H0 V input canonical correlations at varying rank")
    _,_,v0=svd(W["W_v.0"][:64],full_matrices=False)
    _,_,v3=svd(W["W_v.0"][192:256],full_matrices=False)
    for r in [20,22,24,26,28,32]:
        print(r,svdvals(v0[:r]@v3[:r].T))

```

Output:

```
Unadjusted early value-matrix singular values
L0H0 [1.9407 1.8252 1.6812 1.4367 1.3428 1.3139 1.2203 1.1647 1.1506 1.1232 1.0781 1.0133 0.9951 0.9349 0.8817 0.814  0.7843 0.7004 0.6457 0.5851 0.5774 0.5681 0.553  0.543  0.5022 0.4958 0.4797 0.4766
 0.4632 0.4537 0.4478 0.4375 0.431  0.4171 0.414  0.4061 0.4006 0.3958 0.3893 0.3857 0.3792 0.3722 0.3646 0.3557 0.3499 0.3443 0.3402 0.3365 0.3337 0.3276 0.3266 0.3128 0.3095 0.305  0.2997 0.2954
 0.2863 0.2776 0.2701 0.2644 0.2588 0.2541 0.2514 0.2392]
L0H1 [1.8263 1.676  1.0622 0.9647 0.8154 0.6964 0.6599 0.6454 0.625  0.5954 0.5823 0.5646 0.5445 0.5297 0.5173 0.5067 0.4954 0.4913 0.4731 0.4634 0.4518 0.4498 0.4431 0.4354 0.4246 0.4227 0.4158 0.4067
 0.3998 0.3932 0.3881 0.3803 0.3782 0.3736 0.3657 0.3615 0.3516 0.35   0.3486 0.344  0.341  0.3346 0.3276 0.3217 0.3171 0.3136 0.3074 0.3023 0.2993 0.2944 0.2864 0.2849 0.2795 0.2757 0.2752 0.2648
 0.2642 0.2515 0.2497 0.2416 0.2391 0.2273 0.2253 0.219 ]
L0H2 [1.5692 1.3632 0.9447 0.8826 0.7778 0.6878 0.6719 0.64   0.6094 0.5968 0.5583 0.551  0.5479 0.5255 0.5155 0.4951 0.481  0.4715 0.4643 0.4508 0.4406 0.4382 0.432  0.4221 0.4121 0.4094 0.4044 0.4014
 0.3934 0.3871 0.3797 0.3731 0.3706 0.366  0.3599 0.3577 0.353  0.3486 0.346  0.3376 0.3314 0.3268 0.3203 0.3182 0.3127 0.3119 0.3084 0.3002 0.2999 0.2932 0.2885 0.2853 0.2765 0.2744 0.2697 0.2679
 0.2597 0.2555 0.2522 0.2427 0.2317 0.2277 0.2221 0.2182]
L0H3 [2.1024 1.9481 1.9032 1.8657 1.8432 1.8192 1.7694 1.7548 1.6743 1.6512 1.6288 1.6207 1.5888 1.5533 1.5198 1.4649 1.4172 1.3029 1.1603 1.1014 0.9447 0.9297 0.8483 0.8181 0.7903 0.7786 0.745  0.7387
 0.7303 0.7008 0.6882 0.6576 0.6424 0.6217 0.6175 0.6052 0.5753 0.5651 0.5434 0.5218 0.5149 0.4909 0.4806 0.466  0.4463 0.4299 0.4104 0.4    0.3775 0.3661 0.3594 0.3488 0.3453 0.3418 0.3363 0.3291
 0.3183 0.3133 0.3127 0.2983 0.2873 0.2789 0.2713 0.2621]
L0H4 [1.2518 1.1168 0.8176 0.7863 0.7372 0.6582 0.626  0.6118 0.6024 0.5655 0.5424 0.5367 0.5139 0.5076 0.4888 0.4804 0.4636 0.4557 0.4421 0.4374 0.4346 0.4268 0.4161 0.4109 0.405  0.3947 0.3929 0.387
 0.3828 0.3742 0.3695 0.366  0.3633 0.36   0.3528 0.3518 0.3449 0.3436 0.3369 0.3334 0.3279 0.3221 0.3167 0.3138 0.311  0.309  0.3032 0.2965 0.2915 0.287  0.2824 0.2753 0.2726 0.271  0.2611 0.2582
 0.255  0.2504 0.2436 0.2343 0.2324 0.228  0.2256 0.2164]
L0H5 [4.6674 3.0731 2.3513 2.1514 2.0107 1.8571 1.5673 1.3785 1.2482 1.2288 1.1114 1.018  1.0121 0.9648 0.8597 0.8189 0.7872 0.7774 0.7329 0.6697 0.6427 0.6313 0.608  0.5686 0.5361 0.5336 0.5193 0.5071
 0.493  0.4851 0.4707 0.4543 0.4397 0.431  0.4285 0.4155 0.414  0.4026 0.3987 0.3937 0.386  0.3769 0.3709 0.3632 0.359  0.3493 0.3459 0.3333 0.3272 0.3268 0.3227 0.3148 0.3088 0.3034 0.2998 0.2924
 0.2833 0.2811 0.2762 0.2746 0.2692 0.265  0.2639 0.2559]

Restricted matching scores, trace/nuclear norm.
Columns: direct QK; earlier-OV on key; earlier-OV on query; earlier-OV on both.
L1H0 scores [-0.6617  0.0814 -0.0471 -0.0565] norms [0.2839 0.116  0.2112 0.1302]
L1H1 scores [-0.6186  0.1836  0.1708 -0.3229] norms [0.2837 0.1344 0.2361 0.1387]
L1H2 scores [-0.5289  0.127  -0.1549 -0.6371] norms [0.2378 0.1263 0.2211 0.1798]
L1H3 scores [-0.5506  0.0441  0.0365 -0.3096] norms [0.2262 0.1257 0.2084 0.138 ]
L1H4 scores [-0.5593  0.0338 -0.6174 -0.7755] norms [0.1795 0.0955 0.2082 0.1557]
L1H5 scores [-0.6835  0.1593  0.1406 -0.1127] norms [0.2931 0.1307 0.2357 0.122 ]
L2H0 scores [-0.2998  0.0462 -0.1596 -0.6008] norms [0.1151 0.0609 0.0787 0.0578]
L2H1 scores [-0.1263  0.0146  0.096  -0.3013] norms [0.1528 0.0991 0.1205 0.0839]
L2H2 scores [-0.2662  0.084  -0.1981 -0.5464] norms [0.1104 0.0661 0.0864 0.0632]
L2H3 scores [-0.1189  0.087   0.126  -0.3463] norms [0.0993 0.0676 0.0658 0.0508]
L2H4 scores [-0.1086  0.1667 -0.0356 -0.5117] norms [0.1587 0.0964 0.1141 0.0926]
L2H5 scores [-0.27   -0.0132  0.1973 -0.2821] norms [0.1703 0.1043 0.1418 0.0801]
L3H0 scores [ 0.0911 -0.1751  0.1758 -0.1901] norms [0.1278 0.0926 0.097  0.0743]
L3H1 scores [ 0.0828 -0.0375  0.0574  0.0908] norms [0.0726 0.0488 0.045  0.0336]
L3H2 scores [ 0.0549  0.035   0.0199 -0.1394] norms [0.0847 0.0525 0.0596 0.0403]
L3H3 scores [ 0.2541  0.7817 -0.3981  0.645 ] norms [0.1153 0.1434 0.0891 0.0859]
L3H4 scores [ 0.0372  0.1397 -0.0144  0.0423] norms [0.0386 0.023  0.0209 0.0142]
L3H5 scores [ 0.0459 -0.0356  0.2399 -0.1239] norms [0.1012 0.0646 0.0712 0.0475]

Raw L0H3 V input vs L0H0 V input canonical correlations at varying rank
20 [0.9413 0.9322 0.9039 0.8963 0.8896 0.8769 0.8611 0.84   0.8304 0.8209 0.7857 0.7642 0.7303 0.6    0.4376 0.2402 0.1441 0.1198 0.0761 0.0106]
22 [0.9418 0.9336 0.9061 0.8994 0.8922 0.8847 0.8694 0.8531 0.8345 0.8251 0.8196 0.7788 0.7467 0.6902 0.6411 0.4888 0.3772 0.2524 0.1659 0.093  0.0497 0.0374]
24 [0.9425 0.9376 0.9145 0.9022 0.8964 0.8873 0.8704 0.8562 0.8394 0.8311 0.8285 0.784  0.7612 0.6983 0.6519 0.573  0.4821 0.35   0.3118 0.1777 0.1477 0.1227 0.0919 0.0362]
26 [0.9451 0.938  0.9164 0.9069 0.897  0.893  0.8761 0.8648 0.8413 0.8358 0.8292 0.7888 0.7651 0.7182 0.681  0.654  0.5176 0.4024 0.3592 0.3265 0.247  0.1843 0.1408 0.1043 0.0674 0.0147]
28 [0.9463 0.9404 0.9177 0.9123 0.9017 0.8931 0.8768 0.8698 0.8453 0.843  0.8307 0.7947 0.775  0.729  0.6866 0.6624 0.5443 0.4614 0.436  0.3698 0.3158 0.2685 0.2025 0.1644 0.1158 0.0781 0.0343 0.011 ]
32 [0.9476 0.942  0.9229 0.9148 0.906  0.8967 0.8831 0.8773 0.8557 0.8473 0.8398 0.8201 0.7855 0.7525 0.7159 0.6787 0.5567 0.5201 0.4889 0.4776 0.4084 0.3723 0.325  0.2767 0.2391 0.1672 0.1518 0.1074
 0.0873 0.061  0.0486 0.0142]

```

## Call 10

```python
import numpy as np
from scipy.linalg import svd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits
import warnings

np.set_printoptions(precision=3, suppress=True, linewidth=200)
def eff(a,g):
    a=np.asarray(a,dtype=float)*g[None,:]
    return a-a.mean(axis=1,keepdims=True)

with threadpool_limits(limits=4):
    F=eff(W["W_fc.0"],W["ln2_w.0"])
    V0=eff(W["W_v.0"][:64],W["ln1_w.0"])
    V3=eff(W["W_v.0"][192:256],W["ln1_w.0"])
    _,_,vh=svd(np.vstack([V0,V3]),full_matrices=False)
    B3=W["W_o.0"][:,192:256]@V3
    for r in [18,22,26,30]:
        T=vh[:r].T
        X=F@T
        with warnings.catch_warnings(record=True) as ww:
            ica=FastICA(n_components=r,whiten="unit-variance",random_state=7,
                        max_iter=3000,tol=1e-5,fun="logcosh")
            S=ica.fit_transform(X)
        D=T@ica.components_.T
        # These are products between weight matrices and weight-derived directions,
        # not activations and not evaluations of a network layer.
        R=F@D
        U=F@B3@D
        sg=np.where(skew(R,axis=0)>=0,1.,-1.)
        R*=sg;U*=sg
        norm=np.std(R,axis=0)
        R/=norm;U/=norm
        order=np.argsort(-kurtosis(R,axis=0))
        print("\nICA rank",r,"iterations",ica.n_iter_,"warnings",len(ww))
        print("current-read skew",skew(R,axis=0)[order])
        print("current-read excess kurtosis",kurtosis(R,axis=0)[order])
        print("transport-read skew",skew(U,axis=0)[order])
        print("transport/current scale",np.std(U,axis=0)[order])
        C=R.T@U
        print("cross coupling transpose cosine",np.sum(C*C.T)/np.sum(C*C))
        if r==22:
            print("Normalized cross-coupling, ICA coordinates (unlabeled; not transition probabilities)")
            C=C/np.sqrt(np.sum(R*R,axis=0)[:,None]*np.sum(U*U,axis=0)[None,:])
            print(C)

```

Output:

```

ICA rank 18 iterations 71 warnings 0
current-read skew [7.511 6.272 6.794 6.737 5.644 5.221 4.837 3.956 2.685 4.802 4.012 3.156 3.15  3.296 3.524 2.221 2.405 2.108]
current-read excess kurtosis [95.285 88.029 84.542 77.355 54.216 52.633 43.773 42.632 36.657 36.213 32.19  27.309 26.497 24.309 20.596 14.324 14.314 12.631]
transport-read skew [ 2.242  0.848  3.123  2.188  1.895  3.464  1.272  0.475  0.844  2.173  1.213  0.525  1.336  1.055  1.397  1.473  0.097 -0.089]
transport/current scale [0.507 0.819 0.943 0.921 0.869 0.655 0.925 0.668 0.918 0.551 0.919 0.887 0.704 0.787 0.833 0.701 0.809 0.735]
cross coupling transpose cosine 0.12792388659321136

ICA rank 22 iterations 61 warnings 0
current-read skew [9.815 8.264 7.453 7.007 6.039 6.088 5.949 5.819 2.861 4.973 5.109 4.894 4.864 4.668 3.438 3.039 2.638 1.597 3.334 2.978 2.185 2.429]
current-read excess kurtosis [147.441 106.114  95.193  82.05   76.832  62.302  57.191  57.138  48.08   45.29   44.618  38.715  36.551  35.321  28.735  27.209  23.493  22.613  19.275  18.965  17.486  14.888]
transport-read skew [1.023 2.518 3.285 2.45  1.685 4.147 1.627 2.098 1.053 1.294 1.537 1.08  2.227 1.148 0.676 1.201 1.884 0.457 0.005 1.405 1.331 0.457]
transport/current scale [0.674 0.491 0.918 0.903 0.563 0.612 0.73  0.865 0.859 0.905 0.864 0.646 0.53  0.788 0.902 0.688 0.788 0.711 0.751 0.657 0.699 0.78 ]
cross coupling transpose cosine 0.04930985811188526
Normalized cross-coupling, ICA coordinates (unlabeled; not transition probabilities)
[[ 0.167 -0.142  0.087  0.154  0.087 -0.008  0.172 -0.004 -0.087  0.048  0.122 -0.156  0.058 -0.008  0.114  0.039  0.082  0.056  0.116  0.087  0.125 -0.088]
 [ 0.001  0.225  0.026 -0.053 -0.02   0.038 -0.073  0.034  0.16  -0.064 -0.101  0.097  0.083  0.092  0.094  0.025 -0.041  0.013  0.061 -0.073  0.041  0.031]
 [ 0.04  -0.023  0.159  0.083  0.143 -0.046  0.091 -0.027  0.172  0.057 -0.096 -0.102  0.217 -0.046  0.08   0.093  0.019  0.068 -0.004  0.024  0.126 -0.019]
 [ 0.061 -0.105  0.025  0.167  0.107 -0.047  0.117  0.067  0.043  0.05   0.098  0.181  0.297 -0.073  0.155  0.017 -0.122  0.126 -0.017 -0.101 -0.074  0.09 ]
 [-0.043  0.168  0.022 -0.036  0.084 -0.031 -0.038  0.063 -0.02  -0.065 -0.094 -0.086 -0.097  0.031 -0.04  -0.055 -0.019 -0.007  0.08   0.074 -0.004  0.067]
 [-0.211 -0.002  0.048  0.062 -0.121  0.035 -0.104 -0.132  0.105  0.032 -0.026  0.066  0.011 -0.085  0.098 -0.017 -0.012 -0.145  0.061 -0.141 -0.009  0.063]
 [ 0.001  0.001 -0.001 -0.043 -0.02  -0.021  0.056  0.029  0.081  0.039 -0.093  0.081  0.09   0.071  0.121 -0.064 -0.214  0.151  0.148  0.016 -0.012  0.071]
 [ 0.117  0.036 -0.002 -0.011 -0.027 -0.012  0.037 -0.092 -0.093  0.015  0.037 -0.131 -0.06   0.037  0.052 -0.029  0.011  0.001  0.064 -0.056  0.095 -0.091]
 [ 0.138  0.     0.038  0.064  0.048 -0.155  0.233  0.008  0.046 -0.06  -0.14  -0.07   0.125  0.005  0.129  0.037  0.155 -0.098 -0.058 -0.062 -0.065 -0.13 ]
 [-0.03   0.031 -0.064 -0.086 -0.028  0.111  0.011  0.062  0.183  0.107 -0.009  0.048  0.059 -0.105  0.014  0.032  0.104 -0.024 -0.03  -0.093  0.    -0.088]
 [ 0.109 -0.016  0.129  0.131  0.161  0.01   0.106 -0.033  0.167  0.002 -0.013 -0.045  0.154 -0.031  0.045  0.021 -0.141  0.125  0.169  0.004  0.031  0.033]
 [-0.023 -0.121  0.149  0.181  0.159 -0.076  0.028  0.141  0.025 -0.125 -0.05  -0.014  0.064  0.038  0.111  0.05  -0.027  0.098  0.133  0.173  0.174  0.119]
 [ 0.092  0.243 -0.022 -0.135  0.104 -0.145  0.004 -0.035  0.102  0.028 -0.085 -0.05   0.142  0.022 -0.102  0.066  0.013  0.025 -0.219 -0.014  0.07   0.008]
 [ 0.051  0.133 -0.027  0.007  0.073 -0.024  0.086  0.046  0.117  0.019  0.023  0.024  0.142  0.041  0.096 -0.058 -0.096 -0.095  0.002  0.009  0.094  0.096]
 [-0.216 -0.061  0.12  -0.018  0.041  0.076  0.107  0.043  0.124 -0.128 -0.135  0.064 -0.059 -0.005  0.063 -0.334 -0.165 -0.2    0.185  0.134  0.039  0.135]
 [-0.109 -0.084  0.062  0.064 -0.029  0.04  -0.054  0.133 -0.019  0.142 -0.031  0.006  0.115  0.091  0.051  0.195  0.05   0.012  0.116  0.119  0.135  0.027]
 [ 0.197  0.085 -0.003  0.098  0.217 -0.115  0.121 -0.11   0.193  0.133  0.005 -0.12   0.179  0.032  0.041  0.125  0.184  0.067 -0.096 -0.118 -0.031 -0.041]
 [ 0.004 -0.105  0.061  0.069 -0.034 -0.083  0.055  0.091 -0.045 -0.006 -0.013  0.029  0.086  0.013  0.148 -0.06   0.048 -0.076 -0.059  0.016  0.099 -0.035]
 [-0.02  -0.007 -0.039  0.073  0.065 -0.032 -0.002  0.052  0.11   0.019 -0.012 -0.039  0.136 -0.041  0.038  0.121  0.047  0.134 -0.143  0.117  0.042  0.1  ]
 [ 0.125 -0.153  0.049  0.171  0.213 -0.004  0.1    0.008  0.178  0.063 -0.042 -0.013  0.212  0.108  0.15   0.104 -0.06   0.243  0.125 -0.011  0.153  0.161]
 [ 0.015  0.1   -0.125 -0.003  0.079 -0.042 -0.069 -0.1    0.105 -0.001 -0.01  -0.029  0.003 -0.015 -0.089  0.126  0.039  0.115 -0.1   -0.048 -0.028  0.023]
 [ 0.082 -0.076  0.055 -0.07   0.033 -0.003  0.127 -0.142 -0.018 -0.091 -0.076 -0.073 -0.028  0.042  0.013 -0.008  0.043 -0.017  0.054 -0.002 -0.053 -0.02 ]]

ICA rank 26 iterations 47 warnings 0
current-read skew [10.42   8.354  8.369  8.52   7.353  6.4    6.368  5.908  5.757  3.898  5.088  5.485  5.036  4.956  3.96   3.98   2.004  3.826  2.883  1.438  1.788  3.4    2.891  1.84   1.381  0.617]
current-read excess kurtosis [158.847 114.273 110.892 109.887  87.71   65.813  62.089  58.076  52.731  52.164  45.418  44.027  39.828  38.87   34.918  31.197  31.106  24.553  24.085  23.623  22.811  21.2    19.524  17.491
   9.621   5.838]
transport-read skew [ 1.093  1.866  3.502  2.36   2.53   3.702  1.634  2.057  1.657  0.905  1.35   1.76   1.234  1.132  0.666  1.605  0.737  0.07   1.698  0.386  1.235  0.401  1.481  1.146  1.    -0.59 ]
transport/current scale [0.633 0.598 0.899 0.632 0.89  0.597 0.724 0.859 0.782 0.794 0.915 0.509 0.778 0.649 0.914 0.633 0.564 0.777 0.812 0.71  0.512 0.772 0.672 0.697 0.628 0.509]
cross coupling transpose cosine 0.041313983556812414

ICA rank 30 iterations 91 warnings 0
current-read skew [10.762  8.617  8.777  8.411  7.419  4.553  6.516  6.462  6.124  5.
```

## Call 11

```python
import numpy as np
from scipy.linalg import svd, eigh
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=3, suppress=True, linewidth=220)
def eff(a,g):
    a=np.asarray(a,dtype=float)*g[None,:]
    return a-a.mean(axis=1,keepdims=True)

with threadpool_limits(limits=4):
    F=eff(W["W_fc.0"],W["ln2_w.0"])
    V0=eff(W["W_v.0"][:64],W["ln1_w.0"])
    V3=eff(W["W_v.0"][192:256],W["ln1_w.0"])
    _,_,vh=svd(np.vstack([V0,V3]),full_matrices=False)
    T=vh[:30].T
    ica=FastICA(n_components=30,whiten="unit-variance",random_state=7,
                max_iter=3000,tol=1e-5).fit(F@T)
    D=T@ica.components_.T
    R=F@D
    U=F@(W["W_o.0"][:,192:256]@V3)@D
    sg=np.where(skew(R,axis=0)>=0,1.,-1.)
    R*=sg;U*=sg
    order=np.argsort(-kurtosis(R,axis=0))
    R=R[:,order];U=U[:,order]
    R=(R-R.mean(axis=0))/R.std(axis=0)
    U=(U-U.mean(axis=0))/U.std(axis=0)
    m=23
    R=R[:,:m];U=U[:,:m]
    print("Unlabeled components are numbered in decreasing current-read kurtosis.")
    print("For each: number of current/transported coefficients > 2 standard deviations")
    print(np.array([(i,(R[:,i]>2).sum(),(U[:,i]>2).sum()) for i in range(m)]))
    print("\nCo-selectivity counts (rows=current-read component, columns=transported-read component).")
    C=(R>2).astype(float).T@(U>2).astype(float)
    print(C.astype(int))
    print("\nStrongest transported-read partners for each current-read component:")
    for i in range(m):
        js=np.argsort(-C[i])[:5]
        print(i,[(int(j),int(C[i,j])) for j in js])
    print("\nSymmetric co-selectivity graph bipartition diagnostics")
    for thresh in [1.5,2.,2.5,3.]:
        C=(R>thresh).astype(float).T@(U>thresh).astype(float)
        S=C+C.T
        np.fill_diagonal(S,0)
        deg=S.sum(axis=1)
        N=S/np.sqrt(np.maximum(deg[:,None]*deg[None,:],1))
        ev,vec=eigh(N)
        signs=np.where(vec[:,0]>=0,1,-1)
        # Improve a cut of this weight-coefficient graph, not a model evaluation.
        for _ in range(100):
            gains=signs*(S@signs)
            j=np.argmax(gains)
            if gains[j]<=1e-10:
                break
            signs[j]*=-1
        cross=S[signs[:,None]!=signs[None,:]].sum()/max(S.sum(),1)
        print("threshold",thresh,"min eigenvalue",round(ev[0],3),
              "cut fraction",round(cross,3),
              "groups",np.flatnonzero(signs>0).tolist(),np.flatnonzero(signs<0).tolist())

```

Output:

```
Unlabeled components are numbered in decreasing current-read kurtosis.
For each: number of current/transported coefficients > 2 standard deviations
[[ 0 17 66]
 [ 1 26 55]
 [ 2 17 61]
 [ 3 29 51]
 [ 4 33 54]
 [ 5 45 59]
 [ 6 31 44]
 [ 7 29 70]
 [ 8 43 52]
 [ 9 34 59]
 [10 39 61]
 [11 32 54]
 [12 37 69]
 [13 42 61]
 [14 39 63]
 [15 49 62]
 [16 41 56]
 [17 46 61]
 [18 48 58]
 [19 35 58]
 [20 54 64]
 [21 45 56]
 [22 43 47]]

Co-selectivity counts (rows=current-read component, columns=transported-read component).
[[ 8  4  0  1  1  0  1  4  1  4  2  2  0  2  3  2  0  3  0  0  2  0  4]
 [ 3  9  3  3  1  3  2  8  5  7  2  0  0  0  7  2  3  1  4  3  2  3  3]
 [ 3  3  5  0  2  0  1  4  4  3  1  1  0  1  1  2  0  4  0  1  1  1  3]
 [ 4  3  3  9  0  2  0  4  4  8  7  1  3  2  3  4  1  2  1  4  2  2  0]
 [ 1  3  2  5  8  8  3  8  3  5  4  5  7  1  5  3  2  4  1  2  5  4  4]
 [16 14  3  7  9 14 10 13  2 13 10  6  5 13  6  0 11  5  8  3  1  0 11]
 [ 7  5  5  2  3  6  8  3  5  3  4  0  3  5  8  8  5  2  6  4  3  5  2]
 [ 5  1  5  2  2  3  3 10  6  3  3  2  6  3  2  0  2  1  2  3  3  4  4]
 [ 8  4  7  5  7  5  4  9 16  6  7  2  3  5  5  5  4  8  5  4  3  4  3]
 [10  7  5  5  4  2  2  7  2 15  8  3  4  5  7  6  3  5  3  3  3  1  2]
 [ 3  5  8  7  6  1  1  2  6  8 14  2  8  6  5  6  1  8  2  3  9  3  0]
 [ 1  1  4  3  0  5  1  3  1  1  1 10  1  0  2  2  1  2  0  3  5  5  1]
 [ 5  4  6  5  3  2  0  8  6  9  8  2 13  2  3  7  3  4  5  2  3  3  1]
 [10  4  4  5  6  3  3  8  5  9  6  4  7 11  8  8  4  7  5 10  3  5  6]
 [ 8  7 11  3  3  1  6  3  1  8  6  1  3  5 13  8  4  6  4  3 10  2  6]
 [12  9  8  8  6  6  6 10  7  8  4  4  8  6 12  9  5  8  9  9  6  6  5]
 [ 6  8  7  5  6 10  3  5  2  7  6  5  3  4  1  4 13  5  4  9  6  3  6]
 [ 7  8  7  5  7  3  5  8  7  7 10  3  6  6  8  7  5 10  6  5  5  8  4]
 [ 8  8  6  4  5  3  5  4  5 10  8  2  5  6 11  9  4  9 10  4  7  5  2]
 [ 5  3  3  2  3  6  3  4  4  4  7  4  4  6  5  5  2  4  2  8  3  3  4]
 [ 9  9 10  8  4  2  7 12 11  7  6  3 10  7 12  8  7  7  9 13  9  6  3]
 [ 2  6  7  4  5  4  6  3  3  2  9  5  9  4 10  2  4  4  3  2  8  9  4]
 [ 7  4  6  3  6  4  2  9  6  7  7  4  2  2  7  4  7  6  6  5  5  3  9]]

Strongest transported-read partners for each current-read component:
0 [(0, 8), (9, 4), (7, 4), (22, 4), (1, 4)]
1 [(1, 9), (7, 8), (14, 7), (9, 7), (8, 5)]
2 [(2, 5), (17, 4), (8, 4), (7, 4), (0, 3)]
3 [(3, 9), (9, 8), (10, 7), (0, 4), (15, 4)]
4 [(4, 8), (5, 8), (7, 8), (12, 7), (11, 5)]
5 [(0, 16), (1, 14), (5, 14), (13, 13), (9, 13)]
6 [(15, 8), (14, 8), (6, 8), (0, 7), (5, 6)]
7 [(7, 10), (12, 6), (8, 6), (0, 5), (2, 5)]
8 [(8, 16), (7, 9), (0, 8), (17, 8), (2, 7)]
9 [(9, 15), (0, 10), (10, 8), (1, 7), (7, 7)]
10 [(10, 14), (20, 9), (17, 8), (2, 8), (12, 8)]
11 [(11, 10), (20, 5), (5, 5), (21, 5), (2, 4)]
12 [(12, 13), (9, 9), (10, 8), (7, 8), (15, 7)]
13 [(13, 11), (0, 10), (19, 10), (9, 9), (15, 8)]
14 [(14, 13), (2, 11), (20, 10), (0, 8), (15, 8)]
15 [(0, 12), (14, 12), (7, 10), (19, 9), (15, 9)]
16 [(16, 13), (5, 10), (19, 9), (1, 8), (2, 7)]
17 [(17, 10), (10, 10), (1, 8), (14, 8), (21, 8)]
18 [(14, 11), (18, 10), (9, 10), (17, 9), (15, 9)]
19 [(19, 8), (10, 7), (5, 6), (13, 6), (0, 5)]
20 [(19, 13), (14, 12), (7, 12), (8, 11), (2, 10)]
21 [(14, 10), (10, 9), (21, 9), (12, 9), (20, 8)]
22 [(22, 9), (7, 9), (16, 7), (14, 7), (10, 7)]

Symmetric co-selectivity graph bipartition diagnostics
threshold 1.5 min eigenvalue -0.17 cut fraction 0.563 groups [5, 8, 9, 15, 17, 18, 19, 20, 21, 22] [0, 1, 2, 3, 4, 6, 7, 10, 11, 12, 13, 14, 16]
threshold 2.0 min eigenvalue -0.189 cut fraction 0.566 groups [2, 5, 8, 9, 15, 17, 19, 20, 21, 22] [0, 1, 3, 4, 6, 7, 10, 11, 12, 13, 14, 16, 18]
threshold 2.5 min eigenvalue -0.226 cut fraction 0.579 groups [0, 1, 2, 3, 4, 6, 7, 10, 11, 12, 13, 16, 18] [5, 8, 9, 14, 15, 17, 19, 20, 21, 22]
threshold 3.0 min eigenvalue -0.26 cut fraction 0.592 groups [0, 1, 4, 5, 8, 9, 10, 11, 12, 18, 19, 20] [2, 3, 6, 7, 13, 14, 15, 16, 17, 21, 22]

```

## Call 12

```python
import numpy as np
from scipy.linalg import svd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits
import warnings

np.set_printoptions(precision=3, suppress=True, linewidth=200)
with threadpool_limits(limits=4):
    for l in [0,3]:
        P=W[f"W_proj.{l}"].astype(float).T
        u,s,vh=svd(P-P.mean(axis=0),full_matrices=False)
        print(f"\nL{l} projection-column PCA singular values, first 50")
        print(s[:50])
        for r in [24,28,36]:
            X=(P-P.mean(axis=0))@vh[:r].T
            with warnings.catch_warnings(record=True) as ww:
                ica=FastICA(n_components=r,whiten="unit-variance",random_state=11,
                            max_iter=2000,tol=2e-5).fit(X)
            S=ica.transform(X)
            ku=kurtosis(S,axis=0);sk=np.abs(skew(S,axis=0))
            ix=np.argsort(-ku)
            print("rank",r,"iterations",ica.n_iter_,"warnings",len(ww))
            print("excess kurtosis",ku[ix])
            print("abs skew",sk[ix])

```

Output:

```

L0 projection-column PCA singular values, first 50
[10.465  7.634  6.872  6.631  6.457  6.287  6.166  6.124  5.938  5.917  5.758  5.637  5.523  5.473  5.365  5.357  5.259  5.201  5.169  5.146  5.048  5.018  4.954  4.872  4.82   4.788  4.769  4.73
  4.634  4.577  4.541  4.526  4.504  4.491  4.438  4.412  4.366  4.314  4.292  4.26   4.242  4.18   4.133  4.103  4.063  4.048  4.026  3.988  3.967  3.947]
rank 24 iterations 163 warnings 0
excess kurtosis [140.974  58.549  56.787  46.767  39.745  36.419  33.115  31.055  25.99   21.738  21.484  19.51   19.084  18.031  17.217  16.84   15.212  13.907  13.073  12.706  12.083  11.368  10.488   4.64 ]
abs skew [9.643 4.497 2.87  5.096 4.692 3.615 2.275 3.546 2.674 2.603 0.114 4.057 1.767 2.428 2.115 0.917 0.136 1.613 0.53  0.763 1.13  0.766 1.635 1.108]
rank 28 iterations 109 warnings 0
excess kurtosis [158.546  65.732  59.275  52.228  45.72   39.552  39.168  37.366  37.094  35.684  31.628  27.056  25.574  24.488  24.219  22.431  22.103  20.09   18.747  18.105  17.638  16.719  16.663  16.239
  14.916  13.335   8.714   4.964]
abs skew [10.423  4.274  5.786  1.688  4.677  3.957  2.974  2.613  3.95   0.03   2.283  2.832  3.148  3.208  2.139  2.638  2.318  4.054  2.093  1.949  0.754  0.564  0.236  2.097  1.213  0.842  1.353  1.073]
rank 36 iterations 150 warnings 0
excess kurtosis [154.224  98.891  85.158  64.045  62.26   49.272  48.104  43.56   43.476  43.153  42.109  40.525  40.462  35.26   32.16   31.851  31.509  27.25   23.859  23.165  20.31   20.058  19.997  19.474
  19.304  19.209  18.748  16.165  15.425  14.858  14.5    13.609  12.628  12.428  10.294   8.838]
abs skew [10.531  6.473  0.701  6.021  5.558  4.636  4.756  0.71   2.653  4.012  3.618  4.102  4.269  4.124  3.771  3.512  1.01   3.423  2.7    2.765  3.836  2.913  1.62   1.449  1.357  0.184  1.027  0.993
  0.171  1.145  0.636  0.387  0.706  0.543  0.435  0.645]

L3 projection-column PCA singular values, first 50
[27.652 22.212 21.316 20.688 19.943 19.657 18.985 18.743 18.294 17.457 17.13  16.556 15.323 14.869 14.324 13.421 12.665 12.452 12.311 11.523 11.044  9.081  8.334  8.2    6.367  5.834  4.606  4.462
  3.777  3.563  3.235  3.182  3.085  2.876  2.853  2.796  2.596  2.54   2.512  2.426  2.404  2.365  2.321  2.306  2.227  2.164  2.154  2.147  2.08   2.065]
rank 24 iterations 201 warnings 0
excess kurtosis [11.477  3.012  2.711  2.5    2.185  2.137  1.884  1.788  1.748  1.675  1.671  1.607  1.516  1.397  1.212  1.197  1.154  1.144  1.13   0.958 -0.373 -0.499 -0.522 -0.593]
abs skew [0.92  0.667 0.737 0.815 0.729 0.628 0.738 0.775 0.756 0.419 0.714 0.092 0.375 0.488 0.479 0.412 0.274 0.049 0.007 0.101 0.126 0.062 0.171 0.081]
rank 28 iterations 260 warnings 0
excess kurtosis [18.766 12.142  4.748  3.045  2.549  2.489  2.429  2.289  2.1    2.089  2.052  2.033  1.834  1.687  1.573  1.551  1.515  1.308  1.21   1.17   1.137  1.115  1.052  0.97   0.551 -0.529 -0.541 -0.596]
abs skew [3.807 1.091 1.244 0.753 0.56  0.228 0.768 0.801 0.537 0.17  0.557 0.43  0.656 0.148 0.687 0.329 0.201 0.101 0.295 0.269 0.151 0.072 0.23  0.14  0.061 0.027 0.013 0.049]
rank 36 iterations 68 warnings 0
excess kurtosis [31.97  23.112 22.538 11.966 11.817 10.81   8.539  6.188  4.413  3.973  3.522  2.83   2.458  2.434  2.434  2.363  2.237  2.01   1.918  1.905  1.885  1.762  1.747  1.649  1.433  1.349  1.264  1.228
  1.195  1.175  1.102  1.014  1.004  0.935  0.768 -0.532]
abs skew [4.473 4.087 4.216 2.044 0.963 1.679 1.836 1.333 1.502 0.976 0.857 0.615 0.653 0.653 0.718 0.52  0.313 0.566 0.105 0.389 0.601 0.221 0.504 0.158 0.092 0.382 0.021 0.098 0.282 0.306 0.137 0.081 0.103
 0.058 0.196 0.018]

```

## Call 13

```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=200)
def eff(a,g):
    a=np.asarray(a,dtype=float)*g[None,:]
    return a-a.mean(axis=1,keepdims=True)

def score(C):
    return np.trace(C)/svdvals(C).sum()

with threadpool_limits(limits=4):
    V0=eff(W["W_v.0"][:64],W["ln1_w.0"])
    V3=eff(W["W_v.0"][192:256],W["ln1_w.0"])
    _,_,vh=svd(np.vstack([V0,V3]),full_matrices=False)
    T=vh[:22].T
    q,k=[eff(W[f"W_{n}.3"][192:256],W["ln1_w.3"]) for n in ["q","k"]]
    A=q.T@k/8
    print("L3H3 key-composition trace/nuclear with all earlier heads, same fixed 22D read subspace:")
    for l in range(3):
        vals=[]
        for j in range(6):
            sl=slice(64*j,64*(j+1))
            B=W[f"W_o.{l}"][:,sl]@eff(W[f"W_v.{l}"][sl],W[f"ln1_w.{l}"])
            vals.append(score(T.T@A@B@T))
        print("layer",l,np.array(vals))

    rng=np.random.default_rng(42)
    o=W["W_o.0"][:,192:256].astype(float)
    Left=T.T@A@o
    Right=V3@T
    observed=score(Left@Right)
    null=[]
    for _ in range(300):
        perm=rng.permutation(64)
        signs=rng.choice([-1.,1.],64)
        null.append(score((Left[:,perm]*signs[None,:])@Right))
    null=np.array(null)
    print("\nWithin-head signed-permutation control for L0H3->L3H3")
    print("observed",observed,"null mean/std",null.mean(),null.std(),
          "null min/max",null.min(),null.max())

    B5=W["W_o.0"][:,320:384]@eff(W["W_v.0"][320:384],W["ln1_w.0"])
    U,s,_=svd(B5,full_matrices=False)
    print("\nL0H5 adjusted OV leading singular values",s[:10])
    print("Later bias-induced key vectors in L0H5 output principal coordinates:")
    for l in range(1,4):
        for j in range(6):
            sl=slice(64*j,64*(j+1))
            q0=W[f"W_q.{l}"][sl].astype(float)
            k=eff(W[f"W_k.{l}"][sl],W[f"ln1_w.{l}"])
            c=k.T@(q0@W[f"ln1_b.{l}"])/8
            print(f"L{l}H{j}",U[:,:3].T@c)

```

Output:

```
L3H3 key-composition trace/nuclear with all earlier heads, same fixed 22D read subspace:
layer 0 [-0.1587  0.072   0.065   0.7817  0.0562 -0.1681]
layer 1 [ 0.3122  0.1921  0.0872  0.01   -0.0189  0.2684]
layer 2 [0.0395 0.3124 0.0598 0.1752 0.1242 0.2643]

Within-head signed-permutation control for L0H3->L3H3
observed 0.7817483093578845 null mean/std -5.96256671334842e-05 0.0640644242581716 null min/max -0.18386251496310396 0.3065867643563379

L0H5 adjusted OV leading singular values [16.2778  8.7783  2.8331  1.117   1.0385  0.6305  0.5332  0.4476  0.4133  0.3262]
Later bias-induced key vectors in L0H5 output principal coordinates:
L1H0 [-2.9439  2.1776  0.2404]
L1H1 [-2.6872  2.0808  0.2384]
L1H2 [-2.2433  1.699   0.0763]
L1H3 [-2.4165  1.9793  0.2446]
L1H4 [-1.4831  1.4498  0.1604]
L1H5 [-2.8104  2.2373  0.3904]
L2H0 [-1.4317  1.1558  0.0913]
L2H1 [-1.2944  1.0904  0.2191]
L2H2 [-1.0693  0.9259  0.1843]
L2H3 [-0.5863  0.5854  0.0584]
L2H4 [-1.6637  1.2805  0.0442]
L2H5 [-1.5006  1.1066  0.1706]
L3H0 [-0.3036  0.2824  0.0351]
L3H1 [-0.1203  0.1505  0.0017]
L3H2 [-0.2744  0.2752  0.0466]
L3H3 [ 0.0178  0.0064 -0.0092]
L3H4 [-0.0244  0.0132 -0.0107]
L3H5 [-0.3261  0.263   0.0097]

```

## Call 14

```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=200)
def eff(a,g):
    a=np.asarray(a,dtype=float)*g[None,:]
    return a-a.mean(axis=1,keepdims=True)

with threadpool_limits(limits=4):
    for l,j in [(0,3),(3,3),(3,4)]:
        sl=slice(64*j,64*(j+1))
        print(f"\nL{l}H{j} unadjusted V and O singular values")
        print("V",svdvals(W[f"W_v.{l}"][sl]))
        print("O",svdvals(W[f"W_o.{l}"][:,sl]))
    q=eff(W["W_q.3"][192:256],W["ln1_w.3"])
    k=eff(W["W_k.3"][192:256],W["ln1_w.3"])
    v=eff(W["W_v.3"][192:256],W["ln1_w.3"])
    A=q.T@k/8
    V0=eff(W["W_v.0"][:64],W["ln1_w.0"])
    V3=eff(W["W_v.0"][192:256],W["ln1_w.0"])
    _,_,vh=svd(np.vstack([V0,V3]),full_matrices=False)
    T=vh[:40].T
    Bprev=W["W_o.0"][:,192:256]@V3
    print("\nL3H3 matching-map singular spectrum in fixed early-read top40 subspace")
    print(svdvals(T.T@A@Bprev@T))
    print("\nL3H3 V reading the same subspace, singular spectrum")
    print(svdvals(v@T))
    print("\nShared strength of final head output and final MLP output")
    o=W["W_o.3"][:,192:256].astype(float)
    u,_,_=svd(W["W_proj.3"].astype(float),full_matrices=False)
    print("O projected onto top27 final-MLP-output directions",svdvals(u[:,:27].T@o))

```

Output:

```

L0H3 unadjusted V and O singular values
V [2.1024 1.9481 1.9032 1.8657 1.8432 1.8192 1.7694 1.7548 1.6743 1.6512 1.6288 1.6207 1.5888 1.5533 1.5198 1.4649 1.4172 1.3029 1.1603 1.1014 0.9447 0.9297 0.8483 0.8181 0.7903 0.7786 0.745  0.7387
 0.7303 0.7008 0.6882 0.6576 0.6424 0.6217 0.6175 0.6052 0.5753 0.5651 0.5434 0.5218 0.5149 0.4909 0.4806 0.466  0.4463 0.4299 0.4104 0.4    0.3775 0.3661 0.3594 0.3488 0.3453 0.3418 0.3363 0.3291
 0.3183 0.3133 0.3127 0.2983 0.2873 0.2789 0.2713 0.2621]
O [2.1335 2.1097 2.0764 1.9746 1.963  1.9366 1.9132 1.8908 1.8783 1.8451 1.8229 1.813  1.7799 1.7372 1.7132 1.678  1.6324 1.4567 1.3946 1.326  1.2865 1.2717 1.2139 1.2008 1.1897 1.1305 1.1008 1.0781
 1.0648 1.0106 0.9944 0.956  0.9351 0.9158 0.8638 0.8174 0.7556 0.7395 0.7133 0.6884 0.6343 0.5621 0.5353 0.4692 0.4227 0.3726 0.3568 0.3494 0.3206 0.2966 0.2795 0.2671 0.2593 0.2486 0.2347 0.2277
 0.2172 0.216  0.198  0.1866 0.1839 0.1723 0.1446 0.1273]

L3H3 unadjusted V and O singular values
V [3.8615 3.7935 3.7482 3.6414 3.5521 3.5014 3.3992 3.2733 3.2572 3.1757 3.1238 3.0492 2.9993 2.9487 2.8892 2.7927 2.6745 2.6111 2.5797 2.5252 2.458  2.4502 2.4079 2.3755 2.3565 2.2805 2.1695 2.1353
 2.0755 2.0424 1.9888 1.9628 1.9209 1.9121 1.842  1.807  1.7773 1.7013 1.6819 1.6172 1.5951 1.565  1.551  1.4957 1.4046 1.3879 1.3502 1.3338 1.2426 1.1953 1.1533 1.1175 1.0991 1.0888 1.0398 0.9781
 0.9305 0.917  0.8792 0.8373 0.7949 0.7542 0.6661 0.6399]
O [4.3442 3.8695 3.7632 3.7077 3.6203 3.567  3.4837 3.4331 3.3724 3.3003 3.2663 3.1428 3.0387 2.963  2.9537 2.856  2.8127 2.7407 2.6923 2.6415 2.5423 2.4985 2.4608 2.4248 2.4208 2.362  2.2851 2.1965
 2.173  2.1095 2.0312 2.0019 1.9193 1.8642 1.8497 1.8242 1.8094 1.7902 1.7408 1.6963 1.6332 1.6173 1.556  1.5379 1.4816 1.4406 1.3825 1.3654 1.294  1.2423 1.2374 1.1975 1.1422 1.1192 1.0741 1.0073
 0.969  0.9567 0.9286 0.8852 0.8224 0.7716 0.7076 0.6786]

L3H4 unadjusted V and O singular values
V [5.0524 4.9051 4.774  4.5242 4.4543 4.3276 4.13   4.0278 3.4812 3.4083 3.3325 3.0272 2.7757 2.7362 2.5766 2.5101 2.4024 2.3662 2.2077 2.0912 2.0779 2.0208 1.922  1.8737 1.7646 1.7369 1.7099 1.6556
 1.5825 1.55   1.454  1.384  1.3726 1.3306 1.2968 1.2516 1.1967 1.1677 1.1473 1.103  1.0588 1.0152 0.9978 0.9627 0.9445 0.8827 0.8292 0.7875 0.7527 0.7263 0.7056 0.6599 0.6457 0.6188 0.6052 0.5896
 0.5698 0.545  0.5283 0.508  0.4847 0.4469 0.4212 0.4046]
O [6.456  5.125  4.7418 4.6286 4.4315 4.2638 4.1666 3.9562 3.644  3.4835 3.1565 3.105  2.9379 2.7646 2.5575 2.5007 2.3897 2.3676 2.2265 2.1369 2.1059 2.0906 1.9046 1.8556 1.7952 1.6835 1.6437 1.5676
 1.5114 1.4794 1.4182 1.3928 1.3035 1.2601 1.2489 1.2288 1.1151 1.0915 1.0799 1.0359 1.0241 0.9766 0.937  0.9112 0.9004 0.8113 0.7922 0.7474 0.6868 0.6616 0.6552 0.6077 0.5778 0.565  0.5396 0.5241
 0.4975 0.4871 0.4552 0.4441 0.4023 0.3923 0.3667 0.3265]

L3H3 matching-map singular spectrum in fixed early-read top40 subspace
[0.0745 0.0661 0.0612 0.0558 0.0492 0.0439 0.0418 0.0382 0.0351 0.0314 0.0296 0.0243 0.0215 0.0191 0.0169 0.0143 0.0127 0.0111 0.0108 0.0082 0.0068 0.0061 0.0054 0.0047 0.004  0.0036 0.0034 0.0027
 0.0025 0.0021 0.0019 0.0015 0.001  0.0009 0.0007 0.0005 0.0003 0.0002 0.0001 0.0001]

L3H3 V reading the same subspace, singular spectrum
[1.6402 1.5493 1.4379 1.3371 1.2842 1.2323 1.1764 1.1171 1.0753 1.0257 0.9918 0.9648 0.9252 0.8955 0.8584 0.798  0.7557 0.7412 0.7156 0.671  0.6076 0.6066 0.5637 0.5229 0.4969 0.4418 0.4306 0.3912
 0.3584 0.3212 0.3103 0.3015 0.2897 0.2449 0.2302 0.2154 0.2027 0.1916 0.1662 0.1436]

Shared strength of final head output and final MLP output
O projected onto top27 final-MLP-output directions [2.8927 2.4715 2.4554 2.4154 2.3181 2.2771 2.225  2.1927 2.1083 2.0835 2.0605 2.0001 1.9571 1.9302 1.8791 1.8375 1.8081 1.7853 1.7116 1.6453 1.6109 1.5184 1.4929 1.2943 1.2103 0.6922 0.4313]

```
