## Call 1

```python
import numpy as np
from scipy.linalg import svdvals, eigvalsh
from threadpoolctl import threadpool_limits
threadpool_limits(2)

np.set_printoptions(precision=4, suppress=True, linewidth=160)

def erank(s):
    p = s*s / np.sum(s*s)
    return np.exp(-np.sum(p * np.log(p + 1e-30)))

print("KEYS AND SHAPES")
print([(k, v.shape) for k, v in sorted(W.items())])

for l in range(4):
    print("\nLAYER", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        a = W[f"{name}.{l}"].astype(float)
        print(name, "min/mean/max/std/norm",
              np.round([a.min(), a.mean(), a.max(), a.std(), np.linalg.norm(a)], 5))
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        a = W[f"{name}.{l}"].astype(float)
        s = svdvals(a)
        print(name, "Frob", round(np.linalg.norm(a), 4),
              "s[:12]", np.round(s[:12], 3), "energy_rank", round(erank(s), 2))

print("\nHEADS: A=Q.T@K; B=O@V (weight products only)")
print("Apos/neg are norms of positive/negative eigenvalues of sym(A); skew is ||A-A.T||/(2||A||).")
print("Btracecos is <B,I>/(||B|| sqrt(384)); Bsym is ||sym(B)||/||B||.")
for l in range(4):
    for j in range(6):
        sl = slice(j*64, (j+1)*64)
        Q = W[f"W_q.{l}"][sl].astype(float)
        K = W[f"W_k.{l}"][sl].astype(float)
        V = W[f"W_v.{l}"][sl].astype(float)
        O = W[f"W_o.{l}"][:, sl].astype(float)
        A, B = Q.T @ K, O @ V
        sa, sb = svdvals(A), svdvals(B)
        ea = eigvalsh((A+A.T)/2)
        na, nb = np.linalg.norm(A), np.linalg.norm(B)
        bias = W[f"ln1_b.{l}"].astype(float)
        qbias, kbias = Q @ bias, K @ bias
        print(f"L{l}H{j}",
              "QK_norm", round(na,3),
              "QK_s", np.round(sa[:6],3),
              "QK_er", round(erank(sa),2),
              "Apos/neg", np.round([np.linalg.norm(ea[ea>0]),np.linalg.norm(ea[ea<0])],3),
              "skew", round(np.linalg.norm(A-A.T)/(2*na),3),
              "OV_norm", round(nb,3),
              "OV_s", np.round(sb[:4],3),
              "OV_er", round(erank(sb),2),
              "Btracecos", round(np.trace(B)/(nb*np.sqrt(384)),4),
              "Bsym", round(np.linalg.norm(B+B.T)/(2*nb),3),
              "biasQ/K", np.round([np.linalg.norm(qbias),np.linalg.norm(kbias)],3))

```

Output:

```
KEYS AND SHAPES
[('W_fc.0', (1536, 384)), ('W_fc.1', (1536, 384)), ('W_fc.2', (1536, 384)), ('W_fc.3', (1536, 384)), ('W_k.0', (384, 384)), ('W_k.1', (384, 384)), ('W_k.2', (384, 384)), ('W_k.3', (384, 384)), ('W_o.0', (384, 384)), ('W_o.1', (384, 384)), ('W_o.2', (384, 384)), ('W_o.3', (384, 384)), ('W_proj.0', (384, 1536)), ('W_proj.1', (384, 1536)), ('W_proj.2', (384, 1536)), ('W_proj.3', (384, 1536)), ('W_q.0', (384, 384)), ('W_q.1', (384, 384)), ('W_q.2', (384, 384)), ('W_q.3', (384, 384)), ('W_v.0', (384, 384)), ('W_v.1', (384, 384)), ('W_v.2', (384, 384)), ('W_v.3', (384, 384)), ('ln1_b.0', (384,)), ('ln1_b.1', (384,)), ('ln1_b.2', (384,)), ('ln1_b.3', (384,)), ('ln1_w.0', (384,)), ('ln1_w.1', (384,)), ('ln1_w.2', (384,)), ('ln1_w.3', (384,)), ('ln2_b.0', (384,)), ('ln2_b.1', (384,)), ('ln2_b.2', (384,)), ('ln2_b.3', (384,)), ('ln2_w.0', (384,)), ('ln2_w.1', (384,)), ('ln2_w.2', (384,)), ('ln2_w.3', (384,))]

LAYER 0
ln1_w min/mean/max/std/norm [0.0165 0.2677 1.0134 0.2147 6.7241]
ln1_b min/mean/max/std/norm [-0.1215  0.0006  0.1418  0.0366  0.7172]
ln2_w min/mean/max/std/norm [ 0.0468  0.8168  1.6412  0.3312 17.2723]
ln2_b min/mean/max/std/norm [-0.5537  0.018   0.1663  0.0667  1.353 ]
W_q Frob 22.7038 s[:12] [5.039 4.999 4.703 4.377 4.343 4.294 4.058 4.032 3.95  3.923 3.795 3.744] energy_rank 67.58
W_k Frob 23.2143 s[:12] [6.204 5.052 4.572 4.537 4.391 4.289 4.284 4.208 4.119 4.073 3.959 3.917] energy_rank 64.9
W_v Frob 18.1973 s[:12] [4.553 4.024 3.738 3.497 3.465 3.367 3.279 3.215 3.136 3.113 3.073 3.02 ] energy_rank 64.91
W_o Frob 19.5995 s[:12] [6.678 5.237 2.835 2.679 2.62  2.581 2.55  2.515 2.503 2.477 2.449 2.443] energy_rank 77.48
W_fc Frob 51.6496 s[:12] [18.11  12.28   8.473  7.902  7.779  7.442  7.146  7.109  7.008  6.805  6.721  6.54 ] energy_rank 86.42
W_proj Frob 45.8056 s[:12] [10.932  7.576  6.628  6.293  6.166  5.924  5.886  5.769  5.706  5.61   5.495  5.463] energy_rank 144.1

LAYER 1
ln1_w min/mean/max/std/norm [0.1979 0.4754 1.2096 0.0964 9.5046]
ln1_b min/mean/max/std/norm [-1.0505  0.022   0.9103  0.1532  3.0327]
ln2_w min/mean/max/std/norm [ 0.0139  1.0111  1.4707  0.1582 20.0543]
ln2_b min/mean/max/std/norm [-0.3744  0.0275  0.3652  0.1442  2.8769]
W_q Frob 31.6763 s[:12] [8.984 7.137 6.079 5.874 5.19  5.102 4.973 4.885 4.84  4.734 4.567 4.462] energy_rank 75.14
W_k Frob 28.6017 s[:12] [10.967  9.448  5.438  4.842  4.539  4.481  4.235  4.11   4.011  3.969  3.944  3.731] energy_rank 59.24
W_v Frob 28.4686 s[:12] [5.35  5.064 4.979 4.819 4.69  4.585 4.519 4.419 4.328 4.148 4.068 3.934] energy_rank 98.24
W_o Frob 26.151 s[:12] [4.19  3.651 3.385 3.352 3.24  3.216 3.141 3.115 3.109 3.086 3.032 3.005] energy_rank 143.5
W_fc Frob 80.2609 s[:12] [20.796 14.143 13.826 13.428 13.122 12.883 12.764 12.474 12.289 11.884 11.753 11.47 ] energy_rank 98.48
W_proj Frob 75.1193 s[:12] [7.625 7.359 6.908 6.489 6.344 6.223 6.192 6.02  5.949 5.902 5.82  5.756] energy_rank 320.96

LAYER 2
ln1_w min/mean/max/std/norm [ 0.3445  0.7343  0.9047  0.0712 14.4574]
ln1_b min/mean/max/std/norm [-0.9417  0.0244  0.7326  0.1315  2.6214]
ln2_w min/mean/max/std/norm [ 0.0775  1.275   1.5018  0.1126 25.082 ]
ln2_b min/mean/max/std/norm [-0.4157  0.0188  0.3635  0.1474  2.911 ]
W_q Frob 37.4829 s[:12] [5.067 4.742 4.456 4.39  4.314 4.26  4.157 4.097 4.047 4.008 3.996 3.96 ] energy_rank 196.23
W_k Frob 36.3557 s[:12] [8.293 7.348 5.059 4.467 4.306 3.912 3.776 3.773 3.73  3.712 3.627 3.611] energy_rank 177.05
W_v Frob 34.7465 s[:12] [3.89  3.824 3.791 3.654 3.644 3.628 3.602 3.582 3.556 3.524 3.494 3.448] energy_rank 207.98
W_o Frob 34.2053 s[:12] [4.442 4.073 3.768 3.455 3.44  3.365 3.35  3.297 3.254 3.233 3.178 3.174] energy_rank 225.28
W_fc Frob 79.4541 s[:12] [14.829  8.048  7.921  7.76   7.66   7.586  7.522  7.509  7.367  7.364  7.178  7.153] energy_rank 269.74
W_proj Frob 77.4765 s[:12] [10.095  8.985  8.075  7.909  7.791  7.656  7.588  7.375  7.212  7.144  7.089  7.034] energy_rank 295.88

LAYER 3
ln1_w min/mean/max/std/norm [ 0.451   0.7694  0.8778  0.0469 15.1046]
ln1_b min/mean/max/std/norm [-0.5141  0.0102  0.6285  0.0945  1.8619]
ln2_w min/mean/max/std/norm [ 0.2619  1.2552  1.4568  0.0888 24.6574]
ln2_b min/mean/max/std/norm [-0.5112  0.014   0.3808  0.1267  2.4972]
W_q Frob 33.9401 s[:12] [4.154 3.946 3.875 3.601 3.521 3.428 3.419 3.379 3.356 3.339 3.32  3.296] energy_rank 217.22
W_k Frob 33.2214 s[:12] [4.888 3.928 3.62  3.437 3.381 3.287 3.263 3.246 3.226 3.208 3.167 3.151] energy_rank 220.73
W_v Frob 38.9389 s[:12] [4.697 4.356 4.255 4.208 4.171 4.131 4.096 4.028 4.024 4.018 4.008 3.95 ] energy_rank 207.96
W_o Frob 39.0173 s[:12] [9.175 6.63  4.442 4.111 4.054 4.04  4.006 3.975 3.959 3.942 3.877 3.859] energy_rank 183.59
W_fc Frob 78.1887 s[:12] [14.373  8.176  7.973  7.83   7.036  6.757  6.711  6.594  6.468  6.387  6.347  6.28 ] energy_rank 301.6
W_proj Frob 85.0885 s[:12] [28.175 20.293 19.292 19.135 18.708 18.57  18.139 17.776 17.485 16.54  15.913 15.61 ] energy_rank 34.74

HEADS: A=Q.T@K; B=O@V (weight products only)
Apos/neg are norms of positive/negative eigenvalues of sym(A); skew is ||A-A.T||/(2||A||).
Btracecos is <B,I>/(||B|| sqrt(384)); Bsym is ||sym(B)||/||B||.
L0H0 QK_norm 6.456 QK_s [5.195 3.389 0.846 0.657 0.509 0.454] QK_er 2.93 Apos/neg [2.134 4.417] skew 0.65 OV_norm 8.667 OV_s [6.898 3.897 1.936 1.638] OV_er 3.53 Btracecos 0.0095 Bsym 0.724 biasQ/K [0.343 0.22 ]
L0H1 QK_norm 32.533 QK_s [17.223 12.122 11.496  9.727  8.594  8.546] QK_er 12.3 Apos/neg [ 9.537 22.911] skew 0.647 OV_norm 17.067 OV_s [4.703 4.573 4.424 4.248] OV_er 23.89 Btracecos -0.0025 Bsym 0.709 biasQ/K [0.669 0.833]
L0H2 QK_norm 9.085 QK_s [4.486 3.573 2.993 2.852 2.588 2.235] QK_er 12.45 Apos/neg [3.969 5.727] skew 0.642 OV_norm 8.592 OV_s [4.786 3.621 3.514 2.916] OV_er 7.88 Btracecos -0.0044 Bsym 0.708 biasQ/K [0.305 0.615]
L0H3 QK_norm 11.468 QK_s [7.675 5.606 3.45  2.355 2.231 1.797] QK_er 6.29 Apos/neg [4.346 7.245] skew 0.676 OV_norm 5.288 OV_s [2.168 1.997 1.875 1.555] OV_er 14.45 Bt
```

## Call 2

```python
import numpy as np
from scipy.linalg import svd, svdvals, eigvals
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=4, suppress=True, linewidth=180)

def centered(a):
    return a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()

print("SPECTRAL TAILS / POSSIBLE CHARACTER-SUBSPACE DIMENSIONS")
for l in [0, 1, 3]:
    for name in ["W_v", "W_o", "W_fc", "W_proj"]:
        s = svdvals(W[f"{name}.{l}"].astype(float))
        print(f"{name}.{l}: first 50 singular values:", np.round(s[:50],3))

As, Bs = [], []
for j in range(6):
    sl = slice(j*64, (j+1)*64)
    D = W["ln1_w.0"].astype(float)
    Q = W["W_q.0"][sl].astype(float) * D
    K = W["W_k.0"][sl].astype(float) * D
    V = W["W_v.0"][sl].astype(float) * D
    O = W["W_o.0"][:,sl].astype(float)
    A = centered(Q.T @ K)
    B = O @ V
    B -= B.mean(axis=1, keepdims=True)
    As.append(A)
    Bs.append(B)
    e = eigvals(A)
    e = e[np.argsort(-np.abs(e))]
    s = svdvals(A)
    print(f"\nL0H{j}, gain-folded, mean-projected QK:")
    print("singular values", np.round(s[:32],4))
    print("eigenvalues", np.round(e[:30],4))
    print("normality defect", round(np.linalg.norm(A@A.T-A.T@A)/np.linalg.norm(A)**2,4))
    print("gain-folded OV singular values", np.round(svdvals(B)[:35],4))

print("\nL0 QK pairwise Frobenius cosines")
print(np.array([[np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)) for b in As] for a in As]))
print("\nL0 QK transpose pairwise Frobenius cosines")
print(np.array([[np.sum(a*b.T)/(np.linalg.norm(a)*np.linalg.norm(b)) for b in As] for a in As]))

print("\nOverlap of QK top-20 joint input subspaces: ||Ui.T Uj||F^2/20")
Us = []
for A in As:
    u, _, _ = svd(np.concatenate([A, A.T], axis=1), full_matrices=False)
    Us.append(u[:,:20])
print(np.array([[np.linalg.norm(u.T@v)**2/20 for v in Us] for u in Us]))

print("\nL0 gain coordinates")
d = W["ln1_w.0"]
print("quantiles", np.quantile(d,[0,.05,.1,.25,.5,.75,.9,.95,1]))
print("largest gains: (coordinate, gain)")
print([(int(i),round(float(d[i]),4)) for i in np.argsort(d)[-35:][::-1]])

```

Output:

```
SPECTRAL TAILS / POSSIBLE CHARACTER-SUBSPACE DIMENSIONS
W_v.0: first 50 singular values: [4.553 4.024 3.738 3.497 3.465 3.367 3.279 3.215 3.136 3.113 3.073 3.02  2.953 2.893 2.867 2.846 2.803 2.739 2.702 2.674 2.589 2.575 2.536 2.466 2.444 2.408 2.246 2.09  2.035
 1.967 1.621 1.492 1.403 1.324 1.211 1.188 1.129 1.112 1.078 1.059 1.042 1.019 0.999 0.99  0.967 0.949 0.923 0.91  0.897 0.881]
W_o.0: first 50 singular values: [6.678 5.237 2.835 2.679 2.62  2.581 2.55  2.515 2.503 2.477 2.449 2.443 2.424 2.404 2.365 2.361 2.319 2.291 2.269 2.26  2.243 2.218 2.2   2.197 2.174 2.14  2.13  2.116 2.099
 2.086 2.039 2.029 2.012 1.995 1.979 1.972 1.934 1.923 1.885 1.876 1.83  1.823 1.816 1.8   1.751 1.736 1.711 1.692 1.654 1.642]
W_fc.0: first 50 singular values: [18.11  12.28   8.473  7.902  7.779  7.442  7.146  7.109  7.008  6.805  6.721  6.54   6.378  6.329  6.27   6.217  6.204  6.035  6.023  5.944  5.845  5.736  5.702  5.668  5.609
  5.58   5.498  5.447  5.377  5.334  5.244  5.203  5.141  5.126  4.99   4.943  4.92   4.846  4.787  4.716  4.669  4.601  4.519  4.439  4.408  4.384  4.307  4.242  4.208  4.081]
W_proj.0: first 50 singular values: [10.932  7.576  6.628  6.293  6.166  5.924  5.886  5.769  5.706  5.61   5.495  5.463  5.404  5.313  5.193  5.161  5.119  5.07   5.003  4.937  4.877  4.793  4.693  4.664  4.651
  4.637  4.548  4.526  4.487  4.404  4.329  4.283  4.247  4.206  4.178  4.141  4.12   4.105  4.062  4.023  3.997  3.958  3.934  3.904  3.857  3.85   3.798  3.758  3.709  3.694]
W_v.1: first 50 singular values: [5.35  5.064 4.979 4.819 4.69  4.585 4.519 4.419 4.328 4.148 4.068 3.934 3.884 3.852 3.788 3.752 3.672 3.621 3.549 3.5   3.444 3.403 3.363 3.338 3.315 3.23  3.226 3.164 3.141
 3.111 3.091 3.042 3.018 3.006 2.947 2.893 2.842 2.826 2.81  2.759 2.723 2.671 2.646 2.625 2.588 2.562 2.48  2.453 2.416 2.371]
W_o.1: first 50 singular values: [4.19  3.651 3.385 3.352 3.24  3.216 3.141 3.115 3.109 3.086 3.032 3.005 2.981 2.95  2.917 2.902 2.877 2.862 2.85  2.829 2.805 2.798 2.78  2.762 2.723 2.71  2.704 2.691 2.678
 2.662 2.656 2.651 2.624 2.612 2.61  2.575 2.559 2.534 2.527 2.52  2.506 2.491 2.472 2.449 2.438 2.429 2.417 2.388 2.359 2.341]
W_fc.1: first 50 singular values: [20.796 14.143 13.826 13.428 13.122 12.883 12.764 12.474 12.289 11.884 11.753 11.47  11.377 11.25  10.909 10.566 10.436 10.111  9.904  9.73   9.478  9.322  9.318  9.088  8.927
  8.89   8.831  8.578  8.515  8.451  8.316  8.109  7.997  7.881  7.799  7.751  7.667  7.492  7.309  7.235  7.18   7.042  6.902  6.669  6.536  6.451  6.256  6.216  6.064  5.966]
W_proj.1: first 50 singular values: [7.625 7.359 6.908 6.489 6.344 6.223 6.192 6.02  5.949 5.902 5.82  5.756 5.728 5.692 5.674 5.65  5.633 5.608 5.581 5.567 5.544 5.523 5.479 5.457 5.45  5.408 5.404 5.39  5.374
 5.341 5.338 5.309 5.285 5.267 5.257 5.223 5.197 5.193 5.179 5.159 5.154 5.127 5.125 5.112 5.094 5.083 5.082 5.072 5.055 5.045]
W_v.3: first 50 singular values: [4.697 4.356 4.255 4.208 4.171 4.131 4.096 4.028 4.024 4.018 4.008 3.95  3.936 3.924 3.906 3.841 3.811 3.784 3.779 3.742 3.732 3.714 3.681 3.673 3.65  3.618 3.588 3.573 3.559
 3.511 3.493 3.492 3.467 3.456 3.433 3.41  3.378 3.365 3.333 3.331 3.321 3.307 3.273 3.24  3.232 3.224 3.212 3.183 3.171 3.137]
W_o.3: first 50 singular values: [9.175 6.63  4.442 4.111 4.054 4.04  4.006 3.975 3.959 3.942 3.877 3.859 3.845 3.81  3.791 3.775 3.76  3.754 3.713 3.7   3.648 3.64  3.615 3.606 3.572 3.563 3.53  3.527 3.513
 3.475 3.455 3.446 3.407 3.389 3.385 3.364 3.346 3.323 3.319 3.298 3.253 3.234 3.221 3.191 3.183 3.178 3.149 3.137 3.12  3.087]
W_fc.3: first 50 singular values: [14.373  8.176  7.973  7.83   7.036  6.757  6.711  6.594  6.468  6.387  6.347  6.28   6.228  6.179  6.162  6.067  5.995  5.987  5.947  5.895  5.854  5.816  5.791  5.764  5.753
  5.699  5.681  5.652  5.641  5.641  5.588  5.564  5.538  5.509  5.46   5.443  5.411  5.386  5.369  5.357  5.328  5.301  5.293  5.278  5.273  5.249  5.231  5.217  5.213  5.178]
W_proj.3: first 50 singular values: [28.175 20.293 19.292 19.135 18.708 18.57  18.139 17.776 17.485 16.54  15.913 15.61  14.975 14.873 14.417 14.313 13.327 12.971 12.593 12.37  12.225 12.019 11.322 11.115 10.769
  9.227  8.573  7.929  6.5    5.271  3.966  3.184  2.792  2.697  2.653  2.555  2.511  2.471  2.379  2.29   2.25   2.215  2.117  2.081  2.036  2.011  1.984  1.969  1.952  1.926]

L0H0, gain-folded, mean-projected QK:
singular values [0.7095 0.5408 0.1064 0.0641 0.0448 0.0396 0.033  0.0327 0.0307 0.0296 0.0279 0.0265 0.0248 0.0235 0.0212 0.0206 0.0201 0.0188 0.0182 0.0173 0.017  0.0164 0.0153 0.0147 0.0142
 0.0138 0.0131 0.0124 0.0118 0.0116 0.0113 0.011 ]
eigenvalues [-0.3921+0.j      0.0484+0.0689j  0.0484-0.0689j -0.023 +0.j     -0.0086+0.0125j -0.0086-0.0125j -0.0126+0.0024j -0.0126-0.0024j -0.0114+0.j     -0.0056-0.0099j -0.0056+0.0099j
 -0.0102+0.j      0.0068+0.0059j  0.0068-0.0059j -0.0082+0.0033j -0.0082-0.0033j  0.0087+0.j     -0.0024+0.0077j -0.0024-0.0077j -0.0067+0.j     -0.0047-0.0047j -0.0047+0.0047j
  0.0047-0.004j   0.0047+0.004j   0.0006+0.0056j  0.0006-0.0056j -0.0052+0.0015j -0.0052-0.0015j  0.0048+0.002j   0.0048-0.002j ]
normality defect 0.8789
gain-folded OV singular values [3.426  2.4694 0.529  0.4008 0.346  0.2503 0.2048 0.1992 0.1777 0.1348 0.1142 0.1006 0.0936 0.0858 0.0791 0.0733 0.0699 0.0604 0.0569 0.0531 0.0515 0.0482 0.0456 0.0421 0.041
 0.0386 0.0372 0.0345 0.0331 0.0304 0.0288 0.0276 0.0261 0.0255 0.0237]

L0H1, gain-folded, mean-projected QK:
singular values [4.8107 2.9852 2.3749 1.692  1.5914 1.2388 1.1421 0.7181 0.6265 0.5788 0.5226 0.4805 0.4647 0.4441 0.4184 0.3977 0.3853 0.337  0.2988 0.2962 0.276  0.2695 0.2125 0.2017 0.1856
 0.1614 0.1463 0.1264 0.115  0.1115 0.0981 0.0911]
eigenvalues [ 1.7423+2.7606j  1.7423-2.7606j -2.2754+0.j     -0.9851+1.1964j -0.9851-1.1964j -0.7783+0.784j  -0.7783-0.784j  -0.2542+0.3298j -0.2542-0.3298j  0.2046+0.2615j  0.2046-0.2615j
 -0.3193+0.j      0.0011+0.256j   0.0011-0.256j  -0.222 +0.0957j -0.222 -0.0957j -0.1893+0.1445j -0
```

## Call 3

```python
import numpy as np
from scipy.linalg import svd, svdvals, eigh
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=4, suppress=True, linewidth=190)
d = 384
P = np.eye(d) - np.ones((d,d))/d

A, B, Qs, Ks, Vs, Os = {}, {}, {}, {}, {}, {}
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        sl = slice(64*j,64*(j+1))
        key = (l,j)
        Qs[key] = (W[f"W_q.{l}"][sl].astype(float)*g) @ P
        Ks[key] = (W[f"W_k.{l}"][sl].astype(float)*g) @ P
        Vs[key] = (W[f"W_v.{l}"][sl].astype(float)*g) @ P
        Os[key] = P @ W[f"W_o.{l}"][:,sl].astype(float)
        A[key] = Qs[key].T @ Ks[key]
        B[key] = Os[key] @ Vs[key]

def basis(M, r):
    return svd(M, full_matrices=False)[0][:,:r]

# Candidate character-reading subspace, inferred only from the value weights
# of the two high-rank first-layer heads.
T = basis(np.concatenate([Vs[0,1].T, Vs[0,5].T], axis=1), 27)
# Candidate positional subspace, inferred from their Q/K weights.
R = basis(np.concatenate([Qs[0,1].T, Ks[0,1].T, Qs[0,5].T, Ks[0,5].T], axis=1), 24)
print("Candidate token/position subspace overlap:", np.linalg.norm(T.T@R)**2/24)
print("L0 fraction of squared matrix norm in candidate token / position input subspaces")
for j in range(6):
    print("H",j,[(n,round(np.linalg.norm(M@T)**2/np.linalg.norm(M)**2,4),
                  round(np.linalg.norm(M@R)**2/np.linalg.norm(M)**2,4))
                 for n,M in [("Q",Qs[0,j]),("K",Ks[0,j]),("V",Vs[0,j])]])

print("\nInduction-path diagnostics, C = A_lj @ B_0h")
print("Each row lists H0..H5 of layer 0.")
print("trace(C)/nuclear_norm(C): positive values favor matching through that earlier OV path.")
for l in range(1,4):
    for j in range(6):
        vals, coupling, tokenvals = [], [], []
        for h in range(6):
            C = A[l,j] @ B[0,h]
            vals.append(np.trace(C)/svdvals(C).sum())
            coupling.append(np.linalg.norm(C)/(np.linalg.norm(A[l,j])*np.linalg.norm(B[0,h])))
            Ct = T.T @ C @ T
            tokenvals.append(np.trace(Ct)/svdvals(Ct).sum())
        print(f"L{l}H{j}", "full",np.round(vals,3),
              "token",np.round(tokenvals,3),"coupling",np.round(coupling,3))

print("\nDirect QK symmetry and identity alignment (gain-folded)")
for l in range(4):
    for j in range(6):
        a=A[l,j]
        s=svdvals(a)
        e=eigh((a+a.T)/2,eigvals_only=True)
        print(f"L{l}H{j}",
              "norm",round(np.linalg.norm(a),3),
              "trace/nuclear",round(np.trace(a)/s.sum(),3),
              "positive_sym_energy",round(np.sum(e[e>0]**2)/np.sum(e**2),3))

print("\nL1 QK energy reading each L0 output subspace (top 24 left singular vectors of OV)")
U = [basis(B[0,h],24) for h in range(6)]
for j in range(6):
    print(f"L1H{j}",
          "Q",np.round([np.linalg.norm(Qs[1,j]@u)**2/np.linalg.norm(Qs[1,j])**2 for u in U],3),
          "K",np.round([np.linalg.norm(Ks[1,j]@u)**2/np.linalg.norm(Ks[1,j])**2 for u in U],3))

```

Output:

```
Candidate token/position subspace overlap: 0.1125481792404922
L0 fraction of squared matrix norm in candidate token / position input subspaces
H 0 [('Q', np.float64(0.0508), np.float64(0.5115)), ('K', np.float64(0.1925), np.float64(0.1732)), ('V', np.float64(0.1226), np.float64(0.5459))]
H 1 [('Q', np.float64(0.1681), np.float64(0.7718)), ('K', np.float64(0.1261), np.float64(0.8999)), ('V', np.float64(0.777), np.float64(0.1205))]
H 2 [('Q', np.float64(0.1872), np.float64(0.4161)), ('K', np.float64(0.2662), np.float64(0.4437)), ('V', np.float64(0.4452), np.float64(0.1402))]
H 3 [('Q', np.float64(0.1296), np.float64(0.4165)), ('K', np.float64(0.2451), np.float64(0.4453)), ('V', np.float64(0.2759), np.float64(0.2609))]
H 4 [('Q', np.float64(0.0831), np.float64(0.1118)), ('K', np.float64(0.1021), np.float64(0.1482)), ('V', np.float64(0.085), np.float64(0.7477))]
H 5 [('Q', np.float64(0.1141), np.float64(0.8357)), ('K', np.float64(0.0968), np.float64(0.9227)), ('V', np.float64(0.725), np.float64(0.1918))]

Induction-path diagnostics, C = A_lj @ B_0h
Each row lists H0..H5 of layer 0.
trace(C)/nuclear_norm(C): positive values favor matching through that earlier OV path.
L1H0 full [ 0.014 -0.    -0.041  0.02   0.039 -0.013] token [-0.321  0.015 -0.123 -0.084 -0.297 -0.046] coupling [0.318 0.057 0.086 0.144 0.395 0.067]
L1H1 full [-0.008  0.06  -0.086  0.022  0.027  0.08 ] token [0.282 0.156 0.041 0.029 0.372 0.208] coupling [0.431 0.03  0.105 0.168 0.545 0.051]
L1H2 full [ 0.007  0.005 -0.091  0.035  0.046 -0.015] token [-0.205  0.004 -0.162 -0.107 -0.071 -0.066] coupling [0.413 0.031 0.101 0.172 0.519 0.048]
L1H3 full [ 0.012 -0.022 -0.069  0.024  0.057  0.068] token [-0.023 -0.066  0.064 -0.033  0.159  0.183] coupling [0.439 0.029 0.109 0.186 0.547 0.049]
L1H4 full [ 0.016  0.018 -0.094  0.044  0.062  0.049] token [ 0.064  0.062 -0.029 -0.072  0.168  0.137] coupling [0.427 0.03  0.108 0.179 0.527 0.052]
L1H5 full [-0.036  0.034 -0.077 -0.002 -0.006  0.062] token [ 0.247  0.07   0.052 -0.074  0.306  0.17 ] coupling [0.401 0.034 0.091 0.144 0.506 0.061]
L2H0 full [ 0.023 -0.002 -0.015  0.036  0.044 -0.015] token [ 0.112 -0.012  0.014  0.022  0.087 -0.086] coupling [0.128 0.052 0.063 0.066 0.16  0.063]
L2H1 full [0.063 0.005 0.057 0.014 0.048 0.212] token [ 0.131  0.017  0.244 -0.012  0.177  0.719] coupling [0.045 0.063 0.059 0.045 0.043 0.076]
L2H2 full [ 0.047 -0.072 -0.028  0.007  0.075 -0.119] token [ 0.04  -0.296 -0.088 -0.06   0.092 -0.479] coupling [0.147 0.047 0.06  0.077 0.183 0.071]
L2H3 full [ 0.064 -0.003 -0.032  0.027  0.13  -0.009] token [-0.116  0.014 -0.093 -0.037  0.017 -0.038] coupling [0.171 0.049 0.063 0.087 0.214 0.06 ]
L2H4 full [ 0.028 -0.023 -0.016  0.042  0.081  0.006] token [-0.18  -0.095  0.005 -0.001 -0.079  0.043] coupling [0.174 0.049 0.092 0.088 0.217 0.061]
L2H5 full [0.047 0.002 0.    0.022 0.093 0.025] token [-0.051  0.011  0.086 -0.019  0.013  0.133] coupling [0.115 0.047 0.065 0.058 0.146 0.054]
L3H0 full [ 0.07  -0.014  0.035  0.039  0.071  0.066] token [ 0.106 -0.06   0.195  0.112 -0.111  0.305] coupling [0.072 0.051 0.058 0.047 0.094 0.057]
L3H1 full [0.065 0.027 0.03  0.007 0.04  0.179] token [0.26  0.122 0.145 0.005 0.188 0.666] coupling [0.057 0.051 0.058 0.048 0.062 0.069]
L3H2 full [ 0.04  -0.012  0.008 -0.004  0.082 -0.004] token [-0.06  -0.045  0.038  0.002 -0.081 -0.016] coupling [0.085 0.05  0.057 0.048 0.114 0.056]
L3H3 full [0.038 0.012 0.025 0.014 0.07  0.004] token [ 0.025  0.064  0.196  0.032 -0.008  0.011] coupling [0.055 0.051 0.058 0.045 0.065 0.05 ]
L3H4 full [ 0.055 -0.025  0.029  0.015  0.074  0.041] token [-0.005 -0.097  0.198  0.025 -0.082  0.203] coupling [0.089 0.05  0.058 0.049 0.117 0.048]
L3H5 full [ 0.06  -0.026  0.002  0.014  0.033  0.187] token [ 0.103 -0.139 -0.004 -0.049  0.069  0.705] coupling [0.039 0.068 0.056 0.046 0.038 0.078]

Direct QK symmetry and identity alignment (gain-folded)
L0H0 norm 0.91 trace/nuclear -0.183 positive_sym_energy 0.276
L0H1 norm 7.02 trace/nuclear -0.157 positive_sym_energy 0.516
L0H2 norm 1.396 trace/nuclear 0.091 positive_sym_energy 0.839
L0H3 norm 1.916 trace/nuclear -0.274 positive_sym_energy 0.21
L0H4 norm 0.074 trace/nuclear -0.064 positive_sym_energy 0.4
L0H5 norm 7.764 trace/nuclear 0.261 positive_sym_energy 0.901
L1H0 norm 5.152 trace/nuclear -0.104 positive_sym_energy 0.439
L1H1 norm 12.675 trace/nuclear -0.22 positive_sym_energy 0.438
L1H2 norm 9.886 trace/nuclear -0.163 positive_sym_energy 0.483
L1H3 norm 10.475 trace/nuclear -0.149 positive_sym_energy 0.514
L1H4 norm 9.993 trace/nuclear -0.129 positive_sym_energy 0.509
L1H5 norm 11.691 trace/nuclear -0.221 positive_sym_energy 0.423
L2H0 norm 16.364 trace/nuclear -0.04 positive_sym_energy 0.471
L2H1 norm 15.563 trace/nuclear 0.178 positive_sym_energy 0.674
L2H2 norm 22.432 trace/nuclear 0.08 positive_sym_energy 0.595
L2H3 norm 15.807 trace/nuclear -0.028 positive_sym_energy 0.486
L2H4 norm 17.612 trace/nuclear 0.004 positive_sym_energy 0.52
L2H5 norm 13.233 trace/nuclear -0.058 positive_sym_energy 0.451
L3H0 norm 16.621 trace/nuclear 0.11 positive_sym_energy 0.606
L3H1 norm 13.558 trace/nuclear 0.171 positive_sym_energy 0.655
L3H2 norm 16.487 trace/nuclear 0.045 positive_sym_energy 0.547
L3H3 norm 8.424 trace/nuclear 0.0 positive_sym_energy 0.51
L3H4 norm 13.875 trace/nuclear 0.0 positive_sym_energy 0.506
L3H5 norm 16.935 trace/nuclear 0.494 positive_sym_energy 0.902

L1 QK energy reading each L0 output subspace (top 24 left singular vectors of OV)
L1H0 Q [0.096 0.163 0.091 0.074 0.123 0.189] K [0.296 0.097 0.096 0.098 0.334 0.121]
L1H1 Q [0.08  0.132 0.059 0.053 0.124 0.182] K [0.567 0.032 0.124 0.132 0.599 0.079]
L1H2 Q [0.09  0.135 0.062 0.06  0.127 0.188] K [0.559 0.039 0.117 0.124 0.602 0.074]
L1H3 Q [0.094 0.128 0.058 0.059 0.14  0.173] K [0.59  0.036 0.118 0.131 0.629 0.078]
L1H4 Q [0.092 0.133 0.061 0.06  0.132 0.19 ] K [0.582 0.038 0.118 0.129 0.621 0.076]
L1H5 Q [0.086 0.137 0.063 0.056 0.126 0.178] K [0.516 0.04  0.103 0.12 
```

## Call 4

```python
import numpy as np
from scipy.linalg import svd, svdvals, eigh, schur
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=4, suppress=True, linewidth=180)
d=384
P=np.eye(d)-np.ones((d,d))/d
g=W["ln1_w.0"].astype(float)
As=[]
for j in range(6):
    sl=slice(64*j,64*(j+1))
    Q=(W["W_q.0"][sl].astype(float)*g)@P
    K=(W["W_k.0"][sl].astype(float)*g)@P
    As.append(Q.T@K)

print("FIRST-LAYER VALUE SPECTRA (raw weights)")
for j in range(6):
    sl=slice(64*j,64*(j+1))
    print(f"H{j}:",np.round(svdvals(W["W_v.0"][sl].astype(float))[:48],3))
V=np.concatenate([W["W_v.0"][64:128],W["W_v.0"][320:384]],axis=0).astype(float)
print("Joint H1/H5 V spectrum:",np.round(svdvals(V)[:65],3))

print("\nPOSITIONAL-OPERATOR PHASE COMPARISON")
print("Only decompositions and products of weights; no positions or token inputs are supplied.")
for r in [16,20,24,28]:
    U=svd(np.concatenate([As[5],As[5].T],axis=1),full_matrices=False)[0][:,:r]
    M=U.T@As[5]@U
    # A shared congruence metric from the symmetric input/output energy.
    ev,evc=eigh((M@M.T+M.T@M)/2)
    G=(evc * np.maximum(ev,1e-10)**(-0.25))@evc.T
    N=G@M@G
    u,_,vt=svd(N)
    R=u@vt
    Z,T=schur(R,output="real")  # R = T @ Z @ T.T
    transformed=[T.T@G@(U.T@a@U)@G@T for a in As]
    print("\nrank",r,"whitened H5 normality",round(np.linalg.norm(N@N.T-N.T@N)/np.linalg.norm(N)**2,4))
    pairs=[]
    k=0
    while k<r:
        if k+1<r and abs(Z[k+1,k])>1e-6:
            vals=[]
            for H in [0,1,2,3,5]:
                block=transformed[H][k:k+2,k:k+2]
                z=(block[0,0]+block[1,1])/2 + 1j*(block[1,0]-block[0,1])/2
                vals.append((float(np.angle(z)),float(abs(z))))
            pairs.append(vals)
            k+=2
        else:
            k+=1
    for vals in sorted(pairs,key=lambda v:abs(v[-1][0])):
        print("H0,H1,H2,H3,H5 (phase radians, amplitude):",np.round(vals,3))
    for h in [0,1,2,3]:
        C=G@(U.T@As[h]@U)@G
        fits=[]
        for power in range(-4,7):
            Rp=np.linalg.matrix_power(R,power)
            fits.append(np.sum(C*Rp)/(np.linalg.norm(C)*np.linalg.norm(Rp)))
        print("H",h,"cosines to H5 polar powers -4..6:",np.round(fits,3))

```

Output:

```
FIRST-LAYER VALUE SPECTRA (raw weights)
H0: [2.214 2.066 1.611 1.282 1.259 1.034 0.964 0.915 0.843 0.762 0.716 0.68  0.652 0.649 0.609 0.606 0.593 0.551 0.54  0.516 0.51  0.498 0.489 0.473 0.455 0.451 0.447 0.434 0.427
 0.413 0.408 0.404 0.398 0.396 0.386 0.378 0.376 0.372 0.367 0.362 0.357 0.349 0.348 0.34  0.337 0.329 0.327 0.321]
H1: [2.25  2.127 2.089 2.055 1.991 1.973 1.958 1.933 1.869 1.832 1.808 1.785 1.717 1.69  1.639 1.611 1.584 1.554 1.496 1.48  1.419 1.37  1.277 1.193 1.14  1.094 1.028 0.906 0.744
 0.693 0.682 0.659 0.638 0.587 0.538 0.527 0.495 0.482 0.469 0.452 0.448 0.434 0.415 0.408 0.398 0.386 0.383 0.372]
H2: [2.164 2.047 2.012 1.723 1.514 1.361 1.273 1.115 1.036 0.993 0.908 0.857 0.765 0.74  0.681 0.628 0.618 0.592 0.58  0.57  0.535 0.526 0.512 0.477 0.471 0.46  0.458 0.445 0.444
 0.433 0.423 0.409 0.408 0.401 0.397 0.392 0.381 0.374 0.366 0.361 0.356 0.35  0.347 0.342 0.34  0.334 0.33  0.325]
H3: [1.443 1.28  1.237 1.154 1.129 1.076 1.038 1.013 0.958 0.917 0.897 0.841 0.816 0.734 0.696 0.67  0.64  0.604 0.581 0.568 0.529 0.519 0.513 0.475 0.471 0.464 0.457 0.446 0.432
 0.417 0.414 0.404 0.395 0.391 0.386 0.379 0.372 0.368 0.364 0.36  0.356 0.353 0.349 0.339 0.332 0.329 0.323 0.319]
H4: [4.102 3.344 2.539 2.115 1.785 1.548 1.482 1.422 1.387 1.323 1.196 1.164 1.145 1.022 0.967 0.848 0.832 0.801 0.7   0.665 0.644 0.637 0.616 0.566 0.56  0.535 0.529 0.494 0.474
 0.464 0.447 0.445 0.433 0.427 0.413 0.408 0.403 0.399 0.396 0.383 0.38  0.376 0.368 0.363 0.347 0.342 0.335 0.331]
H5: [2.397 2.282 2.191 2.148 2.082 2.049 2.033 1.995 1.901 1.88  1.86  1.788 1.772 1.744 1.703 1.646 1.591 1.538 1.448 1.384 1.317 1.2   1.042 0.923 0.892 0.853 0.817 0.784 0.767
 0.744 0.737 0.722 0.709 0.681 0.663 0.654 0.627 0.599 0.579 0.571 0.565 0.533 0.504 0.49  0.452 0.438 0.434 0.405]
Joint H1/H5 V spectrum: [3.023 2.923 2.898 2.843 2.791 2.735 2.708 2.661 2.614 2.601 2.519 2.44  2.421 2.379 2.364 2.311 2.266 2.214 2.194 2.105 2.019 1.897 1.838 1.726 1.497 1.42  1.348 1.111 1.
 0.935 0.918 0.89  0.865 0.854 0.821 0.81  0.793 0.779 0.76  0.725 0.721 0.702 0.693 0.676 0.64  0.619 0.601 0.546 0.54  0.534 0.524 0.514 0.504 0.494 0.483 0.476 0.469 0.465
 0.464 0.444 0.438 0.434 0.425 0.422 0.413]

POSITIONAL-OPERATOR PHASE COMPARISON
Only decompositions and products of weights; no positions or token inputs are supplied.

rank 16 whitened H5 normality 0.1562
H0,H1,H2,H3,H5 (phase radians, amplitude): [[-2.412  0.003]
 [ 0.567  0.144]
 [ 3.132  0.013]
 [-2.964  0.013]
 [ 0.144  0.931]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[-2.356  0.003]
 [ 1.801  0.171]
 [-2.256  0.009]
 [-2.201  0.003]
 [ 0.614  0.953]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[ 1.951  0.008]
 [ 1.359  0.326]
 [-0.344  0.042]
 [-2.991  0.035]
 [ 0.921  0.984]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[ 1.599  0.029]
 [ 1.13   0.773]
 [-0.241  0.131]
 [-3.097  0.115]
 [ 0.937  0.993]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[-1.573  0.004]
 [-2.243  0.232]
 [ 0.261  0.022]
 [-3.037  0.024]
 [-1.115  0.962]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[ 2.284  0.001]
 [ 2.933  0.236]
 [ 1.663  0.006]
 [-2.704  0.003]
 [-1.439  0.981]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[-3.072  0.001]
 [ 2.709  0.334]
 [-1.805  0.005]
 [ 3.066  0.015]
 [-1.684  0.91 ]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[ 2.487  0.002]
 [-2.44   0.716]
 [ 2.72   0.005]
 [ 2.661  0.006]
 [ 1.975  0.921]]
H 0 cosines to H5 polar powers -4..6: [ 0.14  -0.046 -0.18  -0.202 -0.062  0.144  0.192  0.046 -0.13  -0.212 -0.138]
H 1 cosines to H5 polar powers -4..6: [ 0.102 -0.313 -0.144  0.001 -0.158  0.202  0.555  0.043 -0.36  -0.108  0.006]
H 2 cosines to H5 polar powers -4..6: [-0.291 -0.272 -0.021  0.216  0.256  0.075 -0.204 -0.335 -0.185  0.087  0.258]
H 3 cosines to H5 polar powers -4..6: [ 0.154  0.229  0.119 -0.158 -0.318 -0.173  0.083  0.205  0.161 -0.003 -0.193]

rank 20 whitened H5 normality 0.1079
H0,H1,H2,H3,H5 (phase radians, amplitude): [[ 1.542  0.004]
 [-0.434  0.216]
 [ 3.029  0.007]
 [ 2.658  0.015]
 [-0.088  0.961]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[-1.078  0.002]
 [ 1.375  0.185]
 [-2.449  0.008]
 [-1.855  0.013]
 [ 0.544  0.984]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[-1.432  0.003]
 [-1.472  0.273]
 [ 0.553  0.029]
 [-2.601  0.026]
 [-0.838  0.981]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[-1.631  0.036]
 [-1.085  0.986]
 [ 0.271  0.166]
 [ 3.126  0.148]
 [-0.948  0.991]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[ 2.729  0.001]
 [ 2.407  0.184]
 [-0.839  0.004]
 [-2.632  0.007]
 [ 1.026  0.987]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[ 2.444  0.001]
 [ 2.956  0.18 ]
 [-1.902  0.004]
 [ 2.189  0.005]
 [ 1.382  0.984]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[-2.517  0.002]
 [ 2.964  0.239]
 [-2.972  0.002]
 [-3.069  0.01 ]
 [-1.56   0.965]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[-2.479  0.012]
 [ 2.349  0.783]
 [-2.041  0.006]
 [-1.764  0.019]
 [-1.851  0.981]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[ 2.851  0.012]
 [ 2.347  0.854]
 [-0.712  0.005]
 [ 1.681  0.014]
 [-1.984  0.975]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[ 2.23   0.004]
 [-2.317  0.857]
 [ 1.633  0.002]
 [ 2.207  0.008]
 [ 1.998  0.952]]
H 0 cosines to H5 polar powers -4..6: [ 0.058 -0.073 -0.032 -0.114 -0.105  0.169  0.193 -0.053 -0.082 -0.066 -0.165]
H 1 cosines to H5 polar powers -4..6: [ 0.24  -0.435 -0.123  0.235 -0.229  0.071  0.644 -0.026 -0.423  0.131  0.035]
H 2 cosines to H5 polar powers -4..6: [-0.267 -0.199  0.015  0.191  0.227  0.068 -0.177 -0.276 -0.159  0.059  0.233]
H 3 cosines to H5 polar powers -4..6: [ 0.14   0.189  0.073 -0.15  -0.245 -0.118  0.057  0.141  0.131 -0.004 -0.182]

rank 24 whitened H5 normality 0.1536
H0,H1,H2,H3,H5 (phase radians, amplitude): [[ 0.666  0.008]
 [-0.616  0.293]
 [ 0.862  0.009]
 [ 2.521  0.028]
 [-0.269  0.919]]
H0,H1,H2,H3,H5 (phase radians, amplitude): [[ 0.38   0.003]
 [-1.397  0.197]
 [ 2.027  0.004]
 [ 2.061  0.014]
 [-0.561  0.975]]
H0,H1,H2,H3,H5 (pha
```

## Call 5

```python
import numpy as np
from scipy.linalg import svd, eigh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=3,suppress=True,linewidth=180)

d=384
P=np.eye(d)-np.ones((d,d))/d
g=W["ln1_w.0"].astype(float)
V=[]
B=[]
for j in [1,5]:
    sl=slice(j*64,(j+1)*64)
    v=(W["W_v.0"][sl].astype(float)*g)@P
    V.append(v)
    B.append(P@W["W_o.0"][:,sl].astype(float)@v)
T=svd(np.concatenate([v.T for v in V],axis=1),full_matrices=False)[0][:,:27]
F=(W["W_fc.0"].astype(float)*W["ln2_w.0"])@P

# Read directions in MLP weights, pulled back through earlier OV matrices.
# These are matrix products, not MLP or attention evaluations.
groups=[F@T,F@B[1]@T,F@B[0]@T]
names=["direct","via_H5","via_H1"]
groups=[x/np.sqrt(np.mean(x*x)) for x in groups]
X=np.concatenate(groups)
X-=X.mean(axis=0)
ev,ec=eigh(X.T@X/len(X))
whitener=(ec*(1/np.sqrt(np.maximum(ev,1e-8))))@ec.T
Y=X@whitener
Y/=np.maximum(np.linalg.norm(Y,axis=1,keepdims=True),1e-12)

for k in [20,24,28,32,40]:
    km=KMeans(n_clusters=k,n_init=6,random_state=19,max_iter=300).fit(Y)
    centers=km.cluster_centers_
    centers/=np.linalg.norm(centers,axis=1,keepdims=True)
    cos=Y@centers.T
    best=cos.max(axis=1)
    print("clusters",k,"best-cosine quantiles",np.round(np.quantile(best,[.1,.25,.5,.75,.9,.95]),3))
    if k==28:
        labels=cos.argmax(axis=1).reshape(3,1536)
        confidence=best.reshape(3,1536)
        for i,n in enumerate(names):
            print(n,"cluster counts",np.bincount(labels[i],minlength=28))
        print("center nearest-neighbor cosine quantiles",
              np.quantile((centers@centers.T-2*np.eye(k)).max(axis=1),[0,.25,.5,.75,1]))
        print("Strongly assigned direct/via_H5 pairs (counts, confidence > .55)")
        G=np.zeros((28,28))
        for n in range(1536):
            if min(confidence[0,n],confidence[1,n])>.55:
                G[labels[1,n],labels[0,n]]+=1
        edges=sorted([(G[i,j],i,j) for i in range(28) for j in range(28)],reverse=True)
        print(edges[:70])
        print("qualifying neurons",G.sum(),"diagonal",np.trace(G))
        print("Weighted graph, all neurons:")
        H=np.zeros((28,28))
        for n in range(1536):
            H[labels[1,n],labels[0,n]]+=confidence[0,n]*confidence[1,n]
        print(np.round(H,1))

```

Output:

```
clusters 20 best-cosine quantiles [0.327 0.395 0.476 0.568 0.654 0.708]
clusters 24 best-cosine quantiles [0.343 0.412 0.494 0.582 0.667 0.716]
clusters 28 best-cosine quantiles [0.361 0.427 0.512 0.6   0.684 0.741]
direct cluster counts [ 43  17 156  49  23  63  64  34  24  57  33  24  13  20 209 155  35  22  47  59  50   6   2  50   8 119  61  93]
via_H5 cluster counts [ 53   3  22  42  33  21  20   2  41  40 171   2 184 191  12  36  33   2  40  15  58 186 189  46   9  23  47  15]
via_H1 cluster counts [ 27 215  16  36  73  38  12  99  36  30   6 172   7   9  13  10 138 169  41  23  42   5   6  39 181  19  24  50]
center nearest-neighbor cosine quantiles [0.095 0.172 0.265 0.434 0.607]
Strongly assigned direct/via_H5 pairs (counts, confidence > .55)
[(np.float64(12.0), 13, 14), (np.float64(12.0), 12, 14), (np.float64(8.0), 21, 15), (np.float64(8.0), 21, 14), (np.float64(7.0), 22, 15), (np.float64(7.0), 22, 14), (np.float64(7.0), 21, 25), (np.float64(6.0), 21, 2), (np.float64(6.0), 12, 15), (np.float64(5.0), 22, 3), (np.float64(5.0), 12, 2), (np.float64(5.0), 10, 14), (np.float64(4.0), 26, 26), (np.float64(4.0), 22, 2), (np.float64(4.0), 21, 27), (np.float64(4.0), 21, 20), (np.float64(4.0), 13, 15), (np.float64(4.0), 10, 15), (np.float64(4.0), 6, 6), (np.float64(3.0), 16, 16), (np.float64(3.0), 13, 2), (np.float64(3.0), 12, 25), (np.float64(3.0), 10, 2), (np.float64(3.0), 10, 0), (np.float64(3.0), 4, 11), (np.float64(3.0), 0, 0), (np.float64(2.0), 22, 27), (np.float64(2.0), 22, 26), (np.float64(2.0), 22, 9), (np.float64(2.0), 22, 5), (np.float64(2.0), 21, 23), (np.float64(2.0), 21, 5), (np.float64(2.0), 18, 6), (np.float64(2.0), 13, 25), (np.float64(2.0), 13, 9), (np.float64(2.0), 12, 19), (np.float64(2.0), 12, 8), (np.float64(2.0), 12, 6), (np.float64(2.0), 12, 5), (np.float64(2.0), 10, 25), (np.float64(2.0), 10, 5), (np.float64(2.0), 6, 14), (np.float64(2.0), 3, 14), (np.float64(2.0), 3, 3), (np.float64(2.0), 0, 25), (np.float64(2.0), 0, 8), (np.float64(1.0), 26, 27), (np.float64(1.0), 26, 4), (np.float64(1.0), 25, 25), (np.float64(1.0), 25, 2), (np.float64(1.0), 23, 23), (np.float64(1.0), 23, 14), (np.float64(1.0), 22, 25), (np.float64(1.0), 22, 23), (np.float64(1.0), 22, 19), (np.float64(1.0), 22, 18), (np.float64(1.0), 22, 7), (np.float64(1.0), 22, 4), (np.float64(1.0), 21, 26), (np.float64(1.0), 21, 9), (np.float64(1.0), 21, 7), (np.float64(1.0), 21, 3), (np.float64(1.0), 19, 19), (np.float64(1.0), 19, 14), (np.float64(1.0), 19, 6), (np.float64(1.0), 18, 17), (np.float64(1.0), 18, 2), (np.float64(1.0), 15, 2), (np.float64(1.0), 14, 14), (np.float64(1.0), 13, 26)]
qualifying neurons 224.0 diagonal 23.0
Weighted graph, all neurons:
[[ 1.7  0.   1.   0.2  0.   0.1  0.8  0.8  0.9  0.5  0.   0.7  0.   0.3  1.2  0.   0.   0.2  0.   0.6  0.5  0.1  0.   1.1  0.   1.6  0.5  1. ]
 [ 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.1  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.2  0.   0.   0.3]
 [ 0.   0.   0.3  0.2  0.   0.2  0.8  0.   0.   0.2  0.2  0.5  0.   0.   0.   0.6  0.3  0.2  0.   0.3  0.   0.1  0.   0.   0.5  0.   0.2  0. ]
 [ 0.1  0.2  1.7  2.1  0.4  0.   0.3  0.   0.   0.6  0.   0.   0.   0.   1.6  0.5  0.3  0.   0.5  1.2  0.   0.   0.   0.3  0.   0.8  0.   0.6]
 [ 0.   0.   0.4  0.   0.5  0.   0.6  0.   0.6  0.   0.   2.3  0.2  0.   0.3  1.7  0.2  0.1  0.5  0.1  1.1  0.   0.   0.   0.   0.   0.2  0. ]
 [ 0.2  0.   0.3  0.2  0.1  2.   0.2  0.2  0.   0.   0.   0.   0.2  0.   0.   0.4  0.1  0.   0.   0.   0.   0.   0.   0.1  0.   0.4  0.   0. ]
 [ 0.   0.   0.   0.3  0.   0.2  3.6  0.   0.   0.2  0.   0.   0.   0.   1.4  0.   0.   0.3  0.   0.3  0.   0.   0.   0.   0.   0.4  0.   0. ]
 [ 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.3  0.   0.   0.   0.   0.   0.   0.   0.2]
 [ 1.6  0.   0.4  0.4  0.   0.3  1.   0.1  0.   0.3  0.4  0.1  0.1  0.2  0.4  0.8  0.2  0.2  0.4  0.2  0.6  0.   0.   0.7  0.   0.1  0.6  0.2]
 [ 0.8  0.   0.4  0.   0.   0.   1.1  0.   1.   2.3  0.5  0.   0.   0.1  0.4  0.2  0.1  0.   0.   0.3  0.5  0.4  0.   0.2  0.2  0.9  0.5  0.3]
 [ 2.2  0.6  5.4  0.4  1.2  1.7  1.   0.7  0.9  1.   0.5  0.5  0.1  0.   8.   9.6  0.5  0.   1.7  1.1  0.4  0.   0.   0.2  0.1  4.5  2.1  3.1]
 [ 0.   0.   0.2  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.3  0.   0. ]
 [ 0.2  0.8  6.7  1.6  0.6  2.4  1.   0.9  2.3  0.6  1.4  0.3  0.   0.   9.7  5.4  0.   0.5  3.4  2.2  1.3  0.   0.3  1.3  0.   3.8  2.4  4.3]
 [ 1.   1.6  4.1  1.1  0.2  2.1  1.1  1.3  0.6  2.3  2.3  0.   0.3  0.6 11.2  5.1  0.5  0.4  1.5  3.   1.3  0.3  0.   1.1  0.3  5.2  1.9  1.7]
 [ 0.   0.   0.   0.   0.   0.   0.4  0.   0.   0.   0.   0.   0.   0.   0.7  0.   0.   0.   0.   0.   0.2  0.   0.   0.2  0.   0.2  0.3  0.8]
 [ 0.   0.   0.6  1.   1.5  0.2  0.   0.   0.3  0.   0.2  0.2  0.2  0.2  0.4  0.7  0.   0.3  0.   0.6  0.4  0.2  0.   0.5  0.   0.2  0.4  0.9]
 [ 0.1  0.   0.4  0.   0.   0.   0.4  0.   0.   0.3  0.   0.   0.3  0.   0.5  1.2  3.2  0.5  0.   0.4  0.   0.   0.   0.   0.   0.3  0.3  0. ]
 [ 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.8  0.   0.   0.   0.   0.   0.   0. ]
 [ 0.   0.   0.4  0.2  0.3  0.   2.3  0.   0.3  0.   0.6  0.1  0.2  0.   0.6  1.7  0.7  1.1  1.   0.6  0.2  0.   0.   0.   0.   0.   0.   0. ]
 [ 0.   0.   0.6  0.   0.   0.   0.6  0.   0.   0.   0.   0.   0.   0.   0.8  0.   0.   0.1  0.   1.4  0.   0.   0.   0.   0.   0.3  0.   0.2]
 [ 0.6  0.2  3.2  0.   0.2  0.2  1.   0.3  0.   0.3  0.4  0.   0.   0.   4.1  0.6  0.2  0.   0.5  0.4  0.3  0.   0.   0.6  0.   0.7  0.   0. ]
 [ 0.1  1.  10.   1.4  0.5  2.7  0.5  1.5  0.   1.6  0.3  0.4  0.6  0.3  6.2  8.7  0.   0.2  2.   0.3  3.7  0.   0.   1.3  0.   5.   1.7  5.7]
 [ 0.7  0.   4.9  3.9  0.3  2.8  1.4  1.8  0.8  1.9  1.   0.3  0.6  0.9 10.1  8.   0.5  0.3  0.7  1.   1.1  0.   0.   3.1  0.   3.6  2.7  3.5]
 [ 1.5  0.   1.   0.3  0. 
```

## Call 6

```python
import numpy as np
from scipy.linalg import svd, eigh, qr
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=3,suppress=True,linewidth=170)

d=384
P=np.eye(d)-np.ones((d,d))/d
g=W["ln1_w.0"].astype(float)
Vraw=np.concatenate([W["W_v.0"][64:128],W["W_v.0"][320:384]],axis=0).astype(float)
_,s,Vt=svd(Vraw,full_matrices=False)
F=(W["W_fc.0"].astype(float)*W["ln2_w.0"])@P
Bs=[]
for j in [5,1]:
    sl=slice(j*64,(j+1)*64)
    Bs.append(P@W["W_o.0"][:,sl].astype(float)@(W["W_v.0"][sl].astype(float)*g)@P)

group_ids=np.repeat(np.arange(3),1536)
for power in [-1,0,1]:
    for rank in [24,27,32,40]:
        candidate=P@(Vt[:rank].T*g[:,None]**power)
        T=qr(candidate,mode="economic")[0]
        groups=[F@T]+[F@b@T for b in Bs]
        groups=[(x-x.mean(axis=0))/np.sqrt(np.mean((x-x.mean(axis=0))**2)) for x in groups]
        X=np.concatenate(groups)
        ev,ec=eigh(X.T@X/len(X))
        Y=X@((ec/np.sqrt(np.maximum(ev,1e-8)))@ec.T)
        Y/=np.maximum(np.linalg.norm(Y,axis=1,keepdims=True),1e-12)
        km=KMeans(n_clusters=28,n_init=3,random_state=19,max_iter=250).fit(Y)
        C=km.cluster_centers_
        C/=np.linalg.norm(C,axis=1,keepdims=True)
        sims=Y@C.T
        labs=sims.argmax(axis=1)
        print("unscaling power",power,"rank",rank,
              "cos quantiles",np.round(np.quantile(sims.max(axis=1),[.25,.5,.75,.9]),3),
              "group-label MI",round(mutual_info_score(group_ids,labs),4))
        if power==-1 and rank==27:
            print("Cluster populations by direct / H5 / H1:")
            print(np.array([np.bincount(labs[i*1536:(i+1)*1536],minlength=28) for i in range(3)]))
            print("Joint V singular values",np.round(s[:35],3))

```

Output:

```
unscaling power -1 rank 24 cos quantiles [0.437 0.518 0.606 0.694] group-label MI 0.379
unscaling power -1 rank 27 cos quantiles [0.423 0.506 0.592 0.673] group-label MI 0.4437
Cluster populations by direct / H5 / H1:
[[ 44  40   5  31 185   9  14  20  28 199  41  31  39  12  25   9 189  31  47 205  33   7  39  40  56  37   7 113]
 [ 95  39 172  87  13 192  31 195   0   1   5   9  74   5  89   6  20  11  33   8  79 126  58  19  67  88   1  13]
 [ 28  41  10  41  17  15 112   8 141   2  80 115  44 192  28 150  16  39  17  11  53  25  23  83  27  45 161  12]]
Joint V singular values [3.023 2.923 2.898 2.843 2.791 2.735 2.708 2.661 2.614 2.601 2.519 2.44  2.421 2.379 2.364 2.311 2.266 2.214 2.194 2.105 2.019 1.897 1.838 1.726 1.497 1.42  1.348 1.111
 1.    0.935 0.918 0.89  0.865 0.854 0.821]
unscaling power -1 rank 32 cos quantiles [0.408 0.499 0.6   0.686] group-label MI 0.5319
unscaling power -1 rank 40 cos quantiles [0.371 0.463 0.556 0.637] group-label MI 0.493
unscaling power 0 rank 24 cos quantiles [0.437 0.518 0.612 0.706] group-label MI 0.3362
unscaling power 0 rank 27 cos quantiles [0.421 0.502 0.595 0.683] group-label MI 0.3728
unscaling power 0 rank 32 cos quantiles [0.407 0.513 0.616 0.707] group-label MI 0.5262
unscaling power 0 rank 40 cos quantiles [0.369 0.469 0.569 0.648] group-label MI 0.4776
unscaling power 1 rank 24 cos quantiles [0.435 0.516 0.608 0.7  ] group-label MI 0.3052
unscaling power 1 rank 27 cos quantiles [0.42  0.496 0.587 0.675] group-label MI 0.369
unscaling power 1 rank 32 cos quantiles [0.402 0.505 0.611 0.694] group-label MI 0.4522
unscaling power 1 rank 40 cos quantiles [0.369 0.468 0.561 0.641] group-label MI 0.4931

```

## Call 7

```python
import numpy as np
from scipy.linalg import svd, qr, eigh
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=3,suppress=True,linewidth=170)

d=384
P=np.eye(d)-np.ones((d,d))/d
g=W["ln1_w.0"].astype(float)
V=np.concatenate([W["W_v.0"][64:128],W["W_v.0"][320:384]],axis=0).astype(float)
R=np.concatenate([W[name][sl] for name in ["W_q.0","W_k.0"]
                  for sl in [slice(64,128),slice(320,384)]],axis=0).astype(float)
Ur,sr,Vr=svd(R,full_matrices=False)
F=(W["W_fc.0"].astype(float)*W["ln2_w.0"])@P
Bs=[]
for j in [5,1]:
    sl=slice(j*64,(j+1)*64)
    Bs.append(P@W["W_o.0"][:,sl].astype(float)@(W["W_v.0"][sl].astype(float)*g)@P)
groupids=np.repeat(np.arange(3),1536)

print("Joint positional Q/K spectrum",np.round(sr[:60],3))
for rp in [20,24,28,36]:
    Upos=Vr[:rp].T
    Vclean=V-(V@Upos)@Upos.T
    _,sc,Vc=svd(Vclean,full_matrices=False)
    print("\nRemoved positional rank",rp,"value singular values",np.round(sc[:40],3))
    for method in ["raw","unscaled","unscaled_orthogonal"]:
        candidate=P@(Vc[:27].T/(g[:,None] if method!="raw" else 1))
        if method=="unscaled_orthogonal":
            Rpos=qr(P@(Upos/g[:,None]),mode="economic")[0]
            candidate-=Rpos@(Rpos.T@candidate)
        T=qr(candidate,mode="economic")[0]
        groups=[F@T]+[F@b@T for b in Bs]
        groups=[(x-x.mean(axis=0))/np.sqrt(np.mean((x-x.mean(axis=0))**2)) for x in groups]
        X=np.concatenate(groups)
        ev,ec=eigh(X.T@X/len(X))
        Y=X@((ec/np.sqrt(np.maximum(ev,1e-8)))@ec.T)
        Y/=np.linalg.norm(Y,axis=1,keepdims=True)
        km=KMeans(n_clusters=28,n_init=3,random_state=19,max_iter=250).fit(Y)
        C=km.cluster_centers_
        C/=np.linalg.norm(C,axis=1,keepdims=True)
        sims=Y@C.T
        labs=sims.argmax(axis=1)
        print(method,"cos quartiles",np.round(np.quantile(sims.max(axis=1),[.25,.5,.75,.9]),3),
              "group-label MI",round(mutual_info_score(groupids,labs),4))
        if rp==28 and method=="unscaled_orthogonal":
            print("Cluster populations:",np.array([np.bincount(labs[i*1536:(i+1)*1536],minlength=28) for i in range(3)]))

```

Output:

```
Joint positional Q/K spectrum [6.618 6.557 6.285 6.235 5.884 5.742 5.553 5.475 5.437 5.399 5.352 5.153 5.099 4.964 4.906 4.855 4.739 4.589 4.365 4.327 4.172 3.56  3.538 3.165 3.022 2.79  2.757 2.617
 2.534 2.524 2.421 2.345 2.266 2.196 2.118 2.044 2.01  1.964 1.908 1.895 1.83  1.796 1.745 1.706 1.669 1.657 1.647 1.609 1.529 1.488 1.409 1.394 1.381 1.362 1.324 1.309
 1.267 1.261 1.219 1.191]

Removed positional rank 20 value singular values [3.012 2.902 2.886 2.822 2.779 2.719 2.674 2.645 2.593 2.584 2.503 2.426 2.405 2.356 2.35  2.283 2.232 2.184 2.164 2.08  2.008 1.854 1.802 1.701 1.468 1.389 1.313 0.842
 0.777 0.742 0.677 0.667 0.654 0.63  0.62  0.605 0.583 0.557 0.547 0.527]
raw cos quartiles [0.422 0.503 0.596 0.692] group-label MI 0.3819
unscaled cos quartiles [0.424 0.507 0.593 0.678] group-label MI 0.4391
unscaled_orthogonal cos quartiles [0.422 0.507 0.602 0.683] group-label MI 0.4321

Removed positional rank 24 value singular values [3.008 2.899 2.881 2.818 2.772 2.715 2.653 2.639 2.591 2.582 2.494 2.417 2.4   2.348 2.344 2.271 2.214 2.177 2.14  2.074 2.001 1.85  1.782 1.678 1.456 1.382 1.304 0.758
 0.715 0.66  0.653 0.636 0.61  0.59  0.577 0.559 0.544 0.535 0.529 0.517]
raw cos quartiles [0.424 0.507 0.598 0.691] group-label MI 0.3927
unscaled cos quartiles [0.427 0.507 0.598 0.681] group-label MI 0.426
unscaled_orthogonal cos quartiles [0.423 0.507 0.598 0.68 ] group-label MI 0.4223

Removed positional rank 28 value singular values [2.979 2.893 2.876 2.782 2.727 2.673 2.637 2.622 2.58  2.564 2.492 2.382 2.36  2.328 2.313 2.217 2.176 2.121 2.071 1.987 1.838 1.773 1.718 1.468 1.364 1.281 0.9   0.724
 0.659 0.649 0.609 0.591 0.586 0.557 0.552 0.546 0.53  0.521 0.513 0.504]
raw cos quartiles [0.428 0.519 0.62  0.707] group-label MI 0.4574
unscaled cos quartiles [0.427 0.512 0.602 0.685] group-label MI 0.433
unscaled_orthogonal cos quartiles [0.431 0.513 0.604 0.689] group-label MI 0.4433
Cluster populations: [[ 31  39 219  11  22  13  17  36  32 185  22  29  48  32   3 107  33 122  35  34  12  30 299  37   7  18  28  35]
 [ 60   3  18 163  79 129   3  41 104   3   5 109  42  95 212   4  76   9  51  39   0  10   3  36  29 125  48  40]
 [ 29 121   9   2  72  24 197  71  20  13 151  27  26  34   7   1  64  28  55  54 205  36   0  65 119  35  46  25]]

Removed positional rank 36 value singular values [2.945 2.834 2.776 2.702 2.657 2.618 2.596 2.562 2.527 2.424 2.408 2.3   2.278 2.172 2.106 2.101 2.018 1.849 1.826 1.716 1.435 1.313 1.22  0.87  0.768 0.717 0.682 0.639
 0.61  0.598 0.571 0.562 0.557 0.529 0.522 0.513 0.509 0.5   0.49  0.483]
raw cos quartiles [0.424 0.514 0.615 0.705] group-label MI 0.4827
unscaled cos quartiles [0.429 0.511 0.607 0.694] group-label MI 0.4837
unscaled_orthogonal cos quartiles [0.438 0.52  0.613 0.701] group-label MI 0.5204

```

## Call 8

```python
import numpy as np
from scipy.linalg import svd, svdvals, qr
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=3,suppress=True,linewidth=190)
d=384
P=np.eye(d)-np.ones((d,d))/d
A,B,Q,K,V,O={},{},{},{},{},{}
for l in range(4):
    g=W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        sl=slice(j*64,(j+1)*64)
        key=(l,j)
        Q[key]=(W[f"W_q.{l}"][sl].astype(float)*g)@P
        K[key]=(W[f"W_k.{l}"][sl].astype(float)*g)@P
        V[key]=(W[f"W_v.{l}"][sl].astype(float)*g)@P
        O[key]=P@W[f"W_o.{l}"][:,sl].astype(float)
        A[key]=Q[key].T@K[key]
        B[key]=O[key]@V[key]
def basis(x,r):
    return svd(x,full_matrices=False)[0][:,:r]
def frac(M,U):
    return np.linalg.norm(M@U)**2/np.linalg.norm(M)**2
def cosine(x,y):
    return np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y))
T=basis(np.concatenate([V[0,1].T,V[0,5].T],axis=1),27)
R=basis(np.concatenate([Q[0,1].T,K[0,1].T,Q[0,5].T,K[0,5].T],axis=1),24)
clock=basis(B[0,4],2)
pos=qr(np.concatenate([R,clock,basis(B[0,0],2)],axis=1),mode="economic")[0]
token=qr(np.concatenate([T,basis(B[0,1],27),basis(B[0,5],27)],axis=1),mode="economic")[0]
out=basis(P@W["W_proj.3"].astype(float),27)

print("Reader energy fractions in weight-inferred positional (28D), token/context (81D), and clock (2D) spaces")
for l in range(1,4):
    for j in range(6):
        key=(l,j)
        sl=slice(j*64,(j+1)*64)
        qb=W[f"W_q.{l}"][sl].astype(float)@W[f"ln1_b.{l}"].astype(float)
        pref=K[key].T@qb
        print(f"L{l}H{j}",
              "Q(pos,tok,clock)",np.round([frac(Q[key],pos),frac(Q[key],token),frac(Q[key],clock)],3),
              "K(pos,tok,clock)",np.round([frac(K[key],pos),frac(K[key],token),frac(K[key],clock)],3),
              "bias-key clock frac",round(np.linalg.norm(clock.T@pref)**2/np.linalg.norm(pref)**2,3))

print("\nKey-side versus query-side composition with L0H5, restricted to T")
for l in range(1,4):
    for j in range(6):
        Ck=T.T@A[l,j]@B[0,5]@T
        Cq=T.T@B[0,5].T@A[l,j]@T
        z=lambda M: np.trace(M)/svdvals(M).sum()
        print(f"L{l}H{j}","key",round(z(Ck),3),"query",round(z(Cq),3),
              "V token frac",round(frac(V[l,j],T),3),
              "OV token output-space frac",round(np.linalg.norm(out.T@B[l,j]@T)**2/np.linalg.norm(B[l,j]@T)**2,3))

print("\nAgreement of token OV maps with candidate copying heads L2H1, L3H1, L3H5")
refs=[(2,1),(3,1),(3,5)]
for l in range(1,4):
    for j in range(6):
        M=out.T@B[l,j]@T
        print(f"L{l}H{j}",np.round([cosine(M,out.T@B[k]@T) for k in refs],3))

print("\nL1 pairwise QK cosine (full / positional restriction)")
for U in [np.eye(d),pos]:
    mats=[U.T@A[1,j]@U for j in range(6)]
    print(np.round([[cosine(x,y) for y in mats] for x in mats],3))

```

Output:

```
Reader energy fractions in weight-inferred positional (28D), token/context (81D), and clock (2D) spaces
L1H0 Q(pos,tok,clock) [0.121 0.46  0.011] K(pos,tok,clock) [0.301 0.396 0.221] bias-key clock frac 0.764
L1H1 Q(pos,tok,clock) [0.11  0.498 0.012] K(pos,tok,clock) [0.551 0.334 0.461] bias-key clock frac 0.903
L1H2 Q(pos,tok,clock) [0.11  0.494 0.012] K(pos,tok,clock) [0.538 0.32  0.428] bias-key clock frac 0.906
L1H3 Q(pos,tok,clock) [0.121 0.485 0.016] K(pos,tok,clock) [0.574 0.296 0.47 ] bias-key clock frac 0.93
L1H4 Q(pos,tok,clock) [0.117 0.497 0.013] K(pos,tok,clock) [0.563 0.315 0.448] bias-key clock frac 0.917
L1H5 Q(pos,tok,clock) [0.113 0.488 0.01 ] K(pos,tok,clock) [0.501 0.346 0.411] bias-key clock frac 0.824
L2H0 Q(pos,tok,clock) [0.07  0.243 0.003] K(pos,tok,clock) [0.127 0.243 0.063] bias-key clock frac 0.465
L2H1 Q(pos,tok,clock) [0.068 0.273 0.002] K(pos,tok,clock) [0.07  0.306 0.004] bias-key clock frac 0.034
L2H2 Q(pos,tok,clock) [0.067 0.316 0.005] K(pos,tok,clock) [0.182 0.285 0.122] bias-key clock frac 0.843
L2H3 Q(pos,tok,clock) [0.074 0.237 0.005] K(pos,tok,clock) [0.187 0.252 0.126] bias-key clock frac 0.789
L2H4 Q(pos,tok,clock) [0.072 0.247 0.006] K(pos,tok,clock) [0.169 0.252 0.108] bias-key clock frac 0.664
L2H5 Q(pos,tok,clock) [0.07  0.216 0.002] K(pos,tok,clock) [0.108 0.225 0.045] bias-key clock frac 0.265
L3H0 Q(pos,tok,clock) [0.069 0.225 0.001] K(pos,tok,clock) [0.079 0.228 0.014] bias-key clock frac 0.084
L3H1 Q(pos,tok,clock) [0.067 0.233 0.001] K(pos,tok,clock) [0.073 0.253 0.006] bias-key clock frac 0.075
L3H2 Q(pos,tok,clock) [0.064 0.214 0.002] K(pos,tok,clock) [0.092 0.223 0.029] bias-key clock frac 0.233
L3H3 Q(pos,tok,clock) [0.068 0.196 0.002] K(pos,tok,clock) [0.077 0.209 0.009] bias-key clock frac 0.044
L3H4 Q(pos,tok,clock) [0.068 0.207 0.002] K(pos,tok,clock) [0.085 0.207 0.022] bias-key clock frac 0.16
L3H5 Q(pos,tok,clock) [0.067 0.303 0.002] K(pos,tok,clock) [0.072 0.306 0.003] bias-key clock frac 0.058

Key-side versus query-side composition with L0H5, restricted to T
L1H0 key -0.046 query -0.426 V token frac 0.164 OV token output-space frac 0.069
L1H1 key 0.208 query 0.085 V token frac 0.169 OV token output-space frac 0.051
L1H2 key -0.066 query -0.318 V token frac 0.202 OV token output-space frac 0.066
L1H3 key 0.183 query 0.124 V token frac 0.152 OV token output-space frac 0.054
L1H4 key 0.137 query -0.237 V token frac 0.166 OV token output-space frac 0.056
L1H5 key 0.17 query -0.109 V token frac 0.191 OV token output-space frac 0.055
L2H0 key -0.086 query 0.047 V token frac 0.089 OV token output-space frac 0.062
L2H1 key 0.719 query -0.502 V token frac 0.089 OV token output-space frac 0.104
L2H2 key -0.479 query 0.656 V token frac 0.091 OV token output-space frac 0.046
L2H3 key -0.038 query 0.218 V token frac 0.087 OV token output-space frac 0.058
L2H4 key 0.043 query -0.023 V token frac 0.126 OV token output-space frac 0.046
L2H5 key 0.133 query -0.155 V token frac 0.088 OV token output-space frac 0.085
L3H0 key 0.305 query 0.076 V token frac 0.082 OV token output-space frac 0.162
L3H1 key 0.666 query -0.328 V token frac 0.112 OV token output-space frac 0.39
L3H2 key -0.016 query 0.159 V token frac 0.079 OV token output-space frac 0.142
L3H3 key 0.011 query 0.089 V token frac 0.075 OV token output-space frac 0.127
L3H4 key 0.203 query -0.053 V token frac 0.08 OV token output-space frac 0.137
L3H5 key 0.705 query -0.422 V token frac 0.133 OV token output-space frac 0.375

Agreement of token OV maps with candidate copying heads L2H1, L3H1, L3H5
L1H0 [-0.003  0.016 -0.012]
L1H1 [-0.08  -0.037 -0.054]
L1H2 [ 0.037 -0.117 -0.074]
L1H3 [ 0.009 -0.044 -0.036]
L1H4 [ 0.016 -0.105 -0.017]
L1H5 [0.004 0.006 0.022]
L2H0 [-0.033 -0.082 -0.096]
L2H1 [1.    0.353 0.489]
L2H2 [-0.031  0.005 -0.008]
L2H3 [ 0.019 -0.109 -0.067]
L2H4 [ 0.127 -0.032  0.036]
L2H5 [-0.02   0.114  0.03 ]
L3H0 [0.262 0.298 0.354]
L3H1 [0.353 1.    0.751]
L3H2 [-0.055 -0.213 -0.236]
L3H3 [ 0.003 -0.074 -0.05 ]
L3H4 [-0.032  0.017 -0.018]
L3H5 [0.489 0.751 1.   ]

L1 pairwise QK cosine (full / positional restriction)
[[1.    0.302 0.393 0.327 0.338 0.266]
 [0.302 1.    0.524 0.591 0.568 0.566]
 [0.393 0.524 1.    0.538 0.578 0.45 ]
 [0.327 0.591 0.538 1.    0.607 0.524]
 [0.338 0.568 0.578 0.607 1.    0.479]
 [0.266 0.566 0.45  0.524 0.479 1.   ]]
[[1.    0.839 0.83  0.818 0.789 0.781]
 [0.839 1.    0.9   0.946 0.891 0.944]
 [0.83  0.9   1.    0.939 0.958 0.903]
 [0.818 0.946 0.939 1.    0.952 0.936]
 [0.789 0.891 0.958 0.952 1.    0.893]
 [0.781 0.944 0.903 0.936 0.893 1.   ]]

```

## Call 9

```python
import numpy as np
from scipy.linalg import svd, eigh, qr
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
d=384
P=np.eye(d)-np.ones((d,d))/d

U,s,Vt=svd(P@W["W_proj.3"].astype(float),full_matrices=False)
U=U[:,:27]
X=W["W_proj.3"].T.astype(float)@U
mu=X.mean(axis=0)
X-=mu
ev,ec=eigh(X.T@X/len(X))
root=(ec*np.sqrt(ev))@ec.T
invroot=(ec/np.sqrt(ev))@ec.T
Y=X@invroot
Y/=np.linalg.norm(Y,axis=1,keepdims=True)
km=KMeans(n_clusters=28,n_init=20,max_iter=400,random_state=13).fit(Y)
centers=km.cluster_centers_
centers/=np.linalg.norm(centers,axis=1,keepdims=True)
similarities=Y@centers.T
labels=similarities.argmax(axis=1)
print("Final projection column clustering")
print("best cosine quantiles",np.quantile(similarities.max(axis=1),[.1,.25,.5,.75,.9,.95]))
print("cluster counts",np.bincount(labels,minlength=28))
print("center nearest cosine",np.quantile((centers@centers.T-2*np.eye(28)).max(axis=1),[0,.25,.5,.75,1]))
print("center Gram eigenvalues",np.round(np.linalg.eigvalsh(centers@centers.T),3))

# All following quantities are products or factorizations of weight matrices.
# The directions are derived from projection-weight clusters, not text inputs.
g=W["ln1_w.0"].astype(float)
Vraw=np.concatenate([W["W_v.0"][64:128],W["W_v.0"][320:384]],axis=0).astype(float)
Rraw=np.concatenate([W[n][sl] for n in ["W_q.0","W_k.0"]
                    for sl in [slice(64,128),slice(320,384)]],axis=0).astype(float)
R=svd(Rraw,full_matrices=False)[2][:24].T
Vclean=Vraw-(Vraw@R)@R.T
T0=svd(Vclean,full_matrices=False)[2][:27].T
T=qr(P@(T0/g[:,None]),mode="economic")[0]
def ov(l,j):
    sl=slice(j*64,(j+1)*64)
    return P@W[f"W_o.{l}"][:,sl].astype(float)@(W[f"W_v.{l}"][sl].astype(float)*W[f"ln1_w.{l}"])@P
C=U.T@((ov(3,1)+ov(3,5))/2)@T
print("Copy-weight map singular values",np.round(svd(C,compute_uv=False),3))
prototypes=centers@root
prototypes-=prototypes.mean(axis=0)
D=T@np.linalg.pinv(C,rcond=.03)@prototypes.T
D/=np.linalg.norm(D,axis=0,keepdims=True)
print("Pulled-back dictionary Gram nearest cosine",
      np.quantile((D.T@D-2*np.eye(28)).max(axis=1),[0,.25,.5,.75,1]))

F=(W["W_fc.0"].astype(float)*W["ln2_w.0"])@P
L=[F@D,F@ov(0,5)@D,F@ov(0,1)@D]
assign=[]
strength=[]
for i,z in enumerate(L):
    z=z-z.mean(axis=1,keepdims=True)
    z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    a=z.argmax(axis=1)
    assign.append(a)
    strength.append(z.max(axis=1))
    print(["direct","H5 path","H1 path"][i],"preferred direction counts",np.bincount(a,minlength=28),
          "max-loading quartiles",np.quantile(z.max(axis=1),[.25,.5,.75,.9]))

G=np.zeros((28,28))
for n in range(1536):
    G[assign[1][n],assign[0][n]]+=strength[0][n]*strength[1][n]
print("Synaptic co-preference graph: top edges (weight, H5-direction, direct-direction)")
print(sorted([(round(float(G[i,j]),3),i,j) for i in range(28) for j in range(28)],reverse=True)[:90])
print("row totals",np.round(G.sum(axis=1),2))
print("column totals",np.round(G.sum(axis=0),2))
print("diagonal / total",np.trace(G),G.sum())

```

Output:

```
Final projection column clustering
best cosine quantiles [0.357 0.396 0.454 0.532 0.613 0.69 ]
cluster counts [50 59 38 57 62 65 50 56 60 62 50 65 32 56 51 58 54 50 49 63 58 60 61 54 60 51 58 47]
center nearest cosine [0.101 0.158 0.205 0.28  0.459]
center Gram eigenvalues [-0.     0.     0.005  0.037  0.123  0.203  0.315  0.398  0.577  0.696  0.787  0.846  0.914  1.048  1.075  1.218  1.284  1.341  1.386  1.446  1.546  1.572  1.609  1.692  1.795
  1.889  2.076  2.122]
Copy-weight map singular values [1.719 1.671 1.588 1.5   1.386 1.309 1.278 1.143 1.121 1.03  1.003 0.966 0.922 0.834 0.788 0.673 0.631 0.602 0.521 0.466 0.394 0.377 0.282 0.18  0.144 0.058 0.009]
Pulled-back dictionary Gram nearest cosine [0.604 0.801 0.885 0.939 0.956]
direct preferred direction counts [ 79  27  35  86  53  20   3 139  75   4 100  37  47   3  70   7 159   7  59 221  16  64  17  60  18  74  44  12] max-loading quartiles [0.3   0.357 0.419 0.486]
H5 path preferred direction counts [ 52  12  16  36  76  30  12 109 103   9 250  17  39  35  14  13 172   3 166 150  16  42  21  21  46  19  49   8] max-loading quartiles [0.291 0.339 0.394 0.455]
H1 path preferred direction counts [ 56   8   9 124  92  62  50  81  96   0  40  58  57 102  52   8  89   3  24  65   4  49  96  12   1 194 104   0] max-loading quartiles [0.321 0.365 0.416 0.473]
Synaptic co-preference graph: top edges (weight, H5-direction, direct-direction)
[(5.363, 16, 19), (5.106, 10, 19), (3.653, 10, 16), (3.162, 10, 7), (3.126, 18, 19), (3.035, 16, 16), (2.781, 10, 25), (2.676, 19, 7), (2.475, 16, 7), (2.316, 18, 7), (2.249, 19, 3), (2.137, 10, 3), (1.963, 7, 16), (1.947, 19, 10), (1.939, 26, 19), (1.882, 19, 16), (1.833, 10, 18), (1.757, 10, 4), (1.718, 19, 19), (1.694, 16, 14), (1.638, 16, 8), (1.53, 13, 16), (1.49, 10, 0), (1.475, 8, 16), (1.47, 19, 0), (1.451, 16, 10), (1.447, 4, 19), (1.42, 8, 7), (1.392, 0, 16), (1.384, 18, 14), (1.356, 18, 16), (1.351, 8, 10), (1.32, 4, 16), (1.303, 5, 19), (1.293, 7, 8), (1.29, 18, 3), (1.286, 7, 19), (1.269, 4, 7), (1.188, 10, 14), (1.187, 8, 19), (1.186, 16, 23), (1.182, 26, 16), (1.174, 3, 19), (1.173, 19, 14), (1.147, 7, 7), (1.127, 4, 21), (1.111, 16, 3), (1.096, 7, 10), (1.094, 18, 8), (1.02, 18, 25), (1.015, 10, 8), (0.971, 18, 10), (0.943, 7, 0), (0.942, 16, 0), (0.932, 3, 4), (0.906, 10, 26), (0.893, 10, 10), (0.88, 4, 3), (0.87, 0, 3), (0.837, 22, 19), (0.834, 26, 0), (0.834, 10, 12), (0.831, 19, 23), (0.821, 16, 11), (0.818, 12, 16), (0.816, 18, 21), (0.803, 3, 12), (0.802, 10, 5), (0.787, 12, 19), (0.778, 5, 16), (0.777, 25, 19), (0.777, 4, 12), (0.773, 19, 21), (0.768, 19, 26), (0.764, 16, 25), (0.764, 7, 21), (0.757, 7, 18), (0.757, 4, 8), (0.756, 7, 25), (0.753, 10, 23), (0.751, 12, 22), (0.74, 10, 21), (0.736, 0, 0), (0.733, 0, 19), (0.72, 12, 12), (0.72, 7, 12), (0.718, 16, 22), (0.711, 13, 0), (0.708, 19, 8), (0.706, 7, 23)]
row totals [ 7.    1.17  1.84  5.5  10.28  3.95  1.55 14.95 11.48  0.79 32.15  1.95  6.45  4.78  1.91  1.4  24.91  0.35 17.37 20.34  1.75  5.49  2.96  2.23  4.25  2.55  6.4   0.67]
column totals [10.36  2.61  3.97 12.51  6.7   2.33  0.32 18.43  9.22  0.34 12.57  3.58  7.21  0.23  8.82  0.68 24.75  0.6   6.08 30.03  1.47  7.7   2.6   6.13  1.68  9.19  5.28  1.04]
diagonal / total 9.719949646977788 196.44102240622556

```

## Call 10

```python
import numpy as np
from scipy.linalg import svd, svdvals, qr
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=4,suppress=True,linewidth=180)

d=384
P=np.eye(d)-np.ones((d,d))/d
V=np.concatenate([W["W_v.0"][64:128],W["W_v.0"][320:384]],axis=0).astype(float)
R=np.concatenate([W[n][sl] for n in ["W_q.0","W_k.0"]
                  for sl in [slice(64,128),slice(320,384)]],axis=0).astype(float)
R=svd(R,full_matrices=False)[2][:20].T
extra=[]
for j in [0,4]:
    sl=slice(j*64,(j+1)*64)
    rawOV=W["W_o.0"][:,sl].astype(float)@W["W_v.0"][sl].astype(float)
    extra.append(svd(rawOV,full_matrices=False)[2][:2].T)
print("Value spectra after removing alternative position-like weight subspaces")
for label,U in [
    ("20D QK",R),
    ("20D QK + H4 top2",np.concatenate([R,extra[1]],axis=1)),
    ("20D QK + H0/H4 top2",np.concatenate([R]+extra,axis=1)),
]:
    U=qr(U,mode="economic")[0]
    s=svdvals(V-(V@U)@U.T)
    print(label,"singular values 18..40:",np.round(s[17:40],4))

A,B,Vs={},{},{}
for l in range(4):
    g=W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        sl=slice(j*64,(j+1)*64)
        Q=(W[f"W_q.{l}"][sl].astype(float)*g)@P
        K=(W[f"W_k.{l}"][sl].astype(float)*g)@P
        v=(W[f"W_v.{l}"][sl].astype(float)*g)@P
        A[l,j]=Q.T@K
        B[l,j]=P@W[f"W_o.{l}"][:,sl].astype(float)@v
        Vs[l,j]=v
Tfull=svd(np.concatenate([Vs[0,1].T,Vs[0,5].T],axis=1),full_matrices=False)[0]
print("\nRobustness of key-side / query-side matching through L0H5")
for l,j in [(2,1),(2,2),(3,0),(3,1),(3,2),(3,5)]:
    vals=[]
    for r in [20,24,27,32,40]:
        T=Tfull[:,:r]
        Ck=T.T@A[l,j]@B[0,5]@T
        Cq=T.T@B[0,5].T@A[l,j]@T
        vals.append([np.trace(Ck)/svdvals(Ck).sum(),np.trace(Cq)/svdvals(Cq).sum()])
    print(f"L{l}H{j}","ranks 20,24,27,32,40:",np.round(vals,3))

print("\nExact first-layer high-rank positional operator norms and representative weight entries")
for j in [1,4,5]:
    sl=slice(j*64,(j+1)*64)
    print(f"L0H{j}","||Q||F,||K||F,||V||F,||O||F:",
          np.round([np.linalg.norm(W["W_q.0"][sl]),np.linalg.norm(W["W_k.0"][sl]),
                    np.linalg.norm(W["W_v.0"][sl]),np.linalg.norm(W["W_o.0"][:,sl])],4))
print("ln1_w.0[211], ln1_w.0[273], ln1_w.0[223]:",
      [float(W["ln1_w.0"][i]) for i in [211,273,223]])

```

Output:

```
Value spectra after removing alternative position-like weight subspaces
20D QK singular values 18..40: [2.184  2.1643 2.0802 2.0077 1.8538 1.8016 1.7005 1.4684 1.389  1.3132 0.8423 0.7766 0.7417 0.6773 0.6665 0.6538 0.6302 0.6204 0.6052 0.5834 0.5566 0.5469 0.5269]
20D QK + H4 top2 singular values 18..40: [2.177  2.1629 2.0797 1.9236 1.8537 1.7828 1.6958 1.4653 1.3826 1.3093 0.8361 0.7417 0.7089 0.669  0.6531 0.6415 0.6172 0.6065 0.5948 0.5764 0.5508 0.5422 0.5216]
20D QK + H0/H4 top2 singular values 18..40: [2.1593 2.0763 1.9204 1.8582 1.8199 1.7075 1.6535 1.4458 1.3785 1.3013 0.8323 0.7333 0.6939 0.6661 0.6512 0.6351 0.6071 0.602  0.577  0.5649 0.5425 0.5271 0.5213]

Robustness of key-side / query-side matching through L0H5
L2H1 ranks 20,24,27,32,40: [[ 0.792 -0.577]
 [ 0.745 -0.532]
 [ 0.719 -0.502]
 [ 0.673 -0.448]
 [ 0.606 -0.397]]
L2H2 ranks 20,24,27,32,40: [[-0.55   0.702]
 [-0.506  0.689]
 [-0.479  0.656]
 [-0.437  0.603]
 [-0.391  0.545]]
L3H0 ranks 20,24,27,32,40: [[0.336 0.106]
 [0.31  0.08 ]
 [0.305 0.076]
 [0.285 0.072]
 [0.234 0.057]]
L3H1 ranks 20,24,27,32,40: [[ 0.749 -0.379]
 [ 0.71  -0.352]
 [ 0.666 -0.328]
 [ 0.614 -0.277]
 [ 0.549 -0.243]]
L3H2 ranks 20,24,27,32,40: [[ 0.003  0.197]
 [-0.009  0.169]
 [-0.016  0.159]
 [-0.022  0.143]
 [-0.019  0.129]]
L3H5 ranks 20,24,27,32,40: [[ 0.79  -0.512]
 [ 0.744 -0.459]
 [ 0.705 -0.422]
 [ 0.654 -0.373]
 [ 0.583 -0.325]]

Exact first-layer high-rank positional operator norms and representative weight entries
L0H1 ||Q||F,||K||F,||V||F,||O||F: [12.4632 12.643   9.2571 10.205 ]
L0H4 ||Q||F,||K||F,||V||F,||O||F: [2.1731 2.6235 8.2845 8.9416]
L0H5 ||Q||F,||K||F,||V||F,||O||F: [15.8981 15.965   9.4051 10.6032]
ln1_w.0[211], ln1_w.0[273], ln1_w.0[223]: [1.0134334564208984, 0.9813231229782104, 0.9731485247612]

```
