## Call 1

```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)

print("LAYER STATISTICS")
for l in range(4):
    print("\nLayer", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        a = W[f"{name}.{l}"].astype(np.float64)
        print(name, "mean/std/min/max", np.round([a.mean(), a.std(), a.min(), a.max()], 5))
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        a = W[f"{name}.{l}"].astype(np.float64)
        s = svdvals(a)
        print(name, "rms", round(np.sqrt(np.mean(a*a)), 5),
              "sv", np.round(s[[0,1,2,3,7,15,31,63,127,255,383]], 3))

print("\nHEAD STATISTICS: QK and OV are matrix products, not layer evaluations.")
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(np.float64)
    b = W[f"ln1_b.{l}"].astype(np.float64)
    for j in range(6):
        sl = slice(64*j, 64*(j+1))
        q = W[f"W_q.{l}"][sl].astype(np.float64)
        k = W[f"W_k.{l}"][sl].astype(np.float64)
        v = W[f"W_v.{l}"][sl].astype(np.float64)
        o = W[f"W_o.{l}"][:,sl].astype(np.float64)
        qk = (q*g).T @ (k*g) / 8
        ov = o @ (v*g)
        sq, so = svdvals(qk)[:64], svdvals(ov)[:64]
        effective_rank = lambda s: (s@s)**2 / np.sum(s**4)
        key_bias_direction = (q@b) @ (k*g) / 8
        print(f"L{l}H{j}",
              "qk_norm/rank", np.round([np.linalg.norm(qk), effective_rank(sq)],3),
              "qk_sv", np.round(sq[[0,1,2,3,7,15,31,63]],3),
              "qk_sym", round(np.sum(qk*qk.T)/np.sum(qk*qk),3),
              "key_bias_norm", round(np.linalg.norm(key_bias_direction),3),
              "ov_norm/rank", np.round([np.linalg.norm(ov), effective_rank(so)],3),
              "ov_sv", np.round(so[[0,1,2,3,7,15,31,63]],3),
              "ov_trace", round(np.trace(ov),3))

```

Output:

```
LAYER STATISTICS

Layer 0
ln1_w mean/std/min/max [ 0.2403  0.2436 -0.001   1.0341]
ln1_b mean/std/min/max [ 0.0008  0.0287 -0.0959  0.091 ]
ln2_w mean/std/min/max [0.8006 0.3295 0.0044 1.5072]
ln2_b mean/std/min/max [ 0.0022  0.0683 -0.4986  0.3331]
W_q rms 0.08409 sv [8.252 8.223 7.963 7.819 7.205 4.947 2.136 0.82  0.451 0.192 0.   ]
W_k rms 0.07914 sv [7.646 7.436 7.33  6.912 6.446 4.901 2.11  0.89  0.515 0.226 0.   ]
W_v rms 0.03864 sv [4.339 3.529 3.27  3.166 2.552 2.017 1.267 0.663 0.439 0.2   0.   ]
W_o rms 0.04263 sv [7.103 6.607 4.268 2.77  2.218 1.987 1.349 0.692 0.277 0.09  0.   ]
W_fc rms 0.06022 sv [18.176 11.018  8.737  7.452  6.62   5.918  4.533  2.18   1.062  0.686  0.4  ]
W_proj rms 0.05118 sv [9.826 7.351 6.265 5.869 5.45  4.819 3.847 2.839 1.276 0.427 0.213]

Layer 1
ln1_w mean/std/min/max [0.4575 0.1022 0.1361 1.1239]
ln1_b mean/std/min/max [-0.005   0.1672 -1.4712  1.1544]
ln2_w mean/std/min/max [ 1.0429  0.2003 -0.0019  1.4856]
ln2_b mean/std/min/max [ 0.0005  0.1364 -0.3597  0.3686]
W_q rms 0.08415 sv [9.903 7.797 6.704 5.952 5.313 4.476 3.327 1.461 0.58  0.245 0.001]
W_k rms 0.07617 sv [13.307  9.933  5.899  5.677  4.404  3.475  2.531  1.241  0.549  0.235  0.   ]
W_v rms 0.07123 sv [5.284 4.966 4.87  4.733 4.322 3.733 3.027 1.498 0.697 0.269 0.001]
W_o rms 0.06751 sv [4.966 3.241 3.157 3.112 2.954 2.835 2.607 2.136 1.089 0.219 0.   ]
W_fc rms 0.10303 sv [21.488 13.135 12.804 12.625 11.691  9.928  7.853  5.034  2.392  1.002  0.411]
W_proj rms 0.09858 sv [8.193 7.894 7.199 6.997 6.133 5.675 5.351 4.904 4.194 3.016 0.319]

Layer 2
ln1_w mean/std/min/max [0.7052 0.0749 0.2078 1.0144]
ln1_b mean/std/min/max [-0.001   0.1525 -1.3819  1.0527]
ln2_w mean/std/min/max [1.3123 0.1255 0.0011 1.4655]
ln2_b mean/std/min/max [ 0.005   0.1537 -0.3761  0.3886]
W_q rms 0.10246 sv [5.796 5.196 4.809 4.616 4.432 4.039 3.56  2.898 2.023 0.833 0.001]
W_k rms 0.09836 sv [10.663  8.624  4.989  4.394  3.848  3.546  3.195  2.662  1.888  0.751  0.001]
W_v rms 0.08673 sv [3.706 3.625 3.579 3.533 3.415 3.206 2.934 2.52  1.827 0.763 0.005]
W_o rms 0.08552 sv [4.102 3.971 3.558 3.381 3.169 3.001 2.773 2.442 1.849 0.823 0.006]
W_fc rms 0.10282 sv [15.363  8.021  7.895  7.725  7.216  6.773  6.101  5.228  4.166  2.812  0.645]
W_proj rms 0.10004 sv [9.404 9.187 8.723 8.47  7.45  6.733 5.703 4.943 4.052 2.835 0.362]

Layer 3
ln1_w mean/std/min/max [0.7546 0.0542 0.2944 0.8701]
ln1_b mean/std/min/max [ 0.0121  0.0849 -0.6069  0.6502]
ln2_w mean/std/min/max [1.2778 0.0808 0.4364 1.4306]
ln2_b mean/std/min/max [ 0.0043  0.1276 -0.6728  0.3068]
W_q rms 0.0858 sv [4.56  4.239 4.005 3.924 3.691 3.295 2.917 2.423 1.729 0.748 0.001]
W_k rms 0.08388 sv [6.09  4.899 4.197 3.456 3.161 2.958 2.734 2.337 1.75  0.78  0.   ]
W_v rms 0.10363 sv [5.06  4.904 4.848 4.769 4.519 4.114 3.645 2.924 2.017 0.857 0.   ]
W_o rms 0.10535 sv [9.567 8.759 5.339 5.139 4.736 4.183 3.678 2.814 1.867 0.79  0.002]
W_fc rms 0.10127 sv [14.162  8.125  7.915  7.326  6.691  6.088  5.568  4.95   4.165  3.051  0.683]
W_proj rms 0.10729 sv [26.309 20.685 19.699 19.261 17.667 13.5    3.139  1.593  1.08   0.612  0.318]

HEAD STATISTICS: QK and OV are matrix products, not layer evaluations.
L0H0 qk_norm/rank [0.055 1.265] qk_sv [0.051 0.011 0.007 0.005 0.003 0.002 0.001 0.   ] qk_sym 0.023 key_bias_norm 0.014 ov_norm/rank [9.043 1.523] ov_sv [8.074 3.328 2.097 0.673 0.193 0.072 0.031 0.007] ov_trace 0.024
L0H1 qk_norm/rank [0.03  1.908] qk_sv [0.025 0.01  0.005 0.004 0.003 0.002 0.001 0.   ] qk_sym 0.047 key_bias_norm 0.003 ov_norm/rank [6.966 1.894] ov_sv [5.661 3.827 1.065 0.502 0.166 0.068 0.029 0.006] ov_trace -0.192
L0H2 qk_norm/rank [ 4.078 13.914] qk_sv [1.415 1.275 1.215 1.206 1.022 0.492 0.034 0.001] qk_sym -0.047 key_bias_norm 0.143 ov_norm/rank [ 2.54  19.527] ov_sv [0.787 0.718 0.683 0.671 0.558 0.421 0.118 0.004] ov_trace -0.626
L0H3 qk_norm/rank [0.395 1.441] qk_sv [0.359 0.119 0.078 0.042 0.023 0.009 0.003 0.   ] qk_sym 0.478 key_bias_norm 0.054 ov_norm/rank [1.766 4.306] ov_sv [1.032 0.965 0.617 0.505 0.183 0.083 0.026 0.006] ov_trace -0.25
L0H4 qk_norm/rank [0.106 1.786] qk_sv [0.089 0.054 0.016 0.007 0.003 0.002 0.001 0.   ] qk_sym 0.006 key_bias_norm 0.011 ov_norm/rank [4.034 1.867] ov_sv [3.396 1.642 1.082 0.607 0.205 0.077 0.03  0.006] ov_trace -0.349
L0H5 qk_norm/rank [0.078 1.09 ] qk_sv [0.076 0.011 0.005 0.004 0.003 0.002 0.001 0.   ] qk_sym 0.16 key_bias_norm 0.016 ov_norm/rank [9.311 1.504] ov_sv [8.332 3.605 1.679 0.768 0.305 0.094 0.032 0.007] ov_trace 0.048
L1H0 qk_norm/rank [0.887 3.3  ] qk_sv [0.598 0.482 0.248 0.194 0.098 0.046 0.011 0.002] qk_sym 0.008 key_bias_norm 1.357 ov_norm/rank [10.122 12.552] ov_sv [4.01  3.552 2.947 2.893 2.501 1.095 0.248 0.023] ov_trace -9.552
L1H1 qk_norm/rank [1.316 3.561] qk_sv [0.882 0.666 0.411 0.267 0.154 0.084 0.012 0.002] qk_sym 0.018 key_bias_norm 2.285 ov_norm/rank [10.333 21.583] ov_sv [3.02  2.804 2.655 2.597 2.285 1.689 0.616 0.026] ov_trace -12.532
L1H2 qk_norm/rank [1.551 3.675] qk_sv [0.995 0.86  0.384 0.325 0.203 0.104 0.011 0.002] qk_sym 0.018 key_bias_norm 2.588 ov_norm/rank [ 9.181 18.526] ov_sv [2.867 2.74  2.493 2.379 2.116 1.453 0.348 0.024] ov_trace -9.662
L1H3 qk_norm/rank [1.376 3.411] qk_sv [0.92  0.739 0.408 0.254 0.157 0.088 0.01  0.002] qk_sym 0.017 key_bias_norm 2.351 ov_norm/rank [10.051 14.179] ov_sv [3.705 3.25  3.102 2.786 2.309 1.427 0.171 0.018] ov_trace -8.962
L1H4 qk_norm/rank [1.451 3.159] qk_sv [0.948 0.868 0.36  0.259 0.155 0.083 0.013 0.002] qk_sym 0.011 key_bias_norm 2.29 ov_norm/rank [10.401 18.908] ov_sv [3.081 2.984 2.823 2.677 2.427 1.812 0.313 0.022] ov_trace -11.859
L1H5 qk_norm/rank [1.338 3.332] qk_sv [0.893 0.739 0.372 0.23  0.155 0.081 0.011 0.002] qk_sym -0.003 key_bias_norm 1.753 ov_norm/rank [10.62  20.972] ov_sv [3.027 2.825 2.756 2.672 2.362 1.837 0.414 0.023] ov_trace -13.2
L2H0 qk_norm/rank [ 2.438 28.208] qk_sv [0.695 0.63  0.602 0.563 0.465 0.369 0.233 0.031] qk_sym -0.034 key_bias_norm 1.699 ov_norm/rank [16.307 40.355] 
```

## Call 2

```python
import numpy as np
from scipy.linalg import svd, svdvals, eigvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=160)

P = np.eye(384) - np.ones((384,384))/384
A, B, QR, KR, VR = {}, {}, {}, {}, {}
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        s = slice(64*j,64*(j+1))
        q = (W[f"W_q.{l}"][s]*g) @ P
        k = (W[f"W_k.{l}"][s]*g) @ P
        v = (W[f"W_v.{l}"][s]*g) @ P
        o = W[f"W_o.{l}"][:,s].astype(float)
        A[l,j], B[l,j] = q.T@k/8, o@v
        QR[l,j], KR[l,j], VR[l,j] = q,k,v

print("DETAILED SPECTRA")
for l in [0,3]:
    for name in ["W_fc", "W_proj"]:
        s = svdvals(W[f"{name}.{l}"].astype(float))
        print(f"{name}.{l} first 45:", s[:45])
print("\nL0 ln1 weight quantiles:", np.quantile(W["ln1_w.0"], np.linspace(0,1,11)))
print("L0 ln1 weights, sorted largest 65:", np.sort(W["ln1_w.0"])[-65:])

print("\nL0 head eigenvalues (largest magnitudes); Q/K/V top-20 subspace overlap")
for j in range(6):
    q,k,v=QR[0,j],KR[0,j],VR[0,j]
    uq=svd(q,full_matrices=False)[2][:20].T
    uk=svd(k,full_matrices=False)[2][:20].T
    uv=svd(v,full_matrices=False)[2][:20].T
    e=eigvals(A[0,j]); e=e[np.argsort(-abs(e))]
    print(f"L0H{j}", "QK eigenvalues",e[:12])
    print("QK/QV/KV overlaps", [round(np.linalg.norm(u.T@z)**2/20,3) for u,z in [(uq,uk),(uq,uv),(uk,uv)]])
print("\nL0 QK Frobenius cosine matrix")
print(np.array([[np.sum(A[0,i]*A[0,j])/np.linalg.norm(A[0,i])/np.linalg.norm(A[0,j]) for j in range(6)] for i in range(6)]))
print("L1 QK Frobenius cosine matrix")
print(np.array([[np.sum(A[1,i]*A[1,j])/np.linalg.norm(A[1,i])/np.linalg.norm(A[1,j]) for j in range(6)] for i in range(6)]))

print("\nComposition with L0H2 OV: A_lh @ B_02; algebraic induction diagnostic")
b=B[0,2]
for l in [1,2,3]:
    for j in range(6):
        a=A[l,j]
        c=a@b
        sy=(c+c.T)/2
        ev=np.linalg.eigvalsh(sy)
        print(f"L{l}H{j}", "norm",round(np.linalg.norm(c),3),
              "trace/norm",round(np.trace(c)/np.linalg.norm(c),3),
              "sym_cos",round(np.sum(c*c.T)/np.sum(c*c),3),
              "sym positive/negative mass",np.round([ev[ev>0].sum(),-ev[ev<0].sum()],3))

print("\nL0 column diagnostics, largest ln1 weights")
inds=np.argsort(-W["ln1_w.0"])[:35]
for i in inds:
    print(int(i), "ln1",round(float(W["ln1_w.0"][i]),4),
          "Q2 K2 V2",np.round([np.linalg.norm(W[n+".0"][128:192,i]) for n in ["W_q","W_k","W_v"]],3),
          "otherQV",np.round([np.linalg.norm(np.delete(W[n+".0"][:,i],np.arange(128,192))) for n in ["W_q","W_v"]],3))

```

Output:

```
DETAILED SPECTRA
W_fc.0 first 45: [18.176 11.018  8.737  7.452  7.212  7.062  7.011  6.62   6.57   6.42   6.343  6.29   6.137  6.043  5.974  5.918  5.807  5.779  5.583  5.513  5.5    5.36
  5.16   5.12   5.086  5.032  4.899  4.879  4.819  4.732  4.601  4.533  4.479  4.441  4.268  4.198  4.031  3.974  3.869  3.824  3.72   3.683  3.646  3.574
  3.516]
W_proj.0 first 45: [9.826 7.351 6.265 5.869 5.797 5.657 5.516 5.45  5.333 5.242 5.197 5.094 4.98  4.904 4.863 4.819 4.793 4.689 4.627 4.595 4.557 4.523 4.411 4.378 4.283 4.216
 4.143 4.125 4.048 4.025 3.894 3.847 3.808 3.792 3.764 3.711 3.685 3.636 3.595 3.525 3.499 3.479 3.469 3.422 3.39 ]
W_fc.3 first 45: [14.162  8.125  7.915  7.326  7.148  6.856  6.763  6.691  6.489  6.436  6.387  6.31   6.26   6.177  6.144  6.088  6.066  6.017  5.962  5.934  5.883  5.855
  5.843  5.781  5.752  5.72   5.697  5.657  5.632  5.617  5.593  5.568  5.519  5.517  5.495  5.45   5.431  5.389  5.379  5.361  5.341  5.308  5.303  5.288
  5.242]
W_proj.3 first 45: [26.309 20.685 19.699 19.261 18.594 18.197 17.862 17.667 17.044 16.749 16.039 15.151 14.756 14.326 14.05  13.5   13.293 12.531 12.401 11.958 11.74  11.594
 10.367  8.351  8.064  7.914  7.461  7.166  5.833  4.072  3.221  3.139  2.842  2.8    2.687  2.571  2.403  2.328  2.244  2.185  2.181  2.116  2.081  2.053
  1.996]

L0 ln1 weight quantiles: [-0.001  0.039  0.069  0.093  0.118  0.142  0.177  0.256  0.359  0.715  1.034]
L0 ln1 weights, sorted largest 65: [0.425 0.426 0.443 0.453 0.455 0.456 0.472 0.472 0.477 0.493 0.495 0.501 0.519 0.521 0.535 0.539 0.554 0.554 0.569 0.597 0.605 0.606 0.611 0.631 0.652 0.706
 0.719 0.72  0.726 0.726 0.727 0.738 0.75  0.758 0.772 0.777 0.779 0.78  0.78  0.786 0.793 0.8   0.82  0.824 0.83  0.836 0.843 0.845 0.845 0.85  0.856 0.86
 0.868 0.882 0.89  0.895 0.896 0.929 0.937 0.937 0.938 0.944 0.98  0.984 1.034]

L0 head eigenvalues (largest magnitudes); Q/K/V top-20 subspace overlap
L0H0 QK eigenvalues [ 0.007+0.j    -0.004+0.j    -0.002+0.j    -0.   +0.001j -0.   -0.001j  0.   +0.001j  0.   -0.001j -0.001+0.001j -0.001-0.001j -0.001-0.j    -0.001+0.j
  0.001+0.001j]
QK/QV/KV overlaps [np.float64(0.227), np.float64(0.264), np.float64(0.206)]
L0H1 QK eigenvalues [-0.004+0.001j -0.004-0.001j  0.002+0.j    -0.001+0.001j -0.001-0.001j -0.001+0.j    -0.001+0.001j -0.001-0.001j -0.   +0.001j -0.   -0.001j -0.001+0.j
 -0.   -0.001j]
QK/QV/KV overlaps [np.float64(0.249), np.float64(0.209), np.float64(0.204)]
L0H2 QK eigenvalues [ 0.632+0.813j  0.632-0.813j -0.231+0.927j -0.231-0.927j  0.332+0.871j  0.332-0.871j -0.777+0.334j -0.777-0.334j -0.817+0.j     0.651+0.41j   0.651-0.41j
  0.338-0.51j ]
QK/QV/KV overlaps [np.float64(0.786), np.float64(0.151), np.float64(0.198)]
L0H3 QK eigenvalues [ 0.262+0.j     0.078+0.j     0.01 +0.031j  0.01 -0.031j -0.017+0.005j -0.017-0.005j -0.017+0.j    -0.015+0.003j -0.015-0.003j -0.012+0.j    -0.007-0.001j
 -0.007+0.001j]
QK/QV/KV overlaps [np.float64(0.448), np.float64(0.229), np.float64(0.227)]
L0H4 QK eigenvalues [-0.028+0.028j -0.028-0.028j -0.008+0.j    -0.002+0.j    -0.001+0.001j -0.001-0.001j -0.002+0.j     0.001+0.001j  0.001-0.001j -0.001+0.001j -0.001-0.001j
  0.001-0.j   ]
QK/QV/KV overlaps [np.float64(0.268), np.float64(0.252), np.float64(0.209)]
L0H5 QK eigenvalues [-0.031+0.j     0.001+0.002j  0.001-0.002j -0.001+0.j    -0.001-0.j    -0.001+0.001j -0.001-0.001j  0.001+0.001j  0.001-0.001j -0.001-0.j    -0.001+0.j
  0.001+0.j   ]
QK/QV/KV overlaps [np.float64(0.254), np.float64(0.208), np.float64(0.203)]

L0 QK Frobenius cosine matrix
[[ 1.     0.209 -0.016 -0.014  0.115  0.018]
 [ 0.209  1.    -0.004 -0.036  0.21  -0.008]
 [-0.016 -0.004  1.     0.24  -0.006 -0.026]
 [-0.014 -0.036  0.24   1.    -0.054 -0.058]
 [ 0.115  0.21  -0.006 -0.054  1.     0.172]
 [ 0.018 -0.008 -0.026 -0.058  0.172  1.   ]]
L1 QK Frobenius cosine matrix
[[1.    0.578 0.416 0.448 0.455 0.464]
 [0.578 1.    0.55  0.488 0.556 0.528]
 [0.416 0.55  1.    0.608 0.667 0.564]
 [0.448 0.488 0.608 1.    0.644 0.563]
 [0.455 0.556 0.667 0.644 1.    0.577]
 [0.464 0.528 0.564 0.563 0.577 1.   ]]

Composition with L0H2 OV: A_lh @ B_02; algebraic induction diagnostic
L1H0 norm 0.108 trace/norm -0.113 sym_cos 0.01 sym positive/negative mass [0.151 0.164]
L1H1 norm 0.213 trace/norm -0.003 sym_cos 0.036 sym positive/negative mass [0.299 0.299]
L1H2 norm 0.227 trace/norm 0.253 sym_cos 0.012 sym positive/negative mass [0.389 0.332]
L1H3 norm 0.22 trace/norm 0.233 sym_cos 0.014 sym positive/negative mass [0.38  0.329]
L1H4 norm 0.196 trace/norm 0.251 sym_cos 0.007 sym positive/negative mass [0.332 0.283]
L1H5 norm 0.267 trace/norm 0.077 sym_cos 0.004 sym positive/negative mass [0.387 0.366]
L2H0 norm 0.388 trace/norm -0.08 sym_cos -0.001 sym positive/negative mass [0.811 0.842]
L2H1 norm 0.305 trace/norm -0.097 sym_cos 0.006 sym positive/negative mass [0.68  0.709]
L2H2 norm 0.208 trace/norm 0.02 sym_cos 0.004 sym positive/negative mass [0.483 0.479]
L2H3 norm 0.286 trace/norm 0.054 sym_cos -0.001 sym positive/negative mass [0.654 0.639]
L2H4 norm 0.334 trace/norm -0.142 sym_cos -0.001 sym positive/negative mass [0.736 0.783]
L2H5 norm 0.317 trace/norm 0.007 sym_cos 0.0 sym positive/negative mass [0.726 0.723]
L3H0 norm 0.309 trace/norm -0.042 sym_cos 0.001 sym positive/negative mass [0.736 0.749]
L3H1 norm 0.376 trace/norm 1.173 sym_cos 0.097 sym positive/negative mass [1.087 0.646]
L3H2 norm 0.18 trace/norm 0.063 sym_cos 0.0 sym positive/negative mass [0.437 0.426]
L3H3 norm 0.195 trace/norm 0.024 sym_cos -0.004 sym positive/negative mass [0.478 0.473]
L3H4 norm 0.237 trace/norm -0.095 sym_cos 0.007 sym positive/negative mass [0.561 0.584]
L3H5 norm 0.112 trace/norm 0.048 sym_cos 0.004 sym positive/negative mass [0.259 0.253]

L0 column diagnostics, largest ln1 weights
162 ln1 1.0341 Q2 K2 V2 [1.637 3.153 0.16 ] otherQV [0.62  1.286]
60 ln1 0.9844 Q2 K2 V2 [1.873 2.82  0.148] otherQV [0.607 1.028]
223 ln1 0.98 Q2 K2 V2 [1.518 2.597 0.163] otherQV [0.632 1.294]
311 ln1 0.9443 Q2 K2 
```

## Call 3

```python
import numpy as np
from scipy.linalg import svd, svdvals
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

d=384
P=np.eye(d)-np.ones((d,d))/d
g=W["ln1_w.0"].astype(float)
Bs=[]; As=[]
for j in range(6):
    sl=slice(64*j,64*(j+1))
    q=(W["W_q.0"][sl]*g)@P
    k=(W["W_k.0"][sl]*g)@P
    v=(W["W_v.0"][sl]*g)@P
    Bs.append(W["W_o.0"][:,sl]@v)
    As.append(q.T@k/8)
ua,sa,vha=svd(As[2],full_matrices=False)
ub,sb,vhb=svd(Bs[2],full_matrices=False)
pos=svd(np.concatenate([ua[:,:26],vha[:26].T],axis=1),full_matrices=False)[0][:,:30]
tok=vhb[:27].T
print("L0H2 OV singular values:",sb[:40])
print("L0H2 QK singular values:",sa[:40])
print("L0H2 value readout versus inferred positional span:",np.linalg.norm(tok.T@pos)**2/27)
print("\nL0 head dominant OV read/write directions")
for j in range(6):
    u,s,vh=svd(Bs[j],full_matrices=False)
    print(j,"first 5 singular values",s[:5],
          "right positional energy",np.sum((vh[:5]@pos)**2,axis=1),
          "right token energy",np.sum((vh[:5]@tok)**2,axis=1),
          "left positional energy",np.sum((u[:,:5].T@pos)**2,axis=1))
print("\nPairwise overlap of L0 non-H2 top OV read directions")
rs=[svd(Bs[j],full_matrices=False)[2][:3] for j in [0,1,3,4,5]]
print(np.array([[np.linalg.norm(a@b.T)**2/3 for b in rs] for a in rs]))

# Only linear products of weights: character-sensitive read directions
# of first-MLP neurons, directly and through L0H2's OV matrix.
F=(W["W_fc.0"]*W["ln2_w.0"])@P
C=F@tok
D=F@Bs[2]@tok
for name,X in [("direct",C),("via_H2_OV",D)]:
    X=X-X.mean(0)
    u,s,vh=svd(X,full_matrices=False)
    print("\n",name,"singular values",s)
    for k in [5,10,20,24,28,32]:
        Z=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
        km=KMeans(n_clusters=k,n_init=5,random_state=0).fit(Z)
        score=silhouette_score(Z,km.labels_,sample_size=1000,random_state=0)
        print("k",k,"silhouette",round(score,3),"counts",np.sort(np.bincount(km.labels_))[::-1])

print("\nFinal projection column clustering (weights only)")
X=W["W_proj.3"].T.astype(float)
X-=X.mean(0)
u,s,vh=svd(X,full_matrices=False)
Z=u[:,:27]*s[:27]
Z/=np.linalg.norm(Z,axis=1,keepdims=True)
for k in [5,10,20,24,28,32]:
    km=KMeans(n_clusters=k,n_init=5,random_state=0).fit(Z)
    print("k",k,"silhouette",round(silhouette_score(Z,km.labels_,sample_size=1000,random_state=0),3),
          "counts",np.sort(np.bincount(km.labels_))[::-1])

```

Output:

```
L0H2 OV singular values: [0.787 0.718 0.683 0.671 0.636 0.601 0.569 0.558 0.55  0.522 0.515 0.486 0.48  0.469 0.439 0.421 0.4   0.393 0.347 0.304 0.277 0.275 0.255 0.229 0.218 0.205 0.181 0.17  0.163
 0.159 0.131 0.118 0.11  0.101 0.095 0.081 0.073 0.065 0.051 0.047]
L0H2 QK singular values: [1.408 1.272 1.213 1.199 1.158 1.09  1.084 1.021 0.955 0.916 0.86  0.781 0.728 0.721 0.514 0.492 0.429 0.376 0.322 0.288 0.189 0.165 0.128 0.115 0.109 0.098 0.075 0.056 0.049
 0.043 0.037 0.034 0.03  0.027 0.024 0.02  0.019 0.016 0.014 0.013]
L0H2 value readout versus inferred positional span: 0.2404848967024288

L0 head dominant OV read/write directions
0 first 5 singular values [8.073 3.327 2.097 0.672 0.561] right positional energy [0.965 0.754 0.667 0.142 0.284] right token energy [0.124 0.111 0.101 0.605 0.172] left positional energy [0.343 0.312 0.213 0.163 0.089]
1 first 5 singular values [5.66  3.827 1.064 0.502 0.428] right positional energy [0.954 0.91  0.477 0.156 0.127] right token energy [0.091 0.091 0.259 0.413 0.513] left positional energy [0.331 0.3   0.237 0.171 0.072]
2 first 5 singular values [0.787 0.718 0.683 0.671 0.636] right positional energy [0.117 0.114 0.103 0.094 0.088] right token energy [1. 1. 1. 1. 1.] left positional energy [0.126 0.106 0.144 0.109 0.094]
3 first 5 singular values [1.032 0.964 0.617 0.505 0.448] right positional energy [0.443 0.166 0.115 0.202 0.099] right token energy [0.114 0.708 0.582 0.47  0.308] left positional energy [0.321 0.074 0.08  0.093 0.112]
4 first 5 singular values [3.396 1.642 1.082 0.607 0.379] right positional energy [0.951 0.828 0.535 0.118 0.132] right token energy [0.092 0.073 0.207 0.366 0.382] left positional energy [0.339 0.305 0.231 0.102 0.109]
5 first 5 singular values [8.33  3.604 1.679 0.768 0.511] right positional energy [0.911 0.854 0.514 0.177 0.126] right token energy [0.178 0.062 0.203 0.439 0.582] left positional energy [0.32  0.322 0.231 0.172 0.1  ]

Pairwise overlap of L0 non-H2 top OV read directions
[[1.    0.895 0.258 0.872 0.889]
 [0.895 1.    0.215 0.836 0.884]
 [0.258 0.215 1.    0.208 0.254]
 [0.872 0.836 0.208 1.    0.846]
 [0.889 0.884 0.254 0.846 1.   ]]

 direct singular values [5.657 5.265 4.62  4.439 4.349 4.083 3.761 3.713 3.56  3.371 3.329 3.292 3.049 3.037 2.936 2.878 2.742 2.72  2.491 2.357 2.278 1.97  1.5   1.212 1.083 0.841 0.762]
k 5 silhouette 0.046 counts [369 343 320 300 204]
k 10 silhouette 0.052 counts [243 232 190 152 138 136 134 111 106  94]
k 20 silhouette 0.063 counts [137 136 132 126 120  88  82  73  69  65  64  56  55  54  51  51  49  43  43  42]
k 24 silhouette 0.069 counts [108 104 103 102  96  91  90  74  72  72  54  54  50  49  49  48  48  47  43  41  40  36  36  29]
k 28 silhouette 0.069 counts [98 93 85 85 83 78 76 71 62 60 56 54 52 51 47 45 45 44 43 42 41 39 39 37 31 28 26 25]
k 32 silhouette 0.075 counts [88 87 86 78 78 74 74 74 72 57 57 51 47 44 44 42 42 40 39 39 37 37 36 33 33 28 23 21 20 20 19 16]

 via_H2_OV singular values [3.632 2.711 2.593 2.374 2.308 1.975 1.952 1.902 1.773 1.617 1.606 1.403 1.311 1.285 1.17  1.151 1.031 0.877 0.682 0.469 0.392 0.351 0.295 0.266 0.239 0.206 0.185]
k 5 silhouette 0.047 counts [351 338 305 280 262]
k 10 silhouette 0.044 counts [196 192 173 164 156 143 135 131 127 119]
k 20 silhouette 0.043 counts [107 103 101  92  92  92  83  82  82  79  75  71  68  67  64  63  56  54  53  52]
k 24 silhouette 0.042 counts [91 90 86 86 86 77 71 70 67 64 64 63 62 61 60 58 55 52 51 50 50 44 39 39]
k 28 silhouette 0.046 counts [83 79 74 69 69 68 65 63 59 58 58 57 56 54 53 52 52 51 51 50 49 46 44 42 42 40 30 22]
k 32 silhouette 0.044 counts [74 72 67 60 60 59 56 55 55 54 51 50 50 49 49 48 47 47 46 46 46 44 42 41 40 40 37 37 34 31 30 19]

Final projection column clustering (weights only)
k 5 silhouette 0.044 counts [345 345 327 309 210]
k 10 silhouette 0.049 counts [173 160 160 159 159 151 151 151 136 136]
k 20 silhouette 0.048 counts [112  96  87  85  81  81  81  79  78  78  77  74  71  69  68  68  66  64  63  58]
k 24 silhouette 0.041 counts [79 73 73 73 72 72 71 68 67 67 67 64 63 62 61 59 59 59 59 57 56 54 51 50]
k 28 silhouette 0.048 counts [82 79 78 75 73 70 66 60 57 57 56 55 51 51 51 49 47 47 46 45 44 44 44 44 43 42 40 40]
k 32 silhouette 0.039 counts [65 61 61 58 55 54 53 52 52 52 51 50 49 49 48 48 47 46 46 45 45 45 45 45 43 42 41 40 40 39 36 33]

```

## Call 4

```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=160)
d=384
P=np.eye(d)-np.ones((d,d))/d
A={};B={};Q={};K={}
for l in range(4):
    g=W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        s=slice(64*j,64*(j+1))
        Q[l,j]=(W[f"W_q.{l}"][s]*g)@P
        K[l,j]=(W[f"W_k.{l}"][s]*g)@P
        A[l,j]=Q[l,j].T@K[l,j]/8
        B[l,j]=W[f"W_o.{l}"][:,s]@((W[f"W_v.{l}"][s]*g)@P)

print("PAIRWISE KEY-COMPOSITION SCORES: tr(A_lh B_kg)/||A_lh B_kg||")
for l in [1,2,3]:
    for k in range(l):
        scores=np.zeros((6,6))
        for h in range(6):
            for g in range(6):
                c=A[l,h]@B[k,g]
                scores[h,g]=np.trace(c)/np.linalg.norm(c)
        print(f"destination layer {l}, source layer {k}; rows destination heads, columns source heads")
        print(scores)

# Shared low-dimensional read and write subspaces, extracted only from weights.
u,s,vh=svd(A[0,2],full_matrices=False)
rawpos=svd(np.concatenate([u[:,:24],vh[:24].T],1),full_matrices=False)[0][:,:30]
outs=[]
for j in [0,1,4,5]:
    outs.append(svd(B[0,j],full_matrices=False)[0][:,:3])
writepos=svd(np.concatenate(outs,1),full_matrices=False)[0][:,:4]
readq=svd(np.concatenate([Q[1,j] for j in range(6)],0),full_matrices=False)[2][:6].T
readk=svd(np.concatenate([K[1,j] for j in range(6)],0),full_matrices=False)[2][:6].T
print("\nPosition-write shared singular values:",svdvals(np.concatenate(outs,1)))
print("L1 shared Q and K spectra:")
print(svdvals(np.concatenate([Q[1,j] for j in range(6)],0))[:20])
print(svdvals(np.concatenate([K[1,j] for j in range(6)],0))[:20])
print("L1 top Q/K basis overlaps with L0 positional read/write:")
for name,U in [("q",readq),("k",readk)]:
    print(name,"rawpos",np.sum((U.T@rawpos)**2,axis=1),"writepos",np.sum((U.T@writepos)**2,axis=1))

print("\nPer-head QK energy in the shared L1 positional-read subspaces")
for l in range(4):
    for j in range(6):
        a=A[l,j]
        print(f"L{l}H{j}",
              "query/key/both fractions",
              np.round([np.linalg.norm(readq.T@a)**2/np.linalg.norm(a)**2,
                        np.linalg.norm(a@readk)**2/np.linalg.norm(a)**2,
                        np.linalg.norm(readq.T@a@readk)**2/np.linalg.norm(a)**2],3))

print("\nFinal projection singular values after removing shared L0 positional write subspace")
M=W["W_proj.3"].astype(float)
M=M-writepos@(writepos.T@M)
print(svdvals(M)[:40])

# Covariances of first-MLP read weights for direct and H2-OV-mediated features.
_,_,vh=svd(B[0,2],full_matrices=False)
T=vh[:24].T
F=(W["W_fc.0"]*W["ln2_w.0"])@P
C=F@T; D=F@B[0,2]@T
C-=C.mean(0);D-=D.mean(0)
G=C.T@C+D.T@D
e,U=np.linalg.eigh(G)
H=(U/np.sqrt(e+1e-8))@U.T
S=H@(C.T@D+D.T@C)@H
print("\nSymmetric direct/past weight cross-covariance eigenvalues:",np.linalg.eigvalsh(S))
print("Direct/past canonical correlations:",
      svdvals(np.linalg.solve(np.linalg.cholesky(C.T@C+np.eye(24)*1e-8),C.T@D)
              @np.linalg.inv(np.linalg.cholesky(D.T@D+np.eye(24)*1e-8)).T))

```

Output:

```
PAIRWISE KEY-COMPOSITION SCORES: tr(A_lh B_kg)/||A_lh B_kg||
destination layer 1, source layer 0; rows destination heads, columns source heads
[[ 0.072  0.032 -0.113 -0.119  0.062  0.007]
 [ 0.092  0.066 -0.003 -0.097  0.102  0.033]
 [ 0.101  0.084  0.253 -0.035  0.112  0.04 ]
 [ 0.116  0.094  0.233 -0.065  0.123  0.05 ]
 [ 0.149  0.134  0.251 -0.026  0.163  0.081]
 [ 0.078  0.036  0.077 -0.088  0.076 -0.02 ]]
destination layer 2, source layer 0; rows destination heads, columns source heads
[[ 0.08   0.062 -0.08  -0.193  0.096  0.071]
 [ 0.071  0.046 -0.097 -0.131  0.076  0.071]
 [ 0.11   0.087  0.02  -0.124  0.136  0.034]
 [ 0.111  0.08   0.054 -0.128  0.116  0.099]
 [ 0.144  0.108 -0.142 -0.171  0.152  0.1  ]
 [ 0.112  0.073  0.007 -0.185  0.11   0.078]]
destination layer 2, source layer 1; rows destination heads, columns source heads
[[-0.428 -0.092 -0.093 -0.294 -0.244 -0.248]
 [-0.289 -0.405 -0.171  0.063 -0.423 -0.31 ]
 [-0.069 -0.009  0.05   0.091  0.174  0.054]
 [-0.18  -0.14  -0.16   0.033 -0.201 -0.279]
 [-0.215 -0.132 -0.102 -0.059 -0.181 -0.289]
 [-0.215 -0.087  0.007 -0.285 -0.124 -0.386]]
destination layer 3, source layer 0; rows destination heads, columns source heads
[[ 0.057  0.06  -0.042 -0.059  0.088  0.078]
 [-0.058 -0.093  1.173  0.014 -0.13  -0.089]
 [ 0.087  0.059  0.063 -0.083  0.082  0.101]
 [ 0.06   0.03   0.024 -0.025  0.05   0.07 ]
 [ 0.134  0.142 -0.095 -0.021  0.16   0.164]
 [-0.067 -0.07   0.048 -0.058 -0.064 -0.043]]
destination layer 3, source layer 1; rows destination heads, columns source heads
[[-0.211 -0.359 -0.27  -0.281 -0.37  -0.315]
 [ 0.18   0.63   0.879  0.768  0.413  0.037]
 [ 0.029  0.006 -0.067 -0.021 -0.034 -0.037]
 [-0.035 -0.088 -0.075 -0.022 -0.034 -0.   ]
 [-0.053 -0.122 -0.077 -0.081 -0.195 -0.082]
 [ 0.186  0.096  0.147  0.088  0.056  0.122]]
destination layer 3, source layer 2; rows destination heads, columns source heads
[[-0.356 -0.384 -0.289 -0.348 -0.378 -0.252]
 [ 0.392  1.25   0.475  0.487  0.212  0.309]
 [ 0.027 -0.054 -0.127 -0.053 -0.067 -0.061]
 [ 0.017 -0.046 -0.071 -0.077 -0.061 -0.016]
 [ 0.015 -0.058 -0.093 -0.063 -0.158 -0.004]
 [ 0.161  0.217  0.17   0.318  0.228  0.198]]

Position-write shared singular values: [1.992 1.989 1.909 0.44  0.332 0.25  0.144 0.134 0.106 0.093 0.067 0.055]
L1 shared Q and K spectra:
[4.552 3.742 3.189 2.91  2.747 2.576 2.499 2.448 2.439 2.348 2.302 2.266 2.204 2.175 2.106 2.078 2.07  2.02  1.977 1.946]
[11.206  8.605  3.048  2.723  2.39   2.255  2.089  2.053  1.878  1.838  1.749  1.726  1.697  1.659  1.644  1.616  1.564  1.542  1.512  1.465]
L1 top Q/K basis overlaps with L0 positional read/write:
q rawpos [0.124 0.169 0.055 0.034 0.041 0.064] writepos [0.112 0.05  0.137 0.017 0.131 0.003]
k rawpos [0.337 0.425 0.063 0.067 0.048 0.035] writepos [0.931 0.928 0.006 0.013 0.01  0.008]

Per-head QK energy in the shared L1 positional-read subspaces
L0H0 query/key/both fractions [0.019 0.095 0.002]
L0H1 query/key/both fractions [0.04  0.062 0.005]
L0H2 query/key/both fractions [0.017 0.026 0.001]
L0H3 query/key/both fractions [0.021 0.021 0.001]
L0H4 query/key/both fractions [0.028 0.077 0.002]
L0H5 query/key/both fractions [0.08  0.285 0.023]
L1H0 query/key/both fractions [0.589 0.737 0.483]
L1H1 query/key/both fractions [0.699 0.8   0.64 ]
L1H2 query/key/both fractions [0.693 0.806 0.646]
L1H3 query/key/both fractions [0.718 0.812 0.659]
L1H4 query/key/both fractions [0.736 0.823 0.68 ]
L1H5 query/key/both fractions [0.694 0.79  0.614]
L2H0 query/key/both fractions [0.066 0.111 0.028]
L2H1 query/key/both fractions [0.042 0.061 0.009]
L2H2 query/key/both fractions [0.092 0.191 0.05 ]
L2H3 query/key/both fractions [0.058 0.108 0.026]
L2H4 query/key/both fractions [0.074 0.109 0.031]
L2H5 query/key/both fractions [0.07  0.106 0.027]
L3H0 query/key/both fractions [0.016 0.025 0.001]
L3H1 query/key/both fractions [0.025 0.024 0.001]
L3H2 query/key/both fractions [0.019 0.074 0.003]
L3H3 query/key/both fractions [0.021 0.066 0.004]
L3H4 query/key/both fractions [0.021 0.056 0.002]
L3H5 query/key/both fractions [0.021 0.025 0.001]

Final projection singular values after removing shared L0 positional write subspace
[23.444 20.499 19.447 19.217 18.561 18.047 17.735 17.596 16.958 16.68  15.984 15.116 14.697 14.254 13.953 13.474 13.26  12.502 12.295 11.93  11.682 11.572
 10.336  8.315  8.039  7.902  7.436  7.117  5.649  3.193  2.947  2.763  2.668  2.575  2.512  2.399  2.291  2.199  2.193  2.146]

Symmetric direct/past weight cross-covariance eigenvalues: [-0.365 -0.301 -0.248 -0.204 -0.134 -0.125 -0.089 -0.057 -0.016 -0.002  0.035  0.056  0.13   0.148  0.16   0.191  0.217  0.255  0.278  0.319  0.332  0.349
  0.449  0.532]
Direct/past canonical correlations: [0.792 0.706 0.662 0.645 0.581 0.568 0.546 0.511 0.482 0.419 0.407 0.374 0.354 0.342 0.321 0.256 0.241 0.187 0.16  0.123 0.116 0.089 0.017 0.007]

```

## Call 5

```python
import numpy as np
from scipy.linalg import svd, svdvals
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits
import warnings
warnings.filterwarnings("ignore")
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=np.eye(384)-np.ones((384,384))/384
v=W["W_v.0"][128:192].astype(float)
print("L0H2 raw V singular values:",svdvals(v))
F=(W["W_fc.0"]*W["ln2_w.0"])@P
print("Effective first FC singular values:",svdvals(F)[:100])
B=W["W_o.0"][:,128:192]@((v*W["ln1_w.0"])@P)
u,s,vh=svd(B,full_matrices=False)
T=vh[:32].T
C=F@T
D=F@B@T
C-=C.mean(0);D-=D.mean(0)
for label,X,dim in [("FC",F,54),("current",C,26),("past",D,24),
                     ("joint",np.vstack([C/C.std(),D/D.std()]),26),
                     ("last_write",W["W_proj.3"].T,27)]:
    X=X-X.mean(0)
    ica=FastICA(n_components=dim,whiten="unit-variance",max_iter=2000,tol=0.001,random_state=0)
    Z=ica.fit_transform(X)
    ku=kurtosis(Z)
    sk=skew(Z)
    order=np.argsort(-ku)
    print("\nICA",label,"iterations",ica.n_iter_)
    print("kurtosis",ku[order])
    print("absolute skew",abs(sk[order]))
    if label=="joint":
        Zc,Zd=Z[:1536],Z[1536:]
        cor=np.corrcoef(Zc.T,Zd.T)[:dim,dim:]
        ids=np.argsort(-abs(cor).ravel())[:25]
        print("largest current/past component correlations:",
              [(int(i//dim),int(i%dim),round(float(cor.ravel()[i]),3)) for i in ids])

```

Output:

```
L0H2 raw V singular values: [1.954 1.879 1.847 1.804 1.772 1.733 1.721 1.67  1.648 1.571 1.554 1.52  1.517 1.481 1.442 1.331 1.263 1.231 1.051 0.993 0.938 0.908 0.843 0.811 0.778 0.71  0.695 0.645 0.637
 0.599 0.575 0.565 0.548 0.534 0.526 0.505 0.488 0.477 0.456 0.435 0.427 0.409 0.395 0.382 0.371 0.362 0.35  0.344 0.334 0.323 0.319 0.314 0.302 0.298 0.296 0.284 0.281 0.272
 0.265 0.26  0.259 0.25  0.236 0.23 ]
Effective first FC singular values: [17.903 10.353  7.602  7.51   7.11   6.834  6.74   6.432  6.363  6.22   6.139  6.05   6.015  5.936  5.794  5.626  5.575  5.499  5.421  5.309  5.301  5.117  5.076  4.966  4.945
  4.84   4.814  4.73   4.562  4.559  4.484  4.346  4.149  4.084  4.052  3.912  3.822  3.766  3.706  3.527  3.476  3.411  3.338  3.291  3.237  3.206  3.058  2.967  2.94   2.816
  2.717  2.672  2.609  2.459  2.395  2.325  2.232  2.222  2.162  2.091  2.063  2.     1.928  1.897  1.876  1.804  1.795  1.727  1.679  1.669  1.652  1.612  1.593  1.552  1.548
  1.532  1.481  1.453  1.439  1.429  1.414  1.388  1.375  1.357  1.326  1.303  1.293  1.277  1.258  1.257  1.233  1.214  1.205  1.191  1.184  1.182  1.167  1.162  1.147  1.137]

ICA FC iterations 35
kurtosis [410.707 287.93  244.984 228.622 222.002 201.53  200.773 199.788 179.843 176.924 175.208 172.324 167.663 165.597 158.079 144.102 140.899 140.845 133.872 129.187 126.104 120.601
 119.295 118.648 117.181 115.753 111.124 110.303 108.384 103.183  95.991  95.126  90.013  87.682  87.04   84.241  81.121  64.512  58.818  58.158  58.077  56.8    52.114  48.081
  45.611  43.662  42.56   33.739  27.874  24.944  22.982  22.396  22.153  21.154]
absolute skew [15.709 15.317 13.004 13.465 13.916  1.685 11.586 11.96  11.27  11.253  9.972  9.244 10.937 11.232 11.09  10.492  9.422  9.649  7.989  5.665 10.191  8.699  8.489  7.956  9.137
  8.388  9.18   7.331  9.287  7.376  7.555  7.819  6.921  7.733  7.487  7.551  6.674  3.916  4.176  2.717  5.434  5.708  5.783  5.58   2.01   2.608  0.461  1.948  1.265  3.361
  0.905  1.829  1.437  1.346]

ICA current iterations 26
kurtosis [175.737 151.596 138.563 111.998 111.971 110.889 105.854 102.386  92.335  90.776  89.982  81.607  73.501  72.884  72.84   63.765  60.348  47.977  47.973  46.817  43.919  42.948
  38.417  32.454  13.533  11.846]
absolute skew [10.133  8.86   9.066  8.947  8.136  6.624  7.761  7.962  7.798  7.498  7.795  6.869  7.092  6.098  6.267  6.534  5.311  4.848  4.837  5.069  2.363  2.595  4.55   2.724  0.215
  1.789]

ICA past iterations 44
kurtosis [108.842  95.82   75.77   71.787  70.09   55.079  52.308  48.988  41.701  41.041  40.438  38.785  38.486  37.227  36.378  32.012  26.165  26.059  25.576  25.43   24.474  23.256
  19.452  16.556]
absolute skew [6.624 7.392 3.354 5.637 4.412 4.125 3.33  5.444 0.458 0.329 3.558 1.378 3.929 3.569 0.672 2.261 1.559 0.447 0.901 0.216 1.376 1.55  1.567 0.796]

ICA joint iterations 23
kurtosis [151.511 137.77   80.244  71.88   71.21   71.115  68.468  67.31   57.077  53.58   52.088  51.553  47.705  47.02   46.842  46.132  43.19   42.728  39.921  29.39   29.326  28.324
  26.766  26.452  26.368  20.731]
absolute skew [9.166 7.901 6.355 4.399 4.726 4.472 4.957 4.742 4.073 4.841 3.155 3.918 4.213 3.211 0.42  4.118 2.921 3.259 1.116 1.063 2.961 1.374 2.582 1.542 0.048 2.178]
largest current/past component correlations: [(4, 4, 0.352), (12, 12, -0.339), (11, 11, 0.318), (12, 24, 0.309), (12, 10, 0.293), (12, 19, -0.289), (2, 2, -0.289), (12, 18, 0.285), (17, 3, -0.283), (12, 0, 0.28), (12, 11, 0.277), (12, 1, -0.275), (12, 8, -0.264), (12, 17, -0.263), (25, 10, 0.263), (18, 18, 0.261), (12, 2, -0.26), (20, 20, 0.241), (17, 25, 0.241), (10, 5, 0.24), (21, 2, 0.228), (23, 23, 0.221), (5, 24, 0.219), (13, 12, 0.217), (25, 21, 0.217)]

ICA last_write iterations 539
kurtosis [34.771  4.311  1.372  1.289  1.251  1.232  1.165  1.152  1.073  1.04   1.017  0.981  0.97   0.95   0.94   0.938  0.836 -0.511 -0.522 -0.57  -0.578 -0.581 -0.587 -0.603 -0.638
 -0.645 -0.659]
absolute skew [0.572 0.393 0.278 0.14  0.041 0.018 0.341 0.164 0.119 0.132 0.035 0.019 0.097 0.    0.129 0.153 0.228 0.057 0.029 0.039 0.07  0.014 0.085 0.002 0.022 0.05  0.069]

```

## Call 6

```python
import numpy as np
from scipy.linalg import svd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits
import warnings
warnings.filterwarnings("ignore")
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=np.eye(384)-np.ones((384,384))/384
F=(W["W_fc.0"]*W["ln2_w.0"])@P
O=W["W_proj.0"].astype(float)
fn=np.linalg.norm(F,axis=1); on=np.linalg.norm(O,axis=0)
print("FC0 row norm quantiles:",np.quantile(fn,[0,.1,.25,.5,.75,.9,.95,.98,.99,1]))
print("Proj0 column norm quantiles:",np.quantile(on,[0,.1,.25,.5,.75,.9,.95,.98,.99,1]))
print("FC0 energy fraction in top 10/28/54/100/200 rows:",
      [np.sum(np.sort(fn**2)[-k:])/np.sum(fn**2) for k in [10,28,54,100,200]])
B=W["W_o.0"][:,128:192]@((W["W_v.0"][128:192]*W["ln1_w.0"])@P)
u,s,vh=svd(B,full_matrices=False)
T=vh[:28].T
C=F@T
D=F@B@T
aff=W["W_fc.0"]@W["ln2_b.0"]
inds=np.argsort(-fn)[:70]
print("\nTop first-MLP input rows: id, FC norm, output norm, affine term, direct-token norm, OV-mediated norm")
for i in inds:
    print(int(i),np.round([fn[i],on[i],aff[i],np.linalg.norm(C[i]),np.linalg.norm(D[i])],3))
G=F[inds]/fn[inds,None]
S=G@G.T
np.fill_diagonal(S,0)
ii=np.argsort(-abs(S).ravel())[:30]
print("\nLargest cosines between these read rows:")
print([(int(inds[i//len(inds)]),int(inds[i%len(inds)]),round(float(S.ravel()[i]),3)) for i in ii[::2]])

for label,X,dim in [("normalized_FC",F/(fn[:,None]+1e-12),54),
                    ("normalized_current",C/(fn[:,None]+1e-12),26)]:
    ica=FastICA(n_components=dim,whiten="unit-variance",max_iter=2000,tol=.001,random_state=1)
    Z=ica.fit_transform(X)
    ku=kurtosis(Z);sk=skew(Z);ix=np.argsort(-ku)
    print("\n",label,"ICA iterations",ica.n_iter_)
    print("kurtosis",ku[ix])
    print("abs skew",abs(sk[ix]))

```

Output:

```
FC0 row norm quantiles: [0.281 0.32  0.337 0.37  0.884 2.109 2.495 2.955 3.332 4.336]
Proj0 column norm quantiles: [0.132 0.172 0.194 0.237 1.272 1.99  2.154 2.32  2.465 4.795]
FC0 energy fraction in top 10/28/54/100/200 rows: [np.float64(0.07692312786564152), np.float64(0.17914796322096108), np.float64(0.29078317610775556), np.float64(0.443530691951725), np.float64(0.6856764079832841)]

Top first-MLP input rows: id, FC norm, output norm, affine term, direct-token norm, OV-mediated norm
1140 [4.336 2.905 0.026 2.377 0.942]
461 [ 4.182  2.859 -0.055  2.336  0.835]
468 [ 3.795  2.335 -0.753  1.156  0.637]
418 [ 3.75   2.462 -0.684  2.024  0.784]
1115 [ 3.735  2.27  -0.574  1.124  0.558]
1214 [ 3.711  2.29  -0.502  1.224  0.85 ]
827 [ 3.703  2.421 -0.869  0.98   0.702]
98 [ 3.665  4.795 -0.702  1.598  0.295]
1046 [ 3.573  2.307 -0.616  1.515  0.284]
1181 [ 3.551  2.416 -0.673  0.744  0.394]
477 [3.528 2.653 0.02  1.896 0.824]
356 [ 3.461  2.324 -0.49   0.848  0.646]
888 [ 3.37   2.22  -0.687  1.257  0.551]
720 [ 3.355  2.32  -0.712  0.908  0.603]
466 [ 3.342  2.261 -0.472  1.046  0.462]
1456 [ 3.339  2.394 -0.552  0.766  0.707]
100 [3.319 2.817 0.035 1.585 1.136]
1132 [ 3.312  3.353 -0.37   1.497  0.93 ]
883 [ 3.299  2.36  -0.373  0.798  0.591]
779 [ 3.296  2.544 -0.724  1.595  0.591]
361 [ 3.283  2.248 -0.514  0.803  0.618]
796 [ 3.238  2.198 -0.202  1.33   0.74 ]
727 [ 3.196  2.116 -0.65   0.895  0.511]
222 [ 3.156  2.143 -0.686  0.957  0.647]
808 [ 3.108  2.223 -0.482  1.002  0.431]
571 [ 3.096  2.491 -0.092  1.602  0.869]
52 [ 3.087  2.124 -0.396  0.736  0.586]
1189 [ 3.063  2.241 -0.407  1.053  0.711]
642 [ 3.062  2.207 -0.353  0.832  0.466]
1170 [ 3.056  2.546 -0.243  1.186  0.823]
519 [ 3.013  2.178 -0.519  0.632  0.58 ]
1008 [ 2.93   2.075 -0.33   1.021  0.412]
1107 [ 2.928  2.13  -0.064  1.321  0.136]
637 [ 2.9    2.197 -0.474  0.922  0.413]
947 [ 2.887  2.161 -0.495  1.008  0.141]
1054 [2.886 2.548 0.441 1.985 0.649]
1353 [ 2.883  2.224 -0.561  0.851  0.374]
97 [ 2.84   2.421 -0.26   1.197  0.706]
787 [ 2.839  2.136 -0.357  0.751  0.635]
11 [ 2.824  2.093 -0.81   0.638  0.377]
168 [ 2.816  2.224 -0.174  1.39   0.631]
1001 [ 2.816  2.169 -0.395  0.813  0.402]
498 [ 2.811  2.275 -0.522  0.706  0.621]
510 [ 2.797  2.046 -0.415  0.665  0.553]
1082 [ 2.787  2.103 -0.696  0.576  0.439]
51 [ 2.786  2.353 -0.092  0.654  0.759]
1281 [ 2.777  2.14  -0.5    1.283  0.608]
355 [ 2.772  2.052 -0.501  0.836  0.502]
416 [ 2.77   2.52  -0.067  1.89   0.69 ]
564 [ 2.757  2.081 -0.4    0.862  0.415]
1491 [ 2.756  2.099 -0.399  0.701  0.499]
1430 [ 2.751  2.517 -0.415  0.827  0.352]
748 [ 2.748  2.259 -0.339  0.68   0.597]
1260 [ 2.737  2.152 -0.259  0.779  0.214]
241 [ 2.708  2.216 -0.6    0.655  0.465]
1203 [ 2.707  2.149 -0.267  0.779  0.196]
921 [ 2.693  2.075 -0.716  0.819  0.545]
759 [ 2.688  2.121 -0.792  0.527  0.49 ]
585 [ 2.672  2.125 -0.556  1.028  0.49 ]
1045 [ 2.671  2.011 -0.343  0.741  0.397]
1464 [2.614 2.074 0.12  1.443 0.111]
494 [ 2.605  2.392 -0.581  0.818  0.635]
1485 [ 2.603  2.074 -0.597  0.884  0.49 ]
1383 [2.582 2.618 0.074 0.953 0.579]
1194 [ 2.569  2.178 -0.39   0.739  0.512]
928 [ 2.562  2.086 -0.348  0.752  0.385]
194 [ 2.557  2.337 -0.413  0.84   0.644]
298 [ 2.555  2.101 -0.634  0.753  0.491]
1452 [ 2.554  2.093 -0.467  0.708  0.473]
1351 [ 2.549  2.067 -0.56   0.593  0.409]

Largest cosines between these read rows:
[(808, 466, 0.981), (1260, 1203, 0.978), (1491, 356, 0.977), (637, 466, 0.975), (808, 637, 0.973), (642, 1001, 0.973), (928, 808, 0.972), (466, 928, 0.971), (1008, 564, 0.97), (928, 637, 0.965), (1045, 642, 0.964), (1045, 1001, 0.962), (1214, 1189, 0.959), (361, 52, 0.958), (796, 1214, 0.913)]

 normalized_FC ICA iterations 67
kurtosis [40.292 37.64  33.478 33.229 33.089 32.982 32.562 32.084 31.405 31.307 30.799 30.794 30.523 30.362 30.01  29.332 29.127 29.091 28.48  28.282 27.597 25.962 25.863 25.327 24.798
 24.208 22.343 20.543 14.386 10.96  10.908  8.89   7.927  7.612  7.004  7.001  6.635  6.034  5.988  5.886  5.452  3.946  3.631  2.395  1.668  1.312  1.118  1.082  1.012  1.009
  0.987  0.87  -0.426 -0.606]
abs skew [4.418 4.411 4.089 4.173 3.712 3.888 3.803 3.844 3.524 3.654 3.617 3.731 3.423 3.568 3.253 3.419 3.6   3.512 3.414 3.42  3.488 3.61  3.105 3.235 3.171 3.211 2.985 3.222 0.158
 1.733 1.732 2.02  1.067 1.032 1.099 1.192 0.057 1.212 1.145 0.752 1.045 0.859 0.535 0.278 0.038 0.073 0.039 0.105 0.029 0.013 0.15  0.13  0.014 0.034]

 normalized_current ICA iterations 18
kurtosis [22.808 21.977 21.977 21.744 21.41  21.278 21.199 21.125 20.886 19.673 19.282 17.78  17.129 16.471 16.002 14.421 14.364 14.349 14.215 13.186 11.124  9.508  7.83   6.32   1.595
  0.592]
abs skew [3.11  2.868 2.804 2.996 2.841 2.797 2.861 2.853 2.537 2.656 2.737 2.545 2.255 2.339 2.309 2.059 2.039 1.996 1.973 2.016 2.121 1.826 1.125 1.18  0.476 0.094]

```

## Call 7

```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import AgglomerativeClustering
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=np.eye(384)-np.ones((384,384))/384
F=(W["W_fc.0"]*W["ln2_w.0"])@P
norm=np.linalg.norm(F,axis=1)
g=W["ln1_w.0"]
q=(W["W_q.0"][128:192]*g)@P
k=(W["W_k.0"][128:192]*g)@P
A=q.T@k/8
B=W["W_o.0"][:,128:192]@((W["W_v.0"][128:192]*g)@P)
u,s,vh=svd(A,full_matrices=False)
pos=svd(np.c_[u[:,:26],vh[:26].T],full_matrices=False)[0][:,:30]
ub,sb,vhb=svd(B,full_matrices=False)
tin=vhb[:28].T
tout=ub[:28]
ids=np.where(norm>.8)[0]
Z=F[ids]/norm[ids,None]
for threshold in [.10,.15,.20,.25,.30]:
    lab=AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=threshold).fit_predict(Z)
    counts=np.bincount(lab)
    print("threshold",threshold,"clusters",len(counts),"sizes",np.sort(counts)[::-1][:65])
    if threshold==.20:
        chosen=lab
centers=[]; groups=[]
for c in np.argsort(-np.bincount(chosen)):
    members=ids[chosen==c]
    if len(members)<3:continue
    center=F[members].mean(0);center/=np.linalg.norm(center)
    centers.append(center);groups.append(members)
print("\nClusters with >=3 members: group, count, exemplar, mean FC norm, positional/current/past-write energy, affine term")
for c,(r,ids_c) in enumerate(zip(centers,groups)):
    ex=ids_c[np.argmax(norm[ids_c])]
    aff=(W["W_fc.0"][ids_c]@W["ln2_b.0"]).mean()
    print(c,len(ids_c),int(ex),round(float(norm[ids_c].mean()),3),
          np.round([np.linalg.norm(r@pos)**2,np.linalg.norm(r@tin)**2,np.linalg.norm(r@tout)**2],3),
          round(float(aff),3))
R=np.array(centers)
print("\nCosine similarities of direct cluster read vectors and the same vectors composed with H2 OV")
RB=R@B
cross=(R@RB.T)/(np.linalg.norm(R,axis=1)[:,None]*np.linalg.norm(RB,axis=1)[None,:]+1e-12)
for i in range(len(R)):
    ix=np.argsort(-abs(cross[i]))[:3]
    print(i,[(int(j),round(float(cross[i,j]),3)) for j in ix])

```

Output:

```
threshold 0.1 clusters 317 sizes [4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
threshold 0.15 clusters 297 sizes [6 5 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
threshold 0.2 clusters 276 sizes [7 5 5 5 5 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1]
threshold 0.25 clusters 255 sizes [7 5 5 5 5 5 5 5 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1]
threshold 0.3 clusters 230 sizes [7 5 5 5 5 5 5 5 5 5 5 5 5 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1]

Clusters with >=3 members: group, count, exemplar, mean FC norm, positional/current/past-write energy, affine term

[stderr]
Traceback (most recent call last):
  File "<string>", line 42, in <module>
    np.round([np.linalg.norm(r@pos)**2,np.linalg.norm(r@tin)**2,np.linalg.norm(r@tout)**2],3),
                                                                               ~^~~~~
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 28 is different from 384)
```

## Call 8

```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import AgglomerativeClustering
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=np.eye(384)-np.ones((384,384))/384
F=(W["W_fc.0"]*W["ln2_w.0"])@P
norm=np.linalg.norm(F,axis=1)
g=W["ln1_w.0"]
q=(W["W_q.0"][128:192]*g)@P
k=(W["W_k.0"][128:192]*g)@P
A=q.T@k/8
B=W["W_o.0"][:,128:192]@((W["W_v.0"][128:192]*g)@P)
u,s,vh=svd(A,full_matrices=False)
pos=svd(np.c_[u[:,:26],vh[:26].T],full_matrices=False)[0][:,:30]
ub,sb,vhb=svd(B,full_matrices=False)
tin=vhb[:28].T
tout=ub[:,:28]
ids=np.where(norm>.8)[0]
Z=F[ids]/norm[ids,None]
lab=AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=.20).fit_predict(Z)
centers=[];groups=[]
for c in np.argsort(-np.bincount(lab)):
    members=ids[lab==c]
    if len(members)<3:continue
    center=F[members].mean(0);center/=np.linalg.norm(center)
    centers.append(center);groups.append(members)
print("group/count/exemplar/mean FC norm/position-current-pastwrite energies/affine term")
for c,(r,ids_c) in enumerate(zip(centers,groups)):
    ex=ids_c[np.argmax(norm[ids_c])]
    aff=(W["W_fc.0"][ids_c]@W["ln2_b.0"]).mean()
    print(c,len(ids_c),int(ex),round(float(norm[ids_c].mean()),3),
          np.round([np.linalg.norm(r@pos)**2,np.linalg.norm(r@tin)**2,np.linalg.norm(r@tout)**2],3),
          round(float(aff),3))
R=np.array(centers)
RB=R@B
cross=(R@RB.T)/(np.linalg.norm(R,axis=1)[:,None]*np.linalg.norm(RB,axis=1)[None,:]+1e-12)
print("\nLargest direct-to-OV-mediated cluster read cosines")
for i in range(len(R)):
    ix=np.argsort(-abs(cross[i]))[:3]
    print(i,[(int(j),round(float(cross[i,j]),3)) for j in ix])
print("\nFC0 energies in position/current/pastwrite subspaces, by row-norm band")
for lo,hi in [(0,.5),(.5,1),(1,2),(2,3),(3,10)]:
    sel=(norm>=lo)&(norm<hi)
    f=F[sel]
    print(lo,hi,int(sel.sum()),np.round([np.linalg.norm(f@U)**2/np.linalg.norm(f)**2 for U in [pos,tin,tout]],3))

```

Output:

```
group/count/exemplar/mean FC norm/position-current-pastwrite energies/affine term
0 7 1038 2.053 [0.026 0.039 0.052] -0.426
1 5 947 2.1 [0.01  0.11  0.026] -0.341
2 5 921 2.309 [0.022 0.083 0.114] -0.559
3 5 1486 1.109 [0.004 0.445 0.03 ] 0.113
4 5 575 1.267 [0.004 0.329 0.016] 0.02
5 4 35 1.977 [0.016 0.14  0.04 ] -0.12
6 4 466 2.978 [0.032 0.098 0.093] -0.444
7 4 983 1.358 [0.005 0.547 0.026] 0.14
8 4 1144 1.906 [0.026 0.08  0.11 ] -0.299
9 4 1472 1.466 [0.003 0.512 0.027] 0.112
10 4 642 2.732 [0.028 0.074 0.081] -0.36
11 4 298 2.349 [0.031 0.086 0.131] -0.575
12 4 106 1.413 [0.004 0.528 0.03 ] 0.087
13 4 1022 2.182 [0.032 0.084 0.11 ] -0.344
14 4 361 2.614 [0.025 0.059 0.122] -0.377
15 4 533 1.382 [0.005 0.236 0.057] 0.03
16 3 222 2.349 [0.037 0.09  0.114] -0.491
17 3 605 1.504 [0.005 0.374 0.018] -0.008
18 3 413 1.849 [0.005 0.218 0.016] -0.033
19 3 1382 1.645 [0.004 0.237 0.042] -0.014
20 3 293 1.555 [0.005 0.49  0.016] 0.113
21 3 1074 2.332 [0.028 0.093 0.158] -0.478
22 3 1031 1.554 [0.005 0.324 0.041] 0.06
23 3 11 2.509 [0.024 0.059 0.08 ] -0.783
24 3 1260 2.592 [0.007 0.084 0.026] -0.259
25 3 1107 2.163 [0.012 0.201 0.054] -0.068
26 3 1485 2.131 [0.049 0.109 0.13 ] -0.502
27 3 1214 3.337 [0.02  0.134 0.15 ] -0.37
28 3 720 2.808 [0.02  0.073 0.111] -0.544
29 3 283 1.844 [0.006 0.276 0.039] -0.003
30 3 1135 1.354 [0.004 0.503 0.029] 0.102
31 3 975 1.722 [0.002 0.342 0.027] 0.05
32 3 727 2.57 [0.035 0.078 0.102] -0.465
33 3 355 2.469 [0.029 0.098 0.106] -0.414
34 3 1082 2.675 [0.027 0.042 0.091] -0.683
35 3 5 1.544 [0.005 0.485 0.033] 0.003
36 3 256 1.921 [0.007 0.301 0.043] -0.04
37 3 773 1.773 [0.005 0.369 0.041] 0.272
38 3 468 2.757 [0.031 0.1   0.112] -0.47
39 3 1407 1.579 [0.003 0.368 0.022] 0.119

Largest direct-to-OV-mediated cluster read cosines
0 [(0, 0.186), (21, 0.115), (27, 0.1)]
1 [(1, -0.183), (32, -0.154), (0, -0.145)]
2 [(39, -0.161), (10, 0.142), (2, 0.131)]
3 [(3, -0.518), (13, 0.484), (20, 0.251)]
4 [(8, 0.426), (10, -0.254), (4, -0.216)]
5 [(10, 0.242), (33, -0.172), (0, -0.168)]
6 [(29, -0.19), (6, 0.18), (34, 0.089)]
7 [(33, 0.488), (7, -0.462), (4, 0.225)]
8 [(8, 0.143), (4, -0.103), (33, -0.099)]
9 [(9, -0.593), (10, 0.232), (1, 0.184)]
10 [(32, 0.16), (10, 0.151), (12, -0.121)]
11 [(35, -0.159), (11, 0.126), (0, 0.086)]
12 [(32, 0.518), (12, -0.503), (10, 0.281)]
13 [(3, -0.196), (13, 0.14), (18, -0.089)]
14 [(14, 0.094), (37, -0.087), (15, -0.079)]
15 [(28, 0.29), (15, -0.266), (36, 0.212)]
16 [(16, 0.135), (28, 0.098), (0, 0.085)]
17 [(17, -0.506), (20, 0.182), (36, -0.181)]
18 [(23, -0.259), (21, -0.241), (5, -0.192)]
19 [(19, -0.429), (5, -0.205), (29, -0.191)]
20 [(16, 0.472), (20, -0.224), (18, 0.22)]
21 [(30, -0.198), (21, 0.189), (0, 0.117)]
22 [(22, -0.467), (6, -0.329), (25, 0.27)]
23 [(23, 0.115), (31, 0.078), (15, 0.078)]
24 [(24, -0.138), (27, -0.133), (16, -0.106)]
25 [(25, -0.289), (18, -0.231), (1, -0.2)]
26 [(17, 0.147), (26, 0.121), (18, -0.101)]
27 [(27, 0.193), (1, -0.104), (20, 0.102)]
28 [(28, 0.157), (36, 0.094), (35, 0.087)]
29 [(29, -0.406), (6, 0.348), (25, -0.219)]
30 [(21, 0.483), (30, -0.455), (25, 0.178)]
31 [(31, -0.482), (38, 0.394), (25, -0.227)]
32 [(32, 0.189), (12, -0.136), (10, 0.096)]
33 [(7, -0.175), (34, 0.163), (33, 0.152)]
34 [(36, -0.146), (20, 0.098), (28, -0.093)]
35 [(35, -0.532), (11, 0.514), (18, 0.224)]
36 [(36, -0.416), (34, 0.284), (28, -0.211)]
37 [(37, -0.45), (14, 0.41), (15, -0.244)]
38 [(31, -0.222), (38, 0.201), (25, -0.091)]
39 [(39, -0.474), (2, 0.421), (18, 0.218)]

FC0 energies in position/current/pastwrite subspaces, by row-norm band
0 0.5 1076 [0.042 0.081 0.057]
0.5 1 86 [0.041 0.152 0.101]
1 2 196 [0.023 0.213 0.147]
2 3 147 [0.027 0.142 0.12 ]
3 10 31 [0.025 0.147 0.13 ]

```

## Call 9

```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import AgglomerativeClustering
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=np.eye(384)-np.ones((384,384))/384
F=(W["W_fc.0"]*W["ln2_w.0"])@P
fn=np.linalg.norm(F,axis=1)
B=W["W_o.0"][:,128:192]@((W["W_v.0"][128:192]*W["ln1_w.0"])@P)
u,s,vh=svd(B,full_matrices=False)
T=vh[:28].T
ids=np.where(fn>.8)[0]
Z=F[ids]/fn[ids,None]
lab=AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=.3).fit_predict(Z)
R=[];sizes=[];ex=[]
for c in np.argsort(-np.bincount(lab)):
    members=ids[lab==c]
    if len(members)<2: continue
    r=F[members].mean(0);r/=np.linalg.norm(r)
    R.append(r);sizes.append(len(members));ex.append(int(members[np.argmax(fn[members])]))
R=np.array(R)
eng=np.sum((R@T)**2,1)
bias=np.array([float(W["W_fc.0"][i]@W["ln2_b.0"]) for i in ex])
print("Repeated read-direction groups: group, size, exemplar, current energy, weighted ln1 gain, bias")
for i,r in enumerate(R):
    print(i,sizes[i],ex[i],round(eng[i],3),round(float(np.sum(r*r*W["ln1_w.0"])),3),round(bias[i],3))
ix=np.where(eng>.19)[0]
print("\nCurrent-like group indices:",ix)
G=R[ix]@R[ix].T
print("Their Gram eigenvalues:",np.linalg.eigvalsh(G))
print("Their cosine matrix:")
print(G)
print("\nTop mapping from each current-like group to all OV-composed group directions")
RB=R@B
RB/=np.linalg.norm(RB,axis=1,keepdims=True)
S=R[ix]@RB.T
for row,i in enumerate(ix):
    best=np.argsort(-S[row])[:3]
    print(int(i),[(int(j),round(float(S[row,j]),3)) for j in best])

```

Output:

```
Repeated read-direction groups: group, size, exemplar, current energy, weighted ln1 gain, bias
0 7 1038 0.039 0.135 -0.512
1 5 605 0.35 0.139 -0.056
2 5 355 0.101 0.149 -0.501
3 5 298 0.091 0.141 -0.634
4 5 642 0.07 0.161 -0.353
5 5 921 0.083 0.141 -0.716
6 5 947 0.11 0.141 -0.495
7 5 1486 0.445 0.14 0.144
8 5 575 0.329 0.14 -0.01
9 5 1074 0.095 0.157 -0.623
10 5 11 0.069 0.153 -0.81
11 5 1082 0.046 0.138 -0.696
12 5 418 0.294 0.149 -0.684
13 4 35 0.14 0.151 -0.181
14 4 1485 0.108 0.171 -0.597
15 4 533 0.236 0.126 0.063
16 4 106 0.528 0.14 0.148
17 4 975 0.327 0.129 0.038
18 4 413 0.215 0.135 -0.05
19 4 468 0.089 0.149 -0.753
20 4 720 0.076 0.164 -0.712
21 4 1144 0.08 0.144 -0.336
22 4 888 0.133 0.141 -0.687
23 4 1472 0.512 0.14 0.164
24 4 361 0.059 0.158 -0.514
25 4 727 0.084 0.141 -0.65
26 4 222 0.096 0.155 -0.686
27 4 787 0.057 0.163 -0.357
28 4 1407 0.358 0.13 0.12
29 4 1382 0.237 0.138 -0.03
30 4 51 0.078 0.146 -0.092
31 4 1022 0.084 0.149 -0.408
32 4 411 0.303 0.145 0.077
33 4 983 0.547 0.13 0.19
34 4 466 0.098 0.149 -0.472
35 4 283 0.266 0.14 0.036
36 3 1214 0.134 0.146 -0.502
37 3 1135 0.503 0.127 0.106
38 3 1055 0.153 0.141 0.01
39 3 5 0.485 0.138 -0.005
40 3 256 0.301 0.144 -0.046
41 3 356 0.066 0.152 -0.49
42 3 1302 0.507 0.136 0.109
43 3 1107 0.201 0.155 -0.064
44 3 773 0.369 0.134 0.305
45 3 293 0.49 0.13 0.123
46 3 1046 0.195 0.144 -0.616
47 3 1031 0.324 0.131 0.025
48 3 1464 0.3 0.132 0.12
49 2 782 0.086 0.153 -0.268
50 2 1008 0.11 0.164 -0.33
51 2 827 0.076 0.144 -0.869
52 2 883 0.07 0.138 -0.373
53 2 781 0.032 0.146 -0.296
54 2 1456 0.055 0.164 -0.552
55 2 611 0.068 0.149 -0.067
56 2 196 0.096 0.257 -0.113
57 2 1166 0.135 0.163 -0.354
58 2 1115 0.09 0.145 -0.574
59 2 519 0.051 0.164 -0.519
60 2 441 0.363 0.174 -0.311
61 2 459 0.133 0.149 -0.12
62 2 1288 0.129 0.152 -0.345

Current-like group indices: [ 1  7  8 12 15 16 17 18 23 28 29 32 33 35 37 39 40 42 43 44 45 46 47 48 60]
Their Gram eigenvalues: [0.219 0.397 0.704 0.803 0.828 0.868 0.873 0.895 0.929 0.947 0.959 0.993 1.017 1.043 1.073 1.089 1.124 1.138 1.15  1.186 1.208 1.243 1.262 1.272 1.779]
Their cosine matrix:
[[ 1.    -0.044 -0.034  0.015  0.007 -0.017  0.001 -0.041 -0.049 -0.036 -0.088 -0.061 -0.042 -0.027 -0.015  0.008 -0.005 -0.062 -0.012 -0.01  -0.039  0.054  0.047 -0.056  0.057]
 [-0.044  1.    -0.04  -0.008  0.012 -0.068 -0.027  0.033 -0.032 -0.083  0.056 -0.066 -0.006 -0.099 -0.011 -0.048  0.032 -0.031 -0.029 -0.037 -0.05  -0.051  0.007 -0.01  -0.112]
 [-0.034 -0.04   1.     0.008  0.015 -0.136 -0.056 -0.004 -0.042 -0.034 -0.013 -0.057 -0.065  0.038 -0.006  0.006  0.013 -0.013  0.007  0.026 -0.07   0.024 -0.028  0.011 -0.05 ]
 [ 0.015 -0.008  0.008  1.    -0.031 -0.018  0.006  0.029  0.016 -0.025  0.016 -0.102 -0.007  0.045 -0.05   0.01   0.018 -0.066  0.014  0.027  0.003  0.039  0.047  0.017  0.065]
 [ 0.007  0.012  0.015 -0.031  1.     0.01  -0.029 -0.031 -0.017 -0.004 -0.01  -0.027 -0.003 -0.05   0.003 -0.007 -0.056 -0.059 -0.099 -0.044 -0.006 -0.021 -0.06  -0.071 -0.042]
 [-0.017 -0.068 -0.136 -0.018  0.01   1.     0.033  0.027 -0.113 -0.016  0.002 -0.133 -0.071 -0.002  0.002 -0.082 -0.024 -0.049 -0.002  0.016 -0.047 -0.005 -0.039 -0.056  0.041]
 [ 0.001 -0.027 -0.056  0.006 -0.029  0.033  1.    -0.021 -0.021 -0.008 -0.02  -0.063 -0.008 -0.048 -0.089 -0.04   0.021 -0.028  0.003 -0.013 -0.014 -0.069  0.013 -0.033 -0.022]
 [-0.041  0.033 -0.004  0.029 -0.031  0.027 -0.021  1.    -0.075 -0.051 -0.034 -0.044  0.012 -0.033 -0.027 -0.02  -0.091 -0.015  0.013 -0.052 -0.028 -0.028 -0.031 -0.002 -0.019]
 [-0.049 -0.032 -0.042  0.016 -0.017 -0.113 -0.021 -0.075  1.    -0.003  0.03  -0.058 -0.111 -0.022  0.007 -0.006 -0.001 -0.086 -0.021 -0.03  -0.091  0.01   0.03  -0.032  0.075]
 [-0.036 -0.083 -0.034 -0.025 -0.004 -0.016 -0.008 -0.051 -0.003  1.    -0.004 -0.066 -0.031 -0.037 -0.008 -0.039  0.045 -0.032 -0.042 -0.022 -0.014  0.004 -0.006 -0.005 -0.038]
 [-0.088  0.056 -0.013  0.016 -0.01   0.002 -0.02  -0.034  0.03  -0.004  1.    -0.037 -0.017 -0.098 -0.023 -0.029 -0.104 -0.047 -0.032 -0.012 -0.007  0.009 -0.063 -0.003 -0.03 ]
 [-0.061 -0.066 -0.057 -0.102 -0.027 -0.133 -0.063 -0.044 -0.058 -0.066 -0.037  1.    -0.065 -0.019 -0.082 -0.045 -0.067 -0.065 -0.008 -0.078 -0.082 -0.025 -0.099 -0.046  0.006]
 [-0.042 -0.006 -0.065 -0.007 -0.003 -0.071 -0.008  0.012 -0.111 -0.031 -0.017 -0.065  1.    -0.03  -0.052 -0.009  0.01  -0.048 -0.02  -0.004 -0.065 -0.004 -0.013  0.015  0.039]
 [-0.027 -0.099  0.038  0.045 -0.05  -0.002 -0.048 -0.033 -0.022 -0.037 -0.098 -0.019 -0.03   1.    -0.001  0.025 -0.035 -0.046 -0.009  0.012  0.006  0.038 -0.024  0.03  -0.049]
 [-0.015 -0.011 -0.006 -0.05   0.003  0.002 -0.089 -0.027  0.007 -0.008 -0.023 -0.082 -0.052 -0.001  1.    -0.056 -0.007 -0.044 -0.057 -0.029 -0.004 -0.075  0.037 -0.043 -0.075]
 [ 0.008 -0.048  0.006  0.01  -0.007 -0.082 -0.04  -0.02  -0.006 -0.039 -0.029 -0.045 -0.009  0.025 -0.056  1.     0.026 -0.028 -0.012 -0.011 -0.042  0.007 -0.021 -0.032  0.719]
 [-0.005  0.032  0.013  0.018 -0.056 -0.024  0.021 -0.091 -0.001  0.045 -0.104 -0.067  0.01  -0.035 -0.007  0.026  1.    -0.058  0.004 -0.012 -0.019 -0.084 -0.055 -0.013 -0.024]
 [-0.062 -0.031 -0.013 -0.066 -0.059 -0.049 -0.028 -0.015 -0.086 -0.032 -0.047 -0.065 -0.048 -0.046 -0.044 -0.028 -0.058  1.    -0.009  0.039 -0.031 -0.009 -0.05   0.015 -0.005]
 [-0.012 -0.029  0.007  0.014 -0.099 -0.002  0.003  0.013 -0.021 -0.042 -0.032 -0.008 -0.02  -0.009 -0.057 -0.012  0.004 -0.009  1.    -0.042 -0.    -0.006 -0.038  0.003 -0.012]
 [-0.01  -0.037  0.026  0.027 -0.044  0.016 -0.013 -0.052 -0.03  -0.022 -0.012 -0.078 -0.004  0.012 -0.029 -0.011 -0.012  0.039 -0.042  1.    -0.022 -0.009 -0.044 -0.012 -0.036]
 [-0.039 -0.05  -0.07   0.003 -0.006 -0.047 -0.014 -0.028 -0.091 -0.014 -0.007 -0.082 -0.065  0.006 -0.004 -0.042 -0.019 -0.031 -0.    -0.022  1.    -0.116 -0.032 -0.036  0.015]
 [ 0.054 -0.051  0.024  0.039 -0.021 -0.005 -0.069 -0.028  0.01   0.004  0.009 -0.025 -0.004  0.038 -0.075  0.007 -0.084 -0.009 -0.006 -0.0
```

## Call 10

```python
import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=np.eye(384)-np.ones((384,384))/384
A={}; B={}; kb={}; qs={}; ks={}
for l in range(4):
    g=W[f"ln1_w.{l}"].astype(float)
    b=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        sl=slice(64*j,64*(j+1))
        q=W[f"W_q.{l}"][sl].astype(float)
        k=W[f"W_k.{l}"][sl].astype(float)
        qs[l,j]=(q*g)@P
        ks[l,j]=(k*g)@P
        A[l,j]=qs[l,j].T@ks[l,j]/8
        B[l,j]=W[f"W_o.{l}"][:,sl]@((W[f"W_v.{l}"][sl]*g)@P)
        kb[l,j]=(q@b)@ks[l,j]/8
_,_,vh=svd(np.concatenate([ks[1,j] for j in range(6)]),full_matrices=False)
poskey=vh[:2].T
reference=sum(kb[1,j] for j in range(6));reference/=np.linalg.norm(reference)
print("Per-head affine key-preference direction: norm, positional fraction, cosine with L1 mean, coordinates on shared key basis")
for l in range(4):
    for j in range(6):
        z=kb[l,j]
        print(f"L{l}H{j}",round(np.linalg.norm(z),3),
              round(np.linalg.norm(z@poskey)**2/np.sum(z*z),3),
              round(z@reference/np.linalg.norm(z),3),np.round(z@poskey,3))

print("\nL1 QK restrictions to common positional key basis: singular values and affine coordinates")
for j in range(6):
    z=A[1,j]@poskey
    print(j,"singular values",np.linalg.svd(z,compute_uv=False),
          "affine",kb[1,j]@poskey,
          "QK energy fraction",np.linalg.norm(z)**2/np.linalg.norm(A[1,j])**2)

# Compare L0 positional bilinear forms in a metric supplied by the
# Q/K weights themselves. No position vectors or attention evaluation.
q,k=qs[0,2],ks[0,2]
G=q.T@q+k.T@k
e,U=np.linalg.eigh(G)
ix=np.argsort(-e)[:24]
T=U[:,ix]/np.sqrt(e[ix])
a=T.T@A[0,2]@T
u,s,vh=svd(a)
R=u@vh
powers={n:np.linalg.matrix_power(R,n) for n in range(-8,9)}
print("\nL0 positional bilinear forms: energy in shared positional span and cosine with powers of H2 polar factor")
span=U[:,ix]
for j in range(6):
    z=T.T@A[0,j]@T
    vals=[(n,float(np.sum(z*r)/np.linalg.norm(z)/np.linalg.norm(r))) for n,r in powers.items()]
    print(j,"energy fraction",round(np.linalg.norm(span.T@A[0,j]@span)**2/np.linalg.norm(A[0,j])**2,3),
          "largest correlations",sorted(vals,key=lambda t:-abs(t[1]))[:5])

```

Output:

```
Per-head affine key-preference direction: norm, positional fraction, cosine with L1 mean, coordinates on shared key basis
L0H0 0.014 0.014 -0.115 [0.001 0.001]
L0H1 0.003 0.256 -0.401 [0.001 0.002]
L0H2 0.143 0.034 0.148 [-0.026  0.005]
L0H3 0.054 0.024 -0.038 [-0.001  0.008]
L0H4 0.011 0.024 -0.139 [0.002 0.   ]
L0H5 0.016 0.267 0.388 [-0.004 -0.008]
L1H0 1.357 0.935 0.977 [-1.149 -0.634]
L1H1 2.285 0.963 0.987 [-2.03  -0.953]
L1H2 2.588 0.974 0.99 [-2.316 -1.078]
L1H3 2.351 0.963 0.986 [-2.053 -1.052]
L1H4 2.29 0.977 0.986 [-2.099 -0.847]
L1H5 1.753 0.963 0.985 [-1.529 -0.787]
L2H0 1.699 0.697 0.808 [-1.37  -0.365]
L2H1 0.745 0.66 0.782 [-0.594 -0.114]
L2H2 0.755 0.633 0.788 [-0.566 -0.202]
L2H3 1.462 0.72 0.817 [-1.212 -0.265]
L2H4 1.577 0.712 0.807 [-1.305 -0.259]
L2H5 1.509 0.718 0.816 [-1.246 -0.286]
L3H0 0.258 0.112 0.318 [-0.086 -0.01 ]
L3H1 0.241 0.031 -0.13 [ 0.039 -0.017]
L3H2 0.29 0.426 0.623 [-0.186 -0.033]
L3H3 0.36 0.504 0.655 [-0.255 -0.021]
L3H4 0.307 0.407 0.604 [-0.194 -0.027]
L3H5 0.165 0.025 0.081 [-0.02   0.016]

L1 QK restrictions to common positional key basis: singular values and affine coordinates
0 singular values [0.559 0.469] affine [-1.149 -0.634] QK energy fraction 0.6773024355185239
1 singular values [0.827 0.648] affine [-2.03  -0.953] QK energy fraction 0.6383160610963774
2 singular values [0.973 0.833] affine [-2.316 -1.078] QK energy fraction 0.6832928077625905
3 singular values [0.874 0.722] affine [-2.053 -1.052] QK energy fraction 0.6805527530615861
4 singular values [0.923 0.852] affine [-2.099 -0.847] QK energy fraction 0.7513282732973771
5 singular values [0.855 0.709] affine [-1.529 -0.787] QK energy fraction 0.690588687213962

L0 positional bilinear forms: energy in shared positional span and cosine with powers of H2 polar factor
0 energy fraction 0.093 largest correlations [(6, 0.06612371350005328), (-1, 0.06031062706529956), (-2, 0.05388427922521157), (-8, 0.05341448882942065), (0, 0.05140020182302721)]
1 energy fraction 0.154 largest correlations [(0, -0.09849675877345045), (6, -0.05330293644138276), (-7, -0.05040156918272033), (-1, 0.03546207745529068), (4, -0.03297801846379657)]
2 energy fraction 0.939 largest correlations [(1, 0.938342973486363), (-8, 0.15778670829355615), (-5, 0.10940431939622357), (7, 0.1094043193962234), (-1, 0.09781076256528556)]
3 energy fraction 0.642 largest correlations [(-3, -0.18317491798301833), (-7, 0.16529018226894687), (1, 0.15438986638487048), (8, 0.1372272886189476), (-4, -0.13035988654392894)]
4 energy fraction 0.086 largest correlations [(0, -0.1798125903620195), (6, -0.09457391634488124), (-2, -0.08755623864253599), (2, -0.08203781892971149), (4, -0.078040861136126)]
5 energy fraction 0.262 largest correlations [(0, -0.12106072108049415), (-3, 0.08834114118593185), (5, 0.08823156225109004), (-6, -0.06895045891570958), (2, -0.0614710954777797)]

```

## Call 11

```python
import numpy as np
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=np.eye(384)-np.ones((384,384))/384
F=(W["W_fc.0"]*W["ln2_w.0"])@P
fn=np.linalg.norm(F,axis=1)
B=W["W_o.0"][:,128:192]@((W["W_v.0"][128:192]*W["ln1_w.0"])@P)
ids=np.where(fn>.8)[0]
Z=F[ids]/fn[ids,None]
lab=AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=.3).fit_predict(Z)
R=[];U=[];groups=[]
for c in np.argsort(-np.bincount(lab)):
    members=ids[lab==c]
    if len(members)<2:continue
    r=F[members].mean(0);r/=np.linalg.norm(r)
    o=W["W_proj.0"][:,members].mean(1).astype(float);o/=np.linalg.norm(o)
    R.append(r);U.append(o);groups.append(members)
R=np.array(R);U=np.array(U).T
cur=np.array([1,7,8,15,16,17,18,23,28,29,32,33,35,37,39,40,42,43,44,45,47,48])
past=np.array([i for i in range(len(R)) if i not in cur and i not in [12,46,60]])
RB=R[past]@B
RB/=np.linalg.norm(RB,axis=1,keepdims=True)
S=R[cur]@RB.T
ri,ci=linear_sum_assignment(-S)
good=S[ri,ci]>.23
cur=cur[ri[good]];past=past[ci[good]]
print("Matched direct/past read groups:",[(int(i),int(j),round(float(R[i]@(R[j]@B)/np.linalg.norm(R[j]@B)),3)) for i,j in zip(cur,past)])
Uc=U[:,cur];Up=U[:,past]
print("Current/past writer pair cosines:",np.sum(Uc*Up,axis=0))
n=len(cur)
H=np.eye(n)-np.ones((n,n))/n
for l in range(1,4):
    F=(W[f"W_fc.{l}"]*W[f"ln2_w.{l}"])@P
    C=F@Uc@H;D=F@Up@H
    C-=C.mean(0);D-=D.mean(0)
    C/=np.linalg.norm(C,axis=0,keepdims=True)
    D/=np.linalg.norm(D,axis=0,keepdims=True)
    cross=C.T@D
    sym=(cross+cross.T)/2
    e,v=np.linalg.eigh(sym)
    print("\nLayer",l,"symmetric cross-feature weight-correlation eigenvalues:",e)
    print("most negative eigenvector (group,value):",[(int(cur[i]),round(float(v[i,0]),3)) for i in np.argsort(v[:,0])])
    print("diagonal:",np.diag(cross))
    print("top positive directed correlations:",
          [(int(cur[i//n]),int(cur[i%n]),round(float(cross.ravel()[i]),3)) for i in np.argsort(-cross.ravel())[:15]])

```

Output:

```
Matched direct/past read groups: [(1, 50, 0.345), (7, 31, 0.484), (8, 21, 0.426), (15, 20, 0.289), (16, 25, 0.492), (17, 19, 0.372), (18, 58, 0.269), (23, 22, 0.402), (28, 5, 0.402), (29, 55, 0.242), (33, 2, 0.491), (35, 34, 0.34), (37, 9, 0.466), (39, 3, 0.513), (40, 11, 0.312), (42, 14, 0.493), (44, 24, 0.41), (45, 26, 0.453), (47, 27, 0.327), (48, 36, 0.365)]
Current/past writer pair cosines: [0.117 0.113 0.115 0.162 0.166 0.154 0.172 0.147 0.206 0.034 0.068 0.152 0.211 0.124 0.208 0.207 0.156 0.155 0.123 0.157]

Layer 1 symmetric cross-feature weight-correlation eigenvalues: [-0.687 -0.142 -0.127 -0.039 -0.     0.014  0.033  0.071  0.094  0.115  0.184  0.235  0.289  0.307  0.359  0.402  0.48   0.515  0.657  0.749]
most negative eigenvector (group,value): [(45, -0.433), (16, -0.384), (33, -0.315), (23, -0.284), (8, -0.18), (48, -0.155), (18, -0.079), (29, -0.004), (15, 0.016), (35, 0.026), (37, 0.032), (42, 0.088), (1, 0.112), (44, 0.113), (47, 0.149), (40, 0.168), (28, 0.174), (17, 0.264), (39, 0.333), (7, 0.355)]
diagonal: [0.07  0.012 0.077 0.138 0.154 0.221 0.258 0.167 0.294 0.087 0.053 0.196 0.332 0.092 0.362 0.258 0.164 0.127 0.267 0.18 ]
top positive directed correlations: [(40, 40, 0.362), (37, 37, 0.332), (28, 28, 0.294), (47, 47, 0.267), (1, 42, 0.261), (42, 42, 0.258), (18, 18, 0.258), (23, 40, 0.235), (39, 16, 0.231), (17, 17, 0.221), (16, 39, 0.209), (28, 33, 0.199), (45, 7, 0.196), (35, 35, 0.196), (45, 44, 0.194)]

Layer 2 symmetric cross-feature weight-correlation eigenvalues: [-0.329 -0.133 -0.068 -0.013 -0.009 -0.     0.043  0.09   0.122  0.134  0.181  0.253  0.274  0.292  0.303  0.358  0.426  0.465  0.552  0.651]
most negative eigenvector (group,value): [(45, -0.408), (16, -0.401), (33, -0.386), (23, -0.25), (29, -0.148), (48, -0.082), (18, -0.071), (8, -0.051), (35, -0.032), (1, -0.001), (15, 0.027), (42, 0.081), (44, 0.087), (37, 0.102), (40, 0.154), (47, 0.175), (28, 0.21), (39, 0.282), (17, 0.325), (7, 0.339)]
diagonal: [0.077 0.105 0.137 0.214 0.159 0.178 0.195 0.13  0.322 0.088 0.071 0.229 0.325 0.168 0.315 0.25  0.208 0.111 0.133 0.175]
top positive directed correlations: [(37, 37, 0.325), (28, 28, 0.322), (40, 40, 0.315), (42, 42, 0.25), (35, 35, 0.229), (15, 15, 0.214), (44, 44, 0.208), (18, 18, 0.195), (17, 17, 0.178), (48, 48, 0.175), (39, 39, 0.168), (16, 16, 0.159), (23, 40, 0.145), (16, 39, 0.138), (8, 8, 0.137)]

Layer 3 symmetric cross-feature weight-correlation eigenvalues: [-0.18  -0.092 -0.073 -0.031 -0.012 -0.     0.024  0.057  0.084  0.118  0.169  0.199  0.261  0.285  0.31   0.383  0.409  0.489  0.55   0.743]
most negative eigenvector (group,value): [(7, -0.491), (39, -0.303), (28, -0.227), (23, -0.197), (37, -0.181), (35, -0.154), (1, -0.147), (18, -0.081), (8, -0.053), (17, -0.023), (40, 0.041), (48, 0.057), (15, 0.068), (42, 0.076), (45, 0.14), (47, 0.178), (44, 0.199), (33, 0.248), (16, 0.284), (29, 0.491)]
diagonal: [ 0.119 -0.005  0.12   0.158  0.262  0.217  0.222  0.18   0.228  0.049  0.232  0.194  0.355  0.186  0.255  0.28   0.171  0.18   0.121  0.167]
top positive directed correlations: [(37, 37, 0.355), (42, 42, 0.28), (16, 16, 0.262), (40, 40, 0.255), (33, 33, 0.232), (28, 28, 0.228), (18, 18, 0.222), (17, 17, 0.217), (35, 35, 0.194), (39, 39, 0.186), (16, 39, 0.186), (23, 23, 0.18), (45, 45, 0.18), (44, 44, 0.171), (48, 48, 0.167)]

```

## Call 12

```python
import numpy as np
from scipy.linalg import svd, svdvals
from sklearn.cluster import AgglomerativeClustering
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=np.eye(384)-np.ones((384,384))/384
F=(W["W_fc.0"]*W["ln2_w.0"])@P
fn=np.linalg.norm(F,axis=1)
ids=np.where(fn>.8)[0]
Z=F[ids]/fn[ids,None]
lab=AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=.3).fit_predict(Z)
R=[];groups=[]
for c in np.argsort(-np.bincount(lab)):
    members=ids[lab==c]
    if len(members)<2:continue
    r=F[members].mean(0);r/=np.linalg.norm(r)
    R.append(r);groups.append(members)
R=np.array(R)
cur=np.array([1,7,8,15,16,17,18,23,28,29,32,33,35,37,39,40,42,43,44,45,47,48])
past=np.array([i for i in range(len(R)) if i not in cur and i not in [12,46,60]])
Qcur=np.linalg.qr(R[cur].T)[0]
Qpast=np.linalg.qr(R[past].T)[0]
print("Largest FC rows not close to known direct groups")
aff=W["W_fc.0"]@W["ln2_b.0"]
for i in np.argsort(-fn)[:100]:
    cs=F[i]@R[cur].T/fn[i]
    if cs.max()<.65 and aff[i]>-.25:
        print(int(i),"norm/bias",np.round([fn[i],aff[i]],3),
              "closest",int(cur[cs.argmax()]),round(float(cs.max()),3),
              "energy current/past",np.round([np.linalg.norm(F[i]@Qcur)**2/fn[i]**2,
                                             np.linalg.norm(F[i]@Qpast)**2/fn[i]**2],3))

# Project away a read-direction subspace, not a network layer evaluation.
Fr=F-(F@Qpast)@Qpast.T
nr=np.linalg.norm(Fr,axis=1)
ix=np.where((fn>1)&(nr/fn>.65))[0]
Y=Fr[ix]/nr[ix,None]
print("\nPast-orthogonal row count",len(ix),"singular values",svdvals(Fr)[:45])
for thresh in [.2,.3,.4,.5]:
    ll=AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=thresh).fit_predict(Y)
    sizes=np.bincount(ll)
    print("threshold",thresh,"cluster counts/sizes",len(sizes),np.sort(sizes)[::-1][:55])
    if thresh==.4:
        print("Cluster exemplars for >=2 members:")
        for c in np.argsort(-sizes):
            mm=ix[ll==c]
            if len(mm)<2:continue
            mean=Fr[mm].mean(0);mean/=np.linalg.norm(mean)
            cs=mean@R[cur].T
            ex=int(mm[np.argmax(fn[mm])])
            print("n",len(mm),"ex",ex,"norm/bias",np.round([fn[ex],aff[ex]],3),
                  "closest current",int(cur[cs.argmax()]),round(float(cs.max()),3))

```

Output:

```
Largest FC rows not close to known direct groups
477 norm/bias [3.528 0.02 ] closest 28 0.615 energy current/past [0.506 0.445]
100 norm/bias [3.319 0.035] closest 1 0.606 energy current/past [0.426 0.556]
796 norm/bias [ 3.238 -0.202] closest 48 0.525 energy current/past [0.37  0.957]
1170 norm/bias [ 3.056 -0.243] closest 15 0.59 energy current/past [0.418 0.643]
1054 norm/bias [2.886 0.441] closest 33 0.318 energy current/past [0.573 0.603]
168 norm/bias [ 2.816 -0.174] closest 32 0.563 energy current/past [0.345 0.382]
51 norm/bias [ 2.786 -0.092] closest 35 0.114 energy current/past [0.054 0.856]
416 norm/bias [ 2.77  -0.067] closest 17 0.242 energy current/past [0.496 0.483]
1383 norm/bias [2.582 0.074] closest 32 0.271 energy current/past [0.224 0.37 ]
1129 norm/bias [2.533 0.036] closest 17 0.188 energy current/past [0.249 0.541]
1100 norm/bias [ 2.474 -0.125] closest 18 0.558 energy current/past [0.612 0.274]
1391 norm/bias [ 2.468 -0.189] closest 37 0.518 energy current/past [0.322 0.479]
1055 norm/bias [2.467 0.01 ] closest 23 0.041 energy current/past [0.024 0.985]
1097 norm/bias [ 2.411 -0.06 ] closest 18 0.545 energy current/past [0.432 0.553]
743 norm/bias [2.372 0.015] closest 32 0.275 energy current/past [0.278 0.39 ]
1335 norm/bias [ 2.352 -0.079] closest 37 0.265 energy current/past [0.144 0.586]
1518 norm/bias [2.35  0.075] closest 23 0.413 energy current/past [0.385 0.512]
1489 norm/bias [ 2.349 -0.055] closest 47 0.227 energy current/past [0.241 0.563]

Past-orthogonal row count 193 singular values [7.033 6.276 5.274 4.746 4.554 4.388 4.279 4.143 4.065 3.898 3.848 3.756 3.661 3.537 3.522 3.437 3.339 3.25  3.221 3.197 3.083 3.065 3.006 2.922 2.806 2.782 2.702 2.693 2.533
 2.479 2.375 2.36  2.24  2.11  2.018 1.951 1.922 1.873 1.806 1.786 1.758 1.693 1.651 1.596 1.553]
threshold 0.2 cluster counts/sizes 146 [4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
threshold 0.3 cluster counts/sizes 120 [7 6 6 5 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
threshold 0.4 cluster counts/sizes 101 [12  6  6  6  5  5  5  5  5  4  4  4  4  4  4  4  4  4  4  4  3  3  3  3  2  2  2  2  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1]
Cluster exemplars for >=2 members:
n 12 ex 418 norm/bias [ 3.75  -0.684] closest current 45 0.068
n 6 ex 441 norm/bias [ 2.442 -0.311] closest current 39 0.834
n 6 ex 383 norm/bias [ 1.969 -0.029] closest current 8 0.73
n 6 ex 533 norm/bias [1.478 0.063] closest current 15 0.735
n 5 ex 1302 norm/bias [2.103 0.109] closest current 42 0.849
n 5 ex 1486 norm/bias [1.406 0.144] closest current 7 0.856
n 5 ex 283 norm/bias [2.098 0.036] closest current 35 0.772
n 5 ex 571 norm/bias [ 3.096 -0.092] closest current 32 0.848
n 5 ex 1197 norm/bias [ 2.226 -0.214] closest current 28 0.819
n 4 ex 1107 norm/bias [ 2.928 -0.064] closest current 43 0.713
n 4 ex 106 norm/bias [2.273 0.148] closest current 16 0.847
n 4 ex 1382 norm/bias [ 1.98 -0.03] closest current 29 0.76
n 4 ex 413 norm/bias [ 2.361 -0.05 ] closest current 18 0.759
n 4 ex 983 norm/bias [1.652 0.19 ] closest current 33 0.799
n 4 ex 975 norm/bias [2.191 0.038] closest current 17 0.814
n 4 ex 839 norm/bias [1.74 0.02] closest current 37 0.83
n 4 ex 1472 norm/bias [1.636 0.164] closest current 23 0.755
n 4 ex 605 norm/bias [ 1.907 -0.056] closest current 1 0.82
n 4 ex 293 norm/bias [1.845 0.123] closest current 45 0.713
n 4 ex 1464 norm/bias [2.614 0.12 ] closest current 48 0.739
n 3 ex 1031 norm/bias [1.853 0.025] closest current 47 0.792
n 3 ex 773 norm/bias [2.14  0.305] closest current 44 0.843
n 3 ex 256 norm/bias [ 2.247 -0.046] closest current 40 0.823
n 3 ex 1046 norm/bias [ 3.573 -0.616] closest current 43 0.132
n 2 ex 168 norm/bias [ 2.816 -0.174] closest current 32 0.516
n 2 ex 946 norm/bias [ 1.582 -0.046] closest current 28 0.233
n 2 ex 860 norm/bias [2.079 0.259] closest current 33 0.259
n 2 ex 505 norm/bias [ 1.789 -0.273] closest current 32 0.568
n 2 ex 743 norm/bias [2.372 0.015] closest current 32 0.305
threshold 0.5 cluster counts/sizes 70 [14 10  7  6  6  6  6  6  6  5  5  5  5  5  4  4  4  4  4  4  4  4  3  3  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1]

```

## Call 13

```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=np.eye(384)-np.ones((384,384))/384
F=(W["W_fc.0"]*W["ln2_w.0"])@P
fn=np.linalg.norm(F,axis=1)
B=W["W_o.0"][:,128:192]@((W["W_v.0"][128:192]*W["ln1_w.0"])@P)
ids=np.where(fn>.8)[0]
lab=AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=.3).fit_predict(F[ids]/fn[ids,None])
R=[];groups=[];ex=[]
for c in np.argsort(-np.bincount(lab)):
    mm=ids[lab==c]
    if len(mm)<2:continue
    r=F[mm].mean(0);r/=np.linalg.norm(r)
    R.append(r);groups.append(mm);ex.append(int(mm[np.argmax(fn[mm])]))
R=np.array(R)
RB=R@B
RB/=np.linalg.norm(RB,axis=1,keepdims=True)
S=R@RB.T
diag=np.diag(S)
print("Group / size / exemplar / self OV-read cosine / strongest positive OV-read match")
for i in range(len(R)):
    best=np.argmax(S[i])
    print(i,len(groups[i]),ex[i],round(diag[i],3),int(best),round(S[i,best],3))
cur=np.where(diag<-.10)[0]
past=np.where(diag>.06)[0]
print("\nNegative self-composition groups:",cur,"count",len(cur))
print("Positive self-composition groups:",past,"count",len(past))
ri,ci=linear_sum_assignment(-S[np.ix_(cur,past)])
print("One-to-one matches:",[(int(cur[i]),int(past[j]),round(S[cur[i],past[j]],3)) for i,j in zip(ri,ci)])
print("Direct-group Gram eigenvalues:",np.linalg.eigvalsh(R[cur]@R[cur].T))

for l,h in [(3,1),(3,5)]:
    sl=slice(64*h,64*(h+1))
    b=W[f"W_o.{l}"][:,sl]@((W[f"W_v.{l}"][sl]*W[f"ln1_w.{l}"])@P)
    print(f"L{l}H{h} OV singular values",np.linalg.svd(b,compute_uv=False)[:45])

```

Output:

```
Group / size / exemplar / self OV-read cosine / strongest positive OV-read match
0 7 1038 0.186 0 0.186
1 5 605 -0.397 50 0.345
2 5 355 0.165 2 0.165
3 5 298 0.136 3 0.136
4 5 642 0.144 25 0.155
5 5 921 0.131 4 0.144
6 5 947 -0.183 10 0.222
7 5 1486 -0.518 31 0.484
8 5 575 -0.216 21 0.426
9 5 1074 0.19 9 0.19
10 5 11 0.163 10 0.163
11 5 1082 0.08 45 0.094
12 5 418 -0.418 11 0.21
13 4 35 -0.146 4 0.243
14 4 1485 0.125 57 0.14
15 4 533 -0.266 20 0.289
16 4 106 -0.503 25 0.492
17 4 975 -0.381 19 0.372
18 4 413 -0.275 58 0.269
19 4 468 0.193 19 0.193
20 4 720 0.161 20 0.161
21 4 1144 0.143 62 0.165
22 4 888 0.214 22 0.214
23 4 1472 -0.593 22 0.402
24 4 361 0.094 24 0.094
25 4 727 0.192 25 0.192
26 4 222 0.143 26 0.143
27 4 787 0.118 43 0.121
28 4 1407 -0.47 5 0.402
29 4 1382 -0.319 55 0.242
30 4 51 -0.071 59 0.211
31 4 1022 0.14 31 0.14
32 4 411 -0.349 22 0.247
33 4 983 -0.462 2 0.491
34 4 466 0.18 34 0.18
35 4 283 -0.314 34 0.34
36 3 1214 0.193 36 0.193
37 3 1135 -0.455 9 0.466
38 3 1055 -0.068 54 0.277
39 3 5 -0.532 3 0.513
40 3 256 -0.416 11 0.312
41 3 356 0.061 34 0.087
42 3 1302 -0.58 14 0.493
43 3 1107 -0.289 37 0.193
44 3 773 -0.45 24 0.41
45 3 293 -0.224 26 0.453
46 3 1046 -0.256 51 0.28
47 3 1031 -0.467 27 0.327
48 3 1464 -0.288 36 0.365
49 2 782 0.188 49 0.188
50 2 1008 0.153 50 0.153
51 2 827 0.13 51 0.13
52 2 883 0.009 23 0.094
53 2 781 0.04 36 0.081
54 2 1456 0.036 44 0.059
55 2 611 0.093 55 0.093
56 2 196 -0.077 59 0.146
57 2 1166 0.253 57 0.253
58 2 1115 0.161 58 0.161
59 2 519 0.055 3 0.114
60 2 441 0.138 3 0.431
61 2 459 0.287 61 0.287
62 2 1288 0.252 62 0.252

Negative self-composition groups: [ 1  6  7  8 12 13 15 16 17 18 23 28 29 32 33 35 37 39 40 42 43 44 45 46 47 48] count 26
Positive self-composition groups: [ 0  2  3  4  5  9 10 11 14 19 20 21 22 24 25 26 27 31 34 36 41 49 50 51 55 57 58 60 61 62] count 30
One-to-one matches: [(1, 50, np.float64(0.345)), (6, 10, np.float64(0.222)), (7, 31, np.float64(0.484)), (8, 21, np.float64(0.426)), (12, 49, np.float64(0.148)), (13, 4, np.float64(0.243)), (15, 20, np.float64(0.289)), (16, 25, np.float64(0.492)), (17, 19, np.float64(0.372)), (18, 58, np.float64(0.269)), (23, 22, np.float64(0.402)), (28, 5, np.float64(0.402)), (29, 55, np.float64(0.242)), (32, 41, np.float64(0.182)), (33, 2, np.float64(0.491)), (35, 34, np.float64(0.34)), (37, 9, np.float64(0.466)), (39, 3, np.float64(0.513)), (40, 11, np.float64(0.312)), (42, 14, np.float64(0.493)), (43, 60, np.float64(0.038)), (44, 24, np.float64(0.41)), (45, 26, np.float64(0.453)), (46, 51, np.float64(0.28)), (47, 27, np.float64(0.327)), (48, 36, np.float64(0.365))]
Direct-group Gram eigenvalues: [0.354 0.658 0.784 0.82  0.837 0.864 0.877 0.899 0.921 0.946 0.956 0.991 0.999 1.02  1.049 1.069 1.082 1.102 1.141 1.165 1.181 1.189 1.245 1.254 1.291 1.305]
L3H1 OV singular values [10.634 10.336 10.218  9.489  9.42   9.311  8.995  8.964  8.59   8.405  8.197  7.747  7.614  7.322  7.273  7.114  6.756  6.511  6.461  6.252  5.847  5.648  5.588  5.204  4.296
  4.146  3.959  3.599  3.332  2.956  2.571  2.293  2.182  2.074  1.842  1.754  1.697  1.642  1.559  1.454  1.448  1.341  1.242  1.172  1.15 ]
L3H5 OV singular values [16.891 14.696 13.371 12.86  12.278 12.185 10.813 10.663  9.322  8.905  8.328  7.242  7.14   6.334  5.83   5.617  5.14   4.911  4.639  3.952  3.93   3.684  3.397  3.287  3.09
  2.99   2.806  2.593  2.48   2.365  2.268  2.138  1.976  1.947  1.892  1.786  1.667  1.641  1.604  1.57   1.485  1.422  1.323  1.301  1.234]

```

## Call 14

```python
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
P=np.eye(384)-np.ones((384,384))/384
F=(W["W_fc.0"]*W["ln2_w.0"])@P
fn=np.linalg.norm(F,axis=1)
B=W["W_o.0"][:,128:192]@((W["W_v.0"][128:192]*W["ln1_w.0"])@P)
ids=np.where(fn>.8)[0]
lab=AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=.3).fit_predict(F[ids]/fn[ids,None])
R=[];U=[];groups=[]
for c in np.argsort(-np.bincount(lab)):
    mm=ids[lab==c]
    if len(mm)<2:continue
    r=F[mm].mean(0);r/=np.linalg.norm(r)
    u=W["W_proj.0"][:,mm].mean(1).astype(float);u/=np.linalg.norm(u)
    R.append(r);U.append(u);groups.append(mm)
R=np.array(R);U=np.array(U).T
RB=R@B;RB/=np.linalg.norm(RB,axis=1,keepdims=True)
S=R@RB.T
cur=np.array([i for i in range(len(R)) if S[i,i]<-.05 and i!=56])
past=np.array([i for i in range(len(R)) if i not in cur and i!=56])
ri,ci=linear_sum_assignment(-S[np.ix_(cur,past)])
cur=cur[ri];past=past[ci]
print("Matched groups:",[(int(i),int(j),round(S[i,j],3)) for i,j in zip(cur,past)])
e,V=np.linalg.eigh(R[cur]@R[cur].T)
v=V[:,0]
if v.sum()<0:v=-v
print("Smallest direct-read Gram eigenvalue:",e[0])
print("Corresponding direction:",[(int(cur[i]),round(v[i],3)) for i in np.argsort(-v)])

Uc=U[:,cur];Up=U[:,past]
n=len(cur)
for l in [1,2,3]:
    F=(W[f"W_fc.{l}"]*W[f"ln2_w.{l}"])@P
    C=F@Uc;D=F@Up
    C-=C.mean(0);D-=D.mean(0)
    C/=np.linalg.norm(C,axis=0,keepdims=True)
    D/=np.linalg.norm(D,axis=0,keepdims=True)
    cross=C.T@D
    print("\nLayer",l,"group / matched group / diagonal / row mean / column mean / writer cosine")
    for i in range(n):
        print(int(cur[i]),int(past[i]),np.round([cross[i,i],cross[i].mean(),cross[:,i].mean(),Uc[:,i]@Up[:,i]],3))
    if l==1:
        for group in [32,43,30,38]:
            i=np.where(cur==group)[0][0]
            js=np.argsort(-cross[i])[:6]
            ks=np.argsort(-cross[:,i])[:6]
            print("group",group,"top row",[(int(cur[j]),round(cross[i,j],3)) for j in js],
                  "top column",[(int(cur[j]),round(cross[j,i],3)) for j in ks])

```

Output:

```
Matched groups: [(1, 50, np.float64(0.345)), (6, 10, np.float64(0.222)), (7, 31, np.float64(0.484)), (8, 21, np.float64(0.426)), (12, 49, np.float64(0.148)), (13, 4, np.float64(0.243)), (15, 20, np.float64(0.289)), (16, 25, np.float64(0.492)), (17, 19, np.float64(0.372)), (18, 58, np.float64(0.269)), (23, 22, np.float64(0.402)), (28, 5, np.float64(0.402)), (29, 55, np.float64(0.242)), (30, 59, np.float64(0.211)), (32, 53, np.float64(0.209)), (33, 2, np.float64(0.491)), (35, 34, np.float64(0.34)), (37, 9, np.float64(0.466)), (38, 54, np.float64(0.277)), (39, 3, np.float64(0.513)), (40, 11, np.float64(0.312)), (42, 14, np.float64(0.493)), (43, 52, np.float64(0.082)), (44, 24, np.float64(0.41)), (45, 26, np.float64(0.453)), (46, 51, np.float64(0.28)), (47, 27, np.float64(0.327)), (48, 36, np.float64(0.365))]
Smallest direct-read Gram eigenvalue: 0.35174054786000414
Corresponding direction: [(32, np.float64(0.41)), (16, np.float64(0.307)), (45, np.float64(0.278)), (23, np.float64(0.268)), (42, np.float64(0.268)), (33, np.float64(0.26)), (8, np.float64(0.224)), (7, np.float64(0.21)), (1, np.float64(0.201)), (37, np.float64(0.192)), (28, np.float64(0.184)), (39, np.float64(0.175)), (17, np.float64(0.158)), (18, np.float64(0.151)), (29, np.float64(0.145)), (47, np.float64(0.137)), (15, np.float64(0.135)), (35, np.float64(0.132)), (40, np.float64(0.131)), (48, np.float64(0.124)), (44, np.float64(0.119)), (46, np.float64(0.114)), (43, np.float64(0.11)), (13, np.float64(0.102)), (12, np.float64(0.063)), (38, np.float64(0.058)), (6, np.float64(-0.002)), (30, np.float64(-0.028))]

Layer 1 group / matched group / diagonal / row mean / column mean / writer cosine
1 50 [0.1   0.023 0.023 0.117]
6 10 [-0.014 -0.022 -0.041  0.135]
7 31 [ 0.039  0.009 -0.016  0.113]
8 21 [ 0.069  0.    -0.005  0.115]
12 49 [0.108 0.012 0.082 0.055]
13 4 [0.438 0.031 0.076 0.231]
15 20 [ 0.167 -0.001  0.004  0.162]
16 25 [ 0.09  -0.007 -0.037  0.166]
17 19 [ 0.185 -0.016  0.02   0.154]
18 58 [0.277 0.007 0.042 0.172]
23 22 [ 0.119 -0.002 -0.048  0.147]
28 5 [ 0.268 -0.015  0.009  0.206]
29 55 [ 0.002 -0.013 -0.056  0.034]
30 59 [ 0.099 -0.015  0.037  0.198]
32 53 [0.43  0.007 0.143 0.24 ]
33 2 [ 0.03  -0.005 -0.002  0.068]
35 34 [0.203 0.008 0.016 0.152]
37 9 [ 0.28   0.002 -0.03   0.211]
38 54 [ 0.164  0.004 -0.002  0.125]
39 3 [-0.007 -0.001 -0.069  0.124]
40 11 [0.33  0.009 0.016 0.208]
42 14 [ 0.185  0.015 -0.055  0.207]
43 52 [0.134 0.001 0.071 0.103]
44 24 [ 0.129 -0.006 -0.021  0.156]
45 26 [ 0.145  0.019 -0.012  0.155]
46 51 [ 0.109  0.    -0.067  0.093]
47 27 [ 0.232 -0.011 -0.019  0.123]
48 36 [ 0.136  0.003 -0.024  0.157]
group 32 top row [(32, np.float64(0.43)), (12, np.float64(0.312)), (13, np.float64(0.196)), (40, np.float64(0.141)), (33, np.float64(0.102)), (18, np.float64(0.066))] top column [(32, np.float64(0.43)), (39, np.float64(0.272)), (12, np.float64(0.264)), (35, np.float64(0.241)), (40, np.float64(0.237)), (18, np.float64(0.229))]
group 43 top row [(32, np.float64(0.138)), (43, np.float64(0.134)), (18, np.float64(0.122)), (12, np.float64(0.118)), (33, np.float64(0.092)), (30, np.float64(0.062))] top column [(42, np.float64(0.247)), (48, np.float64(0.149)), (43, np.float64(0.134)), (18, np.float64(0.124)), (39, np.float64(0.124)), (8, np.float64(0.12))]
group 30 top row [(44, np.float64(0.158)), (13, np.float64(0.15)), (33, np.float64(0.129)), (43, np.float64(0.102)), (30, np.float64(0.099)), (1, np.float64(0.073))] top column [(23, np.float64(0.146)), (1, np.float64(0.141)), (45, np.float64(0.12)), (29, np.float64(0.117)), (48, np.float64(0.111)), (12, np.float64(0.11))]
group 38 top row [(38, np.float64(0.164)), (16, np.float64(0.153)), (32, np.float64(0.117)), (43, np.float64(0.09)), (12, np.float64(0.088)), (35, np.float64(0.083))] top column [(38, np.float64(0.164)), (16, np.float64(0.121)), (28, np.float64(0.11)), (7, np.float64(0.075)), (45, np.float64(0.07)), (42, np.float64(0.067))]

Layer 2 group / matched group / diagonal / row mean / column mean / writer cosine
1 50 [0.083 0.021 0.011 0.117]
6 10 [ 0.167 -0.001  0.029  0.135]
7 31 [0.148 0.02  0.02  0.113]
8 21 [ 0.129 -0.005  0.001  0.115]
12 49 [-0.015 -0.001  0.019  0.055]
13 4 [0.316 0.022 0.035 0.231]
15 20 [0.217 0.003 0.002 0.162]
16 25 [ 0.115 -0.008 -0.017  0.166]
17 19 [0.194 0.    0.044 0.154]
18 58 [0.201 0.011 0.022 0.172]
23 22 [ 0.093  0.002 -0.021  0.147]
28 5 [0.304 0.005 0.001 0.206]
29 55 [ 0.075  0.005 -0.006  0.034]
30 59 [0.161 0.005 0.011 0.198]
32 53 [0.351 0.003 0.049 0.24 ]
33 2 [ 0.036  0.005 -0.036  0.068]
35 34 [0.225 0.008 0.005 0.152]
37 9 [ 0.297  0.009 -0.007  0.211]
38 54 [ 0.108  0.01  -0.011  0.125]
39 3 [ 0.122  0.    -0.021  0.124]
40 11 [0.325 0.    0.048 0.208]
42 14 [ 0.213  0.022 -0.023  0.207]
43 52 [0.196 0.02  0.048 0.103]
44 24 [ 0.192 -0.001  0.003  0.156]
45 26 [0.116 0.017 0.002 0.155]
46 51 [ 0.118  0.001 -0.026  0.093]
47 27 [ 0.129 -0.001 -0.002  0.123]
48 36 [ 0.151  0.004 -0.003  0.157]

Layer 3 group / matched group / diagonal / row mean / column mean / writer cosine
1 50 [0.164 0.037 0.052 0.117]
6 10 [0.169 0.024 0.019 0.135]
7 31 [0.057 0.03  0.03  0.113]
8 21 [ 0.089  0.009 -0.008  0.115]
12 49 [-0.     0.01  -0.039  0.055]
13 4 [0.259 0.034 0.068 0.231]
15 20 [0.199 0.028 0.025 0.162]
16 25 [ 0.186  0.004 -0.031  0.166]
17 19 [0.245 0.012 0.068 0.154]
18 58 [0.243 0.027 0.035 0.172]
23 22 [ 0.139  0.    -0.008  0.147]
28 5 [0.226 0.015 0.023 0.206]
29 55 [0.068 0.024 0.014 0.034]
30 59 [0.195 0.019 0.039 0.198]
32 53 [ 0.387 -0.022 -0.036  0.24 ]
33 2 [ 0.174  0.011 -0.035  0.068]
35 34 [0.191 0.026 0.014 0.152]
37 9 [0.354 0.035 0.024 0.211]
38 54 [0.147 0.033 0.02  0.125]
39 3 [0.163 0.019 0.002 0.124]
40 11 [0.265 0.023 0.039 0.208]
42 14 [0.278 0.038 0.016 0.207]
43 52 [0.146 0.025 0.057 0.103]
44 24 [0.197 0.019 0.044 0.156]
45 26 [0.203 0.023 0.042 0.155]
46 51 [0.09  0.018 0.002 0.093]
47 27 [0.15  0.022 0.028 0.123]
48 36 [0.217 0.02  0.062 0.157]

```
