## Call 1

```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

def norm(x):
    return float(la.norm(x))

print("All calculations below are weight-only linear algebra; no network evaluations.")
for l in range(4):
    print("\nLAYER", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        a = W[f"{name}.{l}"].astype(np.float64)
        print(name, "min/mean/max/std/norm",
              np.round([a.min(), a.mean(), a.max(), a.std(), norm(a)], 5))
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        a = W[f"{name}.{l}"].astype(np.float64)
        s = la.svdvals(a)
        print(name, "Frob", round(norm(a), 4),
              "top8_sv", np.round(s[:8], 4),
              "stable_rank", round(np.sum(s*s)/(s[0]*s[0]), 3))
    print("HEAD: QK_frob QK_top4_sv QK_top1_energy QK_sym_fraction QK_trace; "
          "OV_frob OV_top4_sv OV_trace OV_eig_posmax OV_eig_negmin")
    for j in range(6):
        sl = slice(64*j, 64*(j+1))
        q = W[f"W_q.{l}"][sl].astype(np.float64)
        k = W[f"W_k.{l}"][sl].astype(np.float64)
        v = W[f"W_v.{l}"][sl].astype(np.float64)
        o = W[f"W_o.{l}"][:, sl].astype(np.float64)
        a = q.T @ k
        b = o @ v
        sa, sb = la.svdvals(a)[:64], la.svdvals(b)[:64]
        eb = la.eigvals(v @ o)
        print(f"L{l}H{j}",
              round(norm(a), 4), np.round(sa[:4], 4),
              round(sa[0]**2/np.sum(sa**2), 4),
              round(norm((a+a.T)/2)/norm(a), 4), round(np.trace(a), 4),
              ";", round(norm(b), 4), np.round(sb[:4], 4),
              round(np.trace(b), 4), round(eb.real.max(), 4),
              round(eb.real.min(), 4))

```

Output:

```
All calculations below are weight-only linear algebra; no network evaluations.

LAYER 0
ln1_w min/mean/max/std/norm [-0.001   0.2403  1.0341  0.2436  6.7051]
ln1_b min/mean/max/std/norm [-0.0959  0.0008  0.091   0.0287  0.5635]
ln2_w min/mean/max/std/norm [ 0.0044  0.8006  1.5072  0.3295 16.9648]
ln2_b min/mean/max/std/norm [-0.4986  0.0022  0.3331  0.0683  1.3392]
W_q Frob 32.2912 top8_sv [8.2519 8.2232 7.963  7.819  7.68   7.6416 7.5194 7.2054] stable_rank 15.313
W_k Frob 30.3909 top8_sv [7.6464 7.4363 7.3304 6.9118 6.8285 6.6716 6.4724 6.4464] stable_rank 15.797
W_v Frob 14.836 top8_sv [4.339  3.5292 3.2699 3.1656 2.9824 2.6465 2.589  2.5519] stable_rank 11.691
W_o Frob 16.3713 top8_sv [7.1033 6.6074 4.2683 2.7702 2.5025 2.285  2.2638 2.2184] stable_rank 5.312
W_fc Frob 46.2486 top8_sv [18.1759 11.0178  8.7366  7.4517  7.2123  7.0621  7.0112  6.6204] stable_rank 6.474
W_proj Frob 39.3075 top8_sv [9.8257 7.3508 6.2651 5.8693 5.7965 5.6574 5.5163 5.4501] stable_rank 16.004
HEAD: QK_frob QK_top4_sv QK_top1_energy QK_sym_fraction QK_trace; OV_frob OV_top4_sv OV_trace OV_eig_posmax OV_eig_negmin
L0H0 3.0479 [2.5097 0.8949 0.6966 0.4393] 0.678 0.7267 -1.7245 ; 12.9041 [10.2524  5.6035  3.4749  2.8184] -0.1807 1.1732 -0.4325
L0H1 1.8376 [1.0971 0.7651 0.4518 0.343 ] 0.3564 0.7472 -1.2358 ; 10.1485 [7.662  5.1724 2.5819 2.0305] -0.7493 0.9812 -0.9241
L0H2 182.4351 [59.8786 57.2786 55.5815 54.3474] 0.1077 0.6963 32.7753 ; 13.223 [4.0009 3.853  3.7432 3.5437] -4.6421 0.4428 -0.6979
L0H3 11.3515 [6.568  5.9931 3.2141 2.6459] 0.3348 0.7629 -7.9874 ; 7.3079 [4.305  4.0456 2.4933 2.078 ] -2.0478 0.262 -0.6907
L0H4 6.6069 [6.0218 2.0421 1.078  0.5566] 0.8307 0.7629 -4.7252 ; 7.3117 [4.8786 3.2357 2.5484 2.0826] -0.9671 0.3599 -0.4665
L0H5 2.5787 [2.1893 0.543  0.3941 0.328 ] 0.7208 0.7288 -0.9664 ; 13.0384 [10.3142  5.4678  3.1482  2.9348] -0.0858 0.827 -0.41

LAYER 1
ln1_w min/mean/max/std/norm [0.1361 0.4575 1.1239 0.1022 9.1868]
ln1_b min/mean/max/std/norm [-1.4712 -0.005   1.1544  0.1672  3.2771]
ln2_w min/mean/max/std/norm [-0.0019  1.0429  1.4856  0.2003 20.8093]
ln2_b min/mean/max/std/norm [-0.3597  0.0005  0.3686  0.1364  2.6732]
W_q Frob 32.3126 top8_sv [9.9032 7.7974 6.7042 5.9516 5.6721 5.6025 5.4021 5.3125] stable_rank 10.646
W_k Frob 29.2506 top8_sv [13.3072  9.9327  5.8989  5.677   5.0458  4.622   4.4903  4.4044] stable_rank 4.832
W_v Frob 27.3529 top8_sv [5.2845 4.9661 4.87   4.7328 4.6743 4.5853 4.4113 4.3216] stable_rank 26.792
W_o Frob 25.9233 top8_sv [4.9657 3.2412 3.157  3.1119 3.0615 3.0531 2.9942 2.9545] stable_rank 27.253
W_fc Frob 79.1265 top8_sv [21.4881 13.1349 12.8042 12.6247 12.3279 12.1184 11.8504 11.6908] stable_rank 13.56
W_proj Frob 75.7103 top8_sv [8.1935 7.8941 7.1992 6.9972 6.671  6.4329 6.2745 6.1333] stable_rank 85.384
HEAD: QK_frob QK_top4_sv QK_top1_energy QK_sym_fraction QK_trace; OV_frob OV_top4_sv OV_trace OV_eig_posmax OV_eig_negmin
L1H0 23.6654 [12.6145 11.6953  8.2823  7.0164] 0.2841 0.7351 -26.1053 ; 21.7066 [8.0989 7.5796 6.3982 6.2019] -21.6005 1.6279 -2.7755
L1H1 35.5126 [19.649  17.056  10.7527  9.5417] 0.3061 0.7424 -44.9928 ; 22.226 [6.5833 6.0091 5.7392 5.681 ] -28.4652 0.5693 -2.0517
L1H2 41.7812 [23.3819 18.3356 12.0604 11.2132] 0.3132 0.7448 -60.5649 ; 19.5165 [6.1346 5.7881 5.2517 4.9869] -21.8959 0.2894 -1.708
L1H3 36.8369 [21.1075 17.3804 11.6312  9.1371] 0.3283 0.7402 -44.4839 ; 21.4133 [7.7675 6.8644 6.5273 6.0604] -20.2784 0.2159 -1.8799
L1H4 37.3819 [22.9586 17.425  11.442   8.6811] 0.3772 0.7366 -46.5798 ; 22.0658 [6.4984 6.2533 5.8966 5.7835] -26.577 0.1462 -2.0645
L1H5 36.2112 [20.9422 18.3428 10.3483  8.4275] 0.3345 0.7295 -39.7087 ; 22.4638 [6.2081 6.0193 5.803  5.6451] -29.2909 0.2127 -2.1146

LAYER 2
ln1_w min/mean/max/std/norm [ 0.2078  0.7052  1.0144  0.0749 13.8969]
ln1_b min/mean/max/std/norm [-1.3819 -0.001   1.0527  0.1525  2.9893]
ln2_w min/mean/max/std/norm [ 0.0011  1.3123  1.4655  0.1255 25.8329]
ln2_b min/mean/max/std/norm [-0.3761  0.005   0.3886  0.1537  3.
```

## Call 2

```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

A, B = {}, {}
for l in range(4):
    w = W[f"ln1_w.{l}"].astype(float)
    bias = W[f"ln1_b.{l}"].astype(float)
    print("\nLAYER", l, "(gain-weighted matrices; no LayerNorm evaluation)")
    for j in range(6):
        s = slice(64*j, 64*(j+1))
        q0 = W[f"W_q.{l}"][s].astype(float)
        k0 = W[f"W_k.{l}"][s].astype(float)
        v0 = W[f"W_v.{l}"][s].astype(float)
        o = W[f"W_o.{l}"][:,s].astype(float)
        q, k, v = q0*w, k0*w, v0*w
        a, b = q.T@k, o@v
        A[l,j], B[l,j] = a, b
        sv = la.svdvals(a)[:64]
        qb, kb = q0@bias, k0@bias
        key_linear = k.T@qb
        query_linear = q.T@kb
        print(f"L{l}H{j}",
              "QKnorm", round(la.norm(a),4),
              "QKsv", np.round(sv[:6],4),
              "QKtrace", round(np.trace(a),4),
              "OVnorm", round(la.norm(b),4),
              "bias_key/query_norm", np.round([la.norm(key_linear),la.norm(query_linear)],4))

print("\nNormalized trace(A_lh @ B_0j), columns L0H0..L0H5")
for l in range(1,4):
    for j in range(6):
        a = A[l,j]
        vals = [np.trace(a@B[0,k])/(la.norm(a)*la.norm(B[0,k])) for k in range(6)]
        print(f"L{l}H{j}", np.round(vals,4))

print("\nLargest earlier-head trace compositions (signed normalized trace)")
for l in range(2,4):
    for j in range(6):
        a = A[l,j]
        vals = []
        for m in range(l):
            for k in range(6):
                b = B[m,k]
                vals.append((float(np.trace(a@b)/(la.norm(a)*la.norm(b))), f"L{m}H{k}"))
        print(f"L{l}H{j}", sorted(vals, reverse=True)[:5])

print("\nL0H2 effective QK spectrum")
a = A[0,2]
u,s,vt = la.svd(a, full_matrices=False)
print("singular values", np.round(s[:40],5))
ev = la.eigvals(a)
ix = np.argsort(abs(ev))[::-1][:24]
print("largest eigenvalues", np.round(ev[ix],4))
print("left/right top-20 subspace overlap", np.round(la.svdvals(u[:,:20].T@vt[:20].T),4))

print("\nL0 query/key and value subspace overlap with L0H2 dominant QK subspace")
p = la.orth(np.column_stack([u[:,:20],vt[:20].T]))
for j in range(6):
    s0 = slice(64*j,64*(j+1))
    vals=[]
    for name in ["W_q","W_k","W_v"]:
        m = W[f"{name}.0"][s0].astype(float)*W["ln1_w.0"]
        vals.append(la.norm(m@p)**2/la.norm(m)**2)
    print(f"L0H{j}", np.round(vals,4))

```

Output:

```

LAYER 0 (gain-weighted matrices; no LayerNorm evaluation)
L0H0 QKnorm 0.4367 QKsv [0.4114 0.0916 0.0567 0.0368 0.0343 0.0289] QKtrace -0.0379 OVnorm 9.0429 bias_key/query_norm [0.1091 0.1694]
L0H1 QKnorm 0.2362 QKsv [0.1997 0.0761 0.0422 0.0353 0.0279 0.0247] QKtrace -0.1115 OVnorm 6.9661 bias_key/query_norm [0.0267 0.0107]
L0H2 QKnorm 32.6269 QKsv [11.3199 10.1995  9.7194  9.6486  9.272   8.7533] QKtrace 9.5999 OVnorm 2.54 bias_key/query_norm [1.1448 1.5108]
L0H3 QKnorm 3.1569 QKsv [2.8706 0.9529 0.621  0.3398 0.261  0.2181] QKtrace 1.5753 OVnorm 1.7662 bias_key/query_norm [0.4283 0.3216]
L0H4 QKnorm 0.8508 QKsv [0.7136 0.4284 0.1319 0.0522 0.0419 0.0328] QKtrace -0.5674 OVnorm 4.0336 bias_key/query_norm [0.0901 0.0772]
L0H5 QKnorm 0.6206 QKsv [0.6073 0.0861 0.0369 0.0299 0.0271 0.023 ] QKtrace -0.2776 OVnorm 9.3107 bias_key/query_norm [0.1299 0.2023]

LAYER 1 (gain-weighted matrices; no LayerNorm evaluation)
L1H0 QKnorm 7.0989 QKsv [4.7823 3.8586 1.9815 1.5518 1.0526 0.8946] QKtrace -5.1542 OVnorm 10.1223 bias_key/query_norm [10.8591 13.1828]
L1H1 QKnorm 10.5253 QKsv [7.0546 5.3302 3.2915 2.1355 1.6238 1.4283] QKtrace -9.2943 OVnorm 10.3335 bias_key/query_norm [18.2783 18.644 ]
L1H2 QKnorm 12.41 QKsv [7.9573 6.8828 3.0682 2.603  2.115  1.8171] QKtrace -12.955 OVnorm 9.1806 bias_key/query_norm [20.7031 23.8558]
L1H3 QKnorm 11.0075 QKsv [7.3565 5.9091 3.2626 2.029  1.764  1.3744] QKtrace -9.2582 OVnorm 10.0512 bias_key/query_norm [18.809  18.7776]
L1H4 QKnorm 11.6052 QKsv [7.5816 6.9475 2.8775 2.0717 1.5464 1.3804] QKtrace -9.9582 OVnorm 10.401 bias_key/query_norm [18.3219 23.5867]
L1H5 QKnorm 10.7065 QKsv [7.146  5.914  2.9776 1.8365 1.6321 1.4794] QKtrace -8.1163 OVnorm 10.6202 bias_key/query_norm [14.0211 19.6845]

LAYER 2 (gain-weighted matrices; no LayerNorm evaluation)
L2H0 QKnorm 19.5056 QKsv [5.5592 5.0376 4.8195 4.5062 4.2407 4.0516] QKtrace -0.173 OVnorm 16.3068 bias_key/query_norm [13.5928 13.0054]
L2H1 QKnorm 16.5448 QKsv [4.2136 3.5951 3.4598 3.3372 3.2779 3.161 ] QKtrace 4.237 OVnorm 17.7221 bias_key/query_norm [5.9578 7.4963]
L2H2 QKnorm 12.9062 QKsv [4.8575 4.024  3.2103 2.5372 2.3832 2.2932] QKtrace -8.8427 OVnorm 20.5752 bias_key/query_norm [ 6.0447 12.5128]
L2H3 QKnorm 16.6966 QKsv [4.4958 4.2052 3.9324 3.5228 3.2925 3.1557] QKtrace -5.9051 OVnorm 17.5062 bias_key/query_norm [11.702  11.0247]
L2H4 QKnorm 18.7551 QKsv [5.0228 4.8342 4.2578 4.0652 3.7358 3.6109] QKtrace -3.8054 OVnorm 16.624 bias_key/query_norm [12.6211 13.1065]
L2H5 QKnorm 18.0092 QKsv [5.1073 4.3177 3.95   3.8098 3.7436 3.615 ] QKtrace -5.578 OVnorm 16.6842 bias_key/query_norm [12.0719 11.6318]

LAYER 3 (gain-weighted matrices; no LayerNorm evaluation)
L3H0 QKnorm 17.8612 QKsv [3.6121 3.3594 3.345  3.2422 3.2067 3.1565] QKtrace 27.2957 OVnorm 22.5984 bias_key/query_norm [2.0652 2.089 ]
L3H1 QKnorm 14.6476 QKsv [2.6208 2.5393 2.4737 2.4472 2.4061 2.3836] QKtrace 50.449 OVnorm 40.8686 bias_key/query_norm [1.9323 1.0931]
L3H2 QKnorm 12.0463 QKsv [3.7667 3.2887 2.8452 2.6572 2.4694 2.3229] QKtrace 1.7643 OVnorm 26.8815 bias_key/query_norm [2.3165 2.2163]
L3H3 QKnorm 12.7936 QKsv [3.9788 3.2199 2.6861 2.4313 2.3645 2.3594] QKtrace 0.8934 OVnorm 27.4312 bias_key/query_norm [2.8828 2.4488]
L3H4 QKnorm 14.8484 QKsv [4.2301 3.3    3.1849 2.9889 2.8823 2.6888] QKtrace 7.2288 OVnorm 24.1979 bias_key/query_norm [2.4576 2.0855]
L3H5 QKnorm 6.679 QKsv [2.075  1.8792 1.6976 1.57   1.5251 1.2757] QKtrace 0.7931 OVnorm 45.4008 bias_key/query_norm [1.3167 0.7099]

Normalized trace(A_lh @ B_0j), columns L0H0..L0H5
L1H0 [ 0.0388  0.0163 -0.0054 -0.0187  0.032   0.0041]
L1H1 [ 0.0457  0.0329 -0.0001 -0.0143  0.0496  0.0187]
L1H2 [ 0.0497  0.0415  0.0145 -0.0059  0.054   0.0222]
L1H3 [ 0.0584  0.0467  0.0147 -0.0099  0.0601  0.0281]
L1H4 [ 0.0796  0.0707  0.0133 -0.0046  0.0839  0.0459]
L1H5 [ 0.0386  0.0174  0.0059 -0.0139  0.0363 -0.0111]
L2H0 [ 0.0156  0.0124 -0.005  -0.0182  0.0183  0.0159]
L2H1 [ 0.0098  0.0065 -0.0056 -0.0102  0.0104  0.0116]
L2H2 [ 0.0295
```

## Call 3

```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

for l in range(4):
    print("\nLAYER",l)
    for name in ["W_fc","W_proj"]:
        m = W[f"{name}.{l}"].astype(float)
        s = la.svdvals(m)
        print(name,"first 40 SV:",np.round(s[:40],3))
        print("SV indices 40,60,80,100,150,200,300,383:",
              np.round(s[[40,60,80,100,150,200,300,383]],3))

def head(l,j):
    sl=slice(j*64,(j+1)*64)
    w=W[f"ln1_w.{l}"].astype(float)
    q=W[f"W_q.{l}"][sl].astype(float)*w
    k=W[f"W_k.{l}"][sl].astype(float)*w
    v=W[f"W_v.{l}"][sl].astype(float)*w
    o=W[f"W_o.{l}"][:,sl].astype(float)
    return q.T@k,o@v

aa=[head(0,j)[0] for j in range(6)]
bb=[head(0,j)[1] for j in range(6)]
print("\nL0 OV matrix cosine similarities:")
print(np.round([[np.sum(x*y)/(la.norm(x)*la.norm(y)) for y in bb] for x in bb],3))
print("L0 QK matrix cosine similarities:")
print(np.round([[np.sum(x*y)/(la.norm(x)*la.norm(y)) for y in aa] for x in aa],3))

print("\nL0H2 unweighted QK singular values:")
q=W["W_q.0"][128:192].astype(float)
k=W["W_k.0"][128:192].astype(float)
u,s,vt=la.svd(q.T@k,full_matrices=False)
print(np.round(s[:48],3))
p=la.orth(np.column_stack((u[:,:20],vt[:20].T)))
print("L0 raw Q/K/V energy in this subspace:")
for j in range(6):
    print(j, np.round([la.norm(W[f"{name}.0"][64*j:64*(j+1)]@p)**2/
                       la.norm(W[f"{name}.0"][64*j:64*(j+1)])**2
                       for name in ["W_q","W_k","W_v"]],4))

print("\nL1 key-bias weight directions and L0 OV outputs")
for j in range(6):
    sl=slice(j*64,(j+1)*64)
    q=W["W_q.1"][sl].astype(float)
    k=W["W_k.1"][sl].astype(float)
    r=(k*W["ln1_w.1"]).T@(q@W["ln1_b.1"])
    vals=[]
    for b in bb:
        vals.append(la.norm(r@b)/(la.norm(r)*la.norm(b)))
    print(j,np.round(vals,4))

print("\nLinear-algebra-only MLP products: normalized trace with later QK")
# These are matrix products, not MLP evaluations: there is no ReLU or input.
mats={}
for l in range(3):
    f=W[f"W_fc.{l}"].astype(float)*W[f"ln2_w.{l}"]
    mats[l]=W[f"W_proj.{l}"].astype(float)@f
for l in range(1,4):
    for j in range(6):
        a,b=head(l,j)
        print(f"L{l}H{j}",
              np.round([np.trace(a@mats[m])/(la.norm(a)*la.norm(mats[m]))
                        for m in range(l)],4))

```

Output:

```

LAYER 0
W_fc first 40 SV: [18.176 11.018  8.737  7.452  7.212  7.062  7.011  6.62   6.57   6.42   6.343  6.29   6.137  6.043  5.974  5.918  5.807  5.779  5.583  5.513  5.5
  5.36   5.16   5.12   5.086  5.032  4.899  4.879  4.819  4.732  4.601  4.533  4.479  4.441  4.268  4.198  4.031  3.974  3.869  3.824]
SV indices 40,60,80,100,150,200,300,383: [3.72  2.377 1.616 1.25  0.963 0.814 0.596 0.4  ]
W_proj first 40 SV: [9.826 7.351 6.265 5.869 5.797 5.657 5.516 5.45  5.333 5.242 5.197 5.094 4.98  4.904 4.863 4.819 4.793 4.689 4.627 4.595 4.557 4.523 4.411 4.378
 4.283 4.216 4.143 4.125 4.048 4.025 3.894 3.847 3.808 3.792 3.764 3.711 3.685 3.636 3.595 3.525]
SV indices 40,60,80,100,150,200,300,383: [3.499 2.962 2.34  1.767 0.996 0.607 0.343 0.213]

LAYER 1
W_fc first 40 SV: [21.488 13.135 12.804 12.625 12.328 12.118 11.85  11.691 11.43  11.269 11.049 10.846 10.646 10.62  10.324  9.928  9.758  9.699  9.592  9.353  9.219
  9.169  9.074  8.921  8.816  8.729  8.516  8.442  8.433  8.247  8.106  7.853  7.745  7.696  7.617  7.54   7.422  7.349  7.124  7.084]
SV indices 40,60,80,100,150,200,300,383: [6.851 5.28  4.27  3.346 1.842 1.274 0.846 0.411]
W_proj first 40 SV: [8.193 7.894 7.199 6.997 6.671 6.433 6.274 6.133 6.099 6.021 5.942 5.865 5.829 5.789 5.698 5.675 5.646 5.629 5.581 5.564 5.553 5.538 5.528 5.477
 5.456 5.447 5.438 5.403 5.399 5.38  5.367 5.351 5.332 5.327 5.314 5.296 5.252 5.235 5.219 5.208]
SV indices 40,60,80,100,150,200,300,383: [5.197 4.925 4.704 4.479 3.985 3.509 2.629 0.319]

LAYER 2
W_fc first 40 SV: [15.363  8.021  7.895  7.725  7.597  7.49   7.257  7.216  7.141  7.053  6.995  6.974  6.881  6.828  6.8    6.773  6.692  6.634  6.589  6.554  6.512
  6.466  6.411  6.364  6.328  6.312  6.293  6.225  6.17   6.154  6.11   6.101  6.049  6.032  5.963  5.934  5.917  5.91   5.902  5.854]
SV indices 40,60,80,100,150,200,300,383: [5.822 5.301 4.876 4.56  3.917 3.354 2.38  0.645]
W_proj first 40 SV: [9.404 9.187 8.723 8.47  8.04  7.795 7.614 7.45  7.392 7.253 7.223 7.148 7.082 6.972 6.829 6.733 6.637 6.547 6.46  6.404 6.374 6.307 6.209 6.181
 6.12  6.091 5.989 5.9   5.88  5.833 5.758 5.703 5.685 5.645 5.588 5.559 5.491 5.478 5.467 5.429]
SV indices 40,60,80,100,150,200,300,383: [5.406 4.97  4.657 4.389 3.809 3.329 2.449 0.362]

LAYER 3
W_fc first 40 SV: [14.162  8.125  7.915  7.326  7.148  6.856  6.763  6.691  6.489  6.436  6.387  6.31   6.26   6.177  6.144  6.088  6.066  6.017  5.962  5.934  5.883
  5.855  5.843  5.781  5.752  5.72   5.697  5.657  5.632  5.617  5.593  5.568  5.519  5.517  5.495  5.45   5.431  5.389  5.379  5.361]
SV indices 40,60,80,100,150,200,300,383: [5.341 4.986 4.708 4.465 3.94  3.5   2.678 0.683]
W_proj first 40 SV: [26.309 20.685 19.699 19.261 18.594 18.197 17.862 17.667 17.044 16.749 16.039 15.151 14.756 14.326 14.05  13.5   13.293 12.531 12.401 11.958 11.74
 11.594 10.367  8.351  8.064  7.914  7.461  7.166  5.833  4.072  3.221  3.139  2.842  2.8    2.687  2.571  2.403  2.328  2.244  2.185]
SV indices 40,60,80,100,150,200,300,383: [2.181 1.645 1.422 1.244 0.961 0.765 0.509 0.318]

L0 OV matrix cosine similarities:
[[1.    0.885 0.008 0.163 0.903 0.724]
 [0.885 1.    0.006 0.195 0.889 0.802]
 [0.008 0.006 1.    0.012 0.006 0.017]
 [0.163 0.195 0.012 1.    0.195 0.156]
 [0.903 0.889 0.006 0.195 1.    0.728]
 [0.724 0.802 0.017 0.156 0.728 1.   ]]
L0 QK matrix cosine similarities:
[[ 1.     0.209 -0.016 -0.014  0.115  0.018]
 [ 0.209  1.    -0.004 -0.036  0.209 -0.008]
 [-0.016 -0.004  1.     0.239 -0.006 -0.026]
 [-0.014 -0.036  0.239  1.    -0.054 -0.058]
 [ 0.115  0.209 -0.006 -0.054  1.     0.172]
 [ 0.018 -0.008 -0.026 -0.058  0.172  1.   ]]

L0H2 unweighted QK singular values:
[59.879 57.279 55.581 54.347 51.234 49.452 47.184 44.678 43.21  41.651 39.871 35.376 31.577 26.606 23.538 21.633 19.958 18.271 16.837 14.93  13.224
 10.856  9.22   8.099  8.066  6.534  4.851  4.339  3.972  3.207  3.015  2.925  2.44   2.351  1.993  1.798  1.513  1.451  1.349  1.297  1.132  1.074
  0.976  0.837  0.819  0.709  0.665
```

## Call 4

```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

A,B,Rkey = {},{},{}
for l in range(4):
    d=W[f"ln1_w.{l}"].astype(float)
    z=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        sl=slice(64*j,64*(j+1))
        q=W[f"W_q.{l}"][sl].astype(float)
        k=W[f"W_k.{l}"][sl].astype(float)
        v=W[f"W_v.{l}"][sl].astype(float)
        o=W[f"W_o.{l}"][:,sl].astype(float)
        A[l,j]=(q*d).T@(k*d)
        B[l,j]=o@(v*d)
        Rkey[l,j]=(k*d).T@(q@z)

# Common write subspace of the four similar first-layer OV matrices.
common=np.concatenate([B[0,j] for j in [0,1,4,5]],axis=1)
u,s,vt=la.svd(common,full_matrices=False)
print("Shared L0H0/H1/H4/H5 OV singular values:",np.round(s[:20],4))
clock=u[:,:2]
print("Fraction of each L0 OV's squared norm in shared top-2 write subspace:")
for j in range(6):
    print(j,round(la.norm(clock.T@B[0,j])**2/la.norm(B[0,j])**2,4))

print("\nLater QK energy on shared top-2 write subspace and bias-key alignment")
print("head: query_fraction key_fraction key_bias_fraction key_bias_coordinates")
for l in range(1,4):
    for j in range(6):
        a=A[l,j]
        r=Rkey[l,j]
        print(f"L{l}H{j}",
              np.round([la.norm(clock.T@a)**2/la.norm(a)**2,
                        la.norm(a@clock)**2/la.norm(a)**2,
                        la.norm(clock.T@r)**2/la.norm(r)**2],4),
              np.round(clock.T@r,3))

# A character-channel candidate: right singular subspace of L0H2 OV.
u2,s2,vt2=la.svd(B[0,2],full_matrices=False)
print("\nL0H2 gain-weighted OV singular values:",np.round(s2[:40],4))
T=vt2[:27].T
print("Projected compositions on L0H2's top-27 read subspace:")
print("head: trace(A Bprev) projected_trace symmetric_fraction positive/negative_sym_eigenmass")
for l in [1,2,3]:
    for j in range(6):
        a=A[l,j]
        m=T.T@a@B[0,2]@T
        sym=(m+m.T)/2
        es=la.eigvalsh(sym)
        print(f"L{l}H{j}",
              np.round([np.trace(a@B[0,2]),np.trace(m),
                        la.norm(sym)/la.norm(m),
                        es[es>0].sum(),-es[es<0].sum()],4))

print("\nUnweighted L0H2 V singular values:")
print(np.round(la.svdvals(W["W_v.0"][128:192].astype(float)),4))

```

Output:

```
Shared L0H0/H1/H4/H5 OV singular values: [11.996   8.4336  3.3624  1.312   0.9992  0.8615  0.7301  0.6058  0.5661  0.5402  0.4827  0.436   0.3932  0.3667  0.347   0.3209  0.306   0.2954
  0.2622  0.246 ]
Fraction of each L0 OV's squared norm in shared top-2 write subspace:
0 0.9123
1 0.9385
2 0.0028
3 0.2485
4 0.8528
5 0.9345

Later QK energy on shared top-2 write subspace and bias-key alignment
head: query_fraction key_fraction key_bias_fraction key_bias_coordinates
L1H0 [0.0614 0.5748 0.7729] [7.137 6.341]
L1H1 [0.0648 0.5364 0.8146] [13.096 10.032]
L1H2 [0.0606 0.5491 0.8131] [14.859 11.302]
L1H3 [0.0591 0.5547 0.7933] [12.977 10.594]
L1H4 [0.0569 0.6026 0.8268] [13.741  9.42 ]
L1H5 [0.0677 0.5488 0.7836] [9.62  7.843]
L2H0 [0.0275 0.0919 0.8536] [10.353  7.109]
L2H1 [0.0082 0.0444 0.7929] [4.547 2.734]
L2H2 [0.0203 0.1766 0.8019] [4.247 3.356]
L2H3 [0.0381 0.092  0.8678] [9.315 5.662]
L2H4 [0.0304 0.0922 0.8624] [10.055  6.023]
L2H5 [0.0306 0.0882 0.8535] [9.473 5.885]
L3H0 [0.0014 0.0127 0.1481] [0.676 0.418]
L3H1 [0.001  0.0023 0.0316] [-0.327  0.106]
L3H2 [0.0118 0.0693 0.5917] [1.495 0.97 ]
L3H3 [0.0162 0.0655 0.666 ] [2.074 1.111]
L3H4 [0.0054 0.044  0.5194] [1.545 0.865]
L3H5 [0.0029 0.0084 0.0312] [ 0.232 -0.016]

L0H2 gain-weighted OV singular values: [0.7873 0.718  0.6827 0.6713 0.6378 0.6013 0.5692 0.5583 0.5504 0.5222 0.5151 0.4872 0.4805 0.4696 0.4391 0.4211 0.4006 0.3931 0.3471 0.3044 0.2772
 0.2754 0.2565 0.2286 0.2181 0.205  0.1814 0.1707 0.1628 0.1593 0.1314 0.1184 0.1108 0.1014 0.0949 0.0807 0.0726 0.0649 0.0508 0.0469]
Projected compositions on L0H2's top-27 read subspace:
head: trace(A Bprev) projected_trace symmetric_fraction positive/negative_sym_eigenmass
L1H0 [-0.0978 -0.109   0.7328  0.2629  0.372 ]
L1H1 [-0.0033 -0.0118  0.7879  0.7768  0.7886]
L1H2 [0.458  0.4426 0.7303 1.138  0.6953]
L1H3 [0.4099 0.4019 0.7498 0.9661 0.5643]
L1H4 [0.3922 0.3805 0.7224 0.9042 0.5237]
L1H5 [0.1614 0.1521 0.7166 0.9226 0.7705]
L2H0 [-0.248  -0.2914  0.707   1.0645  1.3559]
L2H1 [-0.2372 -0.2557  0.7393  0.8213  1.077 ]
L2H2 [0.0327 0.0371 0.7256 0.635  0.5978]
L2H3 [0.1216 0.1512 0.7026 0.9162 0.765 ]
L2H4 [-0.3803 -0.3931  0.705   0.8113  1.2044]
L2H5 [0.0158 0.0345 0.7101 1.0638 1.0293]
L3H0 [-0.1019 -0.0942  0.7048  0.9675  1.0617]
L3H1 [3.5295 3.5096 0.9062 3.7617 0.252 ]
L3H2 [0.0903 0.0916 0.7098 0.6001 0.5085]
L3H3 [0.0381 0.0281 0.6743 0.5445 0.5164]
L3H4 [-0.1812 -0.1594  0.7495  0.6723  0.8317]
L3H5 [0.0419 0.0467 0.7255 0.3519 0.3052]

Unweighted L0H2 V singular values:
[1.9539 1.8791 1.8467 1.8044 1.7716 1.7334 1.7213 1.6695 1.6478 1.5709 1.5539 1.5198 1.5167 1.4806 1.4422 1.331  1.2633 1.2307 1.0507 0.9934 0.9376
 0.9084 0.843  0.8111 0.7783 0.7104 0.6947 0.6454 0.6367 0.599  0.5752 0.5655 0.5477 0.5344 0.5259 0.5052 0.4878 0.4767 0.4555 0.4349 0.427  0.4089
 0.3946 0.3823 0.3712 0.3616 0.3501 0.3437 0.3341 0.323  0.3187 0.3139 0.302  0.2976 0.2964 0.2836 0.281  0.2722 0.2652 0.2598 0.2594 0.2497 0.2364
 0.2305]

```

## Call 5

```python
import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
threadpool_limits(4)

d=W["ln1_w.0"].astype(float)
v=W["W_v.0"][128:192].astype(float)*d
o=W["W_o.0"][:,128:192].astype(float)
B=o@v
F=W["W_fc.0"].astype(float)*W["ln2_w.0"]
G=F@B
u,s,vt=la.svd(G-G.mean(axis=0),full_matrices=False)
print("Singular values of centered W_fc.0 D_ln2.0 B_L0H2:")
print(np.round(s[:50],4))

# Clustering weight rows, not activations or model inputs.
for r in [20,24,27,32]:
    X=(G-G.mean(axis=0))@vt[:r].T
    X/=np.sqrt(s[:r])[None,:]
    X/=np.maximum(la.norm(X,axis=1,keepdims=True),1e-12)
    km=KMeans(n_clusters=r+1,n_init=10,random_state=0).fit(X)
    sizes=np.bincount(km.labels_,minlength=r+1)
    coh=np.array([la.norm(X[km.labels_==i].mean(axis=0)) for i in range(r+1)])
    sil=silhouette_score(X,km.labels_,sample_size=1000,random_state=0)
    print("rank",r,"clusters",r+1,"silhouette",round(sil,4),
          "mean_cohesion",round(np.average(coh,weights=sizes),4))
    print("sizes descending",np.sort(sizes)[::-1])

print("\nEffective QK symmetric eigenvalue masses, no input evaluation:")
for l in range(4):
    d=W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        sl=slice(64*j,64*(j+1))
        q=W[f"W_q.{l}"][sl].astype(float)*d
        k=W[f"W_k.{l}"][sl].astype(float)*d
        a=q.T@k
        ev=la.eigvalsh((a+a.T)/2)
        pos=ev[ev>0].sum()
        neg=-ev[ev<0].sum()
        print(f"L{l}H{j}", "positive",round(pos,3),"negative",round(neg,3),
              "signed_mass_fraction",round((pos-neg)/(pos+neg),4))

```

Output:

```
Singular values of centered W_fc.0 D_ln2.0 B_L0H2:
[3.6485 2.7127 2.5966 2.3846 2.3094 1.9764 1.9523 1.9044 1.7751 1.6206 1.6077 1.4051 1.3241 1.2869 1.1878 1.1569 1.0534 0.8885 0.6879 0.4863 0.406
 0.3707 0.3187 0.2846 0.2552 0.2371 0.2254 0.1925 0.1892 0.1811 0.1509 0.1421 0.1287 0.113  0.1058 0.0921 0.0829 0.0712 0.0631 0.0553 0.0482 0.0476
 0.04   0.034  0.0309 0.0276 0.0267 0.0212 0.0188 0.0177]
rank 20 clusters 21 silhouette 0.0421 mean_cohesion 0.5708
sizes descending [108  92  91  89  88  87  83  80  79  78  77  75  74  73  71  70  59  50  42  38  32]
rank 24 clusters 25 silhouette 0.0385 mean_cohesion 0.5578
sizes descending [87 82 82 79 79 77 76 69 67 67 66 63 59 58 56 55 55 53 50 48 46 45 45 37 35]
rank 27 clusters 28 silhouette 0.038 mean_cohesion 0.5501
sizes descending [82 80 77 77 75 69 68 68 67 60 60 58 58 57 55 53 53 50 44 42 42 41 39 39 39 35 24 24]
rank 32 clusters 33 silhouette 0.0362 mean_cohesion 0.5417
sizes descending [69 63 62 60 59 59 58 58 54 53 51 50 50 48 48 47 46 46 46 45 45 42 41 40 38 38 37 37 33 31 30 27 25]

Effective QK symmetric eigenvalue masses, no input evaluation:
L0H0 positive 0.526 negative 0.563 signed_mass_fraction -0.0348
L0H1 positive 0.35 negative 0.462 signed_mass_fraction -0.1372
L0H2 positive 57.4 negative 47.8 signed_mass_fraction 0.0913
L0H3 positive 4.184 negative 2.609 signed_mass_fraction 0.2319
L0H4 positive 0.591 negative 1.158 signed_mass_fraction -0.3243
L0H5 positive 0.456 negative 0.733 signed_mass_fraction -0.2334
L1H0 positive 8.996 negative 14.15 signed_mass_fraction -0.2227
L1H1 positive 12.839 negative 22.133 signed_mass_fraction -0.2658
L1H2 positive 14.061 negative 27.016 signed_mass_fraction -0.3154
L1H3 positive 13.096 negative 22.354 signed_mass_fraction -0.2612
L1H4 positive 13.29 negative 23.248 signed_mass_fraction -0.2725
L1H5 positive 13.028 negative 21.144 signed_mass_fraction -0.2375
L2H0 positive 62.04 negative 62.213 signed_mass_fraction -0.0014
L2H1 positive 58.967 negative 54.73 signed_mass_fraction 0.0373
L2H2 positive 38.345 negative 47.188 signed_mass_fraction -0.1034
L2H3 positive 53.32 negative 59.226 signed_mass_fraction -0.0525
L2H4 positive 60.195 negative 64.0 signed_mass_fraction -0.0306
L2H5 positive 56.408 negative 61.986 signed_mass_fraction -0.0471
L3H0 positive 78.679 negative 51.383 signed_mass_fraction 0.2099
L3H1 positive 78.364 negative 27.915 signed_mass_fraction 0.4747
L3H2 positive 42.154 negative 40.39 signed_mass_fraction 0.0214
L3H3 positive 45.205 negative 44.312 signed_mass_fraction 0.01
L3H4 positive 56.44 negative 49.211 signed_mass_fraction 0.0684
L3H5 positive 22.97 negative 22.177 signed_mass_fraction 0.0176

```

## Call 6

```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

d=W["ln1_w.0"].astype(float)
A,B={},{}
for j in range(6):
    sl=slice(64*j,64*(j+1))
    q=W["W_q.0"][sl].astype(float)*d
    k=W["W_k.0"][sl].astype(float)*d
    v=W["W_v.0"][sl].astype(float)*d
    o=W["W_o.0"][:,sl].astype(float)
    A[j]=q.T@k
    B[j]=o@v

# Compare weight matrices with powers of the orthogonal polar factor
# of L0H2's QK matrix. These are matrix alignments, not attention scores.
q=W["W_q.0"][128:192].astype(float)*d
k=W["W_k.0"][128:192].astype(float)*d
c=q.T@q+k.T@k
ev,U=la.eigh(c)
ix=np.argsort(ev)[::-1]
ev,U=ev[ix],U[:,ix]
for r in [16,20,24,32]:
    Z=U[:,:r]/np.sqrt(ev[:r])[None,:]
    t=Z.T@A[2]@Z
    rot,_=la.polar(t)
    print("\nrank",r,"whitened L0H2 singular values",np.round(la.svdvals(t),3))
    print("Alignment with polar-factor powers, columns -4..4:")
    for j in [0,1,3,4,5]:
        m=Z.T@A[j]@Z
        vals=[np.sum(m*np.linalg.matrix_power(rot,p))/
              (la.norm(m)*np.sqrt(r)) for p in range(-4,5)]
        print(f"L0H{j}",np.round(vals,4))

print("\nShared first-layer read subspace and positional-circuit overlap")
stack=np.concatenate([B[j] for j in [0,1,4,5]],axis=0)
_,s,vt=la.svd(stack,full_matrices=False)
print("Shared-read singular values",np.round(s[:10],4))
a_u,a_s,a_vt=la.svd(A[2],full_matrices=False)
P=la.orth(np.column_stack([a_u[:,:20],a_vt[:20].T]))
for i in range(4):
    v=vt[i]
    print("read direction",i,"energy in L0H2 QK subspace",
          round(la.norm(P.T@v)**2,4))

# Check the induction-like composition without choosing one particular rank.
a=(W["W_q.3"][64:128].astype(float)*W["ln1_w.3"]).T @ (
    W["W_k.3"][64:128].astype(float)*W["ln1_w.3"])
_,_,vt=la.svd(B[2],full_matrices=False)
print("\nL3H1 x L0H2 induction-like composition, subspace-rank sensitivity")
for r in [16,20,24,27,32,40,64]:
    T=vt[:r].T
    m=T.T@a@B[2]@T
    e=la.eigvalsh((m+m.T)/2)
    print(r,"trace",round(np.trace(m),4),
          "positive_mass",round(e[e>0].sum(),4),
          "negative_mass",round(-e[e<0].sum(),4),
          "symmetry_fraction",round(la.norm((m+m.T)/2)/la.norm(m),4))

```

Output:

```

rank 16 whitened L0H2 singular values [0.617 0.575 0.566 0.53  0.521 0.507 0.493 0.472 0.468 0.443 0.419 0.405 0.391 0.37  0.083 0.017]
Alignment with polar-factor powers, columns -4..4:
L0H0 [-0.026   0.076   0.1024 -0.018  -0.0802 -0.1375 -0.0964  0.0035  0.1072]
L0H1 [ 0.042   0.0077  0.007  -0.0733 -0.0518 -0.0235  0.0194 -0.0014  0.0007]
L0H3 [-0.2319 -0.2995 -0.1674  0.1027  0.2253  0.2039 -0.0107 -0.2293 -0.2839]
L0H4 [ 0.1195  0.0721 -0.047  -0.1282 -0.1541  0.0002  0.1201  0.0769 -0.0293]
L0H5 [ 0.1541  0.1988  0.0798 -0.0938 -0.2027 -0.1472  0.0305  0.1617  0.1551]

rank 20 whitened L0H2 singular values [0.646 0.605 0.596 0.588 0.558 0.541 0.527 0.525 0.51  0.501 0.461 0.436 0.429 0.427 0.414 0.394 0.38  0.291 0.243 0.235]
Alignment with polar-factor powers, columns -4..4:
L0H0 [-0.0006  0.0326  0.0802  0.0299 -0.0569 -0.0544 -0.0776 -0.0236  0.0441]
L0H1 [ 0.0427  0.0052 -0.0097 -0.0298 -0.0768  0.0005  0.0062  0.005  -0.0008]
L0H3 [-0.1689 -0.2718 -0.164   0.0997  0.1326  0.199  -0.0295 -0.2152 -0.2294]
L0H4 [ 0.0984  0.0222 -0.0847 -0.1366 -0.1324 -0.0288  0.0773  0.0386 -0.0272]
L0H5 [ 0.1218  0.1262  0.0674 -0.0784 -0.1501 -0.1069  0.0388  0.1069  0.1294]

rank 24 whitened L0H2 singular values [0.695 0.655 0.624 0.617 0.606 0.591 0.571 0.565 0.543 0.54  0.521 0.504 0.495 0.488 0.454 0.428 0.412 0.39  0.351 0.338 0.283 0.201 0.095 0.025]
Alignment with polar-factor powers, columns -4..4:
L0H0 [-0.009   0.0091  0.0543  0.0603  0.051  -0.012   0.0038 -0.035   0.0292]
L0H1 [ 0.0074  0.0155  0.0067  0.0354 -0.0988  0.0211  0.0051 -0.0227 -0.0332]
L0H3 [-0.1302 -0.1833 -0.077   0.0664  0.0764  0.1546  0.0565 -0.0982 -0.1215]
L0H4 [ 0.0085  0.0169 -0.0877  0.059  -0.1798  0.055  -0.082   0.0621 -0.077 ]
L0H5 [ 0.027   0.0892  0.0264  0.0462 -0.1206  0.0031 -0.0609  0.0293  0.0147]

rank 32 whitened L0H2 singular values [0.89  0.764 0.721 0.711 0.684 0.67  0.651 0.634 0.623 0.612 0.593 0.586 0.57  0.562 0.539 0.526 0.514 0.474 0.457 0.417 0.385 0.356 0.345 0.311
 0.282 0.256 0.188 0.121 0.049 0.03  0.013 0.   ]
Alignment with polar-factor powers, columns -4..4:
L0H0 [-0.0058 -0.0268 -0.0259  0.0236  0.0453 -0.0054 -0.0143  0.0132  0.0138]
L0H1 [-0.0239 -0.0035 -0.0019 -0.0149 -0.0328 -0.023   0.0047  0.0314  0.0027]
L0H3 [-0.0339  0.0386  0.0535 -0.0087 -0.0414  0.1335  0.013  -0.0334  0.0008]
L0H4 [-0.0246 -0.0487  0.047   0.0855 -0.1421 -0.0777  0.0692 -0.0124 -0.0669]
L0H5 [-0.0346 -0.0258  0.0829  0.0006 -0.1251  0.0222  0.092  -0.0212 -0.0558]

Shared first-layer read subspace and positional-circuit overlap
Shared-read singular values [12.4276  7.5882  3.5107  1.4158  1.1895  0.935   0.8361  0.731   0.6597  0.558 ]
read direction 0 energy in L0H2 QK subspace 0.9829
read direction 1 energy in L0H2 QK subspace 0.9488
read direction 2 energy in L0H2 QK subspace 0.8122
read direction 3 energy in L0H2 QK subspace 0.4707

L3H1 x L0H2 induction-like composition, subspace-rank sensitivity
16 trace 3.2581 positive_mass 3.2621 negative_mass 0.0039 symmetry_fraction 0.9338
20 trace 3.4768 positive_mass 3.5445 negative_mass 0.0677 symmetry_fraction 0.9243
24 trace 3.4982 positive_mass 3.6684 negative_mass 0.1702 symmetry_fraction 0.9143
27 trace 3.5096 positive_mass 3.7617 negative_mass 0.252 symmetry_fraction 0.9062
32 trace 3.5278 positive_mass 3.8829 negative_mass 0.3551 symmetry_fraction 0.8961
40 trace 3.5302 positive_mass 4.081 negative_mass 0.5508 symmetry_fraction 0.8772
64 trace 3.5295 positive_mass 4.643 negative_mass 1.1134 symmetry_fraction 0.8402

```
