## Call 1

```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

def spectrum(A):
    s = la.svdvals(A)
    e = s*s
    er = e.sum()**2 / (e@e)
    return s, er

print("WEIGHT SUMMARY")
for l in range(4):
    print("\nLAYER", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        a = W[f"{name}.{l}"].astype(float)
        print(name, "min/mean/max/std/norm",
              np.round([a.min(), a.mean(), a.max(), a.std(), la.norm(a)], 4))
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        a = W[f"{name}.{l}"].astype(float)
        s, er = spectrum(a)
        print(name, "Frob", round(la.norm(a), 3), "top8", np.round(s[:8], 3),
              "median/min", np.round([np.median(s), s[-1]], 4), "energy_rank", round(er, 2))
    print("HEADS: norms Q,K,V,O; QK top6; OV top6; QK symmetry; trace OV")
    for j in range(6):
        sl = slice(64*j, 64*(j+1))
        q = W[f"W_q.{l}"][sl].astype(float)
        k = W[f"W_k.{l}"][sl].astype(float)
        v = W[f"W_v.{l}"][sl].astype(float)
        o = W[f"W_o.{l}"][:, sl].astype(float)
        a = q.T @ k / 8
        b = o @ v
        sa, ra = spectrum(a)
        sb, rb = spectrum(b)
        sym = np.sum(a*a.T) / np.sum(a*a)
        print(f"L{l}H{j}",
              "norms", np.round([la.norm(q),la.norm(k),la.norm(v),la.norm(o)], 2),
              "QK", np.round(sa[:6],3), "r", round(ra,1),
              "OV", np.round(sb[:6],3), "r", round(rb,1),
              "sym", round(sym,3), "trOV", round(np.trace(b),3))

```

Output:

```
WEIGHT SUMMARY

LAYER 0
ln1_w min/mean/max/std/norm [0.0165 0.2677 1.0134 0.2147 6.7241]
ln1_b min/mean/max/std/norm [-0.1215  0.0007  0.1418  0.0366  0.7172]
ln2_w min/mean/max/std/norm [ 0.0468  0.8168  1.6412  0.3312 17.2723]
ln2_b min/mean/max/std/norm [-0.5537  0.018   0.1663  0.0667  1.353 ]
W_q Frob 22.704 top8 [5.039 4.999 4.703 4.377 4.343 4.294 4.058 4.032] median/min [0.3357 0.0006] energy_rank 42.85
W_k Frob 23.214 top8 [6.204 5.052 4.572 4.537 4.391 4.289 4.284 4.208] median/min [0.3467 0.0001] energy_rank 39.33
W_v Frob 18.197 top8 [4.553 4.024 3.738 3.497 3.465 3.367 3.279 3.215] median/min [0.3117 0.0002] energy_rank 39.74
W_o Frob 19.6 top8 [6.678 5.237 2.835 2.679 2.62  2.581 2.55  2.515] median/min [0.2219 0.0003] energy_rank 36.89
W_fc Frob 51.65 top8 [18.11  12.28   8.473  7.902  7.779  7.442  7.146  7.109] median/min [0.9364 0.4206] energy_rank 36.49
W_proj Frob 45.806 top8 [10.932  7.576  6.628  6.293  6.166  5.924  5.886  5.769] median/min [1.1662 0.2442] energy_rank 86.43
HEADS: norms Q,K,V,O; QK top6; OV top6; QK symmetry; trace OV
L0H0 norms [4.13 4.91 5.35 5.21] QK [0.649 0.424 0.106 0.082 0.064 0.057] r 2.0 OV [6.898 3.897 1.936 1.638 1.469 0.903] r 2.2 sym 0.155 trOV 1.62
L0H1 norms [12.46 12.64  9.26 10.2 ] QK [2.153 1.515 1.437 1.216 1.074 1.068] r 7.4 OV [4.703 4.573 4.424 4.248 4.199 4.159] r 20.8 sym 0.164 trOV -0.82
L0H2 norms [6.46 6.83 5.99 5.54] QK [0.561 0.447 0.374 0.356 0.324 0.279] r 8.2 OV [4.786 3.621 3.514 2.916 2.224 1.742] r 5.6 sym 0.176 trOV -0.743
L0H3 norms [6.63 6.82 4.88 5.47] QK [0.959 0.701 0.431 0.294 0.279 0.225] r 3.7 OV [2.168 1.997 1.875 1.555 1.484 1.369] r 10.7 sym 0.085 trOV -0.071
L0H4 norms [2.17 2.62 8.28 8.94] QK [0.04  0.03  0.024 0.018 0.017 0.017] r 18.6 OV [18.815 15.373  5.587  3.989  2.502  2.324] r 2.4 sym 0.039 trOV 6.681
L0H5 norms [15.9  15.96  9.41 10.6 ] QK [1.76  1.72  1.649 1.558 1.543 1.451] r 17.7 OV [5.61  5.503 5.123 4.867 4.799 4.698] r 17.9 sym -0.228 trOV -11.398

LAYER 1
ln1_w min/mean/max/std/norm [0.1979 0.4754 1.2096 0.0964 9.5046]
ln1_b min/mean/max/std/norm [-1.0505  0.022   0.9102  0.1532  3.0327]
ln2_w min/mean/max/std/norm [ 0.0139  1.0111  1.4707  0.1582 20.0543]
ln2_b min/mean/max/std/norm [-0.3744  0.0275  0.3652  0.1442  2.8769]
W_q Frob 31.676 top8 [8.984 7.137 6.079 5.874 5.19  5.102 4.973 4.885] median/min [0.4135 0.0009] energy_rank 45.0
W_k Frob 28.602 top8 [10.967  9.448  5.438  4.842  4.539  4.481  4.235  4.11 ] median/min [0.4007 0.0025] energy_rank 23.16
W_v Frob 28.469 top8 [5.35  5.064 4.979 4.819 4.69  4.585 4.519 4.419] median/min [0.474  0.0006] energy_rank 66.98
W_o Frob 26.151 top8 [4.19  3.651 3.385 3.352 3.24  3.216 3.141 3.115] median/min [0.5507 0.0004] energy_rank 110.15
W_fc Frob 80.261 top8 [20.796 14.143 13.826 13.428 13.122 12.883 12.764 12.474] median/min [1.4081 0.4014] energy_rank 57.23
W_proj Frob 75.119 top8 [7.625 7.359 6.908 6.489 6.344 6.223 6.192 6.02 ] median/min [3.5764 0.3416] energy_rank 278.49
HEADS: norms Q,K,V,O; QK top6; OV top6; QK symmetry; trace OV
L1H0 norms [10.56  9.83 12.95 11.17] QK [1.116 0.861 0.756 0.719 0.664 0.626] r 10.9 OV [9.927 9.271 8.872 7.793 7.506 6.69 ] r 14.2 sym 0.042 trOV -34.568
L1H1 norms [14.53 12.8  11.06 10.66] QK [2.392 1.846 1.725 1.378 1.068 0.946] r 8.2 OV [6.211 5.658 5.403 5.113 4.867 4.702] r 20.2 sym 0.082 trOV -23.812
L1H2 norms [12.69 11.35 12.12 11.19] QK [1.811 1.583 1.277 0.905 0.808 0.711] r 7.8 OV [6.799 6.707 6.454 6.192 6.    5.739] r 20.1 sym 0.07 trOV -31.928
L1H3 norms [12.7  11.75 11.32 10.14] QK [1.998 1.66  1.278 0.901 0.796 0.733] r 7.1 OV [7.119 6.39  5.875 5.68  5.417 5.131] r 16.7 sym 0.062 trOV -22.474
L1H4 norms [12.59 11.51 10.92  9.91] QK [1.853 1.468 1.368 0.938 0.865 0.722] r 7.9 OV [6.354 5.996 5.642 5.541 5.215 5.171] r 16.7 sym 0.052 trOV -21.931
L1H5 norms [14.14 12.58 11.23 10.92] QK [2.523 2.023 1.535 1.4   1.068 0.943] r 6.9 OV [5.781 5.643 5.336 4.976 4.846 4.795] r 21.9 sym 0.072 trOV -27.708

LAYER 2
ln1_w min/mean/max/std/norm [ 0.3445  0.7343  0.9047  0.0712 14.4574]
ln1_b min/mean/max/std/norm [-0.9417  0.0244  0.7326  0.1315  2.6214]
ln2_w min/mean/max/std/norm [ 0.0775  1.275   1.5018  0.1126 25.082 ]
ln2_b min/mean/max/std/norm [-0.4157  0.0188  0.3635  0.1474  2.911 ]
W_q Frob 37.483 top8 [5.067 4.742 4.456 4.39  4.314 4.26  4.157 4.097] median/min [1.3121 0.0012] energy_rank 150.03
W_k Frob 36.356 top8 [8.293 7.348 5.059 4.467 4.306 3.912 3.776 3.773] median/min [1.2598 0.0043] energy_rank 102.35
W_v Frob 34.746 top8 [3.89  3.824 3.791 3.654 3.644 3.628 3.602 3.582] median/min [1.27   0.0023] energy_rank 166.35
W_o Frob 34.205 top8 [4.442 4.073 3.768 3.455 3.44  3.365 3.35  3.297] median/min [1.3879 0.0041] energy_rank 182.06
W_fc Frob 79.454 top8 [14.829  8.048  7.921  7.76   7.66   7.586  7.522  7.509] median/min [3.4107 0.4996] energy_rank 187.52
W_proj Frob 77.477 top8 [10.095  8.985  8.075  7.909  7.791  7.656  7.588  7.375] median/min [3.4879 0.3586] energy_rank 231.72
HEADS: norms Q,K,V,O; QK top6; OV top6; QK symmetry; trace OV
L2H0 norms [15.22 14.53 13.81 13.64] QK [0.935 0.887 0.852 0.809 0.745 0.728] r 34.8 OV [5.556 5.132 5.005 4.795 4.773 4.7  ] r 41.0 sym -0.006 trOV -48.29
L2H1 norms [14.68 14.47 14.68 14.21] QK [0.907 0.818 0.747 0.716 0.675 0.655] r 37.5 OV [8.652 6.394 6.333 6.217 6.016 5.725] r 31.5 sym 0.04 trOV -54.375
L2H2 norms [17.26 16.72 12.99 13.  ] QK [1.354 1.284 1.219 1.175 1.148 1.13 ] r 29.1 OV [4.742 4.585 4.492 4.275 4.22  4.205] r 41.6 sym 0.034 trOV -39.941
L2H3 norms [15.01 14.61 14.46 14.25] QK [1.012 0.878 0.775 0.766 0.712 0.684] r 34.0 OV [6.179 5.765 5.693 5.322 5.194 5.009] r 40.9 sym -0.015 trOV -51.741
L2H4 norms [15.58 15.2  13.47 13.5 ] QK [1.451 1.211 0.897 0.843 0.772 0.756] r 24.5 OV [5.445 5.184 5.128 4.988 4.903 4.768] r 38.0 sym -0.001 trOV -52.123
L2H5 norms [13.84 13.3  15.55 15.08] QK [0.788 0.74  0.685 0.588 0.574 0.552] r 36.8 OV [8.467 7.339 6.951 6.407 6.182 5.987] r 35.8 sym -0.005 trOV -45.69

LAYER 3
ln1
```

## Call 2

```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

d = 384
P = np.eye(d) - np.ones((d,d))/d
A, B = {}, {}
print("STATIC PRODUCTS (LayerNorm gains and centering only; no inputs or layer evaluations)")
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    D = g[:,None] * P
    for j in range(6):
        sl = slice(64*j,64*(j+1))
        q,k,v = [W[f"{n}.{l}"][sl].astype(float) for n in ["W_q","W_k","W_v"]]
        o = W[f"W_o.{l}"][:,sl].astype(float)
        a = D.T @ q.T @ k @ D / 8
        ov = P @ o @ v @ D
        A[l,j], B[l,j] = a, ov
        ss = la.svdvals(a)
        ee = la.eigvalsh((a+a.T)/2)
        kb = D.T @ k.T @ q @ b / 8
        qb = D.T @ q.T @ k @ b / 8
        print(f"L{l}H{j}", "QKnorm",round(la.norm(a),3),
              "top",np.round(ss[:5],3),
              "sym",round(np.sum(a*a.T)/np.sum(a*a),3),
              "sym_eigs",np.round(np.r_[ee[:3],ee[-3:]],3),
              "key/query_bias_norm",np.round([la.norm(kb),la.norm(qb)],3),
              "OVnorm/tr",np.round([la.norm(ov),np.trace(ov)],3))

print("\nL0 QK singular subspace overlap with each L0 OV input subspace (top 24 each)")
for j in range(6):
    u,s,vt=la.svd(A[0,j],full_matrices=False)
    vals=[]
    for k in range(6):
        _,_,vbt=la.svd(B[0,k],full_matrices=False)
        vals.append([la.norm(u[:,:24].T@vbt[:24].T)**2/24,
                     la.norm(vt[:24]@vbt[:24].T)**2/24])
    print(j, np.round(vals,3))

print("\nKEY-SIDE COMPOSITION: A_l @ OV_earlier")
print("For each head: top earlier heads by positive trace / nuclear norm; then largest Frobenius alignments")
for l in range(1,4):
    for j in range(6):
        rows=[]
        for m in range(l):
            for k in range(6):
                c=A[l,j]@B[m,k]
                sv=la.svdvals(c)
                rows.append((np.trace(c)/max(sv.sum(),1e-12),
                             la.norm(c)/(la.norm(A[l,j])*la.norm(B[m,k])),
                             f"L{m}H{k}", np.trace(c), la.norm(c)))
        pos=sorted(rows,reverse=True)[:4]
        mag=sorted(rows,key=lambda x:x[1],reverse=True)[:3]
        print(f"L{l}H{j}",
              "positive",[(r[2],round(r[0],3),round(r[3],3)) for r in pos],
              "strength",[(r[2],round(r[1],3)) for r in mag])

print("\nL0 effective QK eigenvalues of largest magnitude")
for j in range(6):
    e=la.eigvals(A[0,j])
    e=e[np.argsort(abs(e))[::-1]][:16]
    print(j, np.round(e,3))

```

Output:

```
STATIC PRODUCTS (LayerNorm gains and centering only; no inputs or layer evaluations)
L0H0 QKnorm 0.114 top [0.089 0.068 0.013 0.008 0.006] sym 0.181 sym_eigs [-0.067 -0.03  -0.006  0.006  0.017  0.041] key/query_bias_norm [0.018 0.016] OVnorm/tr [4.321 1.666]
L0H1 QKnorm 0.878 top [0.601 0.373 0.297 0.211 0.199] sym -0.1 sym_eigs [-0.291 -0.146 -0.127  0.05   0.168  0.377] key/query_bias_norm [0.074 0.106] OVnorm/tr [3.488 0.164]
L0H2 QKnorm 0.175 top [0.132 0.08  0.044 0.031 0.028] sym 0.403 sym_eigs [-0.032 -0.023 -0.022  0.018  0.062  0.115] key/query_bias_norm [0.023 0.079] OVnorm/tr [2.023 0.015]
L0H3 QKnorm 0.239 top [0.165 0.14  0.081 0.031 0.029] sym 0.209 sym_eigs [-0.113 -0.107 -0.042  0.016  0.033  0.073] key/query_bias_norm [0.057 0.149] OVnorm/tr [1.621 0.242]
L0H4 QKnorm 0.009 top [0.004 0.003 0.002 0.002 0.002] sym 0.023 sym_eigs [-0.003 -0.002 -0.001  0.001  0.001  0.002] key/query_bias_norm [0.001 0.001] OVnorm/tr [17.158  6.049]
L0H5 QKnorm 0.97 top [0.41  0.362 0.273 0.254 0.245] sym -0.228 sym_eigs [-0.083 -0.078 -0.064  0.21   0.22   0.273] key/query_bias_norm [0.051 0.066] OVnorm/tr [ 3.707 -2.463]
L1H0 QKnorm 0.644 top [0.354 0.252 0.218 0.167 0.158] sym 0.02 sym_eigs [-0.186 -0.13  -0.114  0.1    0.131  0.183] key/query_bias_norm [0.839 0.841] OVnorm/tr [ 13.436 -16.299]
L1H1 QKnorm 1.584 top [1.146 0.661 0.459 0.326 0.249] sym -0.003 sym_eigs [-0.544 -0.279 -0.261  0.175  0.349  0.581] key/query_bias_norm [3.15  2.183] OVnorm/tr [  9.592 -11.205]
L1H2 QKnorm 1.236 top [0.807 0.693 0.332 0.205 0.18 ] sym -0.016 sym_eigs [-0.375 -0.307 -0.192  0.134  0.365  0.412] key/query_bias_norm [2.2   1.897] OVnorm/tr [ 11.496 -14.892]
L1H3 QKnorm 1.309 top [0.913 0.699 0.309 0.202 0.178] sym -0.044 sym_eigs [-0.391 -0.299 -0.182  0.147  0.362  0.476] key/query_bias_norm [2.719 2.207] OVnorm/tr [ 10.043 -10.571]
L1H4 QKnorm 1.249 top [0.848 0.654 0.34  0.213 0.191] sym -0.037 sym_eigs [-0.378 -0.277 -0.196  0.141  0.342  0.441] key/query_bias_norm [2.518 1.942] OVnorm/tr [  9.557 -10.446]
L1H5 QKnorm 1.461 top [1.009 0.555 0.482 0.45  0.242] sym 0.01 sym_eigs [-0.495 -0.307 -0.237  0.211  0.3    0.506] key/query_bias_norm [2.46 1.81] OVnorm/tr [  9.826 -13.045]
L2H0 QKnorm 2.046 top [0.518 0.485 0.459 0.416 0.407] sym -0.003 sym_eigs [-0.265 -0.253 -0.25   0.251  0.278  0.293] key/query_bias_norm [0.895 0.774] OVnorm/tr [ 18.698 -35.97 ]
L2H1 QKnorm 1.945 top [0.458 0.421 0.396 0.393 0.367] sym 0.04 sym_eigs [-0.231 -0.195 -0.188  0.251  0.258  0.264] key/query_bias_norm [0.638 0.333] OVnorm/tr [ 21.448 -40.225]
L2H2 QKnorm 2.804 top [0.709 0.693 0.676 0.634 0.611] sym 0.032 sym_eigs [-0.406 -0.357 -0.315  0.405  0.414  0.449] key/query_bias_norm [1.584 1.581] OVnorm/tr [ 16.782 -29.462]
L2H3 QKnorm 1.976 top [0.571 0.423 0.412 0.392 0.388] sym -0.012 sym_eigs [-0.262 -0.234 -0.225  0.227  0.241  0.307] key/query_bias_norm [1.037 1.265] OVnorm/tr [ 20.617 -38.568]
L2H4 QKnorm 2.202 top [0.783 0.587 0.484 0.429 0.417] sym 0.001 sym_eigs [-0.354 -0.288 -0.264  0.286  0.324  0.431] key/query_bias_norm [1.199 1.34 ] OVnorm/tr [ 18.415 -38.753]
L2H5 QKnorm 1.654 top [0.419 0.414 0.359 0.32  0.315] sym -0.006 sym_eigs [-0.22  -0.217 -0.195  0.192  0.214  0.237] key/query_bias_norm [0.636 0.492] OVnorm/tr [ 23.766 -33.94 ]
L3H0 QKnorm 2.078 top [0.405 0.399 0.385 0.377 0.365] sym 0.02 sym_eigs [-0.218 -0.197 -0.194  0.237  0.241  0.255] key/query_bias_norm [0.299 0.302] OVnorm/tr [ 24.502 -48.319]
L3H1 QKnorm 1.695 top [0.322 0.315 0.309 0.304 0.3  ] sym 0.022 sym_eigs [-0.176 -0.175 -0.16   0.189  0.192  0.206] key/query_bias_norm [0.263 0.253] OVnorm/tr [ 25.724 -49.613]
L3H2 QKnorm 2.061 top [0.434 0.411 0.392 0.362 0.358] sym -0.007 sym_eigs [-0.216 -0.207 -0.201  0.221  0.242  0.26 ] key/query_bias_norm [0.342 0.286] OVnorm/tr [ 26.86  -50.565]
L3H3 QKnorm 1.053 top [0.262 0.237 0.208 0.202 0.193] sym -0.007 sym_eigs [-0.131 -0.118 -0.111  0.122  0.136  0.142] key/query_bias_norm [0.258 0.16 ] OVnorm/tr [ 37.938 -51.153]
L3H4 QKnorm 1.734 top [0.401 0.369 0.357 0.323 0.313] sym 0.004 sym_eigs [-0.217 -0.199 -0.191  0.191  0.216  0.246] key/query_bias_norm [0.307 0.239] OVnorm/tr [ 28.44  -35.013]
L3H5 QKnorm 2.117 top [0.404 0.39  0.384 0.37  0.365] sym 0.221 sym_eigs [-0.153 -0.141 -0.119  0.318  0.322  0.323] key/query_bias_norm [0.31  0.307] OVnorm/tr [ 23.687 -56.843]

L0 QK singular subspace overlap with each L0 OV input subspace (top 24 each)
0 [[0.226 0.188]
 [0.078 0.137]
 [0.153 0.175]
 [0.171 0.159]
 [0.237 0.181]
 [0.081 0.135]]
1 [[0.273 0.267]
 [0.084 0.101]
 [0.197 0.184]
 [0.203 0.203]
 [0.234 0.225]
 [0.108 0.136]]
2 [[0.277 0.25 ]
 [0.224 0.293]
 [0.254 0.263]
 [0.289 0.302]
 [0.289 0.279]
 [0.236 0.289]]
3 [[0.257 0.278]
 [0.188 0.267]
 [0.21  0.232]
 [0.247 0.266]
 [0.287 0.307]
 [0.206 0.273]]
4 [[0.129 0.163]
 [0.067 0.085]
 [0.112 0.147]
 [0.118 0.146]
 [0.133 0.151]
 [0.069 0.094]]
5 [[0.273 0.269]
 [0.057 0.065]
 [0.187 0.173]
 [0.186 0.193]
 [0.222 0.226]
 [0.086 0.103]]

KEY-SIDE COMPOSITION: A_l @ OV_earlier
For each head: top earlier heads by positive trace / nuclear norm; then largest Frobenius alignments
L1H0 positive [('L0H4', np.float64(0.039), np.float64(0.229)), ('L0H3', np.float64(0.02), np.float64(0.005)), ('L0H0', np.float64(0.014), np.float64(0.018)), ('L0H1', np.float64(-0.0), np.float64(-0.0))] strength [('L0H4', np.float64(0.395)), ('L0H0', np.float64(0.318)), ('L0H3', np.float64(0.144))]
L1H1 positive [('L0H5', np.float64(0.08), np.float64(0.09)), ('L0H1', np.float64(0.06), np.float64(0.039)), ('L0H4', np.float64(0.027), np.float64(0.519)), ('L0H3', np.float64(0.022), np.float64(0.014))] strength [('L0H4', np.float64(0.545)), ('L0H0', np.float64(0.431)), ('L0H3', np.float64(0.168))]
L1H2 positive [('L0H4', np.float64(0.046), np.float64(0.692)), ('L0H3', np.float64(0.035), np.float64(0.018)), ('L0H0', np.float64(0.007), np.float64(0.021)), ('L0H1', np.float64(0.005), np.float64(0.003))] strength [('L0H4', np.float64(0.519)), ('L0H0', np.
```

## Call 3

```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

print("DETAILED SINGULAR SPECTRA")
for key in ["W_v.0","W_fc.0","W_proj.0","W_proj.3"]:
    s=la.svdvals(W[key].astype(float))
    print(key, "first 60:", np.round(s[:60],3))
    print("largest successive drops in first 80:",
          sorted([(i+1,round(s[i]/s[i+1],3)) for i in range(80)],key=lambda z:z[1],reverse=True)[:12])

P=np.eye(384)-np.ones((384,384))/384
D=W["ln1_w.0"].astype(float)[:,None]*P
V=W["W_v.0"].astype(float)@D
u,s,vt=la.svd(V,full_matrices=False)
print("\nEffective L0 stacked V first 60:",np.round(s[:60],3))
print("Individual L0 V first 35:")
for j in range(6):
    print(j,np.round(la.svdvals(V[64*j:64*(j+1)])[:35],3))

print("\nFirst-layer QK energy outside top-r value-read subspace")
for r in [16,24,27,32,40,64]:
    T=vt[:r].T
    R=P-T@T.T
    row=[]
    for j in range(6):
        q=W["W_q.0"][64*j:64*(j+1)].astype(float)@D
        k=W["W_k.0"][64*j:64*(j+1)].astype(float)@D
        a=q.T@k/8
        row.append([la.norm(R@a@R)**2/la.norm(a)**2,
                    la.norm(T.T@a@T)**2/la.norm(a)**2])
    print("r",r,np.round(row,3))

print("\nL0 H1/H5 value-read subspace overlap")
for r in [8,16,24,27,32]:
    vs=[]
    for j in [1,5]:
        vs.append(la.svd(V[64*j:64*(j+1)],full_matrices=False)[2][:r].T)
    print(r,np.round(la.svdvals(vs[0].T@vs[1]),3))

print("\nL0 H1/H5 raw QK principal-space geometry")
As=[]
for j in [1,5]:
    q=W["W_q.0"][64*j:64*(j+1)].astype(float)@D
    k=W["W_k.0"][64*j:64*(j+1)].astype(float)@D
    As.append(q.T@k/8)
for r in [16,24,32,40,48]:
    uv=[]
    for a in As:
        u,s,v=la.svd(a,full_matrices=False)
        uv.append((u[:,:r],v[:r].T))
    print("r",r,"query overlap",round(la.norm(uv[0][0].T@uv[1][0])**2/r,3),
          "key overlap",round(la.norm(uv[0][1].T@uv[1][1])**2/r,3),
          "H1 Q/K overlap",round(la.norm(uv[0][0].T@uv[0][1])**2/r,3),
          "H5 Q/K overlap",round(la.norm(uv[1][0].T@uv[1][1])**2/r,3))

```

Output:

```
DETAILED SINGULAR SPECTRA
W_v.0 first 60: [4.553 4.024 3.738 3.497 3.465 3.367 3.279 3.215 3.136 3.113 3.073 3.02  2.953 2.893 2.867 2.846 2.803 2.739 2.702 2.674 2.589 2.575 2.536 2.466 2.444 2.408
 2.246 2.09  2.035 1.967 1.621 1.492 1.403 1.324 1.211 1.188 1.129 1.112 1.078 1.059 1.042 1.019 0.999 0.99  0.967 0.949 0.923 0.91  0.897 0.881 0.872 0.859
 0.829 0.811 0.782 0.778 0.769 0.747 0.73  0.727]
largest successive drops in first 80: [(30, np.float64(1.214)), (1, np.float64(1.131)), (34, np.float64(1.093)), (31, np.float64(1.086)), (2, np.float64(1.077)), (27, np.float64(1.075)), (26, np.float64(1.073)), (3, np.float64(1.069)), (32, np.float64(1.064)), (33, np.float64(1.06)), (36, np.float64(1.053)), (54, np.float64(1.038))]
W_fc.0 first 60: [18.11  12.28   8.473  7.902  7.779  7.442  7.146  7.109  7.008  6.805  6.721  6.54   6.378  6.329  6.27   6.217  6.204  6.035  6.023  5.944  5.845  5.736
  5.702  5.668  5.609  5.58   5.498  5.447  5.377  5.334  5.244  5.203  5.141  5.126  4.99   4.943  4.92   4.846  4.787  4.716  4.669  4.601  4.519  4.439
  4.408  4.384  4.307  4.242  4.208  4.081  4.038  3.984  3.901  3.87   3.791  3.776  3.673  3.574  3.557  3.502]
largest successive drops in first 80: [(1, np.float64(1.475)), (2, np.float64(1.449)), (74, np.float64(1.074)), (3, np.float64(1.072)), (70, np.float64(1.048)), (5, np.float64(1.045)), (6, np.float64(1.041)), (66, np.float64(1.038)), (64, np.float64(1.036)), (62, np.float64(1.032)), (49, np.float64(1.031)), (9, np.float64(1.03))]
W_proj.0 first 60: [10.932  7.576  6.628  6.293  6.166  5.924  5.886  5.769  5.706  5.61   5.495  5.463  5.404  5.313  5.193  5.161  5.119  5.07   5.003  4.937  4.877  4.793
  4.693  4.664  4.651  4.637  4.548  4.526  4.487  4.404  4.329  4.283  4.247  4.206  4.178  4.141  4.12   4.105  4.062  4.023  3.997  3.958  3.934  3.904
  3.857  3.85   3.798  3.758  3.709  3.694  3.669  3.622  3.618  3.591  3.574  3.516  3.476  3.445  3.421  3.387]
largest successive drops in first 80: [(1, np.float64(1.443)), (2, np.float64(1.143)), (3, np.float64(1.053)), (5, np.float64(1.041)), (77, np.float64(1.025)), (14, np.float64(1.023)), (72, np.float64(1.022)), (4, np.float64(1.021)), (10, np.float64(1.021)), (22, np.float64(1.021)), (63, np.float64(1.021)), (7, np.float64(1.02))]
W_proj.3 first 60: [28.175 20.293 19.292 19.135 18.708 18.57  18.139 17.776 17.485 16.54  15.913 15.61  14.975 14.873 14.417 14.313 13.327 12.971 12.593 12.37  12.225 12.019
 11.322 11.115 10.769  9.227  8.573  7.929  6.5    5.271  3.966  3.184  2.792  2.697  2.653  2.555  2.511  2.471  2.379  2.29   2.25   2.215  2.117  2.081
  2.036  2.011  1.984  1.969  1.952  1.926  1.888  1.868  1.853  1.833  1.817  1.796  1.792  1.781  1.763  1.735]
largest successive drops in first 80: [(1, np.float64(1.388)), (30, np.float64(1.329)), (31, np.float64(1.245)), (29, np.float64(1.233)), (28, np.float64(1.22)), (25, np.float64(1.167)), (32, np.float64(1.141)), (27, np.float64(1.081)), (26, np.float64(1.076)), (16, np.float64(1.074)), (22, np.float64(1.062)), (9, np.float64(1.057))]

Effective L0 stacked V first 60: [3.195 2.304 1.316 0.88  0.809 0.771 0.73  0.704 0.685 0.662 0.638 0.629 0.623 0.621 0.604 0.595 0.582 0.567 0.554 0.55  0.542 0.541 0.522 0.513 0.496 0.487
 0.479 0.475 0.464 0.457 0.447 0.431 0.428 0.424 0.41  0.399 0.389 0.383 0.373 0.369 0.358 0.345 0.334 0.333 0.319 0.313 0.308 0.305 0.302 0.3   0.295 0.292
 0.286 0.281 0.278 0.275 0.27  0.265 0.263 0.258]
Individual L0 V first 35:
0 [1.331 1.075 0.446 0.374 0.304 0.294 0.256 0.247 0.237 0.224 0.218 0.205 0.203 0.199 0.188 0.181 0.18  0.174 0.169 0.162 0.162 0.155 0.151 0.147 0.146 0.138
 0.136 0.132 0.13  0.126 0.125 0.122 0.121 0.118 0.117]
1 [0.48  0.455 0.448 0.429 0.421 0.41  0.403 0.385 0.379 0.373 0.37  0.36  0.353 0.341 0.33  0.327 0.322 0.313 0.309 0.303 0.295 0.287 0.281 0.278 0.27  0.255
 0.253 0.241 0.226 0.216 0.209 0.201 0.198 0.171 0.164]
2 [0.5   0.482 0.462 0.399 0.361 0.337 0.296 0.283 0.268 0.261 0.245 0.223 0.217 0.21  0.199 0.19  0.182 0.179 0.177 0.17  0.167 0.158 0.156 0.153 0.15  0.147
 0.145 0.137 0.135 0.129 0.127 0.124 0.12  0.117 0.115]
3 [0.588 0.386 0.342 0.308 0.294 0.277 0.273 0.246 0.242 0.24  0.23  0.216 0.209 0.206 0.195 0.192 0.187 0.178 0.175 0.172 0.168 0.158 0.153 0.151 0.149 0.143
 0.143 0.141 0.138 0.13  0.128 0.125 0.124 0.121 0.118]
4 [2.898 2.029 1.104 0.576 0.48  0.427 0.379 0.362 0.315 0.31  0.3   0.274 0.264 0.262 0.245 0.235 0.218 0.207 0.202 0.194 0.187 0.183 0.177 0.171 0.167 0.164
 0.161 0.158 0.15  0.144 0.143 0.142 0.135 0.133 0.128]
5 [0.493 0.473 0.446 0.437 0.426 0.408 0.404 0.393 0.386 0.376 0.363 0.361 0.356 0.345 0.333 0.329 0.322 0.316 0.31  0.304 0.299 0.289 0.285 0.277 0.274 0.267
 0.252 0.245 0.241 0.235 0.233 0.221 0.216 0.212 0.208]

First-layer QK energy outside top-r value-read subspace
r 16 [[0.118 0.458]
 [0.317 0.555]
 [0.137 0.53 ]
 [0.108 0.466]
 [0.776 0.02 ]
 [0.638 0.247]]
r 24 [[0.104 0.502]
 [0.286 0.568]
 [0.088 0.598]
 [0.085 0.522]
 [0.721 0.031]
 [0.603 0.259]]
r 27 [[0.087 0.546]
 [0.27  0.58 ]
 [0.076 0.647]
 [0.073 0.548]
 [0.698 0.037]
 [0.585 0.269]]
r 32 [[0.081 0.567]
 [0.239 0.597]
 [0.058 0.698]
 [0.059 0.603]
 [0.661 0.048]
 [0.552 0.279]]
r 40 [[0.054 0.651]
 [0.12  0.679]
 [0.04  0.757]
 [0.037 0.687]
 [0.574 0.079]
 [0.389 0.317]]
r 64 [[0.026 0.748]
 [0.021 0.84 ]
 [0.018 0.837]
 [0.016 0.787]
 [0.338 0.197]
 [0.086 0.608]]

L0 H1/H5 value-read subspace overlap
8 [0.895 0.867 0.762 0.69  0.629 0.372 0.23  0.123]
16 [0.937 0.928 0.926 0.897 0.878 0.874 0.855 0.843 0.759 0.73  0.6   0.534 0.379 0.304 0.095 0.08 ]
24 [0.966 0.958 0.957 0.946 0.936 0.932 0.923 0.912 0.909 0.895 0.891 0.874 0.862 0.843 0.839 0.785 0.675 0.639 0.587 0.549 0.425 0.287 0.156 0.005]
27 [0.972 0.964 0.959 0.955 0.943 0.936 0.935 0.933 0.92  0.913 0.901 0.893 0.884 0.88  0.857 0.849 0.826 0.801 0.792 0.748 0.695 0.6   0.45  0.381 0.198 0.094
 0.013]
32 [0.977 0.969 0.961 0.958 0.951 0.946 0.945 0.943 0.939 0.931 0.925 0.917 0
```

## Call 4

```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

P=np.eye(384)-np.ones((384,384))/384
g=W["ln1_w.0"].astype(float)
D=g[:,None]*P
As=[]
Bs=[]
for j in range(6):
    sl=slice(64*j,64*(j+1))
    As.append(D.T@W["W_q.0"][sl].T.astype(float)@W["W_k.0"][sl].astype(float)@D/8)
    Bs.append(P@W["W_o.0"][:,sl].astype(float)@W["W_v.0"][sl].astype(float)@D)
V=np.concatenate([W["W_v.0"][64:128],W["W_v.0"][320:384]]).astype(float)@D
_,sv,vt=la.svd(V,full_matrices=False)
print("H1/H5 joint V spectrum:",np.round(sv[:45],3))

print("\nQK after removing top 27 H1/H5 value-read directions")
T=vt[:27].T
R=P-T@T.T
aa=[R@a@R for a in As]
for j,a in enumerate(aa):
    s=la.svdvals(a)
    print(j,"retained",round(la.norm(a)**2/la.norm(As[j])**2,3),
          "spectrum",np.round(s[:30],3))
G=sum(a@a.T+a.T@a for a in [aa[1],aa[5]])
ev,U=la.eigh(G)
U=U[:,::-1]
print("\nCongruence phase spectra: eigenvalues of pinv(A) A.T")
for r in [16,20,24,28]:
    print("rank",r)
    for j in [1,5]:
        C=U[:,:r].T@aa[j]@U[:,:r]
        e=la.eigvals(la.pinv(C)@C.T)
        es=sorted([(float(abs(z)),float(np.angle(z))) for z in e if z.imag>=-1e-8],key=lambda z:z[1])
        print("H",j, "abs,phase",np.round(es,3))
    C1=U[:,:r].T@aa[1]@U[:,:r]
    C5=U[:,:r].T@aa[5]@U[:,:r]
    e=la.eigvals(la.pinv(C5)@C1)
    es=sorted([(float(abs(z)),float(np.angle(z))) for z in e if z.imag>=-1e-8],key=lambda z:z[1])
    print("H5^-1 H1 abs,phase",np.round(es,3))

print("\nL0 OV top write/read directions and their alignments")
for j in [0,2,3,4]:
    u,s,vtb=la.svd(Bs[j],full_matrices=False)
    print("H",j,"s",np.round(s[:6],3))
    for z in range(3):
        read=vtb[z]; write=u[:,z]
        print(" component",z,
              "read energy on token proxy",round(la.norm(T.T@read)**2,3),
              "write->L0 QK norms",np.round([la.norm(a@write) for a in As],3),
              "write/read cosine",round(float(write@read),3),
              "largest read coordinates",[(int(i),round(read[i],3),round(g[i],3)) for i in np.argsort(abs(read))[-6:][::-1]])

print("\nGain outliers")
for name in ["ln1_w.0","ln1_b.1","ln2_b.0"]:
    x=W[name]
    inds=np.argsort(abs(x))[::-1][:18]
    print(name,[(int(i),round(float(x[i]),4)) for i in inds])

```

Output:

```
H1/H5 joint V spectrum: [0.639 0.622 0.594 0.59  0.571 0.543 0.537 0.529 0.516 0.514 0.494 0.488 0.475 0.471 0.467 0.453 0.448 0.445 0.436 0.422 0.415 0.396 0.381 0.38  0.362 0.356
 0.353 0.334 0.327 0.309 0.299 0.289 0.286 0.276 0.273 0.259 0.256 0.252 0.244 0.237 0.23  0.226 0.223 0.221 0.202]

QK after removing top 27 H1/H5 value-read directions
0 retained 0.616 spectrum [0.075 0.044 0.011 0.007 0.005 0.004 0.004 0.004 0.003 0.003 0.003 0.003 0.003 0.003 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002 0.002
 0.001 0.001 0.001 0.001]
1 retained 0.811 spectrum [0.579 0.356 0.198 0.184 0.151 0.119 0.117 0.078 0.071 0.065 0.059 0.055 0.052 0.051 0.045 0.044 0.04  0.037 0.034 0.033 0.031 0.023 0.022 0.017 0.013 0.012
 0.01  0.01  0.008 0.007]
2 retained 0.683 spectrum [0.116 0.071 0.025 0.018 0.016 0.012 0.012 0.01  0.009 0.009 0.008 0.007 0.007 0.006 0.006 0.006 0.005 0.005 0.004 0.004 0.004 0.004 0.004 0.004 0.003 0.003
 0.003 0.003 0.003 0.003]
3 retained 0.659 spectrum [0.141 0.111 0.061 0.023 0.02  0.016 0.011 0.011 0.009 0.008 0.008 0.007 0.006 0.005 0.005 0.005 0.004 0.004 0.004 0.003 0.003 0.003 0.003 0.003 0.002 0.002
 0.002 0.002 0.002 0.002]
4 retained 0.813 spectrum [0.004 0.003 0.002 0.002 0.002 0.002 0.002 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001
 0.001 0.001 0.001 0.001]
5 retained 0.843 spectrum [0.382 0.339 0.257 0.241 0.224 0.206 0.203 0.192 0.183 0.169 0.167 0.15  0.144 0.134 0.119 0.117 0.113 0.106 0.1   0.095 0.066 0.063 0.046 0.04  0.034 0.029
 0.024 0.019 0.017 0.016]

Congruence phase spectra: eigenvalues of pinv(A) A.T
rank 16
H 1 abs,phase [[1.    0.008]
 [1.632 0.844]
 [0.613 0.844]
 [1.    1.398]
 [1.    1.799]
 [1.245 2.046]
 [0.803 2.046]
 [7.359 3.142]
 [0.136 3.142]]
H 5 abs,phase [[1.    0.115]
 [1.    1.448]
 [1.    1.898]
 [1.    2.153]
 [1.    2.377]
 [0.376 2.6  ]
 [2.659 2.6  ]
 [1.    2.893]]
H5^-1 H1 abs,phase [[1.477 0.   ]
 [1.218 0.   ]
 [0.186 0.395]
 [0.125 0.969]
 [0.165 1.177]
 [0.203 1.542]
 [1.346 2.15 ]
 [0.3   2.195]
 [2.19  2.842]]
rank 20
H 1 abs,phase [[  1.536   0.   ]
 [  0.651   0.   ]
 [  1.      0.148]
 [  1.      1.081]
 [  0.379   1.233]
 [  2.637   1.233]
 [  1.      1.502]
 [  1.      1.859]
 [  1.      2.384]
 [247.745   3.142]
 [  1.346   3.142]
 [  0.743   3.142]
 [  0.004   3.142]]
H 5 abs,phase [[1.    0.01 ]
 [1.    1.   ]
 [1.    1.524]
 [1.288 1.959]
 [0.776 1.959]
 [0.802 2.326]
 [1.247 2.326]
 [1.    2.511]
 [1.    2.772]
 [1.611 3.142]
 [0.621 3.142]]
H5^-1 H1 abs,phase [[1.65  0.   ]
 [1.154 0.   ]
 [0.065 0.   ]
 [0.208 0.47 ]
 [0.217 0.974]
 [0.182 1.408]
 [0.885 1.667]
 [0.114 1.723]
 [1.02  1.878]
 [1.331 2.112]
 [0.134 2.241]
 [0.128 3.142]]
rank 24
H 1 abs,phase [[36.326  0.   ]
 [ 0.028  0.   ]
 [ 1.     0.085]
 [ 1.     0.362]
 [ 5.3    1.06 ]
 [ 0.189  1.06 ]
 [ 1.     1.41 ]
 [ 1.     1.534]
 [ 1.     1.882]
 [ 1.     2.139]
 [ 0.45   2.499]
 [ 2.221  2.499]
 [ 1.     2.909]]
H 5 abs,phase [[11.527  0.   ]
 [ 2.046  0.   ]
 [ 0.489  0.   ]
 [ 0.087  0.   ]
 [ 1.     0.917]
 [ 1.     1.602]
 [ 1.     1.931]
 [ 1.319  2.234]
 [ 0.758  2.234]
 [ 1.     2.407]
 [ 1.081  2.675]
 [ 0.925  2.675]
 [ 7.191  3.142]
 [ 0.139  3.142]
 [ 1.079  3.142]
 [ 0.927  3.142]]
H5^-1 H1 abs,phase [[1.544 0.   ]
 [1.406 0.   ]
 [0.251 0.   ]
 [0.456 0.589]
 [0.182 0.778]
 [0.198 0.916]
 [1.659 1.084]
 [0.103 1.496]
 [0.18  1.52 ]
 [0.179 1.579]
 [1.082 1.981]
 [1.43  2.005]
 [0.231 2.135]
 [1.297 3.142]]
rank 28
H 1 abs,phase [[143.328   0.   ]
 [  0.007   0.   ]
 [  1.      0.014]
 [  1.      0.628]
 [  0.112   0.743]
 [  8.928   0.743]
 [  1.      0.908]
 [  1.      1.217]
 [  1.      1.566]
 [  1.      2.009]
 [  1.      2.294]
 [ 12.152   2.621]
 [  0.082   2.621]
 [  1.912   3.142]
 [  1.241   3.142]
 [  0.806   3.142]
 [  0.523   3.142]]
H 5 abs,phase [[27.337  0.   ]
 [ 3.569  0.   ]
 [ 1.431  0.   ]
 [ 0.699  0.   ]
 [ 0.28   0.   ]
 [ 0.037  0.   ]
 [ 1.     1.232]
 [ 1.     1.524]
 [ 1.373  2.058]
 [ 0.728  2.058]
 [ 1.     2.487]
 [ 0.898  2.585]
 [ 1.113  2.585]
 [ 1.419  2.617]
 [ 0.705  2.617]
 [ 8.569  3.142]
 [ 1.667  3.142]
 [ 0.117  3.142]
 [ 0.6    3.142]]
H5^-1 H1 abs,phase [[3.573 0.   ]
 [1.14  0.   ]
 [0.526 0.   ]
 [0.373 0.   ]
 [0.218 0.181]
 [0.148 1.067]
 [1.908 1.142]
 [0.221 1.166]
 [0.177 1.505]
 [0.253 1.507]
 [0.182 1.879]
 [0.1   1.887]
 [1.452 1.972]
 [0.502 2.34 ]
 [1.092 2.549]
 [0.097 3.095]]

L0 OV top write/read directions and their alignments
H 0 s [3.424 2.469 0.529 0.401 0.346 0.25 ]
 component 0 read energy on token proxy 0.037 write->L0 QK norms [0.024 0.095 0.009 0.043 0.002 0.096] write/read cosine 0.317 largest read coordinates [(307, np.float64(-0.272), np.float64(0.844)), (180, np.float64(-0.268), np.float64(0.948)), (184, np.float64(-0.246), np.float64(0.828)), (125, np.float64(0.242), np.float64(0.809)), (12, np.float64(0.212), np.float64(0.653)), (320, np.float64(0.212), np.float64(0.653))]
 component 1 read energy on token proxy 0.069 write->L0 QK norms [0.01  0.15  0.021 0.018 0.001 0.131] write/read cosine 0.318 largest read coordinates [(273, np.float64(-0.394), np.float64(0.981)), (223, np.float64(-0.32), np.float64(0.973)), (211, np.float64(0.319), np.float64(1.013)), (359, np.float64(0.249), np.float64(0.833)), (162, np.float64(0.237), np.float64(0.913)), (132, np.float64(0.231), np.float64(0.807))]
 component 2 read energy on token proxy 0.327 write->L0 QK norms [0.011 0.081 0.008 0.023 0.001 0.086] write/read cosine -0.059 largest read coordinates [(240, np.float64(0.241), np.float64(0.818)), (169, np.float64(-0.201), np.float64(0.806)), (273, np.float64(-0.192), np.float64(0.981)), (180, np.float64(0.192), np.float64(0.948)), (224, np.float64(-0.185), np.float64(0.69)), (47, np.float64(0.183), np.float64(0.586))]
H 2 s [1.055 0.901 0.807 0.66  0.537 0.47 ]
 component 0 read energy on token proxy 0.695 write->L0 QK norms [0.006 0.057 0.012 0.011 0.    0.049] write/read cosine -
```

## Call 5

```python
import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from threadpoolctl import threadpool_limits
threadpool_limits(4)

print("RAW L0 H1/H5 value spectrum")
V=np.concatenate([W["W_v.0"][64:128],W["W_v.0"][320:384]]).astype(float)
u,s,vt=la.svd(V,full_matrices=False)
print(np.round(s[:65],3))

print("\nLAST MLP OUTPUT-COLUMN GEOMETRY (neurons, not network inputs)")
O=W["W_proj.3"].astype(float)
u,s,vt=la.svd(O,full_matrices=False)
for r in [24,27,30]:
    X=(u[:,:r].T@O).T
    X=X/(la.norm(X,axis=1,keepdims=True)+1e-12)
    km=KMeans(n_clusters=28,n_init=15,random_state=0).fit(X)
    C=km.cluster_centers_
    nc=la.norm(C,axis=1)
    counts=np.bincount(km.labels_,minlength=28)
    print("rank",r,"cluster counts",sorted(counts.tolist(),reverse=True),
          "center norms",np.round(np.sort(nc)[::-1],3),
          "inertia",round(km.inertia_,2))
    if r==27:
        cn=C/(nc[:,None]+1e-12)
        sims=cn@cn.T
        np.fill_diagonal(sims,-2)
        print("top cluster-pair cosines",
              sorted([(round(sims[i,j],3),int(counts[i]),int(counts[j]))
                      for i in range(28) for j in range(i)],reverse=True)[:24])

print("\nICA of last-MLP output columns")
for r in [24,27,30]:
    X=(u[:,:r].T@O).T
    ica=FastICA(n_components=r,whiten="unit-variance",max_iter=1500,tol=0.0005,random_state=0)
    Z=ica.fit_transform(X)
    ku=kurtosis(Z,axis=0)
    print("rank",r,"kurtoses",np.round(np.sort(ku)[::-1],2))
    inds=np.argsort(ku)[::-1]
    print("tail counts |z|>2.5",
          [(int(np.sum(Z[:,i]>2.5)),int(np.sum(Z[:,i]<-2.5))) for i in inds])

print("\nL0 MLP input/output norms, and output bias-like direction")
for l in range(4):
    fc=W[f"W_fc.{l}"].astype(float)
    pr=W[f"W_proj.{l}"].astype(float)
    fn=la.norm(fc,axis=1); pn=la.norm(pr,axis=0)
    print(l,"fc row norm quantiles",np.round(np.quantile(fn,[0,.01,.1,.5,.9,.99,1]),3),
          "proj col norm quantiles",np.round(np.quantile(pn,[0,.01,.1,.5,.9,.99,1]),3),
          "mean-col ratio",round(la.norm(pr.mean(axis=1))/np.mean(pn),3))

```

Output:

```
RAW L0 H1/H5 value spectrum
[3.023 2.923 2.898 2.843 2.791 2.735 2.708 2.661 2.614 2.601 2.519 2.44  2.421 2.379 2.364 2.311 2.266 2.214 2.194 2.105 2.019 1.897 1.838 1.726 1.497 1.42
 1.348 1.111 1.    0.935 0.918 0.89  0.865 0.854 0.821 0.81  0.793 0.779 0.76  0.725 0.721 0.702 0.693 0.676 0.64  0.619 0.601 0.546 0.54  0.534 0.524 0.514
 0.504 0.494 0.483 0.476 0.469 0.465 0.464 0.444 0.438 0.434 0.425 0.422 0.413]

LAST MLP OUTPUT-COLUMN GEOMETRY (neurons, not network inputs)
rank 24 cluster counts [109, 69, 69, 67, 66, 62, 60, 58, 58, 57, 56, 56, 55, 53, 52, 52, 52, 51, 50, 50, 49, 46, 43, 41, 41, 41, 38, 35] center norms [0.701 0.665 0.594 0.566 0.565 0.562 0.56  0.556 0.555 0.552 0.551 0.546 0.541 0.54  0.54  0.537 0.531 0.522 0.52  0.515 0.513 0.512 0.509 0.506 0.505 0.504
 0.501 0.498] inertia 1061.01
rank 27 cluster counts [120, 75, 69, 67, 66, 65, 65, 62, 61, 61, 57, 56, 55, 53, 53, 53, 52, 51, 49, 45, 45, 41, 40, 39, 39, 36, 34, 27] center norms [0.674 0.67  0.649 0.568 0.556 0.548 0.548 0.539 0.537 0.534 0.533 0.533 0.533 0.528 0.527 0.524 0.522 0.521 0.518 0.513 0.511 0.511 0.501 0.498 0.498 0.494
 0.49  0.49 ] inertia 1079.54
top cluster-pair cosines [(np.float64(0.559), 51, 53), (np.float64(0.528), 120, 53), (np.float64(0.524), 120, 61), (np.float64(0.519), 65, 120), (np.float64(0.499), 75, 62), (np.float64(0.488), 120, 53), (np.float64(0.474), 69, 120), (np.float64(0.457), 66, 75), (np.float64(0.447), 39, 51), (np.float64(0.446), 53, 61), (np.float64(0.445), 49, 41), (np.float64(0.438), 120, 39), (np.float64(0.388), 40, 67), (np.float64(0.381), 69, 61), (np.float64(0.374), 120, 52), (np.float64(0.373), 65, 39), (np.float64(0.359), 53, 52), (np.float64(0.342), 34, 56), (np.float64(0.34), 56, 75), (np.float64(0.34), 51, 40), (np.float64(0.334), 27, 120), (np.float64(0.328), 61, 39), (np.float64(0.32), 45, 53), (np.float64(0.316), 39, 53)]
rank 30 cluster counts [121, 74, 67, 65, 61, 60, 59, 57, 56, 56, 55, 55, 53, 53, 52, 51, 51, 51, 51, 49, 47, 47, 46, 46, 45, 45, 39, 24] center norms [0.691 0.665 0.663 0.57  0.549 0.547 0.543 0.54  0.533 0.53  0.53  0.528 0.526 0.526 0.525 0.523 0.521 0.518 0.515 0.51  0.509 0.506 0.491 0.487 0.486 0.485
 0.485 0.476] inertia 1085.98

ICA of last-MLP output columns
rank 24 kurtoses [24.25  1.56  1.43  1.37  1.28  1.19  1.03  1.02  1.    1.    0.94  0.86  0.84  0.8   0.8   0.76  0.74  0.43 -0.22 -0.5  -0.54 -0.57 -0.59 -0.63]
tail counts |z|>2.5 [(10, 25), (10, 26), (9, 26), (25, 7), (26, 9), (23, 9), (18, 20), (21, 10), (28, 14), (18, 17), (17, 16), (23, 10), (21, 15), (13, 23), (23, 15), (19, 15), (22, 14), (8, 18), (1, 10), (2, 7), (4, 3), (6, 2), (2, 2), (2, 4)]
rank 27 kurtoses [24.53  1.71  1.32  1.31  1.23  1.15  1.14  1.11  1.08  1.08  1.07  1.05  1.04  1.02  0.99  0.79  0.78  0.77  0.72  0.33  0.24  0.03 -0.13 -0.14 -0.54 -0.55
 -0.62]
tail counts |z|>2.5 [(24, 9), (25, 7), (14, 13), (18, 18), (12, 25), (12, 16), (17, 13), (17, 20), (21, 13), (23, 17), (15, 15), (23, 18), (17, 21), (19, 15), (15, 17), (19, 11), (14, 18), (17, 13), (17, 20), (10, 13), (11, 16), (7, 14), (10, 4), (6, 7), (5, 4), (5, 1), (2, 2)]
rank 30 kurtoses [28.38  6.25  3.85  2.33  1.43  1.38  1.29  1.27  1.26  1.23  1.14  1.13  1.13  1.11  1.11  1.09  1.05  1.02  1.01  0.99  0.94  0.86  0.85  0.76  0.76  0.75
  0.74  0.72 -0.58 -0.58]
tail counts |z|>2.5 [(9, 23), (9, 43), (21, 11), (17, 21), (15, 17), (17, 15), (25, 12), (17, 17), (15, 20), (16, 21), (25, 11), (13, 20), (14, 23), (24, 16), (12, 23), (32, 7), (17, 19), (15, 21), (12, 24), (17, 22), (15, 23), (17, 18), (16, 14), (14, 16), (23, 13), (21, 17), (25, 15), (25, 11), (3, 3), (2, 3)]

L0 MLP input/output norms, and output bias-like direction
0 fc row norm quantiles [0.33  0.347 0.371 0.437 2.393 3.213 4.287] proj col norm quantiles [0.127 0.146 0.174 0.263 2.098 2.672 3.366] mean-col ratio 0.165
1 fc row norm quantiles [0.441 1.7   1.869 2.023 2.206 2.45  8.68 ] proj col norm quantiles [0.325 1.641 1.776 1.908 2.056 2.257 2.909] mean-col ratio 0.043
2 fc row norm quantiles [1.678 1.856 1.933 2.024 2.121 2.209 2.731] proj col norm quantiles [1.597 1.723 1.837 1.965 2.12  2.25  2.85 ] mean-col ratio 0.038
3 fc row norm quantiles [1.746 1.806 1.902 1.995 2.072 2.176 2.947] proj col norm quantiles [1.305 1.505 1.799 2.132 2.511 2.932 4.621] mean-col ratio 0.053

[stderr]
/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/sklearn/decomposition/_fastica.py:132: ConvergenceWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
  warnings.warn(
/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/sklearn/decomposition/_fastica.py:132: ConvergenceWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
  warnings.warn(
```

## Call 6

```python
import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
threadpool_limits(4)

d=384
P=np.eye(d)-np.ones((d,d))/d
g=W["ln1_w.0"].astype(float)
D=g[:,None]*P
g2=W["ln2_w.0"].astype(float)
F=W["W_fc.0"].astype(float)@(g2[:,None]*P)
norms=la.norm(W["W_fc.0"],axis=1)
sel=norms>1.0
print("First-MLP row norm counts:",[(t,int(np.sum(norms>t))) for t in [.5,.75,1,1.5,2,2.5,3]])
V=np.concatenate([W["W_v.0"][64:128],W["W_v.0"][320:384]]).astype(float)
_,s,vt=la.svd(V,full_matrices=False)
bases={
    "raw_V":P@vt[:27].T,
    "inverse_gain_V":P@(vt[:27].T/g[:,None]),
    "effective_V":la.svd(V@D,full_matrices=False)[2][:27].T,
}
B={}
for j in [1,5]:
    sl=slice(64*j,64*(j+1))
    B[j]=P@W["W_o.0"][:,sl].astype(float)@W["W_v.0"][sl].astype(float)@D

for name,E in bases.items():
    E=la.orth(E)
    channels=[F[sel]@E,F[sel]@B[1]@E,F[sel]@B[5]@E]
    print("\nBASIS",name)
    for whiten in [False,True]:
        XX=np.concatenate([x-x.mean(axis=0) for x in channels])
        if whiten:
            ev,U=la.eigh(XX.T@XX/len(XX))
            T=U@np.diag(1/np.sqrt(np.maximum(ev,ev.max()*.03)))@U.T
        else:
            T=np.eye(27)
        feats=[]
        for x in channels:
            x=(x-x.mean(axis=0))@T
            x=x/(la.norm(x,axis=1,keepdims=True)+1e-12)
            feats.append(x)
        X=np.concatenate(feats)
        km=KMeans(n_clusters=28,n_init=10,random_state=0).fit(X)
        C=km.cluster_centers_
        cn=la.norm(C,axis=1)
        counts=np.bincount(km.labels_,minlength=28)
        print("whiten",whiten,"mean/median/max center norm",np.round([cn.mean(),np.median(cn),cn.max()],3),
              "counts",sorted(counts.tolist(),reverse=True))
        sim=X@(C/(cn[:,None]+1e-12)).T
        maxsim=sim.max(axis=1).reshape(3,-1)
        print("max-cos quantiles by channel",np.round(np.quantile(maxsim,[.1,.5,.9],axis=1).T,3))
        if name=="inverse_gain_V" and whiten:
            labels=km.labels_.reshape(3,-1)
            print("channel cluster counts")
            print(np.array([np.bincount(z,minlength=28) for z in labels]))
            for k in [1,2]:
                M=np.zeros((28,28),int)
                for a,b,ca,cb in zip(labels[0],labels[k],maxsim[0],maxsim[k]):
                    if min(ca,cb)>.6:
                        M[a,b]+=1
                edges=sorted([(int(M[a,b]),int(a),int(b)) for a in range(28) for b in range(28)],reverse=True)
                print("current vs channel",k,"strong-pair count",M.sum(),"top pairs",edges[:35])

```

Output:

```
First-MLP row norm counts: [(0.5, 615), (0.75, 534), (1, 518), (1.5, 446), (2, 286), (2.5, 117), (3, 33)]

BASIS raw_V
whiten False mean/median/max center norm [0.617 0.627 0.756] counts [102, 95, 84, 74, 73, 71, 66, 56, 55, 55, 54, 54, 54, 53, 52, 52, 49, 48, 48, 48, 47, 44, 44, 42, 40, 38, 31, 25]
max-cos quantiles by channel [[0.442 0.658 0.932]
 [0.408 0.568 0.778]
 [0.4   0.577 0.797]]
whiten True mean/median/max center norm [0.602 0.604 0.719] counts [76, 69, 66, 63, 63, 63, 62, 62, 61, 60, 59, 58, 57, 57, 57, 55, 54, 54, 52, 52, 51, 51, 50, 50, 44, 40, 40, 28]
max-cos quantiles by channel [[0.421 0.637 0.928]
 [0.408 0.552 0.785]
 [0.377 0.541 0.758]]

BASIS inverse_gain_V
whiten False mean/median/max center norm [0.62  0.619 0.711] counts [99, 89, 81, 76, 72, 69, 69, 67, 66, 64, 62, 61, 56, 55, 54, 48, 46, 46, 44, 43, 43, 42, 42, 39, 36, 34, 28, 23]
max-cos quantiles by channel [[0.447 0.633 0.903]
 [0.421 0.582 0.798]
 [0.437 0.601 0.789]]
whiten True mean/median/max center norm [0.605 0.61  0.722] counts [96, 80, 73, 70, 67, 66, 61, 60, 60, 59, 59, 58, 58, 58, 57, 55, 55, 54, 49, 47, 46, 43, 43, 42, 42, 37, 33, 26]
max-cos quantiles by channel [[0.389 0.601 0.904]
 [0.419 0.571 0.8  ]
 [0.411 0.576 0.771]]
channel cluster counts
[[27 24 24 20 20 42 18 24  5 15 21 16  8 24 10 18 12 16 21 25 23 16  4 25 23 17  5 15]
 [19 25  9 24 11 13 19 16  8 28 19  6 11 17 31 10 30 76 13 13 25 14 11 16 20 11 12 11]
 [20  9 26 16 11 18  6 27 46 12 17  4 24  6 14  9  7  4 26 20 22 12 46 17 11 18 63  7]]
current vs channel 1 strong-pair count 118 top pairs [(6, 21, 21), (6, 7, 7), (5, 24, 24), (5, 13, 16), (5, 4, 4), (4, 27, 27), (4, 15, 15), (4, 11, 11), (3, 20, 20), (3, 18, 18), (3, 6, 17), (3, 6, 6), (3, 5, 16), (3, 3, 3), (2, 25, 25), (2, 23, 23), (2, 19, 19), (2, 17, 17), (2, 11, 14), (2, 10, 24), (2, 10, 10), (2, 9, 9), (2, 7, 17), (2, 5, 13), (2, 5, 5), (2, 3, 17), (2, 2, 6), (2, 0, 17), (1, 27, 15), (1, 27, 6), (1, 26, 26), (1, 25, 14), (1, 25, 3), (1, 24, 17), (1, 23, 17)]
current vs channel 2 strong-pair count 109 top pairs [(6, 5, 22), (4, 7, 22), (4, 4, 4), (3, 25, 26), (3, 24, 24), (3, 21, 21), (3, 13, 26), (3, 9, 22), (3, 7, 8), (3, 7, 7), (3, 6, 22), (3, 3, 26), (3, 2, 12), (3, 0, 26), (2, 21, 8), (2, 20, 20), (2, 19, 7), (2, 15, 15), (2, 5, 19), (2, 5, 8), (2, 5, 5), (2, 2, 7), (1, 27, 27), (1, 27, 8), (1, 26, 8), (1, 25, 7), (1, 24, 26), (1, 24, 8), (1, 24, 2), (1, 23, 25), (1, 23, 23), (1, 23, 15), (1, 21, 22), (1, 21, 12), (1, 20, 8)]

BASIS effective_V
whiten False mean/median/max center norm [0.612 0.615 0.727] counts [91, 73, 72, 71, 70, 66, 65, 62, 62, 62, 60, 59, 56, 56, 56, 55, 55, 52, 51, 48, 47, 45, 39, 39, 38, 37, 37, 30]
max-cos quantiles by channel [[0.455 0.649 0.922]
 [0.406 0.566 0.785]
 [0.407 0.572 0.772]]
whiten True mean/median/max center norm [0.588 0.584 0.727] counts [84, 81, 70, 69, 66, 63, 62, 59, 59, 58, 57, 57, 56, 54, 54, 54, 52, 52, 51, 50, 47, 47, 45, 45, 44, 44, 43, 31]
max-cos quantiles by channel [[0.392 0.629 0.894]
 [0.389 0.543 0.776]
 [0.373 0.524 0.742]]

```

## Call 7

```python
import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
threadpool_limits(4)

P=np.eye(384)-np.ones((384,384))/384
D1=W["ln1_w.0"].astype(float)[:,None]*P
D2=W["ln2_w.0"].astype(float)[:,None]*P
F=W["W_fc.0"].astype(float)@D2
V=np.concatenate([W["W_v.0"][64:128],W["W_v.0"][320:384]]).astype(float)
_,_,vt=la.svd(V,full_matrices=False)
E=la.orth(P@(vt[:27].T/W["ln1_w.0"][:,None]))
sel=la.norm(W["W_fc.0"],axis=1)>1
inds=np.flatnonzero(sel)
X=F[sel]@E
X=X-X.mean(axis=0)
Xn=X/(la.norm(X,axis=1,keepdims=True)+1e-12)
km=KMeans(n_clusters=28,n_init=40,random_state=0).fit(Xn)
C=km.cluster_centers_
cn=la.norm(C,axis=1)
C=C/cn[:,None]
cos=Xn@C.T
strength=cos[np.arange(len(Xn)),km.labels_]
print("CURRENT-READ CLUSTERS")
for a in range(28):
    ii=np.flatnonzero(km.labels_==a)
    print(a,"count",len(ii),"center norm",round(cn[a],3),
          "median cos",round(np.median(strength[ii]),3),
          "median fc/proj norms",np.round([np.median(la.norm(W["W_fc.0"][inds[ii]],axis=1)),
                                          np.median(la.norm(W["W_proj.0"][:,inds[ii]],axis=0))],3))
G=C@C.T
np.fill_diagonal(G,-2)
print("centroid top pair cosines",
      sorted([(round(G[a,b],3),a,b) for a in range(28) for b in range(a)],reverse=True)[:35])
print("centroid Gram eigenvalues",np.round(la.eigvalsh(C@C.T),3))

print("\nREAD-WEIGHT ASSOCIATIONS BETWEEN CURRENT-READ GROUPS AND EARLIER-HEAD CHANNELS")
for j in [1,5]:
    sl=slice(64*j,64*(j+1))
    B=P@W["W_o.0"][:,sl].astype(float)@W["W_v.0"][sl].astype(float)@D1
    Z=F[sel]@B@E
    Z=Z-Z.mean(axis=0)
    Zn=Z/(la.norm(Z,axis=1,keepdims=True)+1e-12)
    scores=Zn@C.T
    groups=np.array([scores[km.labels_==a].mean(axis=0) for a in range(28)])
    print("HEAD",j)
    for a in range(28):
        best=np.argsort(groups[a])[::-1][:4]
        worst=np.argsort(groups[a])[:2]
        print(a,"positive",[(int(b),round(groups[a,b],2)) for b in best],
              "negative",[(int(b),round(groups[a,b],2)) for b in worst])
    vals=la.svdvals(groups-groups.mean(axis=0))
    print("association singular values",np.round(vals,3))

```

Output:

```
CURRENT-READ CLUSTERS
0 count 19 center norm 0.695 median cos 0.711 median fc/proj norms [1.987 1.837]
1 count 24 center norm 0.697 median cos 0.672 median fc/proj norms [2.061 1.891]
2 count 12 center norm 0.751 median cos 0.78 median fc/proj norms [2.177 1.999]
3 count 14 center norm 0.603 median cos 0.611 median fc/proj norms [2.007 2.003]
4 count 13 center norm 0.81 median cos 0.882 median fc/proj norms [2.129 1.904]
5 count 15 center norm 0.74 median cos 0.8 median fc/proj norms [2.033 1.872]
6 count 44 center norm 0.721 median cos 0.717 median fc/proj norms [2.085 1.873]
7 count 17 center norm 0.757 median cos 0.757 median fc/proj norms [2.146 1.951]
8 count 44 center norm 0.639 median cos 0.649 median fc/proj norms [2.338 1.966]
9 count 16 center norm 0.718 median cos 0.73 median fc/proj norms [1.948 1.93 ]
10 count 34 center norm 0.595 median cos 0.585 median fc/proj norms [2.141 2.046]
11 count 14 center norm 0.726 median cos 0.708 median fc/proj norms [2.217 1.941]
12 count 27 center norm 0.817 median cos 0.822 median fc/proj norms [2.03  1.864]
13 count 12 center norm 0.823 median cos 0.79 median fc/proj norms [1.957 1.869]
14 count 13 center norm 0.78 median cos 0.845 median fc/proj norms [1.72  1.765]
15 count 22 center norm 0.66 median cos 0.661 median fc/proj norms [1.536 1.641]
16 count 16 center norm 0.751 median cos 0.764 median fc/proj norms [2.312 2.062]
17 count 11 center norm 0.792 median cos 0.736 median fc/proj norms [1.838 1.801]
18 count 17 center norm 0.703 median cos 0.721 median fc/proj norms [1.829 1.808]
19 count 22 center norm 0.522 median cos 0.538 median fc/proj norms [2.498 2.079]
20 count 19 center norm 0.721 median cos 0.794 median fc/proj norms [1.881 1.825]
21 count 10 center norm 0.767 median cos 0.798 median fc/proj norms [2.144 1.967]
22 count 10 center norm 0.827 median cos 0.8 median fc/proj norms [2.639 2.1  ]
23 count 11 center norm 0.813 median cos 0.778 median fc/proj norms [2.366 1.994]
24 count 18 center norm 0.696 median cos 0.716 median fc/proj norms [2.309 1.976]
25 count 14 center norm 0.741 median cos 0.677 median fc/proj norms [2.043 1.826]
26 count 16 center norm 0.791 median cos 0.802 median fc/proj norms [1.918 1.844]
27 count 14 center norm 0.732 median cos 0.7 median fc/proj norms [2.195 2.035]
centroid top pair cosines [(np.float64(0.53), 4, 3), (np.float64(0.415), 5, 0), (np.float64(0.361), 8, 0), (np.float64(0.33), 7, 3), (np.float64(0.313), 18, 8), (np.float64(0.248), 14, 3), (np.float64(0.234), 22, 6), (np.float64(0.23), 15, 14), (np.float64(0.227), 15, 10), (np.float64(0.219), 21, 8), (np.float64(0.213), 21, 18), (np.float64(0.212), 22, 16), (np.float64(0.21), 26, 7), (np.float64(0.207), 15, 3), (np.float64(0.205), 11, 4), (np.float64(0.203), 11, 7), (np.float64(0.198), 6, 2), (np.float64(0.198), 3, 0), (np.float64(0.193), 16, 8), (np.float64(0.178), 8, 2), (np.float64(0.176), 1, 0), (np.float64(0.174), 24, 19), (np.float64(0.17), 21, 0), (np.float64(0.17), 20, 0), (np.float64(0.159), 23, 15), (np.float64(0.156), 26, 15), (np.float64(0.156), 24, 4), (np.float64(0.155), 8, 6), (np.float64(0.152), 21, 11), (np.float64(0.15), 10, 5), (np.float64(0.148), 19, 7), (np.float64(0.147), 8, 5), (np.float64(0.143), 14, 5), (np.float64(0.142), 19, 3), (np.float64(0.141), 14, 10)]
centroid Gram eigenvalues [0.    0.002 0.006 0.225 0.276 0.358 0.418 0.512 0.569 0.614 0.696 0.743 0.783 0.845 0.893 1.001 1.084 1.155 1.213 1.3   1.359 1.444 1.563 1.614 1.85  2.024
 2.416 3.038]

READ-WEIGHT ASSOCIATIONS BETWEEN CURRENT-READ GROUPS AND EARLIER-HEAD CHANNELS
HEAD 1
0 positive [(0, np.float64(0.27)), (3, np.float64(0.17)), (12, np.float64(0.15)), (20, np.float64(0.1))] negative [(13, np.float64(-0.15)), (15, np.float64(-0.09))]
1 positive [(1, np.float64(0.22)), (19, np.float64(0.06)), (24, np.float64(0.06)), (23, np.float64(0.05))] negative [(5, np.float64(-0.14)), (14, np.float64(-0.14))]
2 positive [(2, np.float64(0.22)), (4, np.float64(0.14)), (8, np.float64(0.09)), (9, np.float64(0.07))] negative [(20, np.float64(-0.16)), (14, np.float64(-0.12))]
3 positive [(21, np.float64(0.12)), (1, np.float64(0.08)), (8, np.float64(0.08)), (24, np.float64(0.06))] negative [(14, np.float64(-0.18)), (5, np.float64(-0.1))]
4 positive [(4, np.float64(0.26)), (3, np.float64(0.22)), (10, np.float64(0.15)), (5, np.float64(0.12))] negative [(18, np.float64(-0.18)), (13, np.float64(-0.14))]
5 positive [(12, np.float64(0.14)), (5, np.float64(0.12)), (14, np.float64(0.11)), (11, np.float64(0.09))] negative [(9, np.float64(-0.16)), (15, np.float64(-0.14))]
6 positive [(15, np.float64(0.05)), (13, np.float64(0.05)), (19, np.float64(0.04)), (20, np.float64(0.04))] negative [(12, np.float64(-0.09)), (18, np.float64(-0.04))]
7 positive [(12, np.float64(0.14)), (16, np.float64(0.11)), (7, np.float64(0.08)), (5, np.float64(0.05))] negative [(15, np.float64(-0.16)), (9, np.float64(-0.14))]
8 positive [(17, np.float64(0.08)), (6, np.float64(0.06)), (4, np.float64(0.05)), (26, np.float64(0.04))] negative [(21, np.float64(-0.1)), (16, np.float64(-0.08))]
9 positive [(9, np.float64(0.18)), (15, np.float64(0.11)), (21, np.float64(0.07)), (0, np.float64(0.07))] negative [(11, np.float64(-0.08)), (20, np.float64(-0.08))]
10 positive [(10, np.float64(0.09)), (18, np.float64(0.09)), (12, np.float64(0.04)), (5, np.float64(0.04))] negative [(8, np.float64(-0.08)), (24, np.float64(-0.07))]
11 positive [(11, np.float64(0.33)), (5, np.float64(0.12)), (4, np.float64(0.07)), (10, np.float64(0.07))] negative [(27, np.float64(-0.12)), (1, np.float64(-0.11))]
12 positive [(19, np.float64(0.09)), (3, np.float64(0.08)), (7, np.float64(0.08)), (5, np.float64(0.07))] negative [(12, np.float64(-0.17)), (1, np.float64(-0.14))]
13 positive [(21, np.float64(0.16)), (8, np.float64(0.15)), (27, np.float64(0.11)), (16, np.float64(0.1))] negative [(15, np.float64(-0.14)), (20, np.float64(-0.11))]
14 positive [(12, np.float64(0.22)), (6, np.float64(0.09)), (2, np.float64(0.07)), (23, np.float64(0.07
```

## Call 8

```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

P=np.eye(384)-np.ones((384,384))/384
D=W["ln1_w.0"].astype(float)[:,None]*P
A=[]
for j in range(6):
    sl=slice(64*j,64*(j+1))
    A.append(D.T@W["W_q.0"][sl].T.astype(float)@W["W_k.0"][sl].astype(float)@D/8)
V=np.concatenate([W["W_v.0"][64:128],W["W_v.0"][320:384]]).astype(float)
_,_,vt=la.svd(V,full_matrices=False)
token_read=la.orth(D.T@vt[:27].T)
for remove_token in [False,True]:
    R=P-token_read@token_read.T if remove_token else P
    a1,a5=R@A[1]@R,R@A[5]@R
    H=a5@a5.T+a5.T@a5
    ev,U=la.eigh(H)
    ev,U=ev[::-1],U[:,::-1]
    print("\nROTATION-POWER TEST; remove token-read subspace:",remove_token)
    for r in [8,12,16,20,24,28,32]:
        T=U[:,:r]@np.diag(ev[:r]**(-.25))
        c1=T.T@a1@T
        c5=T.T@a5@T
        q,_=la.polar(c5)
        vals=[]
        for power in range(-4,5):
            qp=np.linalg.matrix_power(q,power)
            vals.append(np.sum(c1*qp)/(la.norm(c1)*la.norm(qp)))
        residual=la.norm(c5@c5.T-c5.T@c5)/(la.norm(c5)**2)
        print("r",r,"cos(H1,polar(H5)^p), p=-4..4",np.round(vals,3),
              "H5 nonnormality",round(residual,3))

print("\nATTENTION SUBSPACE AND COMPOSITION SIGNATURES")
As={};Bs={}
for l in range(4):
    D=W[f"ln1_w.{l}"].astype(float)[:,None]*P
    for j in range(6):
        sl=slice(64*j,64*(j+1))
        As[l,j]=D.T@W[f"W_q.{l}"][sl].T.astype(float)@W[f"W_k.{l}"][sl].astype(float)@D/8
        Bs[l,j]=P@W[f"W_o.{l}"][:,sl].astype(float)@W[f"W_v.{l}"][sl].astype(float)@D
u,s,vt=la.svd(Bs[0,4],full_matrices=False)
position_write=u[:,:3]
for l in range(1,4):
    for j in range(6):
        a=As[l,j]
        qfrac=la.norm(position_write.T@a)**2/la.norm(a)**2
        kfrac=la.norm(a@position_write)**2/la.norm(a)**2
        comps=[]
        for m in range(l):
            for h in range(6):
                c=a@Bs[m,h]
                nuclear=la.svdvals(c).sum()
                score=np.trace(c)/(nuclear+1e-12)
                sym=(c+c.T)/2
                eig=la.eigvalsh(sym)
                positive=eig[eig>0].sum()/(abs(eig).sum()+1e-12)
                comps.append((score,m,h,positive))
        best=max(comps)
        print(f"L{l}H{j}",
              "H0H4-write Q/K energy",np.round([qfrac,kfrac],3),
              "best K-comp",f"L{best[1]}H{best[2]}",
              "trace/nuclear",round(best[0],3),
              "symmetric positive-eigen mass",round(best[3],3))

```

Output:

```

ROTATION-POWER TEST; remove token-read subspace: False
r 8 cos(H1,polar(H5)^p), p=-4..4 [ 0.178 -0.198 -0.417 -0.266  0.162  0.517  0.485  0.074 -0.366] H5 nonnormality 0.104
r 12 cos(H1,polar(H5)^p), p=-4..4 [ 0.011 -0.24  -0.227 -0.141  0.019  0.333  0.496  0.161 -0.366] H5 nonnormality 0.126
r 16 cos(H1,polar(H5)^p), p=-4..4 [ 0.092 -0.31  -0.158 -0.01  -0.134  0.211  0.56   0.068 -0.382] H5 nonnormality 0.136
r 20 cos(H1,polar(H5)^p), p=-4..4 [ 0.234 -0.435 -0.124  0.232 -0.225  0.071  0.642 -0.021 -0.426] H5 nonnormality 0.11
r 24 cos(H1,polar(H5)^p), p=-4..4 [ 0.15  -0.414 -0.107  0.262 -0.23   0.076  0.532  0.065 -0.383] H5 nonnormality 0.153
r 28 cos(H1,polar(H5)^p), p=-4..4 [ 0.025 -0.046 -0.169  0.306 -0.265  0.189  0.19   0.179 -0.311] H5 nonnormality 0.159
r 32 cos(H1,polar(H5)^p), p=-4..4 [-0.037 -0.013 -0.141  0.233 -0.232  0.182  0.181  0.095 -0.241] H5 nonnormality 0.17

ROTATION-POWER TEST; remove token-read subspace: True
r 8 cos(H1,polar(H5)^p), p=-4..4 [ 0.152 -0.215 -0.414 -0.251  0.174  0.522  0.493  0.093 -0.35 ] H5 nonnormality 0.116
r 12 cos(H1,polar(H5)^p), p=-4..4 [ 0.006 -0.206 -0.241 -0.17   0.044  0.373  0.502  0.177 -0.322] H5 nonnormality 0.099
r 16 cos(H1,polar(H5)^p), p=-4..4 [ 0.089 -0.339 -0.174  0.004 -0.125  0.224  0.577  0.067 -0.376] H5 nonnormality 0.129
r 20 cos(H1,polar(H5)^p), p=-4..4 [ 0.217 -0.446 -0.121  0.234 -0.223  0.066  0.649 -0.003 -0.442] H5 nonnormality 0.101
r 24 cos(H1,polar(H5)^p), p=-4..4 [ 0.068 -0.359 -0.109  0.304 -0.226  0.071  0.547  0.074 -0.343] H5 nonnormality 0.149
r 28 cos(H1,polar(H5)^p), p=-4..4 [ 0.005 -0.046 -0.179  0.313 -0.266  0.186  0.201  0.181 -0.326] H5 nonnormality 0.158
r 32 cos(H1,polar(H5)^p), p=-4..4 [-0.022 -0.037 -0.109  0.275 -0.237  0.18   0.199  0.146 -0.237] H5 nonnormality 0.169

ATTENTION SUBSPACE AND COMPOSITION SIGNATURES
L1H0 H0H4-write Q/K energy [0.061 0.342] best K-comp L0H4 trace/nuclear 0.039 symmetric positive-eigen mass 0.519
L1H1 H0H4-write Q/K energy [0.107 0.627] best K-comp L0H5 trace/nuclear 0.08 symmetric positive-eigen mass 0.542
L1H2 H0H4-write Q/K energy [0.108 0.631] best K-comp L0H4 trace/nuclear 0.046 symmetric positive-eigen mass 0.523
L1H3 H0H4-write Q/K energy [0.137 0.678] best K-comp L0H5 trace/nuclear 0.068 symmetric positive-eigen mass 0.535
L1H4 H0H4-write Q/K energy [0.119 0.645] best K-comp L0H4 trace/nuclear 0.062 symmetric positive-eigen mass 0.531
L1H5 H0H4-write Q/K energy [0.084 0.525] best K-comp L0H5 trace/nuclear 0.062 symmetric positive-eigen mass 0.531
L2H0 H0H4-write Q/K energy [0.01  0.051] best K-comp L0H4 trace/nuclear 0.044 symmetric positive-eigen mass 0.522
L2H1 H0H4-write Q/K energy [0.005 0.005] best K-comp L0H5 trace/nuclear 0.212 symmetric positive-eigen mass 0.608
L2H2 H0H4-write Q/K energy [0.014 0.07 ] best K-comp L0H4 trace/nuclear 0.075 symmetric positive-eigen mass 0.538
L2H3 H0H4-write Q/K energy [0.022 0.095] best K-comp L0H4 trace/nuclear 0.13 symmetric positive-eigen mass 0.565
L2H4 H0H4-write Q/K energy [0.023 0.095] best K-comp L0H4 trace/nuclear 0.081 symmetric positive-eigen mass 0.54
L2H5 H0H4-write Q/K energy [0.006 0.041] best K-comp L0H4 trace/nuclear 0.093 symmetric positive-eigen mass 0.547
L3H0 H0H4-write Q/K energy [0.002 0.017] best K-comp L2H2 trace/nuclear 0.072 symmetric positive-eigen mass 0.537
L3H1 H0H4-write Q/K energy [0.003 0.009] best K-comp L2H2 trace/nuclear 0.288 symmetric positive-eigen mass 0.648
L3H2 H0H4-write Q/K energy [0.003 0.025] best K-comp L0H4 trace/nuclear 0.082 symmetric positive-eigen mass 0.541
L3H3 H0H4-write Q/K energy [0.005 0.01 ] best K-comp L0H4 trace/nuclear 0.07 symmetric positive-eigen mass 0.535
L3H4 H0H4-write Q/K energy [0.004 0.025] best K-comp L0H4 trace/nuclear 0.074 symmetric positive-eigen mass 0.537
L3H5 H0H4-write Q/K energy [0.003 0.005] best K-comp L2H2 trace/nuclear 0.39 symmetric positive-eigen mass 0.7

```

## Call 9

```python
import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
threadpool_limits(4)

P=np.eye(384)-np.ones((384,384))/384
g=W["ln1_w.0"].astype(float)
D=g[:,None]*P
V=np.concatenate([W["W_v.0"][64:128],W["W_v.0"][320:384]]).astype(float)
_,_,vt=la.svd(V,full_matrices=False)
Tr=vt[:27].T
Te=la.orth(D.T@Tr)
E=la.orth(P@(Tr/g[:,None]))
As=[]
print("L0 raw QK energy in/outside raw V's top-27 read subspace")
for j in range(6):
    sl=slice(64*j,64*(j+1))
    a=W["W_q.0"][sl].T.astype(float)@W["W_k.0"][sl].astype(float)/8
    ar=Tr.T@a@Tr
    rr=np.eye(384)-Tr@Tr.T
    print(j,"token/token energy",round(la.norm(ar)**2/la.norm(a)**2,3),
          "outside/outside",round(la.norm(rr@a@rr)**2/la.norm(a)**2,3),
          "token block singular values",np.round(la.svdvals(ar)[:12],3))
    As.append(D.T@a@D)

print("\nEffective QK energy on uniform-head read subspace")
sl=slice(256,320)
B4=P@W["W_o.0"][:,sl].astype(float)@W["W_v.0"][sl].astype(float)@D
u,s,vtb=la.svd(B4,full_matrices=False)
R=vtb[:3].T
for j,a in enumerate(As):
    print(j,"Q/K energy",np.round([la.norm(R.T@a)**2/la.norm(a)**2,
                                   la.norm(a@R)**2/la.norm(a)**2],3),
          "3x3 coefficient matrix",np.round(R.T@a@R,3))

print("\nL0 MLP current-read groups: static head-factor alignments")
F=W["W_fc.0"].astype(float)@(W["ln2_w.0"][:,None]*P)
sel=la.norm(W["W_fc.0"],axis=1)>1
X=F[sel]@E
X-=X.mean(axis=0)
X/=la.norm(X,axis=1,keepdims=True)
km=KMeans(n_clusters=28,n_init=40,random_state=0).fit(X)
C=km.cluster_centers_
C/=la.norm(C,axis=1,keepdims=True)
counts=np.bincount(km.labels_,minlength=28)
for j in [0,2,3,4]:
    ae=E.T@As[j]@E
    uu,ss,vv=la.svd(ae,full_matrices=False)
    print("head",j,"token-proxy singular values",np.round(ss[:8],5))
    for z in range(2):
        qr=C@uu[:,z]
        kr=C@vv[z]
        qi=np.argsort(abs(qr))[::-1][:5]
        ki=np.argsort(abs(kr))[::-1][:5]
        print("mode",z,"Q",[(int(i),round(qr[i],3),int(counts[i])) for i in qi],
              "K",[(int(i),round(kr[i],3),int(counts[i])) for i in ki])

```

Output:

```
L0 raw QK energy in/outside raw V's top-27 read subspace
0 token/token energy 0.032 outside/outside 0.534 token block singular values [0.135 0.045 0.017 0.012 0.01  0.008 0.008 0.007 0.006 0.005 0.004 0.003]
1 token/token energy 0.021 outside/outside 0.872 token block singular values [0.329 0.282 0.208 0.168 0.151 0.127 0.124 0.103 0.1   0.079 0.065 0.053]
2 token/token energy 0.308 outside/outside 0.248 token block singular values [0.302 0.279 0.253 0.222 0.206 0.149 0.119 0.094 0.09  0.066 0.06  0.056]
3 token/token energy 0.113 outside/outside 0.434 token block singular values [0.308 0.233 0.144 0.129 0.121 0.094 0.085 0.074 0.059 0.045 0.036 0.031]
4 token/token energy 0.008 outside/outside 0.858 token block singular values [0.005 0.004 0.003 0.002 0.002 0.002 0.001 0.001 0.001 0.001 0.001 0.001]
5 token/token energy 0.008 outside/outside 0.923 token block singular values [0.282 0.24  0.192 0.177 0.152 0.108 0.09  0.085 0.059 0.054 0.05  0.043]

Effective QK energy on uniform-head read subspace
0 Q/K energy [0.755 0.223] 3x3 coefficient matrix [[ 0.01   0.003  0.024]
 [ 0.024 -0.018  0.004]
 [ 0.007 -0.018 -0.015]]
1 Q/K energy [0.567 0.608] 3x3 coefficient matrix [[ 0.227 -0.209  0.156]
 [ 0.497  0.18  -0.034]
 [ 0.03  -0.046  0.017]]
2 Q/K energy [0.68  0.511] 3x3 coefficient matrix [[ 0.094  0.028 -0.005]
 [ 0.01   0.044 -0.031]
 [ 0.017  0.004  0.003]]
3 Q/K energy [0.623 0.414] 3x3 coefficient matrix [[-0.067  0.002 -0.027]
 [ 0.006 -0.08   0.035]
 [ 0.003 -0.032  0.029]]
4 Q/K energy [0.03  0.086] 3x3 coefficient matrix [[-0.     0.     0.   ]
 [-0.     0.     0.   ]
 [ 0.    -0.    -0.001]]
5 Q/K energy [0.249 0.281] 3x3 coefficient matrix [[ 0.192 -0.179  0.139]
 [ 0.276  0.138 -0.044]
 [-0.023 -0.114  0.052]]

L0 MLP current-read groups: static head-factor alignments
head 0 token-proxy singular values [0.0016 0.0006 0.0002 0.0001 0.0001 0.0001 0.0001 0.0001]
mode 0 Q [(15, np.float64(0.687), 22), (6, np.float64(-0.645), 44), (12, np.float64(-0.474), 27), (22, np.float64(-0.374), 10), (3, np.float64(0.347), 14)] K [(15, np.float64(-0.573), 22), (12, np.float64(0.498), 27), (9, np.float64(-0.368), 16), (6, np.float64(0.347), 44), (18, np.float64(-0.276), 17)]
mode 1 Q [(12, np.float64(0.535), 27), (25, np.float64(-0.437), 14), (9, np.float64(-0.319), 16), (13, np.float64(-0.3), 12), (7, np.float64(0.29), 17)] K [(3, np.float64(-0.463), 14), (18, np.float64(0.381), 17), (7, np.float64(-0.35), 17), (12, np.float64(0.302), 27), (13, np.float64(-0.296), 12)]
head 2 token-proxy singular values [0.0044 0.0038 0.0034 0.0026 0.0025 0.0017 0.0015 0.001 ]
mode 0 Q [(19, np.float64(0.613), 22), (17, np.float64(0.329), 11), (13, np.float64(-0.309), 12), (8, np.float64(-0.293), 44), (1, np.float64(-0.292), 24)] K [(17, np.float64(-0.495), 11), (1, np.float64(0.446), 24), (13, np.float64(0.392), 12), (20, np.float64(0.37), 19), (19, np.float64(-0.286), 22)]
mode 1 Q [(18, np.float64(-0.211), 17), (8, np.float64(-0.208), 44), (21, np.float64(-0.19), 10), (22, np.float64(0.181), 10), (13, np.float64(-0.16), 12)] K [(21, np.float64(0.344), 10), (13, np.float64(0.146), 12), (18, np.float64(-0.137), 17), (25, np.float64(-0.134), 14), (22, np.float64(-0.126), 10)]
head 3 token-proxy singular values [0.0038 0.0031 0.002  0.0016 0.0014 0.0011 0.0009 0.0008]
mode 0 Q [(6, np.float64(-0.567), 44), (19, np.float64(0.418), 22), (8, np.float64(-0.347), 44), (7, np.float64(0.334), 17), (15, np.float64(0.245), 22)] K [(19, np.float64(-0.384), 22), (21, np.float64(-0.341), 10), (6, np.float64(0.286), 44), (10, np.float64(0.281), 34), (17, np.float64(-0.221), 11)]
mode 1 Q [(12, np.float64(-0.438), 27), (24, np.float64(0.397), 18), (25, np.float64(0.28), 14), (15, np.float64(0.278), 22), (23, np.float64(-0.214), 11)] K [(25, np.float64(-0.471), 14), (22, np.float64(0.431), 10), (24, np.float64(0.391), 18), (23, np.float64(-0.361), 11), (15, np.float64(-0.358), 22)]
head 4 token-proxy singular values [0.0001 0.0001 0.     0.     0.     0.     0.     0.    ]
mode 0 Q [(22, np.float64(-0.501), 10), (19, np.float64(-0.458), 22), (12, np.float64(-0.248), 27), (8, np.float64(0.233), 44), (0, np.float64(0.2), 19)] K [(22, np.float64(0.487), 10), (13, np.float64(-0.373), 12), (27, np.float64(0.264), 14), (11, np.float64(-0.257), 14), (8, np.float64(-0.235), 44)]
mode 1 Q [(12, np.float64(0.593), 27), (22, np.float64(-0.49), 10), (15, np.float64(-0.36), 22), (14, np.float64(0.255), 13), (11, np.float64(0.228), 14)] K [(22, np.float64(0.411), 10), (21, np.float64(0.321), 10), (24, np.float64(-0.274), 18), (19, np.float64(-0.272), 22), (12, np.float64(-0.259), 27)]

```

## Call 10

```python
import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
threadpool_limits(4)

v1=W["W_v.0"][64:128].astype(float)
v5=W["W_v.0"][320:384].astype(float)
print("RAW VALUE-HEAD SPECTRA")
for j,v in [(1,v1),(5,v5)]:
    print(j,np.round(la.svdvals(v)[:45],4))
print("Cross-head V1 V5.T singular values:",np.round(la.svdvals(v1@v5.T)[:45],4))
print("Shared-read singular directions: mean canonical overlap")
for r in [20,21,22,23,24,25,26,27,28,29,30,32]:
    t1=la.svd(v1,full_matrices=False)[2][:r].T
    t5=la.svd(v5,full_matrices=False)[2][:r].T
    cc=la.svdvals(t1.T@t5)
    print(r,"squared overlap",round(float(cc@cc),3),"smallest 6",np.round(cc[-6:],3))

P=np.eye(384)-np.ones((384,384))/384
g=W["ln1_w.0"].astype(float)
_,_,vt=la.svd(np.concatenate([v1,v5]),full_matrices=False)
E=la.orth(P@(vt[:30].T/g[:,None]))
F=W["W_fc.0"].astype(float)@(W["ln2_w.0"][:,None]*P)
sel=la.norm(W["W_fc.0"],axis=1)>1
X=F[sel]@E
X-=X.mean(axis=0)
X/=la.norm(X,axis=1,keepdims=True)
print("\nCURRENT-READ CLUSTER RESOLUTION")
for k in range(20,33):
    km=KMeans(n_clusters=k,n_init=25,random_state=0).fit(X)
    C=km.cluster_centers_
    cn=la.norm(C,axis=1)
    counts=np.bincount(km.labels_,minlength=k)
    print(k,"inertia",round(km.inertia_,3),"mean center norm",round(cn.mean(),3),
          "minimum counts",sorted(counts.tolist())[:5],
          "smallest centroid singular values",np.round(la.svdvals(C)[-5:],3))

print("\nLAST-MLP OUTPUT PRINCIPAL DIRECTIONS: alignment with other block writers")
u,s,vt=la.svd(W["W_proj.3"].astype(float),full_matrices=False)
for i in range(35):
    z=u[:,i]
    vals=[]
    for name in ["W_proj.0","W_proj.1","W_proj.2","W_o.3"]:
        M=W[name].astype(float)
        vals.append(la.norm(z@M))
    print(i+1,round(s[i],3),np.round(vals,3))

```

Output:

```
RAW VALUE-HEAD SPECTRA
1 [2.25   2.1268 2.089  2.0551 1.9905 1.9732 1.9581 1.9328 1.8686 1.8322 1.8079 1.7853 1.7174 1.6904 1.639  1.6107 1.5841 1.5536 1.4959 1.48   1.4186 1.3703
 1.2771 1.1934 1.1397 1.094  1.0284 0.9062 0.7443 0.693  0.6822 0.6586 0.6385 0.5868 0.538  0.527  0.4948 0.482  0.4692 0.4523 0.4476 0.4338 0.4151 0.4077
 0.3976]
5 [2.3965 2.2816 2.1907 2.1483 2.0817 2.0488 2.0328 1.995  1.9012 1.8797 1.8603 1.7884 1.772  1.7442 1.7033 1.6459 1.5908 1.5382 1.448  1.3842 1.3168 1.2004
 1.0419 0.9231 0.8922 0.853  0.8173 0.7835 0.7673 0.7444 0.7368 0.7223 0.7088 0.6811 0.6632 0.6542 0.6266 0.5993 0.5787 0.5714 0.5648 0.5334 0.5038 0.4895
 0.4524]
Cross-head V1 V5.T singular values: [4.3729 4.1122 4.0138 3.895  3.7268 3.613  3.5185 3.336  3.2493 3.2041 3.0362 2.8391 2.7213 2.6217 2.5862 2.4899 2.351  2.2643 2.2241 1.9374 1.8663 1.4553
 1.239  1.1216 0.8115 0.7126 0.5596 0.4437 0.3926 0.3162 0.2873 0.2734 0.2632 0.2167 0.2076 0.1851 0.169  0.1468 0.1403 0.1255 0.1069 0.0987 0.0938 0.0856
 0.0705]
Shared-read singular directions: mean canonical overlap
20 squared overlap 14.551 smallest 6 [0.807 0.76  0.671 0.344 0.318 0.001]
21 squared overlap 15.808 smallest 6 [0.818 0.754 0.731 0.497 0.294 0.007]
22 squared overlap 17.325 smallest 6 [0.899 0.795 0.744 0.67  0.495 0.06 ]
23 squared overlap 18.36 smallest 6 [0.894 0.785 0.745 0.677 0.613 0.092]
24 squared overlap 19.165 smallest 6 [0.89  0.813 0.692 0.671 0.472 0.354]
25 squared overlap 20.086 smallest 6 [0.889 0.815 0.78  0.606 0.495 0.346]
26 squared overlap 20.799 smallest 6 [0.872 0.812 0.67  0.607 0.49  0.407]
27 squared overlap 21.337 smallest 6 [0.838 0.785 0.65  0.545 0.461 0.3  ]
28 squared overlap 21.725 smallest 6 [0.821 0.695 0.62  0.504 0.357 0.168]
29 squared overlap 22.026 smallest 6 [0.7   0.653 0.525 0.416 0.371 0.118]
30 squared overlap 22.314 smallest 6 [0.681 0.553 0.429 0.398 0.321 0.034]
32 squared overlap 22.991 smallest 6 [0.581 0.45  0.426 0.297 0.097 0.024]

CURRENT-READ CLUSTER RESOLUTION
20 inertia 300.888 mean center norm 0.66 minimum counts [13, 13, 15, 16, 17] smallest centroid singular values [0.452 0.417 0.343 0.262 0.046]
21 inertia 294.85 mean center norm 0.664 minimum counts [11, 14, 14, 15, 16] smallest centroid singular values [0.462 0.438 0.36  0.277 0.044]
22 inertia 287.021 mean center norm 0.679 minimum counts [9, 14, 15, 15, 15] smallest centroid singular values [0.467 0.32  0.314 0.253 0.035]
23 inertia 281.637 mean center norm 0.689 minimum counts [9, 10, 13, 14, 14] smallest centroid singular values [0.423 0.386 0.306 0.182 0.035]
24 inertia 277.496 mean center norm 0.696 minimum counts [11, 12, 12, 14, 14] smallest centroid singular values [0.482 0.28  0.207 0.1   0.055]
25 inertia 270.96 mean center norm 0.705 minimum counts [10, 11, 11, 13, 14] smallest centroid singular values [0.426 0.373 0.336 0.196 0.043]
26 inertia 265.829 mean center norm 0.71 minimum counts [10, 10, 11, 11, 12] smallest centroid singular values [0.429 0.369 0.283 0.145 0.043]
27 inertia 260.265 mean center norm 0.721 minimum counts [10, 11, 11, 11, 13] smallest centroid singular values [0.401 0.391 0.184 0.133 0.042]
28 inertia 257.41 mean center norm 0.728 minimum counts [6, 10, 11, 11, 11] smallest centroid singular values [0.394 0.363 0.148 0.126 0.023]
29 inertia 250.098 mean center norm 0.738 minimum counts [6, 7, 8, 8, 10] smallest centroid singular values [0.382 0.325 0.186 0.114 0.004]
30 inertia 247.405 mean center norm 0.738 minimum counts [6, 8, 9, 10, 10] smallest centroid singular values [0.312 0.118 0.085 0.023 0.02 ]
31 inertia 243.963 mean center norm 0.747 minimum counts [4, 6, 6, 9, 10] smallest centroid singular values [0.414 0.358 0.207 0.038 0.021]
32 inertia 238.966 mean center norm 0.747 minimum counts [7, 8, 9, 10, 10] smallest centroid singular values [0.449 0.405 0.273 0.077 0.02 ]

LAST-MLP OUTPUT PRINCIPAL DIRECTIONS: alignment with other block writers
1 28.175 [7.152 5.099 6.691 4.76 ]
2 20.293 [3.126 3.864 5.639 2.653]
3 19.292 [3.825 3.956 5.321 2.821]
4 19.135 [2.838 3.744 5.101 3.065]
5 18.708 [2.962 3.825 5.023 3.048]
6 18.57 [2.82  3.749 4.648 3.17 ]
7 18.139 [3.248 3.748 4.816 2.872]
8 17.776 [3.248 3.809 4.928 3.004]
9 17.485 [2.825 3.857 5.163 2.902]
10 16.54 [3.355 3.925 4.91  3.063]
11 15.913 [2.988 4.034 4.922 2.941]
12 15.61 [3.193 3.854 5.076 2.949]
13 14.975 [2.804 3.89  4.902 3.221]
14 14.873 [3.212 3.961 5.017 3.065]
15 14.417 [3.139 3.958 4.996 3.085]
16 14.313 [3.016 4.034 5.049 3.144]
17 13.327 [3.088 3.811 4.645 3.147]
18 12.971 [3.06  3.73  4.756 3.151]
19 12.593 [2.647 3.846 4.632 3.104]
20 12.37 [3.081 3.793 4.871 2.878]
21 12.225 [2.851 3.982 5.018 2.915]
22 12.019 [3.294 3.93  4.598 2.961]
23 11.322 [2.693 3.983 4.65  2.699]
24 11.115 [3.095 3.963 4.677 2.989]
25 10.769 [2.702 3.848 4.82  3.033]
26 9.227 [2.617 3.697 4.332 2.86 ]
27 8.573 [2.674 3.962 4.419 2.652]
28 7.929 [2.597 3.929 4.259 2.731]
29 6.5 [2.286 3.118 3.958 7.199]
30 5.271 [4.675 3.721 3.898 4.39 ]
31 3.966 [5.555 4.752 4.299 3.233]
32 3.184 [3.078 4.067 3.868 2.725]
33 2.792 [3.348 3.654 3.892 2.408]
34 2.697 [2.502 3.732 4.057 3.401]
35 2.653 [2.878 3.752 4.085 3.338]

```

## Call 11

```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

P=np.eye(384)-np.ones((384,384))/384
A={}; B={}
for l in range(4):
    D=W[f"ln1_w.{l}"].astype(float)[:,None]*P
    for j in range(6):
        sl=slice(64*j,64*(j+1))
        A[l,j]=D.T@W[f"W_q.{l}"][sl].T.astype(float)@W[f"W_k.{l}"][sl].astype(float)@D/8
        B[l,j]=P@W[f"W_o.{l}"][:,sl].astype(float)@W[f"W_v.{l}"][sl].astype(float)@D

a5=A[0,5]
ev,U=la.eigh(a5@a5.T+a5.T@a5)
ev,U=ev[::-1],U[:,::-1]
print("STATIC POSITION-SUBSPACE ROTATION ALIGNMENTS")
print("These are correlations of weight products, not attention distributions or measured lags.")
for r in [8,20]:
    T=U[:,:r]@np.diag(ev[:r]**(-.25))
    q,_=la.polar(T.T@a5@T)
    powers={p:np.linalg.matrix_power(q,p) for p in range(-16,17)}
    print("\nrank",r)
    for l in range(4):
        for j in range(6):
            C=T.T@A[l,j]@T
            vals={p:np.sum(C*qp)/(la.norm(C)*la.norm(qp)+1e-12) for p,qp in powers.items()}
            best=sorted(vals.items(),key=lambda z:z[1],reverse=True)[:3]
            retained=la.norm(U[:,:r].T@A[l,j]@U[:,:r])**2/la.norm(A[l,j])**2
            print(f"L{l}H{j}","energy",round(retained,4),
                  "top powers/cos",[(p,round(v,3)) for p,v in best],
                  "p0/1/2",np.round([vals[p] for p in [0,1,2]],3))

print("\nINDUCTION-CANDIDATE CONTROLS")
for target,source in [((2,1),(0,5)),((3,1),(0,5)),((3,5),(0,5)),
                      ((3,1),(2,2)),((3,5),(2,2))]:
    a=A[target]; b=B[source]
    C=a@b
    singular=la.svdvals(C)
    observed=np.trace(C)/singular.sum()
    # A null distribution by residual-coordinate permutation, preserving B's spectrum.
    rng=np.random.default_rng(17)
    vals=[]
    for _ in range(250):
        perm=rng.permutation(384)
        bp=b[np.ix_(perm,perm)]
        vals.append(np.sum(a*bp.T))
    vals=np.array(vals)
    print("target",target,"source",source,
          "trace",round(np.trace(C),4),"trace/nuclear",round(observed,4),
          "null trace mean/sd/max",np.round([vals.mean(),vals.std(),vals.max()],4),
          "z",round((np.trace(C)-vals.mean())/vals.std(),2))

```

Output:

```
STATIC POSITION-SUBSPACE ROTATION ALIGNMENTS
These are correlations of weight products, not attention distributions or measured lags.

rank 8
L0H0 energy 0.0475 top powers/cos [(2, np.float64(0.377)), (-5, np.float64(0.355)), (8, np.float64(0.354))] p0/1/2 [-0.003  0.316  0.377]
L0H1 energy 0.4939 top powers/cos [(8, np.float64(0.558)), (1, np.float64(0.517)), (2, np.float64(0.485))] p0/1/2 [0.162 0.517 0.485]
L0H2 energy 0.3907 top powers/cos [(0, np.float64(0.424)), (-7, np.float64(0.41)), (6, np.float64(0.405))] p0/1/2 [ 0.424  0.118 -0.3  ]
L0H3 energy 0.1513 top powers/cos [(3, np.float64(0.452)), (10, np.float64(0.427)), (-3, np.float64(0.406))] p0/1/2 [-0.489 -0.272  0.162]
L0H4 energy 0.001 top powers/cos [(-9, np.float64(0.196)), (-2, np.float64(0.192)), (-3, np.float64(0.188))] p0/1/2 [-0.118 -0.187 -0.088]
L0H5 energy 0.5646 top powers/cos [(1, np.float64(0.992)), (7, np.float64(0.776)), (-5, np.float64(0.776))] p0/1/2 [0.636 0.992 0.636]
L1H0 energy 0.0008 top powers/cos [(9, np.float64(0.252)), (16, np.float64(0.226)), (15, np.float64(0.197))] p0/1/2 [-0.096  0.021  0.13 ]
L1H1 energy 0.0016 top powers/cos [(3, np.float64(0.235)), (9, np.float64(0.203)), (10, np.float64(0.165))] p0/1/2 [-0.177 -0.084  0.116]
L1H2 energy 0.0018 top powers/cos [(9, np.float64(0.232)), (3, np.float64(0.197)), (15, np.float64(0.175))] p0/1/2 [-0.227 -0.035  0.17 ]
L1H3 energy 0.0017 top powers/cos [(9, np.float64(0.167)), (3, np.float64(0.161)), (2, np.float64(0.118))] p0/1/2 [-0.132 -0.024  0.118]
L1H4 energy 0.0024 top powers/cos [(3, np.float64(0.212)), (-4, np.float64(0.168)), (-3, np.float64(0.165))] p0/1/2 [-0.163 -0.035  0.148]
L1H5 energy 0.0014 top powers/cos [(3, np.float64(0.322)), (9, np.float64(0.305)), (10, np.float64(0.208))] p0/1/2 [-0.234 -0.096  0.18 ]
L2H0 energy 0.0005 top powers/cos [(3, np.float64(0.223)), (-4, np.float64(0.208)), (-3, np.float64(0.174))] p0/1/2 [-0.206 -0.046  0.155]
L2H1 energy 0.0005 top powers/cos [(-6, np.float64(0.478)), (1, np.float64(0.473)), (-5, np.float64(0.458))] p0/1/2 [0.433 0.473 0.168]
L2H2 energy 0.0005 top powers/cos [(0, np.float64(0.501)), (-6, np.float64(0.477)), (-1, np.float64(0.396))] p0/1/2 [ 0.501  0.218 -0.167]
L2H3 energy 0.0008 top powers/cos [(-12, np.float64(0.23)), (-6, np.float64(0.23)), (-13, np.float64(0.222))] p0/1/2 [ 0.107  0.049 -0.09 ]
L2H4 energy 0.0006 top powers/cos [(-12, np.float64(0.266)), (-6, np.float64(0.215)), (-5, np.float64(0.204))] p0/1/2 [0.067 0.135 0.108]
L2H5 energy 0.0005 top powers/cos [(-12, np.float64(0.253)), (-13, np.float64(0.244)), (16, np.float64(0.147))] p0/1/2 [-0.087 -0.159 -0.183]
L3H0 energy 0.0004 top powers/cos [(0, np.float64(0.386)), (-6, np.float64(0.35)), (6, np.float64(0.291))] p0/1/2 [ 0.386  0.258 -0.044]
L3H1 energy 0.0004 top powers/cos [(0, np.float64(0.604)), (-6, np.float64(0.579)), (1, np.float64(0.506))] p0/1/2 [0.604 0.506 0.132]
L3H2 energy 0.0005 top powers/cos [(-1, np.float64(0.223)), (5, np.float64(0.217)), (0, np.float64(0.179))] p0/1/2 [ 0.179 -0.027 -0.179]
L3H3 energy 0.0003 top powers/cos [(-11, np.float64(0.236)), (-12, np.float64(0.23)), (-4, np.float64(0.202))] p0/1/2 [-0.173 -0.069  0.086]
L3H4 energy 0.0002 top powers/cos [(-11, np.float64(0.253)), (-12, np.float64(0.234)), (-4, np.float64(0.202))] p0/1/2 [-0.141  0.002  0.079]
L3H5 energy 0.0015 top powers/cos [(0, np.float64(0.841)), (-6, np.float64(0.667)), (6, np.float64(0.656))] p0/1/2 [ 0.841  0.569 -0.048]

rank 20
L0H0 energy 0.0734 top powers/cos [(8, np.float64(0.211)), (-5, np.float64(0.211)), (2, np.float64(0.195))] p0/1/2 [-0.103  0.168  0.195]
L0H1 energy 0.6802 top powers/cos [(2, np.float64(0.642)), (15, np.float64(0.437)), (-11, np.float64(0.374))] p0/1/2 [-0.225  0.071  0.642]
L0H2 energy 0.4311 top powers/cos [(6, np.float64(0.232)), (-7, np.float64(0.231)), (7, np.float64(0.231))] p0/1/2 [ 0.228  0.069 -0.176]
L0H3 energy 0.2073 top powers/cos [(-3, np.float64(0.189)), (10, np.float64(0.189)), (-10, np.float64(0.162))] p0/1/2 [-0.249 -0.119  0.056]
L0H4 energy 0.0092 top powers/cos [(-11, np.float64(0.108)), (5, np.float64(0.101)), (12, np.float64(0.078))] p0/1/2 [-0.086 -0.015  0.043]
L0H5 energy 0.8959 top powers/cos [(1, np.float64(0.987)), (-12, np.float64(0.546)), (14, np.float64(0.546))] p0/1/2 [0.272 0.987 0.272]
L1H0 energy 0.0052 top powers/cos [(13, np.float64(0.083)), (1, np.float64(0.07)), (9, np.float64(0.065))] p0/1/2 [ 0.026  0.07  -0.03 ]
L1H1 energy 0.0084 top powers/cos [(-10, np.float64(0.089)), (15, np.float64(0.073)), (9, np.float64(0.068))] p0/1/2 [-0.001 -0.019  0.051]
L1H2 energy 0.0099 top powers/cos [(15, np.float64(0.052)), (-10, np.float64(0.045)), (16, np.float64(0.035))] p0/1/2 [-0.023 -0.026  0.009]
L1H3 energy 0.0096 top powers/cos [(-10, np.float64(0.103)), (15, np.float64(0.054)), (14, np.float64(0.052))] p0/1/2 [-0.014 -0.053  0.034]
L1H4 energy 0.0124 top powers/cos [(-10, np.float64(0.075)), (3, np.float64(0.053)), (16, np.float64(0.035))] p0/1/2 [-0.025 -0.044  0.006]
L1H5 energy 0.0084 top powers/cos [(-10, np.float64(0.089)), (15, np.float64(0.077)), (-7, np.float64(0.061))] p0/1/2 [-0.013 -0.068  0.058]
L2H0 energy 0.0027 top powers/cos [(-16, np.float64(0.132)), (-3, np.float64(0.121)), (-10, np.float64(0.076))] p0/1/2 [-0.043  0.01   0.001]
L2H1 energy 0.0024 top powers/cos [(0, np.float64(0.28)), (13, np.float64(0.166)), (-13, np.float64(0.16))] p0/1/2 [ 0.28   0.017 -0.146]
L2H2 energy 0.0033 top powers/cos [(0, np.float64(0.166)), (13, np.float64(0.093)), (-13, np.float64(0.074))] p0/1/2 [ 0.166  0.056 -0.09 ]
L2H3 energy 0.0039 top powers/cos [(14, np.float64(0.113)), (1, np.float64(0.07)), (7, np.float64(0.067))] p0/1/2 [-0.004  0.07  -0.046]
L2H4 energy 0.0033 top powers/cos [(-16, np.float64(0.094)), (-7, np.float64(0.081)), (-13, np.float64(0.081))] p0/1/2 [ 0.047  0.015 -0.021]
L2H5 energy 0.003 top powers/cos [(-14, np.float64(0.057)), (-4, np.float64(0.048)), (-10, np.float64(0.039))] p0/1/2 [-0.033 -0.115 -0.04 ]
L3H0 energy 0.0027 
```
