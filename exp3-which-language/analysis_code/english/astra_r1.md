## Call 1

```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=160)
threadpool_limits(4)

def spectrum(a):
    s = svdvals(a.astype(np.float64))
    e = s*s
    p = e/e.sum()
    return [float(np.linalg.norm(a)), float(s[0]),
            float(p[:1].sum()), float(p[:5].sum()), float(p[:20].sum()),
            float(np.exp(-(p*np.log(p+1e-300)).sum()))]

print("All calculations below are weight-matrix algebra; no network inputs or activations.")
print("Keys and shapes:", [(k, v.shape) for k, v in sorted(W.items())])
for l in range(4):
    print("\nLAYER", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        a = W[f"{name}.{l}"]
        ix = np.argsort(np.abs(a))[-8:][::-1]
        print(name, "mean/std/min/max/norm",
              np.round([a.mean(), a.std(), a.min(), a.max(), np.linalg.norm(a)], 5),
              "largest_abs", list(zip(ix.tolist(), np.round(a[ix], 4).tolist())))
    print("matrix: norm, largest_sv, energy_top1/top5/top20, effective_rank")
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        print(name, np.round(spectrum(W[f"{name}.{l}"]), 5))

    print("head: ||Q|| ||K|| ||V|| ||O||; QK[frob,trace,transpose_cos]; OV[frob,trace,transpose_cos]; Q/K/V/O effective ranks")
    for j in range(6):
        s = slice(64*j, 64*(j+1))
        q = W[f"W_q.{l}"][s].astype(np.float64)
        k = W[f"W_k.{l}"][s].astype(np.float64)
        v = W[f"W_v.{l}"][s].astype(np.float64)
        o = W[f"W_o.{l}"][:, s].astype(np.float64)
        def product_stats(a,b):
            ba = b @ a
            f2 = np.sum((a.T @ a) * (b @ b.T).T)
            return [np.sqrt(max(f2,0)), np.trace(ba), np.trace(ba @ ba)/(f2+1e-30)]
        print(f"L{l}H{j}",
              np.round([np.linalg.norm(x) for x in [q,k,v,o]], 4),
              "QK", np.round(product_stats(q.T,k), 5),
              "OV", np.round(product_stats(o,v), 5),
              "ranks", np.round([spectrum(x)[-1] for x in [q,k,v,o]], 2))

```

Output:

```
All calculations below are weight-matrix algebra; no network inputs or activations.
Keys and shapes: [('W_fc.0', (1536, 384)), ('W_fc.1', (1536, 384)), ('W_fc.2', (1536, 384)), ('W_fc.3', (1536, 384)), ('W_k.0', (384, 384)), ('W_k.1', (384, 384)), ('W_k.2', (384, 384)), ('W_k.3', (384, 384)), ('W_o.0', (384, 384)), ('W_o.1', (384, 384)), ('W_o.2', (384, 384)), ('W_o.3', (384, 384)), ('W_proj.0', (384, 1536)), ('W_proj.1', (384, 1536)), ('W_proj.2', (384, 1536)), ('W_proj.3', (384, 1536)), ('W_q.0', (384, 384)), ('W_q.1', (384, 384)), ('W_q.2', (384, 384)), ('W_q.3', (384, 384)), ('W_v.0', (384, 384)), ('W_v.1', (384, 384)), ('W_v.2', (384, 384)), ('W_v.3', (384, 384)), ('ln1_b.0', (384,)), ('ln1_b.1', (384,)), ('ln1_b.2', (384,)), ('ln1_b.3', (384,)), ('ln1_w.0', (384,)), ('ln1_w.1', (384,)), ('ln1_w.2', (384,)), ('ln1_w.3', (384,)), ('ln2_b.0', (384,)), ('ln2_b.1', (384,)), ('ln2_b.2', (384,)), ('ln2_b.3', (384,)), ('ln2_w.0', (384,)), ('ln2_w.1', (384,)), ('ln2_w.2', (384,)), ('ln2_w.3', (384,))]

LAYER 0
ln1_w mean/std/min/max/norm [ 0.2403  0.2436 -0.001   1.0341  6.7051] largest_abs [(162, 1.03410005569458), (60, 0.9843999743461609), (223, 0.9800000190734863), (311, 0.9442999958992004), (239, 0.9384999871253967), (300, 0.9370999932289124), (35, 0.9369999766349792), (180, 0.9294000267982483)]
ln1_b mean/std/min/max/norm [ 0.0008  0.0287 -0.0959  0.091   0.5635] largest_abs [(104, -0.09589999914169312), (239, -0.09539999812841415), (37, 0.09099999815225601), (221, 0.08730000257492065), (153, 0.08529999852180481), (364, 0.08330000191926956), (278, -0.08229999989271164), (292, 0.08100000023841858)]
ln2_w mean/std/min/max/norm [ 0.8006  0.3295  0.0044  1.5072 16.9648] largest_abs [(161, 1.507200002670288), (164, 1.5055999755859375), (155, 1.4536999464035034), (348, 1.419100046157837), (376, 1.4045000076293945), (200, 1.4042999744415283), (23, 1.3960000276565552), (379, 1.3837000131607056)]
ln2_b mean/std/min/max/norm [ 0.0022  0.0683 -0.4986  0.3331  1.3392] largest_abs [(222, -0.4986000061035156), (197, 0.33309999108314514), (360, 0.31380000710487366), (103, -0.30469998717308044), (307, -0.27300000190734863), (78, 0.26919999718666077), (254, 0.2655999958515167), (72, -0.21310000121593475)]
matrix: norm, largest_sv, energy_top1/top5/top20, effective_rank
W_q [32.2912  8.2519  0.0653  0.3062  0.8179 37.4667]
W_k [30.3909  7.6464  0.0633  0.2836  0.7835 43.2073]
W_v [14.836   4.339   0.0855  0.2766  0.6129 74.8005]
W_o [16.3713  7.1033  0.1883  0.4711  0.7081 36.7761]
W_fc [46.2486 18.1759  0.1545  0.2972  0.5686 71.244 ]
W_proj [ 39.3075   9.8257   0.0625   0.1669   0.4156 111.0325]
head: ||Q|| ||K|| ||V|| ||O||; QK[frob,trace,transpose_cos]; OV[frob,trace,transpose_cos]; Q/K/V/O effective ranks
L0H0 [3.3053 4.1805 6.0393 6.6661] QK [ 3.0479 -1.7245  0.0561] OV [12.9041 -0.1807  0.0078] ranks [35.37 40.   23.35  9.05]
L0H1 [3.0895 3.6576 5.3817 6.0246] QK [ 1.8376 -1.2358  0.1166] OV [10.1485 -0.7493  0.0184] ranks [43.18 47.45 26.35  9.28]
L0H2 [30.8183 28.2998  7.7648  8.9155] QK [182.4351  32.7752  -0.0305] OV [13.223  -4.6421  0.0098] ranks [26.95 27.51 33.52 30.77]
L0H3 [6.8213 7.2435 5.4336 5.029 ] QK [11.3515 -7.9874  0.1641] OV [ 7.3079 -2.0478  0.0159] ranks [27.79 28.31 27.34 15.54]
L0H4 [3.9711 4.9951 5.2079 5.5871] QK [ 6.6069 -4.7252  0.1641] OV [ 7.3117 -0.9671  0.0082] ranks [24.88 20.75 31.16 13.1 ]
L0H5 [3.1903 3.8022 6.1428 7.1621] QK [ 2.5787 -0.9665  0.0624] OV [13.0384 -0.0858  0.0039] ranks [37.47 40.85 26.19  9.59]

LAYER 1
ln1_w mean/std/min/max/norm [0.4575 0.1022 0.1361 1.1239 9.1868] largest_abs [(222, 1.123900055885315), (72, 0.7994999885559082), (78, 0.7810999751091003), (302, 0.7670999765396118), (118, 0.7135000228881836), (351, 0.6988999843597412), (111, 0.6916999816894531), (200, 0.6776000261306763)]
ln1_b mean/std/min/max/norm [-0.005   0.1672 -1.4713  1.1544  3.2771] largest_abs [(222, -1.4711999893188477), (360, 1.1543999910354614), (254, 1.0657999515533447), (25, -0.9016000032424927), (78, 0.866100013256073), (72, -0.7670000195503235), (197, 0.667900025844574), (273, -0.6603999733924866)]
ln2_w mean/std/min/max/norm [ 1.0429  0.2003 -0.0019  1.4856 20.8093] largest_abs [(375, 1.4855999946594238), (186, 1.409999966621399), (198, 1.3732999563217163), (14, 1.3538000583648682), (53, 1.3459999561309814), (131, 1.3413000106811523), (346, 1.333400011062622), (361, 1.332900047302246)]
ln2_b mean/std/min/max/norm [ 0.0005  0.1364 -0.3597  0.3686  2.6732] largest_abs [(254, 0.3686000108718872), (137, -0.3596999943256378), (25, -0.3517000079154968), (80, 0.34619998931884766), (375, 0.34209999442100525), (297, 0.3073999881744385), (360, 0.3003999888896942), (61, -0.29030001163482666)]
matrix: norm, largest_sv, energy_top1/top5/top20, effective_rank
W_q [32.3126  9.9032  0.0939  0.26    0.5931 62.0766]
W_k [29.2506 13.3072  0.207   0.4304  0.6833 42.6837]
W_v [27.3529  5.2845  0.0373  0.1611  0.4782 88.9684]
W_o [ 25.9233   4.9657   0.0367   0.0955   0.2817 138.4507]
W_fc [ 79.1265  21.4881   0.0737   0.1772   0.4521 104.5661]
W_proj [ 75.7103   8.1935   0.0117   0.0479   0.1385 317.6697]
head: ||Q|| ||K|| ||V|| ||O||; QK[frob,trace,transpose_cos]; OV[frob,trace,transpose_cos]; Q/K/V/O effective ranks
L1H0 [10.716   9.8266 11.1557  9.9835] QK [ 23.6654 -26.1053   0.0806] OV [ 21.7066 -21.6005   0.0885] ranks [27.46 22.8  34.18 26.61]
L1H1 [13.2924 12.0678 11.7199 11.303 ] QK [ 35.5126 -44.9928   0.1022] OV [ 22.226  -28.4652   0.0682] ranks [26.96 20.01 41.63 35.57]
L1H2 [14.4756 13.0922 10.6678 10.2717] QK [ 41.7812 -60.5649   0.1095] OV [ 19.5165 -21.8959   0.0493] ranks [26.67 18.97 39.71 32.03]
L1H3 [13.3579 12.1529 10.9036  9.9282] QK [ 36.8369 -44.4839   0.0956] OV [ 21.4133 -20.2784   0.0452] ranks [25.99 19.05 33.4  26.12]
L1H4 [13.5181 12.1295 11.1481 10.794 ] QK [ 37.3819 -46.5798   0.0851] OV [ 22.0658 -26.577    0.0675] ranks [27.11 18.75 37.12 30.37]
L1H5 [13.4859 12.132  11.3756 11.1355] QK [ 36.2112 -39.7087   0.0644] OV [ 22.4639 -29.2909   0.0683] r
```

## Call 2

```python
import numpy as np
from scipy.linalg import svdvals, eigvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

def centered_weight(a, gamma):
    b = np.asarray(a, dtype=np.float64) * gamma[None, :]
    return b - b.mean(axis=1, keepdims=True)

M, T = {}, {}
print("Folded matrices use only the known LayerNorm scale and the linear mean-removal projection.")
print("No LayerNorm, attention, or MLP is evaluated.")
for l in range(4):
    g = W[f"ln1_w.{l}"]
    for j in range(6):
        s = slice(j*64, (j+1)*64)
        q = centered_weight(W[f"W_q.{l}"][s], g)
        k = centered_weight(W[f"W_k.{l}"][s], g)
        v = centered_weight(W[f"W_v.{l}"][s], g)
        o = W[f"W_o.{l}"][:,s].astype(np.float64)
        m, t = q.T @ k, o @ v
        M[l,j], T[l,j] = m, t
        sm, st = svdvals(m), svdvals(t)
        evm = eigvals(k @ q.T)
        evt = eigvals(v @ o)
        def info(a, sv, ev):
            en = sv**2
            return [
                np.linalg.norm(a), np.trace(a)/np.linalg.norm(a),
                np.sum(a*a.T)/np.sum(a*a),
                en[:1].sum()/en.sum(), en[:5].sum()/en.sum(),
                np.sum(ev.real)/(np.sum(np.abs(ev))+1e-30),
                np.max(ev.real), np.min(ev.real)
            ]
        print(f"L{l}H{j}", "QK", np.round(info(m, sm, evm), 3),
              "OV", np.round(info(t, st, evt), 3))
print("Columns above: frob, trace/frob, transpose_cos, energy_top1, energy_top5, signed_eigen_ratio, max_real_eigen, min_real_eigen")

for a,b in [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3)]:
    key, query, tr = np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6))
    for j in range(6):
        for i in range(6):
            m, t = M[b,j], T[a,i]
            kt = m @ t
            key[j,i] = np.linalg.norm(kt)/(np.linalg.norm(m)*np.linalg.norm(t))
            query[j,i] = np.linalg.norm(t.T @ m)/(np.linalg.norm(m)*np.linalg.norm(t))
            tr[j,i] = np.trace(kt)/(np.linalg.norm(kt)+1e-30)
    print(f"\nL{a} -> L{b}; rows=later heads, columns=earlier heads")
    print("key composition normalized norm:\n", key)
    print("query composition normalized norm:\n", query)
    print("key composition trace/frob:\n", tr)

print("\nL0H2 dominant QK singular values and eigenvalues")
print("singular values", np.round(svdvals(M[0,2])[:45],4))
ev = eigvals(M[0,2])
print("eigenvalues by magnitude", np.round(ev[np.argsort(np.abs(ev))[::-1]][:45],4))
print("\nLast MLP output singular spectrum")
print(np.round(svdvals(W["W_proj.3"].astype(np.float64))[:65],4))

```

Output:

```
Folded matrices use only the known LayerNorm scale and the linear mean-removal projection.
No LayerNorm, attention, or MLP is evaluated.
L0H0 QK [ 0.437 -0.086  0.023  0.888  0.962 -0.099  0.058 -0.032] OV [ 9.041  0.003  0.01   0.797  0.996  0.008  0.986 -0.384]
L0H1 QK [ 0.236 -0.474  0.047  0.715  0.887 -0.325  0.016 -0.035] OV [ 6.965 -0.027  0.019  0.66   0.995 -0.074  0.728 -0.68 ]
L0H2 QK [32.551  0.295 -0.045  0.12   0.474  0.081  5.208 -6.535] OV [ 2.538 -0.247  0.003  0.096  0.381 -0.2    0.116 -0.119]
L0H3 QK [ 3.156  0.499  0.478  0.827  0.975  0.33   2.094 -0.14 ] OV [ 1.765 -0.142  0.024  0.342  0.908 -0.195  0.233 -0.103]
L0H4 QK [ 0.849 -0.666  0.004  0.703  0.987 -0.571  0.009 -0.223] OV [ 4.033 -0.087  0.013  0.709  0.978 -0.195  0.278 -0.253]
L0H5 QK [ 0.62  -0.447  0.16   0.958  0.985 -0.528  0.006 -0.249] OV [ 9.309  0.005  0.006  0.801  0.993  0.014  0.728 -0.383]
L1H0 QK [ 7.091 -0.726  0.009  0.454  0.897 -0.581  0.225 -0.627] OV [10.119 -0.944  0.082  0.157  0.522 -0.706  0.704 -1.247]
L1H1 QK [10.517 -0.883  0.019  0.449  0.868 -0.576  0.558 -1.226] OV [10.331 -1.213  0.062  0.085  0.349 -0.79   0.251 -0.931]
L1H2 QK [12.393 -1.044  0.018  0.412  0.853 -0.616  0.775 -1.343] OV [ 9.179 -1.052  0.043  0.098  0.391 -0.787  0.116 -0.747]
L1H3 QK [10.994 -0.841  0.017  0.447  0.882 -0.559  0.752 -1.403] OV [10.05  -0.892  0.04   0.136  0.485 -0.781  0.123 -0.896]
L1H4 QK [11.597 -0.858  0.011  0.427  0.896 -0.608  0.494 -1.081] OV [10.4   -1.14   0.06   0.088  0.371 -0.851  0.066 -0.953]
L1H5 QK [10.69  -0.758 -0.003  0.446  0.88  -0.523  0.707 -0.966] OV [10.619 -1.243  0.063  0.081  0.341 -0.88   0.104 -0.97 ]
L2H0 QK [19.5   -0.009 -0.034  0.081  0.31  -0.004  0.948 -0.962] OV [16.304 -2.366  0.118  0.047  0.201 -0.871  0.053 -1.545]
L2H1 QK [16.541  0.256  0.01   0.065  0.236  0.119  1.007 -0.826] OV [17.72  -2.011  0.079  0.049  0.205 -0.795  0.194 -1.542]
L2H2 QK [12.902 -0.685 -0.005  0.142  0.373 -0.324  0.539 -0.846] OV [20.572 -1.621  0.072  0.077  0.285 -0.733  0.576 -2.117]
L2H3 QK [16.69  -0.353 -0.027  0.072  0.275 -0.157  0.698 -0.834] OV [17.504 -2.031  0.097  0.065  0.251 -0.819  0.157 -2.013]
L2H4 QK [18.751 -0.203 -0.025  0.072  0.276 -0.094  0.699 -0.945] OV [16.622 -1.948  0.079  0.047  0.202 -0.808  0.19  -1.373]
L2H5 QK [18.003 -0.309 -0.028  0.08   0.274 -0.138  0.93  -1.129] OV [16.682 -2.171  0.099  0.043  0.193 -0.835  0.187 -1.647]
L3H0 QK [17.858  1.529  0.05   0.041  0.177  0.613  1.457 -0.545] OV [22.594 -2.45   0.117  0.08   0.276 -0.86   0.142 -2.404]
L3H1 QK [14.636  3.449  0.211  0.032  0.146  0.854  1.391 -0.62 ] OV [40.856 -1.311  0.062  0.068  0.301 -0.783  0.73  -4.046]
L3H2 QK [12.04   0.147 -0.004  0.098  0.319  0.072  0.658 -0.612] OV [26.877 -1.275  0.049  0.116  0.359 -0.654  0.476 -2.425]
L3H3 QK [12.788  0.07  -0.017  0.097  0.275  0.033  0.795 -0.537] OV [27.428 -1.453  0.049  0.12   0.364 -0.695  0.499 -2.424]
L3H4 QK [14.844  0.487 -0.001  0.081  0.255  0.232  0.82  -0.526] OV [24.196 -1.387  0.039  0.09   0.31  -0.65   0.401 -2.241]
L3H5 QK [ 6.671  0.121  0.024  0.097  0.347  0.057  0.623 -0.297] OV [45.391 -1.012  0.066  0.138  0.483 -0.68   0.853 -6.574]
Columns above: frob, trace/frob, transpose_cos, energy_top1, energy_top5, signed_eigen_ratio, max_real_eigen, min_real_eigen

L0 -> L1; rows=later heads, columns=earlier heads
key composition normalized norm:
 [[0.546 0.529 0.048 0.155 0.522 0.559]
 [0.5   0.503 0.064 0.146 0.485 0.561]
 [0.499 0.503 0.058 0.16  0.483 0.553]
 [0.511 0.507 0.063 0.15  0.491 0.565]
 [0.54  0.53  0.053 0.171 0.516 0.57 ]
 [0.502 0.503 0.079 0.154 0.483 0.559]]
query composition normalized norm:
 [[0.163 0.182 0.066 0.107 0.164 0.164]
 [0.167 0.185 0.068 0.093 0.165 0.171]
 [0.159 0.183 0.073 0.087 0.154 0.166]
 [0.157 0.18  0.079 0.09  0.154 0.162]
 [0.152 0.178 0.079 0.09  0.149 0.164]
 [0.168 0.192 0.083 0.115 0.168 0.181]]
key composition trace/frob:
 [[ 0.072  0.032 -0.113 -0.119  0.062  0.007]
 [ 0.092  0.066 -0.003 -0.097  0.102  0.033]
 [ 0.101  0.084  0.253 -0.035  0.112  0.04 ]
 [ 0.116  0.094  0.233 -0.065  0.123  0.05 ]
 [ 0.149  0.134  0.251 -0.026  0.163  0.081]
 [ 0.078  0.036  0.077 -0.088  0.076 -0.02 ]]

L0 -> L2; rows=later heads, columns=earlier heads
key composition normalized norm:
 [[0.198 0.202 0.063 0.094 0.191 0.222]
 [0.139 0.143 0.058 0.078 0.137 0.162]
 [0.27  0.279 0.051 0.138 0.263 0.302]
 [0.207 0.207 0.054 0.093 0.2   0.219]
 [0.205 0.205 0.056 0.098 0.197 0.211]
 [0.2   0.201 0.056 0.093 0.194 0.214]]
query composition normalized norm:
 [[0.111 0.12  0.056 0.074 0.11  0.111]
 [0.059 0.065 0.06  0.056 0.062 0.065]
 [0.095 0.102 0.049 0.077 0.096 0.106]
 [0.133 0.137 0.054 0.079 0.128 0.131]
 [0.116 0.125 0.055 0.076 0.113 0.117]
 [0.118 0.124 0.055 0.075 0.115 0.118]]
key composition trace/frob:
 [[ 0.08   0.062 -0.08  -0.193  0.096  0.071]
 [ 0.071  0.046 -0.097 -0.131  0.076  0.071]
 [ 0.11   0.087  0.02  -0.124  0.136  0.034]
 [ 0.111  0.08   0.054 -0.128  0.116  0.099]
 [ 0.144  0.108 -0.142 -0.171  0.152  0.1  ]
 [ 0.112  0.073  0.007 -0.185  0.11   0.078]]

L1 -> L2; rows=later heads, columns=earlier heads
key composition normalized norm:
 [[0.065 0.047 0.05  0.051 0.052 0.052]
 [0.059 0.049 0.05  0.049 0.049 0.051]
 [0.07  0.047 0.043 0.044 0.045 0.044]
 [0.051 0.048 0.05  0.05  0.052 0.052]
 [0.057 0.049 0.049 0.051 0.051 0.052]
 [0.059 0.046 0.05  0.049 0.052 0.051]]
query composition normalized norm:
 [[0.066 0.054 0.056 0.059 0.06  0.064]
 [0.066 0.061 0.064 0.068 0.065 0.064]
 [0.07  0.056 0.055 0.054 0.054 0.052]
 [0.061 0.059 0.067 0.069 0.062 0.059]
 [0.059 0.054 0.057 0.065 0.061 0.063]
 [0.059 0.053 0.057 0.061 0.06  0.062]]
key composition trace/frob:
 [[-0.428 -0.092 -0.093 -0.294 -0.244 -0.248]
 [-0.289 -0.405 -0.171  0.063 -0.423 -0.31 ]
 [-0.069 -0.009  0.05   0.091  0.174  0.054]
 [-0.18  -0.14  -0.16   0.033 -0.201 -0.279]
 [-0.215 -0.132 -0.102 -0.059 -0.181 -0.289]
 [-0.215 -0.087  0.007 -0.285 -0.124 -0.386]]

L0 -> L3; rows=l
```

## Call 3

```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=160)

def fold(a,g):
    b=np.asarray(a,dtype=np.float64)*g[None,:]
    return b-b.mean(1,keepdims=True)

F=fold(W["W_fc.0"], W["ln2_w.0"])
g=W["ln1_w.0"]
Q=fold(W["W_q.0"][128:192],g)
K=fold(W["W_k.0"][128:192],g)
V=fold(W["W_v.0"][128:192],g)
O=W["W_o.0"][:,128:192].astype(float)
Uo,so,_=svd(O,full_matrices=False)
Uv,sv,Vv=svd(V,full_matrices=False)
Up,sp,_=svd(np.concatenate([Q.T,K.T],axis=1),full_matrices=False)
print("L0H2 O singular values:",so[:45])
print("L0H2 folded V singular values:",sv[:45])
print("L0H2 joint Q/K singular values:",sp[:45])
print("L0 ln1 gamma quantiles:",np.quantile(g,[0,.1,.25,.5,.75,.9,1]))
print("L0 F singular values:",svd(F,compute_uv=False)[:70])

nuisance=[Up[:,:26], Uo[:,:32]]
for j in [0,1,3,4,5]:
    o=W["W_o.0"][:,64*j:64*(j+1)].astype(float)
    v=fold(W["W_v.0"][64*j:64*(j+1)],g)
    u,ss,_=svd(o@v,full_matrices=False)
    nuisance.append(u[:,:5])
N=np.concatenate(nuisance,axis=1)
Un,sn,_=svd(N,full_matrices=False)
N=Un[:,sn>1e-6]
current=F-(F@N)@N.T
previous=F@Uo[:,:28]
value_read=F@Vv[:28].T
spaces={"current_residual":current,"previous_output":previous,"value_input":value_read,
        "last_projection_columns":W["W_proj.3"].T.astype(float)}
for name,a in spaces.items():
    a=a-a.mean(0,keepdims=True)
    u,s,vh=svd(a,full_matrices=False)
    print("\nSPACE",name, "singular", s[:40])
    z=u[:,:28]*s[:28]
    z=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-20)
    print("k, cosine silhouette, cluster sizes descending")
    for nc in [2,3,5,8,12,16,20,24,28,32]:
        km=KMeans(n_clusters=nc,n_init=8,random_state=0).fit(z)
        score=silhouette_score(z,km.labels_,metric="cosine",sample_size=1000,random_state=0)
        counts=np.sort(np.bincount(km.labels_))[::-1]
        print(nc,round(score,4),counts.tolist())

```

Output:

```
L0H2 O singular values: [2.254 2.161 2.127 2.101 2.052 2.007 1.972 1.96  1.893 1.855 1.8   1.782 1.708 1.668 1.631 1.542 1.509 1.402 1.263 1.209 1.127 1.112 1.04  1.015 0.957 0.941
 0.873 0.844 0.819 0.748 0.702 0.684 0.646 0.626 0.581 0.515 0.471 0.464 0.414 0.405 0.38  0.351 0.333 0.304 0.287]
L0H2 folded V singular values: [0.431 0.416 0.384 0.37  0.362 0.348 0.328 0.315 0.309 0.302 0.3   0.296 0.292 0.274 0.265 0.264 0.26  0.253 0.247 0.245 0.235 0.232 0.23  0.224 0.216 0.213
 0.206 0.198 0.191 0.188 0.185 0.181 0.173 0.161 0.158 0.152 0.151 0.137 0.127 0.123 0.114 0.114 0.106 0.101 0.097]
L0H2 joint Q/K singular values: [5.087 4.666 4.48  4.414 4.294 4.144 4.038 3.983 3.849 3.788 3.647 3.569 3.485 3.226 2.902 2.841 2.624 2.441 2.346 2.191 1.912 1.872 1.653 1.509 1.44  1.415
 1.295 1.214 1.11  1.089 0.998 0.943 0.886 0.868 0.826 0.772 0.705 0.696 0.668 0.655 0.62  0.594 0.575 0.56  0.525]
L0 ln1 gamma quantiles: [-0.001  0.039  0.08   0.142  0.321  0.715  1.034]
L0 F singular values: [17.903 10.353  7.602  7.51   7.11   6.834  6.74   6.432  6.363  6.22   6.139  6.05   6.015  5.936  5.794  5.626  5.575  5.499  5.421  5.309  5.301  5.117
  5.076  4.966  4.945  4.84   4.814  4.73   4.562  4.559  4.484  4.346  4.149  4.084  4.052  3.912  3.822  3.766  3.706  3.527  3.476  3.411  3.338  3.291
  3.237  3.206  3.058  2.967  2.94   2.816  2.717  2.672  2.609  2.459  2.395  2.325  2.232  2.222  2.162  2.091  2.063  2.     1.928  1.897  1.876  1.804
  1.795  1.727  1.679  1.669]

SPACE current_residual singular [12.665  7.07   6.622  6.474  6.333  6.105  5.99   5.856  5.821  5.711  5.605  5.579  5.405  5.324  5.238  5.188  5.088  4.972  4.894  4.749  4.663  4.638
  4.589  4.477  4.386  4.338  4.221  4.007  3.914  3.552  3.206  3.022  2.838  2.775  2.768  2.685  2.626  2.555  2.466  2.41 ]
k, cosine silhouette, cluster sizes descending
2 0.3843 [1096, 440]
3 0.3107 [997, 390, 149]
5 0.0702 [560, 384, 262, 195, 135]
8 0.0665 [414, 358, 203, 169, 146, 136, 61, 49]
12 0.0611 [256, 240, 231, 153, 132, 127, 91, 85, 81, 64, 48, 28]
16 0.0881 [319, 269, 214, 181, 109, 83, 66, 66, 51, 32, 31, 27, 25, 25, 24, 14]
20 0.085 [235, 218, 198, 156, 133, 109, 78, 59, 58, 48, 35, 32, 31, 29, 27, 21, 20, 18, 16, 15]
24 0.0944 [194, 175, 147, 137, 114, 104, 91, 79, 71, 58, 51, 41, 41, 31, 30, 27, 25, 22, 21, 20, 18, 15, 12, 12]
28 0.1102 [197, 167, 115, 113, 112, 110, 94, 85, 65, 65, 47, 39, 33, 31, 31, 29, 26, 22, 20, 19, 17, 16, 16, 15, 15, 15, 14, 8]
32 0.1063 [143, 127, 120, 103, 91, 89, 88, 85, 80, 70, 49, 45, 44, 38, 36, 32, 26, 24, 23, 23, 22, 22, 21, 21, 19, 18, 17, 17, 14, 12, 9, 8]

SPACE previous_output singular [5.796 4.402 4.212 4.019 3.941 3.618 3.389 3.284 3.217 3.109 3.027 2.927 2.762 2.585 2.512 2.451 2.35  2.168 1.835 1.599 1.499 1.238 1.189 1.127 1.024 0.908
 0.885 0.837]
k, cosine silhouette, cluster sizes descending
2 0.1752 [1138, 398]
3 0.0826 [702, 504, 330]
5 0.071 [366, 339, 305, 273, 253]
8 0.0623 [244, 218, 197, 187, 183, 177, 165, 165]
12 0.0596 [189, 146, 142, 139, 137, 130, 127, 122, 109, 101, 98, 96]
16 0.0609 [135, 127, 115, 110, 110, 105, 101, 99, 95, 94, 90, 84, 80, 74, 66, 51]
20 0.0607 [115, 98, 96, 92, 91, 89, 88, 88, 86, 76, 74, 73, 71, 68, 64, 62, 59, 56, 56, 34]
24 0.0631 [99, 93, 91, 90, 82, 79, 79, 72, 72, 72, 71, 66, 64, 62, 61, 57, 49, 47, 47, 43, 38, 35, 35, 32]
28 0.0649 [79, 73, 72, 72, 70, 69, 67, 67, 65, 64, 64, 62, 57, 57, 56, 52, 50, 48, 46, 44, 44, 42, 39, 39, 38, 36, 32, 32]
32 0.0676 [65, 63, 61, 60, 59, 58, 58, 58, 57, 57, 57, 56, 56, 56, 54, 52, 52, 48, 46, 44, 44, 43, 41, 39, 37, 36, 36, 33, 31, 29, 25, 25]

SPACE value_input singular [5.51  4.926 4.441 4.338 4.235 4.021 3.628 3.573 3.397 3.289 3.224 3.119 3.03  2.918 2.827 2.8   2.687 2.58  2.355 2.186 2.092 1.845 1.474 1.24  1.117 0.986
 0.822 0.663]
k, cosine silhouette, cluster sizes descending
2 0.1644 [976, 560]
3 0.0939 [601, 500, 435]
5 0.0926 [415, 380, 325, 264, 152]
8 0.0995 [301, 290, 205, 181, 178, 161, 111, 109]
12 0.1016 [196, 190, 162, 155, 140, 140, 120, 111, 89, 87, 84, 62]
16 0.1087 [166, 144, 139, 135, 130, 108, 87, 80, 78, 76, 75, 75, 72, 64, 58, 49]
20 0.108 [125, 121, 121, 95, 83, 82, 80, 79, 77, 74, 74, 69, 69, 67, 61, 56, 53, 53, 49, 48]
24 0.1109 [94, 94, 86, 84, 83, 79, 78, 72, 68, 66, 66, 64, 60, 60, 59, 59, 55, 54, 51, 48, 47, 42, 34, 33]
28 0.1208 [100, 96, 90, 89, 89, 77, 66, 65, 63, 63, 61, 55, 54, 51, 50, 48, 47, 45, 41, 40, 38, 36, 35, 34, 33, 26, 22, 22]
32 0.1179 [101, 76, 73, 72, 68, 67, 63, 60, 59, 56, 56, 56, 52, 50, 50, 46, 44, 44, 41, 39, 38, 37, 36, 33, 33, 32, 29, 29, 29, 24, 23, 20]

SPACE last_projection_columns singular [26.021 20.684 19.697 19.259 18.594 18.197 17.861 17.666 17.043 16.744 16.033 15.151 14.754 14.311 14.047 13.5   13.285 12.53  12.401 11.958 11.733 11.594
 10.364  8.348  8.04   7.906  7.461  7.161  5.788  3.933  3.142  3.082  2.833  2.8    2.687  2.571  2.397  2.327  2.214  2.181]
k, cosine silhouette, cluster sizes descending
2 0.1159 [888, 648]
3 0.0961 [538, 522, 476]
5 0.0857 [343, 330, 320, 301, 242]
8 0.0851 [216, 204, 201, 200, 191, 185, 170, 169]
12 0.0866 [155, 152, 152, 138, 127, 127, 124, 116, 115, 114, 110, 106]
16 0.0867 [122, 113, 107, 103, 102, 102, 98, 97, 95, 93, 90, 89, 88, 81, 80, 76]
20 0.0888 [96, 95, 95, 94, 92, 91, 82, 82, 81, 76, 75, 74, 71, 69, 66, 62, 60, 59, 58, 58]
24 0.0779 [83, 76, 76, 69, 69, 68, 66, 66, 66, 65, 65, 64, 64, 63, 63, 62, 61, 60, 59, 59, 57, 53, 53, 49]
28 0.0824 [84, 77, 71, 67, 65, 63, 60, 59, 58, 57, 57, 56, 55, 54, 53, 52, 52, 50, 49, 48, 46, 46, 45, 45, 44, 43, 41, 39]
32 0.0861 [85, 69, 66, 59, 57, 54, 53, 51, 51, 50, 49, 48, 48, 47, 47, 47, 47, 47, 46, 45, 44, 44, 44, 42, 41, 41, 40, 39, 39, 34, 31, 31]

```

## Call 4

```python
import numpy as np
from scipy.linalg import svd, subspace_angles
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
def fold(a,g):
    b=np.asarray(a,dtype=float)*g[None,:]
    return b-b.mean(1,keepdims=True)
g=W["ln1_w.0"]
q2=fold(W["W_q.0"][128:192],g)
k2=fold(W["W_k.0"][128:192],g)
P,_,_=svd(np.concatenate([q2.T,k2.T],1),full_matrices=False)
P=P[:,:28]
Rs,Us,Ms,Ts=[],[],[],[]
print("L0: folded QK/OV singular values; OV-input overlap with H2 joint-QK subspace")
for j in range(6):
    s=slice(64*j,64*(j+1))
    q=fold(W["W_q.0"][s],g); k=fold(W["W_k.0"][s],g)
    v=fold(W["W_v.0"][s],g); o=W["W_o.0"][:,s].astype(float)
    m=q.T@k; t=o@v
    u,st,r=svd(t,full_matrices=False)
    Rs.append(r[:5]);Us.append(u[:,:5]);Ms.append(m);Ts.append(t)
    print(j,"QK",svd(m,compute_uv=False)[:8],"OV",st[:10],
          "OV input positional overlaps",np.sum((r[:5]@P)**2,axis=1))
    print("Q/K/V fraction on positional subspace",
          [round(np.linalg.norm(a@P)**2/np.linalg.norm(a)**2,4) for a in [q,k,v]])
    print("OV top-input and top-output largest coordinates")
    for tag,a in [("in",r[0]),("out",u[:,0])]:
        ix=np.argsort(np.abs(a))[-8:][::-1]
        print(tag,[(int(i),round(float(a[i]),4)) for i in ix])

print("\nCosines between L0 OV top input singular vectors")
print(np.array([[Rs[i][0]@Rs[j][0] for j in range(6)] for i in range(6)]))
print("Cosines between L0 OV top output singular vectors")
print(np.array([[Us[i][:,0]@Us[j][:,0] for j in range(6)] for i in range(6)]))

print("\nAffine weight terms (no input evaluation):")
print("head: QK norm; ||(Q*b)^T Kdiag||; ||Qdiag.T(K*b)||; bias-bias scalar")
for l in range(4):
    g=W[f"ln1_w.{l}"];b=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s=slice(64*j,64*(j+1))
        qr=W[f"W_q.{l}"][s].astype(float);kr=W[f"W_k.{l}"][s].astype(float)
        q=fold(qr,g);k=fold(kr,g)
        qb=qr@b;kb=kr@b
        print(f"L{l}H{j}",np.round([np.linalg.norm(q.T@k),
              np.linalg.norm(qb@k),np.linalg.norm(q.T@kb),qb@kb],4))

print("\nLayer-0 head value-read relations to LN bias and later key-read directions:")
for l in [1,2,3]:
    g=W[f"ln1_w.{l}"]
    print("layer",l,"rows=head; columns=overlap of key rowspace with L0 OV top output directions H0..H5")
    table=[]
    for j in range(6):
        k=fold(W[f"W_k.{l}"][64*j:64*(j+1)],g)
        table.append([np.linalg.norm(k@Us[i][:,:1])**2/np.linalg.norm(k)**2 for i in range(6)])
    print(np.array(table))

```

Output:

```
L0: folded QK/OV singular values; OV-input overlap with H2 joint-QK subspace
0 QK [0.411 0.092 0.057 0.037 0.034 0.029 0.023 0.023] OV [8.073 3.327 2.097 0.672 0.561 0.314 0.257 0.193 0.145 0.141] OV input positional overlaps [0.976 0.737 0.792 0.197 0.276]
Q/K/V fraction on positional subspace [np.float64(0.5905), np.float64(0.2192), np.float64(0.7818)]
OV top-input and top-output largest coordinates
in [(162, -0.3585), (223, 0.3121), (311, 0.3038), (300, -0.2932), (59, -0.2326), (351, -0.2269), (152, 0.2025), (178, -0.1989)]
out [(222, -0.9218), (360, 0.2572), (254, 0.143), (25, -0.0993), (197, 0.0816), (103, -0.0705), (273, -0.0669), (136, 0.0492)]
1 QK [0.2   0.076 0.042 0.035 0.028 0.025 0.023 0.021] OV [5.66  3.827 1.064 0.502 0.428 0.27  0.217 0.166 0.151 0.112] OV input positional overlaps [0.959 0.919 0.661 0.15  0.118]
Q/K/V fraction on positional subspace [np.float64(0.3797), np.float64(0.2617), np.float64(0.7841)]
OV top-input and top-output largest coordinates
in [(162, -0.3428), (223, 0.3325), (300, -0.2939), (311, 0.2491), (178, -0.2449), (59, -0.2295), (114, -0.199), (152, 0.1845)]
out [(222, -0.8722), (360, 0.3588), (103, -0.1802), (254, 0.1697), (25, -0.1214), (197, 0.0829), (136, 0.0665), (72, 0.0546)]
2 QK [11.263 10.177  9.702  9.593  9.261  8.718  8.668  8.165] OV [0.787 0.718 0.683 0.671 0.636 0.601 0.569 0.558 0.55  0.522] OV input positional overlaps [0.096 0.076 0.094 0.108 0.097]
Q/K/V fraction on positional subspace [np.float64(0.9409), np.float64(0.9609), np.float64(0.2539)]
OV top-input and top-output largest coordinates
in [(301, -0.4622), (59, 0.2462), (240, 0.202), (40, -0.1832), (311, 0.1788), (103, -0.1787), (278, -0.175), (4, 0.1695)]
out [(210, 0.2208), (302, 0.2112), (347, -0.16), (217, -0.1503), (39, 0.1303), (311, -0.1296), (299, 0.1284), (119, 0.1233)]
3 QK [2.871 0.951 0.621 0.34  0.261 0.218 0.195 0.181] OV [1.032 0.964 0.617 0.505 0.448 0.227 0.201 0.183 0.172 0.147] OV input positional overlaps [0.493 0.214 0.126 0.198 0.09 ]
Q/K/V fraction on positional subspace [np.float64(0.5582), np.float64(0.6882), np.float64(0.2811)]
OV top-input and top-output largest coordinates
in [(239, 0.3392), (360, 0.3229), (35, 0.321), (180, -0.2321), (273, -0.1991), (320, 0.1907), (76, -0.1596), (279, 0.159)]
out [(360, 0.6564), (78, 0.423), (103, -0.3747), (222, -0.2994), (254, 0.1446), (197, 0.1132), (307, -0.0934), (25, -0.0862)]
4 QK [0.712 0.428 0.132 0.052 0.042 0.033 0.028 0.026] OV [3.396 1.642 1.082 0.607 0.379 0.327 0.231 0.204 0.17  0.141] OV input positional overlaps [0.954 0.81  0.58  0.103 0.125]
Q/K/V fraction on positional subspace [np.float64(0.3974), np.float64(0.3003), np.float64(0.6435)]
OV top-input and top-output largest coordinates
in [(162, -0.3446), (223, 0.3365), (300, -0.3013), (311, 0.2712), (59, -0.259), (178, -0.221), (319, -0.1841), (114, -0.1831)]
out [(222, -0.8892), (360, 0.3486), (103, -0.1699), (254, 0.1412), (25, -0.0857), (197, 0.0684), (273, -0.0593), (136, 0.0432)]
5 QK [0.607 0.086 0.037 0.03  0.027 0.023 0.022 0.02 ] OV [8.33  3.604 1.679 0.768 0.511 0.391 0.344 0.305 0.247 0.193] OV input positional overlaps [0.94  0.881 0.623 0.189 0.086]
Q/K/V fraction on positional subspace [np.float64(0.5855), np.float64(0.3196), np.float64(0.754)]
OV top-input and top-output largest coordinates
in [(223, -0.3174), (60, -0.2912), (311, -0.2883), (300, 0.2726), (162, 0.2441), (246, 0.2326), (59, 0.2091), (319, 0.2087)]
out [(222, 0.6503), (78, 0.6418), (72, -0.3031), (360, 0.1721), (103, -0.088), (273, -0.0784), (25, -0.0756), (254, 0.0671)]

Cosines between L0 OV top input singular vectors
[[ 1.     0.945 -0.066 -0.044  0.962 -0.693]
 [ 0.945  1.    -0.076  0.181  0.957 -0.639]
 [-0.066 -0.076  1.     0.005 -0.037  0.077]
 [-0.044  0.181  0.005  1.     0.134  0.259]
 [ 0.962  0.957 -0.037  0.134  1.    -0.71 ]
 [-0.693 -0.639  0.077  0.259 -0.71   1.   ]]
Cosines between L0 OV top output singular vectors
[[ 1.     0.959  0.011  0.513  0.969 -0.524]
 [ 0.959  1.     0.035  0.615  0.987 -0.483]
 [ 0.011  0.035  1.     0.077  0.037  0.065]
 [ 0.513  0.615  0.077  1.     0.596  0.26 ]
 [ 0.969  0.987  0.037  0.596  1.    -0.494]
 [-0.524 -0.483  0.065  0.26  -0.494  1.   ]]

Affine weight terms (no input evaluation):
head: QK norm; ||(Q*b)^T Kdiag||; ||Qdiag.T(K*b)||; bias-bias scalar
L0H0 [ 0.436  0.109  0.169 -0.044]
L0H1 [0.236 0.027 0.011 0.001]
L0H2 [32.551  1.143  1.51  -0.002]
L0H3 [ 3.156  0.428  0.322 -0.043]
L0H4 [ 0.85   0.09   0.077 -0.013]
L0H5 [ 0.62   0.13   0.202 -0.043]
L1H0 [ 7.091 10.858 13.164 17.996]
L1H1 [10.517 18.278 18.629 22.48 ]
L1H2 [12.393 20.703 23.812 26.718]
L1H3 [10.994 18.809 18.756 26.954]
L1H4 [11.597 18.322 23.567 21.137]
L1H5 [10.69  14.021 19.639 19.214]
L2H0 [19.5   13.592 13.001 10.175]
L2H1 [16.541  5.957  7.495  3.58 ]
L2H2 [12.902  6.043 12.497  6.12 ]
L2H3 [16.69  11.7   10.992  6.294]
L2H4 [18.751 12.619 13.097  7.233]
L2H5 [18.003 12.069 11.624  6.705]
L3H0 [17.858  2.065  2.088  0.793]
L3H1 [14.636  1.927  1.092  0.772]
L3H2 [12.04   2.316  2.209  0.877]
L3H3 [12.788  2.883  2.434  0.784]
L3H4 [14.844  2.458  2.085  0.66 ]
L3H5 [ 6.671  1.316  0.71  -0.061]

Layer-0 head value-read relations to LN bias and later key-read directions:
layer 1 rows=head; columns=overlap of key rowspace with L0 OV top output directions H0..H5
[[0.258 0.24  0.004 0.045 0.247 0.287]
 [0.288 0.271 0.009 0.05  0.276 0.313]
 [0.307 0.286 0.004 0.054 0.291 0.333]
 [0.29  0.264 0.006 0.05  0.272 0.314]
 [0.316 0.289 0.009 0.053 0.296 0.344]
 [0.307 0.285 0.016 0.048 0.29  0.35 ]]
layer 2 rows=head; columns=overlap of key rowspace with L0 OV top output directions H0..H5
[[0.093 0.084 0.005 0.025 0.086 0.111]
 [0.054 0.049 0.004 0.016 0.05  0.059]
 [0.097 0.085 0.003 0.027 0.089 0.105]
 [0.114 0.105 0.003 0.029 0.109 0.132]
 [0.105 0.094 0.003 0.028 0.097 0.118]
 [0.098 0.089 0.003 0.027 0.092 0.111]]
layer 3 rows=head; columns=overlap of key rowspace with L0 OV top output directions H0..H5
[[0.013 0.012 0.002 0.005 0.
```

## Call 5

```python
import numpy as np
from scipy.linalg import svd, eigvals
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4,suppress=True,linewidth=180)

def fold(a,g):
    b=np.asarray(a,dtype=float)*g[None,:]
    return b-b.mean(1,keepdims=True)
F=fold(W["W_fc.0"],W["ln2_w.0"])
g=W["ln1_w.0"]
Q=fold(W["W_q.0"][128:192],g)
K=fold(W["W_k.0"][128:192],g)
V=fold(W["W_v.0"][128:192],g)
O=W["W_o.0"][:,128:192].astype(float)
P,_,_=svd(np.concatenate([Q.T,K.T],1),full_matrices=False)
Uo,_,_=svd(O,full_matrices=False)
other=[]
for j in [0,1,3,4,5]:
    s=slice(64*j,64*(j+1))
    u,ss,_=svd(W["W_o.0"][:,s]@fold(W["W_v.0"][s],g),full_matrices=False)
    other.append(u[:,:5])
N=np.concatenate([P[:,:40],Uo]+other,1)
N,sn,_=svd(N,full_matrices=False)
N=N[:,sn>1e-7]
A=F-(F@N)@N.T
A-=A.mean(0)
u,s,vh=svd(A,full_matrices=False)
print("Current-read quotient spectrum:",s[:60])
for skip in [0,1,2,3]:
    z=u[:,skip:skip+27]*s[skip:skip+27]
    z/=np.linalg.norm(z,axis=1,keepdims=True)+1e-30
    print("Removed leading PCs:",skip)
    for nc in [2,5,20,24,28,32]:
        km=KMeans(n_clusters=nc,n_init=10,random_state=42).fit(z)
        print(nc,round(silhouette_score(z,km.labels_,metric="cosine",sample_size=1000,random_state=0),4),
              np.sort(np.bincount(km.labels_))[::-1].tolist())

print("\nICA of current-read neuron weights: component kurtosis and absolute skew")
ica=FastICA(n_components=28,whiten="unit-variance",max_iter=2500,tol=1e-4,random_state=1)
Z=ica.fit_transform(A)
ku=kurtosis(Z); sk=skew(Z)
print("kurtosis",np.sort(ku)[::-1])
print("abs skew",np.sort(np.abs(sk))[::-1])

# Both factors are just neuron read-weights expressed in a shared source space.
B=F@(O@V)
B=B-(B@N)@N.T
B-=B.mean(0)
for skip in [0,1,2]:
    basis=vh[skip:skip+27].T
    X=A@basis
    Y=B@basis
    Cxx=X.T@X
    Cyy=Y.T@Y
    Cxy=X.T@Y
    evals=eigvals(np.linalg.solve(Cxx+1e-7*np.eye(27),Cxy))
    print("\nRead-weight regression eigenvalues, skip",skip,
          np.round(evals[np.argsort(np.abs(evals))[::-1]][:15],4))
    gram=Cxx+Cyy*(np.trace(Cxx)/(np.trace(Cyy)+1e-30))
    e,U=np.linalg.eigh(gram)
    H=(U/np.sqrt(np.maximum(e,1e-10)))@U.T
    C=H@Cxy@H
    ev=np.linalg.eigvalsh((C+C.T)/2)
    print("Symmetric normalized cross-weight eigenvalues:",np.round(ev,4))
    print("transpose cosine",np.sum(C*C.T)/np.sum(C*C))

```

Output:

```
Current-read quotient spectrum: [11.7513  6.6374  6.4615  6.2212  6.1365  5.9583  5.759   5.6521  5.6397  5.5227  5.4217  5.3827  5.2491  5.136   5.0711  4.9656  4.8245  4.7834  4.7065  4.5834  4.5473  4.4865
  4.4482  4.3008  4.2002  4.1889  4.0566  3.7862  3.5745  3.2598  3.0134  2.8489  2.7181  2.6321  2.5105  2.4817  2.4614  2.3611  2.3597  2.2423  2.2286  2.2123  2.0868  2.0743
  2.037   1.9699  1.9534  1.9207  1.8733  1.854   1.8071  1.7831  1.757   1.7261  1.6714  1.647   1.6239  1.5707  1.5257  1.5089]
Removed leading PCs: 0
2 0.3785 [1110, 426]
5 0.0905 [627, 334, 263, 176, 136]
20 0.0834 [182, 178, 172, 152, 111, 92, 92, 88, 84, 71, 59, 52, 31, 29, 29, 28, 24, 24, 23, 15]
24 0.0938 [168, 145, 136, 130, 113, 112, 102, 86, 77, 74, 51, 46, 43, 33, 32, 29, 24, 23, 22, 21, 20, 18, 16, 15]
28 0.116 [181, 158, 120, 118, 109, 96, 91, 90, 77, 68, 65, 36, 31, 28, 27, 25, 23, 23, 22, 21, 21, 18, 17, 16, 15, 14, 14, 12]
32 0.1095 [112, 104, 99, 98, 93, 90, 90, 89, 86, 80, 79, 62, 48, 40, 31, 29, 28, 28, 27, 25, 23, 21, 21, 17, 17, 15, 15, 15, 15, 14, 13, 12]
Removed leading PCs: 1
2 0.1382 [887, 649]
5 0.0669 [433, 338, 281, 248, 236]
20 0.1033 [159, 125, 123, 104, 101, 91, 90, 83, 79, 76, 75, 65, 58, 56, 53, 47, 45, 43, 34, 29]
24 0.1192 [184, 172, 102, 101, 73, 71, 67, 64, 63, 58, 54, 52, 52, 52, 48, 44, 44, 39, 39, 38, 36, 33, 27, 23]
28 0.1268 [139, 125, 110, 108, 83, 80, 71, 58, 56, 52, 51, 47, 44, 44, 43, 39, 39, 38, 36, 35, 34, 34, 33, 31, 29, 29, 26, 22]
32 0.1252 [94, 88, 87, 85, 82, 81, 68, 67, 63, 62, 57, 55, 44, 39, 38, 38, 38, 37, 36, 35, 33, 33, 32, 32, 30, 29, 28, 28, 27, 26, 25, 19]
Removed leading PCs: 2
2 0.111 [858, 678]
5 0.0639 [415, 320, 300, 251, 250]
20 0.0999 [160, 147, 113, 97, 85, 85, 84, 83, 82, 79, 70, 63, 62, 60, 54, 53, 47, 41, 37, 34]
24 0.1166 [142, 140, 94, 91, 87, 83, 79, 73, 60, 59, 57, 57, 52, 51, 50, 50, 44, 44, 43, 43, 43, 39, 29, 26]
28 0.122 [127, 116, 86, 72, 70, 67, 66, 65, 63, 60, 56, 55, 51, 50, 49, 48, 48, 45, 41, 40, 38, 38, 38, 37, 35, 30, 24, 21]
32 0.1265 [93, 80, 79, 78, 70, 69, 63, 60, 58, 54, 53, 53, 49, 48, 47, 46, 42, 41, 37, 37, 36, 36, 36, 35, 34, 34, 34, 32, 29, 28, 24, 21]
Removed leading PCs: 3
2 0.104 [853, 683]
5 0.0645 [399, 379, 324, 266, 168]
20 0.1032 [168, 154, 94, 93, 84, 81, 76, 74, 74, 72, 68, 67, 67, 65, 62, 52, 51, 46, 45, 43]
24 0.111 [122, 106, 92, 87, 79, 72, 72, 69, 69, 68, 63, 61, 59, 57, 57, 55, 47, 47, 45, 44, 43, 41, 41, 40]
28 0.1198 [109, 102, 98, 96, 65, 64, 62, 60, 60, 60, 58, 58, 58, 55, 51, 46, 46, 45, 45, 42, 39, 38, 35, 33, 30, 29, 28, 24]
32 0.1251 [90, 79, 76, 73, 68, 68, 67, 64, 62, 57, 55, 53, 50, 49, 47, 45, 44, 43, 41, 41, 39, 37, 36, 34, 32, 31, 30, 30, 26, 26, 22, 21]

ICA of current-read neuron weights: component kurtosis and absolute skew
kurtosis [235.4026 217.8788 195.9558 174.6764 164.7658 154.8066 144.9882 140.7653 128.7277 126.0953 124.3322 115.9021 111.8307 111.6914 110.5239 109.7803  99.5358  89.6654  82.4746
  79.7524  74.2802  71.2379  64.3916  63.1462  51.972   51.2592  27.8161  27.2398]
abs skew [13.8006 11.7524 11.6865 11.5641 10.9074 10.7852 10.7418 10.0429  9.6431  9.6057  9.5705  9.4548  9.4378  9.4353  9.1425  8.995   8.7133  7.7784  7.3057  7.2028  7.1213  6.8888
  6.8704  5.8721  5.4689  4.9218  2.7844  2.076 ]

Read-weight regression eigenvalues, skip 0 [ 0.1077+0.0197j  0.1077-0.0197j  0.0889+0.j      0.082 +0.j      0.0768+0.0249j  0.0768-0.0249j  0.0582+0.0534j  0.0582-0.0534j  0.0467+0.029j   0.0467-0.029j   0.0396+0.02j
  0.0396-0.02j    0.0426+0.j      0.0333+0.j     -0.021 +0.j    ]
Symmetric normalized cross-weight eigenvalues: [-0.0473 -0.0299 -0.0253 -0.0172 -0.0113 -0.0091 -0.0053 -0.0044 -0.0001  0.001   0.003   0.0063  0.0122  0.0149  0.018   0.0195  0.023   0.0263  0.0287  0.0292  0.0327  0.0346
  0.0365  0.0404  0.0442  0.0464  0.0561]
transpose cosine 0.18120370749430184

Read-weight regression eigenvalues, skip 1 [0.1131+0.0225j 0.1131-0.0225j 0.0941+0.j     0.0822+0.j     0.058 +0.0524j 0.058 -0.0524j 0.0722+0.0215j 0.0722-0.0215j 0.0526+0.0271j 0.0526-0.0271j 0.0476+0.j
 0.0289+0.0329j 0.0289-0.0329j 0.0267+0.0162j 0.0267-0.0162j]
Symmetric normalized cross-weight eigenvalues: [-0.033  -0.0314 -0.0194 -0.0177 -0.0126 -0.0091 -0.0047 -0.0005  0.0005  0.001   0.0024  0.0072  0.0082  0.0167  0.0186  0.0205  0.0227  0.027   0.0292  0.0325  0.0346  0.036
  0.0388  0.0429  0.0451  0.048   0.0528]
transpose cosine 0.2184402372233516

Read-weight regression eigenvalues, skip 2 [0.1152+0.j     0.101 +0.0281j 0.101 -0.0281j 0.0936+0.j     0.0611+0.0471j 0.0611-0.0471j 0.0746+0.0177j 0.0746-0.0177j 0.0546+0.j     0.0445+0.03j   0.0445-0.03j
 0.0328-0.0186j 0.0328+0.0186j 0.0229-0.0295j 0.0229+0.0295j]
Symmetric normalized cross-weight eigenvalues: [-0.0335 -0.0309 -0.023  -0.0164 -0.0125 -0.0085 -0.007  -0.0043  0.0008  0.0018  0.0063  0.0079  0.015   0.0168  0.0182  0.0212  0.025   0.0279  0.0295  0.0331  0.0361  0.0374
  0.0379  0.0459  0.0475  0.0501  0.0542]
transpose cosine 0.2205060417770196

```

## Call 6

```python
import numpy as np
from scipy.linalg import svd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)
def fold(a,g):
    b=np.asarray(a,dtype=float)*g[None,:]
    return b-b.mean(1,keepdims=True)
F=fold(W["W_fc.0"],W["ln2_w.0"])
g=W["ln1_w.0"]
Q=fold(W["W_q.0"][128:192],g);K=fold(W["W_k.0"][128:192],g)
V=fold(W["W_v.0"][128:192],g);O=W["W_o.0"][:,128:192].astype(float)
P,_,_=svd(np.concatenate([Q.T,K.T],1),full_matrices=False)
Uo,_,_=svd(O,full_matrices=False)
other=[]
for j in [0,1,3,4,5]:
    s=slice(64*j,64*(j+1))
    u,_,_=svd(W["W_o.0"][:,s]@fold(W["W_v.0"][s],g),full_matrices=False)
    other.append(u[:,:5])
N=np.concatenate([P[:,:40],Uo]+other,1)
N,sn,_=svd(N,full_matrices=False);N=N[:,sn>1e-7]
A=F-(F@N)@N.T
norms=np.linalg.norm(A,axis=1)
ix=np.argsort(norms)[::-1]
bias=W["W_fc.0"].astype(float)@W["ln2_b.0"].astype(float)
print("Neuron row-norm quantiles:",np.quantile(norms,[0,.1,.5,.8,.9,.95,.98,.99,1]))
print("Top 75 L0 neurons: index, quotient norm, full read norm, output norm, affine bias")
for i in ix[:75]:
    print(int(i),np.round([norms[i],np.linalg.norm(F[i]),np.linalg.norm(W["W_proj.0"][:,i]),bias[i]],3))
top=A[ix[:40]]
cos=(top@top.T)/(np.linalg.norm(top,axis=1)[:,None]*np.linalg.norm(top,axis=1)[None,:])
np.fill_diagonal(cos,0)
print("Top-row nearest cosine:",np.max(cos,axis=1))
print("Top-row smallest cosine:",np.min(cos,axis=1))

Ac=A-A.mean(0)
ica=FastICA(n_components=28,whiten="unit-variance",max_iter=2500,tol=1e-4,random_state=1)
X=ica.fit_transform(Ac)
sign=np.sign(skew(X));X*=sign
D=ica.mixing_*sign[None,:]
B=F@(O@V)
B-= (B@N)@N.T
B-= B.mean(0)
Y=B@np.linalg.pinv(D).T
print("\nICA components: strongest 4 neuron coefficients; previous-read weight coefficient std; read-direction norm")
for c in range(28):
    ids=np.argsort(X[:,c])[-4:][::-1]
    print(c,[(int(i),round(float(X[i,c]),2)) for i in ids],
          "prev_std",round(float(Y[:,c].std()),3),"direction_norm",round(float(np.linalg.norm(D[:,c])),3))
print("Previous-read ICA representation relative residual error:",
      np.linalg.norm(B-Y@D.T)/np.linalg.norm(B))
print("\nCross-products of ICA coefficient columns, normalized by column norms")
C=X.T@Y/(np.linalg.norm(X,axis=0)[:,None]*np.linalg.norm(Y,axis=0)[None,:])
for i in range(28):
    ii=np.argsort(C[i])[::-1][:5]
    print(i,[(int(j),round(float(C[i,j]),3)) for j in ii],
          "negative",int(np.argmin(C[i])),round(float(np.min(C[i])),3))

```

Output:

```
Neuron row-norm quantiles: [0.226 0.262 0.301 1.167 1.634 1.973 2.357 2.559 3.561]
Top 75 L0 neurons: index, quotient norm, full read norm, output norm, affine bias
1140 [3.561 4.336 2.905 0.026]
461 [ 3.408  4.182  2.859 -0.055]
1046 [ 3.264  3.573  2.307 -0.616]
468 [ 2.934  3.795  2.335 -0.753]
477 [2.898 3.528 2.653 0.02 ]
1107 [ 2.81   2.928  2.13  -0.064]
1214 [ 2.796  3.711  2.29  -0.502]
418 [ 2.79   3.75   2.462 -0.684]
827 [ 2.768  3.703  2.421 -0.869]
1115 [ 2.767  3.735  2.27  -0.574]
466 [ 2.705  3.342  2.261 -0.472]
947 [ 2.697  2.887  2.161 -0.495]
356 [ 2.663  3.461  2.324 -0.49 ]
1132 [ 2.645  3.312  3.353 -0.37 ]
1181 [ 2.618  3.551  2.416 -0.673]
888 [ 2.566  3.37   2.22  -0.687]
720 [ 2.545  3.355  2.32  -0.712]
796 [ 2.545  3.238  2.198 -0.202]
808 [ 2.528  3.108  2.223 -0.482]
361 [ 2.502  3.283  2.248 -0.514]
1456 [ 2.475  3.339  2.394 -0.552]
1260 [ 2.469  2.737  2.152 -0.259]
1464 [2.455 2.614 2.074 0.12 ]
1203 [ 2.449  2.707  2.149 -0.267]
100 [2.424 3.319 2.817 0.035]
1054 [2.417 2.886 2.548 0.441]
1170 [ 2.411  3.056  2.546 -0.243]
727 [ 2.404  3.196  2.116 -0.65 ]
779 [ 2.397  3.296  2.544 -0.724]
1281 [ 2.372  2.777  2.14  -0.5  ]
883 [ 2.361  3.299  2.36  -0.373]
52 [ 2.356  3.087  2.124 -0.396]
519 [ 2.351  3.013  2.178 -0.519]
1055 [2.348 2.467 2.034 0.01 ]
222 [ 2.346  3.156  2.143 -0.686]
642 [ 2.327  3.062  2.207 -0.353]
637 [ 2.321  2.9    2.197 -0.474]
571 [ 2.31   3.096  2.491 -0.092]
585 [ 2.274  2.672  2.125 -0.556]
416 [ 2.272  2.77   2.52  -0.067]
1189 [ 2.27   3.063  2.241 -0.407]
1082 [ 2.25   2.787  2.103 -0.696]
97 [ 2.234  2.84   2.421 -0.26 ]
1008 [ 2.226  2.93   2.075 -0.33 ]
953 [ 2.216  2.322  1.992 -0.025]
11 [ 2.184  2.824  2.093 -0.81 ]
51 [ 2.166  2.786  2.353 -0.092]
1353 [ 2.162  2.883  2.224 -0.561]
1001 [ 2.16   2.816  2.169 -0.395]
413 [ 2.155  2.361  1.964 -0.05 ]
241 [ 2.153  2.708  2.216 -0.6  ]
1491 [ 2.153  2.756  2.099 -0.399]
1430 [ 2.143  2.751  2.517 -0.415]
510 [ 2.135  2.797  2.046 -0.415]
106 [2.128 2.273 1.835 0.148]
355 [ 2.128  2.772  2.052 -0.501]
975 [2.118 2.191 1.893 0.038]
35 [ 2.11   2.337  2.039 -0.181]
194 [ 2.106  2.557  2.337 -0.413]
759 [ 2.105  2.688  2.121 -0.792]
498 [ 2.105  2.811  2.275 -0.522]
256 [ 2.094  2.247  1.913 -0.046]
98 [ 2.089  3.665  4.795 -0.702]
748 [ 2.077  2.748  2.259 -0.339]
204 [ 2.072  2.331  2.026 -0.252]
787 [ 2.069  2.839  2.136 -0.357]
494 [ 2.068  2.605  2.392 -0.581]
564 [ 2.061  2.757  2.081 -0.4  ]
928 [ 2.046  2.562  2.086 -0.348]
1045 [ 2.037  2.671  2.011 -0.343]
1351 [ 2.032  2.549  2.067 -0.56 ]
40 [ 2.017  2.501  2.124 -0.179]
773 [2.014 2.14  1.903 0.305]
1302 [1.999 2.103 1.824 0.109]
1527 [ 1.996  2.191  1.999 -0.379]
Top-row nearest cosine: [0.411 0.257 0.957 0.542 0.123 0.647 0.916 0.833 0.633 0.444 0.979 0.164 0.449 0.571 0.518 0.449 0.613 0.916 0.979 0.946 0.633 0.981 0.634 0.981 0.486 0.371 0.613 0.447 0.833
 0.957 0.647 0.946 0.674 0.633 0.458 0.378 0.971 0.184 0.944 0.257]
Top-row smallest cosine: [-0.158 -0.076 -0.104 -0.104 -0.104 -0.243 -0.192 -0.073 -0.243 -0.077 -0.052 -0.073 -0.073 -0.114 -0.02  -0.094 -0.124 -0.241 -0.063 -0.139 -0.108 -0.092 -0.143 -0.071 -0.108
 -0.381 -0.139 -0.078 -0.06  -0.077 -0.109 -0.108 -0.064 -0.092 -0.066 -0.224 -0.081 -0.251 -0.1   -0.381]

ICA components: strongest 4 neuron coefficients; previous-read weight coefficient std; read-direction norm
0 [(100, 16.53), (605, 13.35), (1008, 11.88), (1131, 10.35)] prev_std 0.224 direction_norm 0.133
1 [(461, 23.33), (1391, 10.47), (1135, 9.86), (78, 8.52)] prev_std 0.226 direction_norm 0.134
2 [(787, 13.46), (1031, 12.85), (510, 11.02), (696, 10.86)] prev_std 0.226 direction_norm 0.123
3 [(571, 13.64), (168, 10.22), (411, 10.01), (505, 8.12)] prev_std 0.209 direction_norm 0.143
4 [(1464, 17.01), (566, 13.82), (796, 13.61), (1214, 13.15)] prev_std 0.158 direction_norm 0.143
5 [(1107, 21.19), (883, 14.25), (284, 14.09), (1348, 9.66)] prev_std 0.132 direction_norm 0.127
6 [(1302, 11.57), (1485, 8.38), (986, 7.29), (990, 6.89)] prev_std 0.198 direction_norm 0.185
7 [(474, 12.81), (575, 10.4), (1136, 10.09), (383, 9.74)] prev_std 0.199 direction_norm 0.128
8 [(1046, 21.67), (1281, 15.14), (585, 14.4), (827, 12.7)] prev_std 0.132 direction_norm 0.148
9 [(947, 16.67), (1527, 12.17), (311, 11.56), (494, 10.51)] prev_std 0.092 direction_norm 0.159
10 [(97, 15.83), (468, 15.67), (975, 14.78), (996, 9.38)] prev_std 0.188 direction_norm 0.136
11 [(35, 14.17), (617, 13.35), (642, 13.02), (1001, 12.35)] prev_std 0.097 direction_norm 0.137
12 [(888, 15.37), (1472, 11.63), (658, 11.54), (1451, 9.8)] prev_std 0.222 direction_norm 0.127
13 [(1170, 16.13), (720, 14.75), (1194, 12.23), (800, 10.67)] prev_std 0.131 direction_norm 0.135
14 [(1260, 16.54), (1203, 16.53), (204, 13.86), (51, 12.86)] prev_std 0.116 direction_norm 0.145
15 [(773, 14.61), (361, 12.95), (1206, 12.61), (52, 11.85)] prev_std 0.168 direction_norm 0.138
16 [(1486, 8.66), (1022, 8.57), (633, 8.25), (784, 7.46)] prev_std 0.244 direction_norm 0.143
17 [(466, 16.09), (808, 14.7), (637, 13.54), (283, 12.53)] prev_std 0.133 direction_norm 0.152
18 [(441, 12.99), (250, 12.51), (5, 11.22), (423, 11.04)] prev_std 0.214 direction_norm 0.136
19 [(256, 15.34), (125, 10.99), (689, 10.53), (1082, 10.46)] prev_std 0.157 direction_norm 0.138
20 [(1140, 21.81), (293, 11.15), (222, 10.17), (1290, 9.15)] prev_std 0.257 direction_norm 0.149
21 [(418, 17.48), (779, 13.69), (98, 11.02), (1171, 10.45)] prev_std 0.096 direction_norm 0.153
22 [(355, 11.7), (983, 11.42), (1309, 10.48), (676, 9.65)] prev_std 0.296 direction_norm 0.14
23 [(1115, 15.13), (413, 14.89), (142, 12.49), (971, 10.97)] prev_std 0.148 direction_norm 0.14
24 [(477, 18.66), (1407, 11.84), (1197, 10.88), (817, 10.3)] prev_std 0.22 direction_norm 0.134
25 [(106, 14.26), (727, 12.71), (1339, 9.53), (1258, 9.21)] prev_std 0.235 direction_norm 0.154
26 [(1055, 17.45), (953, 16.4), (1456, 14.23), (498, 12.07)] prev_std 0.14 direction_norm 0.132
27 [(356, 15.25), (1382, 13.34), (1491, 12
```

## Call 7

```python
import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4,suppress=True,linewidth=180)
def fold(a,g):
    b=np.asarray(a,dtype=float)*g[None,:]
    return b-b.mean(1,keepdims=True)

M,T={},{}
for l in range(4):
    for j in range(6):
        s=slice(64*j,64*(j+1));g=W[f"ln1_w.{l}"]
        q=fold(W[f"W_q.{l}"][s],g)
        k=fold(W[f"W_k.{l}"][s],g)
        v=fold(W[f"W_v.{l}"][s],g)
        M[l,j]=q.T@k
        T[l,j]=W[f"W_o.{l}"][:,s].astype(float)@v

clockparts=[]
for j in [0,1,4,5]:
    u,s,v=svd(T[0,j],full_matrices=False)
    clockparts.append(u[:,:3])
C,ss,_=svd(np.concatenate(clockparts,1),full_matrices=False)
print("Singular values of joint early low-rank write bases:",ss)
C=C[:,:4]

print("\nhead: QK-energy with key in early-write subspace; query in it; both in it; trace of restriction; positive/negative restricted eigenvalues")
for l in range(1,4):
    for j in range(6):
        m=M[l,j];r=C.T@m@C;f2=np.sum(m*m)
        er=np.linalg.eigvalsh((r+r.T)/2)
        print(f"L{l}H{j}",np.round([np.linalg.norm(m@C)**2/f2,
              np.linalg.norm(C.T@m)**2/f2,np.linalg.norm(r)**2/f2,
              np.trace(r)],4),np.round(er,4))

print("\nL1 QK pairwise cosines:")
print(np.array([[np.sum(M[1,i]*M[1,j])/(np.linalg.norm(M[1,i])*np.linalg.norm(M[1,j]))
                 for j in range(6)] for i in range(6)]))
print("\nL0H2 QK normality and alignment of left/right principal subspaces")
m=M[0,2];u,s,vh=svd(m,full_matrices=False)
print("commutator ratio",np.linalg.norm(m@m.T-m.T@m)/np.linalg.norm(m@m.T))
for r in [10,20,28]:
    print(r, np.linalg.norm(u[:,:r].T@vh[:r].T)**2/r)

# Nulls permute residual coordinates in one factor of a matrix product.
# These are randomized weight-only controls, not random network inputs.
rng=np.random.default_rng(1234)
print("\nInduction-composition coordinate-permutation controls")
for target,source in [((3,1),(0,2)),((3,1),(2,1)),((3,0),(0,2)),((1,2),(0,2))]:
    m=M[target];t=T[source]
    obs=np.sum(m*t.T)/np.linalg.norm(m@t)
    null=[]
    for k in range(120):
        p=rng.permutation(384)
        tt=t[p][:,p]
        null.append(np.sum(m*tt.T)/np.linalg.norm(m@tt))
    null=np.array(null)
    print(target,source,"observed",round(obs,4),
          "null mean/std/min/max",np.round([null.mean(),null.std(),null.min(),null.max()],4))

```

Output:

```
Singular values of joint early low-rank write bases: [1.9917 1.9891 1.909  0.4398 0.3321 0.2501 0.1443 0.1337 0.106  0.093  0.0665 0.0547]

head: QK-energy with key in early-write subspace; query in it; both in it; trace of restriction; positive/negative restricted eigenvalues
L1H0 [0.6509 0.1033 0.0867 1.1204] [-0.4367  0.0632  0.6241  0.8697]
L1H1 [0.6058 0.0974 0.082  1.4688] [-0.5222  0.0753  0.8956  1.02  ]
L1H2 [0.6398 0.093  0.0823 1.4563] [-0.6784  0.0676  0.9391  1.1279]
L1H3 [0.6391 0.0906 0.0771 1.47  ] [-0.5025  0.0785  0.8991  0.9949]
L1H4 [0.6945 0.0942 0.0809 1.5332] [-0.613   0.0532  0.8973  1.1957]
L1H5 [0.6304 0.1251 0.1039 1.5033] [-0.7259  0.0128  0.8635  1.3529]
L2H0 [0.1015 0.0353 0.0249 1.0841] [-0.1251  0.0438  0.4289  0.7365]
L2H1 [0.0556 0.0141 0.0078 0.6226] [-0.0085  0.0212  0.1629  0.447 ]
L2H2 [0.1973 0.0343 0.0247 0.7417] [-0.3549  0.0601  0.3838  0.6528]
L2H3 [0.1001 0.0465 0.0351 1.0201] [-0.2148 -0.0385  0.5609  0.7124]
L2H4 [0.103  0.0381 0.0277 0.9474] [-0.1126 -0.102   0.4773  0.6847]
L2H5 [0.0974 0.0395 0.028  0.9434] [-0.1171 -0.0418  0.4642  0.6382]
L3H0 [0.0167 0.0044 0.0008 0.3111] [-0.0644  0.066   0.111   0.1984]
L3H1 [0.0066 0.0071 0.0007 0.5959] [0.0305 0.0592 0.2224 0.2839]
L3H2 [0.0742 0.0184 0.0087 0.3316] [-0.0123  0.0031  0.1097  0.2311]
L3H3 [0.0715 0.0217 0.0121 0.4026] [-0.0123  0.0354  0.1667  0.2127]
L3H4 [0.0488 0.0101 0.0038 0.314 ] [-0.0285  0.0611  0.0715  0.21  ]
L3H5 [0.0138 0.0127 0.0006 0.035 ] [-0.0535 -0.0137  0.0286  0.0736]

L1 QK pairwise cosines:
[[1.     0.5776 0.416  0.4477 0.455  0.4643]
 [0.5776 1.     0.5498 0.4884 0.5561 0.5277]
 [0.416  0.5498 1.     0.6081 0.6673 0.5641]
 [0.4477 0.4884 0.6081 1.     0.6439 0.5627]
 [0.455  0.5561 0.6673 0.6439 1.     0.5774]
 [0.4643 0.5277 0.5641 0.5627 0.5774 1.    ]]

L0H2 QK normality and alignment of left/right principal subspaces
commutator ratio 0.7214927026994833
10 0.6351182952401924
20 0.779115232397931
28 0.7936113998310376

Induction-composition coordinate-permutation controls
(3, 1) (0, 2) observed 1.1734 null mean/std/min/max [-0.0358  0.049  -0.1684  0.0867]
(3, 1) (2, 1) observed 1.2504 null mean/std/min/max [-0.3537  0.0549 -0.5213 -0.1521]
(3, 0) (0, 2) observed -0.0424 null mean/std/min/max [-0.0205  0.0562 -0.1675  0.125 ]
(1, 2) (0, 2) observed 0.2528 null mean/std/min/max [ 0.0055  0.0552 -0.1588  0.1272]

```
