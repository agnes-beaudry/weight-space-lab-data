## Call 1

```python
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)

def t(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float64)

def spec(a):
    s = torch.linalg.svdvals(a)
    energy = s.square()
    rank90 = int(torch.searchsorted(torch.cumsum(energy, 0), .9 * energy.sum())) + 1
    return float(torch.linalg.norm(a)), float(s[0]), float(energy.sum()/energy[0]), rank90

print("All calculations below are on weights, not network activations.")
for l in range(4):
    print("\nLAYER", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        a = np.asarray(W[f"{name}.{l}"])
        print(name, "mean/std/min/max/norm",
              np.round([a.mean(), a.std(), a.min(), a.max(), np.linalg.norm(a)], 4))
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        print(name, "Frob/topSV/stableRank/rank90",
              tuple(round(x, 3) if isinstance(x, float) else x
                    for x in spec(t(W[f"{name}.{l}"]))))

print("\nHEADS: raw QK = Q.T @ K / 8; OV = O @ V")
print("QK: Frob, topSV, stableRank, rank90, trace, symmetric min/max eigenvalue")
print("OV: Frob, topSV, stableRank, rank90, trace, largest-real eigenvalues")
for l in range(4):
    for j in range(6):
        sl = slice(64*j, 64*(j+1))
        q = t(W[f"W_q.{l}"])[sl]
        k = t(W[f"W_k.{l}"])[sl]
        v = t(W[f"W_v.{l}"])[sl]
        o = t(W[f"W_o.{l}"])[:, sl]
        a = q.T @ k / 8
        b = o @ v
        ae = torch.linalg.eigvalsh((a+a.T)/2)
        be = torch.linalg.eigvals(v @ o)
        idx = torch.argsort(be.real, descending=True)[:4]
        print(f"L{l}H{j}",
              "QK", tuple(round(x, 3) if isinstance(x, float) else x for x in spec(a)),
              "tr", round(float(torch.trace(a)), 3),
              "sym", np.round([float(ae[0]), float(ae[-1])], 3),
              "OV", tuple(round(x, 3) if isinstance(x, float) else x for x in spec(b)),
              "tr", round(float(torch.trace(b)), 3),
              "eig", [(round(float(be[z].real), 3), round(float(be[z].imag), 3)) for z in idx])

```

Output:

```
All calculations below are on weights, not network activations.

LAYER 0
ln1_w mean/std/min/max/norm [0.2677 0.2147 0.0165 1.0134 6.7241]
ln1_b mean/std/min/max/norm [ 0.0007  0.0366 -0.1215  0.1418  0.7172]
ln2_w mean/std/min/max/norm [ 0.8168  0.3312  0.0468  1.6412 17.2723]
ln2_b mean/std/min/max/norm [ 0.018   0.0667 -0.5537  0.1663  1.353 ]
W_q Frob/topSV/stableRank/rank90 (22.704, 5.039, 20.305, 61)
W_k Frob/topSV/stableRank/rank90 (23.214, 6.204, 14.001, 60)
W_v Frob/topSV/stableRank/rank90 (18.197, 4.553, 15.972, 66)
W_o Frob/topSV/stableRank/rank90 (19.6, 6.678, 8.614, 75)
W_fc Frob/topSV/stableRank/rank90 (51.65, 18.11, 8.134, 87)
W_proj Frob/topSV/stableRank/rank90 (45.806, 10.932, 17.557, 135)

LAYER 1
ln1_w mean/std/min/max/norm [0.4754 0.0964 0.1979 1.2096 9.5046]
ln1_b mean/std/min/max/norm [ 0.022   0.1532 -1.0505  0.9102  3.0327]
ln2_w mean/std/min/max/norm [ 1.0111  0.1582  0.0139  1.4707 20.0543]
ln2_b mean/std/min/max/norm [ 0.0275  0.1442 -0.3744  0.3652  2.8769]
W_q Frob/topSV/stableRank/rank90 (31.676, 8.984, 12.43, 66)
W_k Frob/topSV/stableRank/rank90 (28.602, 10.967, 6.801, 68)
W_v Frob/topSV/stableRank/rank90 (28.469, 5.35, 28.31, 84)
W_o Frob/topSV/stableRank/rank90 (26.151, 4.19, 38.953, 119)
W_fc Frob/topSV/stableRank/rank90 (80.261, 20.796, 14.896, 93)
W_proj Frob/topSV/stableRank/rank90 (75.119, 7.625, 97.059, 280)

LAYER 2
ln1_w mean/std/min/max/norm [ 0.7343  0.0712  0.3445  0.9047 14.4574]
ln1_b mean/std/min/max/norm [ 0.0244  0.1315 -0.9417  0.7326  2.6214]
ln2_w mean/std/min/max/norm [ 1.275   0.1126  0.0775  1.5018 25.082 ]
ln2_b mean/std/min/max/norm [ 0.0188  0.1474 -0.4157  0.3635  2.911 ]
W_q Frob/topSV/stableRank/rank90 (37.483, 5.067, 54.718, 170)
W_k Frob/topSV/stableRank/rank90 (36.356, 8.293, 19.216, 169)
W_v Frob/topSV/stableRank/rank90 (34.746, 3.89, 79.785, 176)
W_o Frob/topSV/stableRank/rank90 (34.205, 4.442, 59.308, 190)
W_fc Frob/topSV/stableRank/rank90 (79.454, 14.829, 28.71, 252)
W_proj Frob/topSV/stableRank/rank90 (77.477, 10.095, 58.906, 272)

LAYER 3
ln1_w mean/std/min/max/norm [ 0.7694  0.0469  0.451   0.8778 15.1046]
ln1_b mean/std/min/max/norm [ 0.0102  0.0945 -0.5141  0.6285  1.8619]
ln2_w mean/std/min/max/norm [ 1.2552  0.0888  0.2619  1.4568 24.6574]
ln2_b mean/std/min/max/norm [ 0.014   0.1267 -0.5112  0.3808  2.4972]
W_q Frob/topSV/stableRank/rank90 (33.94, 4.154, 66.743, 186)
W_k Frob/topSV/stableRank/rank90 (33.221, 4.888, 46.201, 190)
W_v Frob/topSV/stableRank/rank90 (38.939, 4.697, 68.734, 179)
W_o Frob/topSV/stableRank/rank90 (39.017, 9.175, 18.083, 172)
W_fc Frob/topSV/stableRank/rank90 (78.189, 14.373, 29.593, 279)
W_proj Frob/topSV/stableRank/rank90 (85.089, 28.175, 9.12, 25)

HEADS: raw QK = Q.T @ K / 8; OV = O @ V
QK: Frob, topSV, stableRank, rank90, trace, symmetric min/max eigenvalue
OV: Frob, topSV, stableRank, rank90, trace, largest-real eigenvalues
L0H0 QK (0.807, 0.649, 1.545, 2) tr -0.529 sym [-0.482  0.207] OV (8.667, 6.898, 1.579, 4) tr 1.62 eig [(1.827, 0.0), (0.517, 0.0), (0.125, 0.058), (0.125, -0.058)]
L0H1 QK (4.067, 2.153, 3.568, 14) tr -5.152 sym [-2.094  0.759] OV (17.067, 4.703, 13.166, 20) tr -0.82 eig [(1.036, 0.0), (0.285, 0.0), (0.252, -0.117), (0.252, 0.117)]
L0H2 QK (1.136, 0.561, 4.101, 11) tr -0.964 sym [-0.356  0.328] OV (8.592, 4.786, 3.222, 7) tr -0.743 eig [(0.182, 0.101), (0.182, -0.101), (0.165, 0.0), (0.104, 0.152)]
L0H3 QK (1.434, 0.959, 2.233, 7) tr -1.144 sym [-0.563  0.406] OV (5.288, 2.168, 5.949, 12) tr -0.071 eig [(0.406, 0.0), (0.154, 0.0), (0.125, 0.0), (0.092, -0.036)]
L0H4 QK (0.097, 0.04, 5.859, 41) tr -0.059 sym [-0.027  0.013] OV (25.795, 18.815, 1.88, 3) tr 6.681 eig [(3.714, -0.514), (3.714, 0.514), (0.212, 0.2), (0.212, -0.2)]
L0H5 QK (5.785, 1.76, 10.809, 18) tr 6.516 sym [-0.532  1.477] OV (18.558, 5.61, 10.941, 17) tr -11.398 eig [(0.45, 0.0), (0.124, 0.52), (0.124, -0.52), (0.079, 0.0)]
L1H0 QK (2.471, 1.116, 4.901, 15) tr -1.587 sym [-0.688  0.475] OV (27.504, 9.927, 7.677, 16) tr -34.568 eig [(0.83
```

## Call 2

```python
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
P = torch.eye(384) - torch.ones((384,384))/384
M, B = {}, {}

print("Effective linear maps include LayerNorm gain and mean-centering only.")
print("No normalization of inputs, attention, or MLP evaluation is performed.")
for l in range(4):
    g = T(W[f"ln1_w.{l}"])
    b = T(W[f"ln1_b.{l}"])
    for j in range(6):
        s = slice(j*64,(j+1)*64)
        q, k, v = [T(W[f"{name}.{l}"])[s] for name in ["W_q","W_k","W_v"]]
        o = T(W[f"W_o.{l}"])[:,s]
        qe, ke, ve = (q*g)@P, (k*g)@P, (v*g)@P
        a, c = qe.T@ke/8, P@o@ve
        M[l,j], B[l,j] = a,c
        sv = torch.linalg.svdvals(a)
        cs = torch.linalg.svdvals(c)
        keybias = ke.T@(q@b)/8
        print(f"L{l}H{j}",
              "QK_F/top/r90/tr",
              np.round([float(a.norm()),float(sv[0]),
                        int(torch.searchsorted(sv.square().cumsum(0),.9*sv.square().sum()))+1,
                        float(torch.trace(a))],4),
              "QK_svs", np.round(sv[:6].numpy(),3),
              "OV_F/tr",np.round([float(c.norm()),float(torch.trace(c))],3),
              "key_bias_norm",round(float(keybias.norm()),4))

print("\nWithin-layer QK cosine matrices:")
for l in range(4):
    print("L",l)
    print(np.round([[float((M[l,i]*M[l,j]).sum()/(M[l,i].norm()*M[l,j].norm()))
                     for j in range(6)] for i in range(6)],3))

print("\nKey-composition trace cosine: trace(QK_later @ OV_earlier)/(norms)")
for l in range(1,4):
    for j in range(6):
        a=M[l,j]
        scores=[]
        for p in range(l):
            for h in range(6):
                c=B[p,h]
                z=float(torch.trace(a@c)/(a.norm()*c.norm()))
                scores.append((z,f"L{p}H{h}"))
        scores.sort(reverse=True)
        print(f"L{l}H{j}", "highest",[(n,round(z,4)) for z,n in scores[:4]],
              "lowest",[(n,round(z,4)) for z,n in scores[-2:]])

print("\nL0 gain extremes and corresponding linear-map column norms:")
g=np.asarray(W["ln1_w.0"])
idx=np.r_[np.argsort(g)[:12],np.argsort(g)[-12:]]
for i in idx:
    print(int(i), "g1/g2",round(float(g[i]),4),round(float(W["ln2_w.0"][i]),4),
          "q/k/v/fc", np.round([np.linalg.norm(W[n+".0"][:,i])
                                 for n in ["W_q","W_k","W_v","W_fc"]],3))

print("\nLinear MLP weight products (not MLP evaluations):")
for l in range(4):
    a=P@T(W[f"W_proj.{l}"])@(T(W[f"W_fc.{l}"])*T(W[f"ln2_w.{l}"]))@P
    ev=torch.linalg.eigvals(a)
    print(l,"Frob/trace",round(float(a.norm()),3),round(float(torch.trace(a)),3),
          "eig real min/max",round(float(ev.real.min()),3),round(float(ev.real.max()),3))

```

Output:

```
Effective linear maps include LayerNorm gain and mean-centering only.
No normalization of inputs, attention, or MLP evaluation is performed.
L0H0 QK_F/top/r90/tr [ 0.1137  0.0887  2.     -0.0508] QK_svs [0.089 0.068 0.013 0.008 0.006 0.005] OV_F/tr [4.321 1.666] key_bias_norm 0.0185
L0H1 QK_F/top/r90/tr [ 0.8775  0.6013  6.     -0.4868] QK_svs [0.601 0.373 0.297 0.211 0.199 0.155] OV_F/tr [3.488 0.164] key_bias_norm 0.0736
L0H2 QK_F/top/r90/tr [0.1745 0.132  5.     0.0546] QK_svs [0.132 0.08  0.044 0.031 0.028 0.025] OV_F/tr [2.023 0.015] key_bias_norm 0.0234
L0H3 QK_F/top/r90/tr [ 0.2395  0.1645  3.     -0.1871] QK_svs [0.165 0.14  0.081 0.031 0.029 0.021] OV_F/tr [1.621 0.242] key_bias_norm 0.057
L0H4 QK_F/top/r90/tr [ 0.0093  0.0045 28.     -0.0037] QK_svs [0.004 0.003 0.002 0.002 0.002 0.002] OV_F/tr [17.158  6.049] key_bias_norm 0.0009
L0H5 QK_F/top/r90/tr [ 0.9704  0.4099 16.      1.2135] QK_svs [0.41  0.362 0.273 0.254 0.245 0.222] OV_F/tr [ 3.707 -2.463] key_bias_norm 0.0509
L1H0 QK_F/top/r90/tr [ 0.644   0.3538 13.     -0.308 ] QK_svs [0.354 0.252 0.218 0.167 0.158 0.139] OV_F/tr [ 13.436 -16.299] key_bias_norm 0.8392
L1H1 QK_F/top/r90/tr [ 1.5843  1.1456  9.     -1.2729] QK_svs [1.146 0.661 0.459 0.326 0.249 0.215] OV_F/tr [  9.592 -11.205] key_bias_norm 3.1498
L1H2 QK_F/top/r90/tr [ 1.2358  0.8069  8.     -0.7401] QK_svs [0.807 0.693 0.332 0.205 0.18  0.159] OV_F/tr [ 11.496 -14.892] key_bias_norm 2.1998
L1H3 QK_F/top/r90/tr [ 1.3094  0.9127  8.     -0.6948] QK_svs [0.913 0.699 0.309 0.202 0.178 0.167] OV_F/tr [ 10.043 -10.571] key_bias_norm 2.7194
L1H4 QK_F/top/r90/tr [ 1.2492  0.848   8.     -0.5846] QK_svs [0.848 0.654 0.34  0.213 0.191 0.162] OV_F/tr [  9.557 -10.446] key_bias_norm 2.518
L1H5 QK_F/top/r90/tr [ 1.4614  1.0088  8.     -1.2054] QK_svs [1.009 0.555 0.482 0.45  0.242 0.215] OV_F/tr [  9.826 -13.045] key_bias_norm 2.4602
L2H0 QK_F/top/r90/tr [ 2.0456  0.5184 39.     -0.5814] QK_svs [0.518 0.485 0.459 0.416 0.407 0.397] OV_F/tr [ 18.698 -35.97 ] key_bias_norm 0.8945
L2H1 QK_F/top/r90/tr [ 1.9454  0.4582 39.      2.5132] QK_svs [0.458 0.421 0.396 0.393 0.367 0.355] OV_F/tr [ 21.448 -40.225] key_bias_norm 0.6377
L2H2 QK_F/top/r90/tr [ 2.804   0.7089 31.      1.4941] QK_svs [0.709 0.693 0.676 0.634 0.611 0.598] OV_F/tr [ 16.782 -29.462] key_bias_norm 1.5836
L2H3 QK_F/top/r90/tr [ 1.9759  0.5708 39.     -0.4059] QK_svs [0.571 0.423 0.412 0.392 0.388 0.37 ] OV_F/tr [ 20.617 -38.568] key_bias_norm 1.0368
L2H4 QK_F/top/r90/tr [ 2.2016  0.7827 37.      0.0551] QK_svs [0.783 0.587 0.484 0.429 0.417 0.401] OV_F/tr [ 18.415 -38.753] key_bias_norm 1.1994
L2H5 QK_F/top/r90/tr [ 1.6541  0.4194 42.     -0.7004] QK_svs [0.419 0.414 0.359 0.32  0.315 0.296] OV_F/tr [ 23.766 -33.94 ] key_bias_norm 0.6361
L3H0 QK_F/top/r90/tr [ 2.0777  0.4054 48.      1.7527] QK_svs [0.405 0.399 0.385 0.377 0.365 0.359] OV_F/tr [ 24.502 -48.319] key_bias_norm 0.2987
L3H1 QK_F/top/r90/tr [ 1.6948  0.3223 48.      2.225 ] QK_svs [0.322 0.315 0.309 0.304 0.3   0.295] OV_F/tr [ 25.724 -49.613] key_bias_norm 0.2634
L3H2 QK_F/top/r90/tr [ 2.0608  0.4337 48.      0.7176] QK_svs [0.434 0.411 0.392 0.362 0.358 0.353] OV_F/tr [ 26.86  -50.565] key_bias_norm 0.3422
L3H3 QK_F/top/r90/tr [ 1.053   0.2624 47.      0.0028] QK_svs [0.262 0.237 0.208 0.202 0.193 0.191] OV_F/tr [ 37.938 -51.153] key_bias_norm 0.2583
L3H4 QK_F/top/r90/tr [ 1.7344  0.4007 47.      0.0047] QK_svs [0.401 0.369 0.357 0.323 0.313 0.306] OV_F/tr [ 28.44  -35.013] key_bias_norm 0.3067
L3H5 QK_F/top/r90/tr [ 2.1169  0.4041 48.      8.0447] QK_svs [0.404 0.39  0.384 0.37  0.365 0.362] OV_F/tr [ 23.687 -56.843] key_bias_norm 0.3101

Within-layer QK cosine matrices:
L 0
[[ 1.     0.179  0.112  0.206  0.063  0.093]
 [ 0.179  1.     0.21  -0.15   0.005  0.414]
 [ 0.112  0.21   1.    -0.247 -0.006  0.114]
 [ 0.206 -0.15  -0.247  1.    -0.027 -0.098]
 [ 0.063  0.005 -0.006 -0.027  1.    -0.004]
 [ 0.093  0.414  0.114 -0.098 -0.004  1.   ]]
L 1
[[1.    0.302 0.393 0.327 0.338 0.266]
 [0.3
```

## Call 3

```python
import numpy as np
import torch
from sklearn.cluster import KMeans
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
T=lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
P=torch.eye(384)-torch.ones((384,384))/384
M,B={},{}

for l in range(4):
    g=T(W[f"ln1_w.{l}"])
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        q,k,v=[(T(W[f"{n}.{l}"])[s]*g)@P for n in ["W_q","W_k","W_v"]]
        o=P@T(W[f"W_o.{l}"])[:,s]
        M[l,j]=q.T@k/8
        B[l,j]=o@v

print("L0 effective OV singular spectra (first 32):")
for j in range(6):
    print(j,np.round(torch.linalg.svdvals(B[0,j])[:32].numpy(),3))
print("\nL0H5 symmetric-QK eigenvalues, descending:")
e=torch.linalg.eigvalsh((M[0,5]+M[0,5].T)/2).flip(0)
print(np.round(e[:45].numpy(),4))
print("Smallest:",np.round(e[-15:].numpy(),4))

print("\nQK energy reading leading 3 OUTPUT directions of each L0 OV:")
U={}
for j in range(6):
    U[j]=torch.linalg.svd(B[0,j],full_matrices=False).U[:,:3]
for l in range(1,4):
    for j in range(6):
        a=M[l,j]
        print(f"L{l}H{j}",
              "query",np.round([float((u.T@a).norm().square()/a.norm().square())
                                for u in U.values()],3),
              "key",np.round([float((a@u).norm().square()/a.norm().square())
                              for u in U.values()],3))

print("\nSelected key-composition spectra, on weight products only:")
for later,earlier in [((2,1),(0,5)),((3,1),(2,2)),((3,5),(2,2)),
                      ((1,4),(0,4)),((2,2),(0,1)),((2,2),(1,2))]:
    a=M[later]@B[earlier]
    ev=torch.linalg.eigvals(a)
    se=torch.linalg.eigvalsh((a+a.T)/2)
    sv=torch.linalg.svdvals(a)
    print(later,earlier,"trace/Frob/nuclear",
          np.round([float(torch.trace(a)),float(a.norm()),float(sv.sum())],4),
          "sym_positive_fraction",round(float(se.clamp(min=0).sum()/se.abs().sum()),4),
          "eig_top_real",np.round(torch.sort(ev.real,descending=True).values[:12].numpy(),4))

print("\nFirst MLP row-weight geometry:")
f=(T(W["W_fc.0"])*T(W["ln2_w.0"]))@P
u,s,vh=torch.linalg.svd(f,full_matrices=False)
print("effective FC0 singular values (first 65)",np.round(s[:65].numpy(),3))
# Whitening and clustering weight rows, not evaluating neurons.
for r in [24,32,48]:
    z=u[:,:r].numpy()
    z=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12)
    for k in [20,24,28,32,40]:
        km=KMeans(n_clusters=k,n_init=5,random_state=0).fit(z)
        centers=km.cluster_centers_
        norms=np.linalg.norm(centers,axis=1)
        counts=np.bincount(km.labels_,minlength=k)
        print("rank",r,"k",k,"mean within-cluster cosine",
              round(float(np.average(norms,weights=counts)),3),
              "coherent clusters (centroid norm > .65)",
              sorted([(int(counts[i]),round(float(norms[i]),3))
                      for i in range(k) if norms[i]>.65],reverse=True))

```

Output:

```
L0 effective OV singular spectra (first 32):
0 [3.424 2.469 0.529 0.401 0.346 0.25  0.205 0.199 0.178 0.135 0.114 0.101 0.094 0.086 0.079 0.073 0.07  0.06  0.057 0.053 0.051 0.048 0.046 0.042
 0.041 0.039 0.037 0.035 0.033 0.03  0.029 0.027]
1 [0.996 0.923 0.89  0.862 0.836 0.815 0.803 0.767 0.731 0.71  0.698 0.669 0.648 0.64  0.626 0.6   0.586 0.572 0.521 0.517 0.492 0.47  0.441 0.358
 0.349 0.311 0.299 0.257 0.247 0.208 0.201 0.172]
2 [1.055 0.901 0.807 0.66  0.537 0.47  0.373 0.347 0.277 0.236 0.21  0.163 0.133 0.12  0.105 0.097 0.089 0.081 0.073 0.068 0.061 0.058 0.053 0.051
 0.048 0.041 0.038 0.036 0.035 0.034 0.033 0.029]
3 [1.025 0.558 0.459 0.438 0.353 0.341 0.325 0.292 0.277 0.259 0.232 0.213 0.185 0.15  0.142 0.132 0.126 0.11  0.097 0.093 0.079 0.077 0.067 0.062
 0.056 0.055 0.047 0.045 0.043 0.038 0.036 0.036]
4 [13.877  9.644  2.507  0.968  0.594  0.554  0.483  0.433  0.345  0.295  0.28   0.228  0.205  0.187  0.16   0.126  0.111  0.091  0.079  0.074  0.069
  0.068  0.063  0.061  0.05   0.047  0.044  0.042  0.039  0.038  0.033  0.033]
5 [1.059 1.044 0.96  0.948 0.909 0.899 0.855 0.836 0.817 0.77  0.753 0.731 0.709 0.676 0.656 0.646 0.604 0.552 0.533 0.454 0.42  0.346 0.332 0.307
 0.299 0.292 0.274 0.259 0.237 0.227 0.223 0.216]

L0H5 symmetric-QK eigenvalues, descending:
[0.2734 0.2195 0.2099 0.1954 0.1774 0.1566 0.1421 0.1202 0.1088 0.1023 0.0519 0.0495 0.0375 0.0297 0.029  0.0158 0.0133 0.0126 0.0107 0.0101 0.0092
 0.0083 0.0074 0.0065 0.0058 0.005  0.0048 0.0038 0.0034 0.0033 0.0031 0.0025 0.0024 0.0021 0.002  0.0019 0.0016 0.0015 0.0013 0.0012 0.0012 0.0011
 0.0009 0.0009 0.0009]
Smallest: [-0.0174 -0.0197 -0.0224 -0.0268 -0.031  -0.0334 -0.0414 -0.0426 -0.046  -0.0496 -0.0502 -0.0549 -0.0642 -0.0776 -0.0826]

QK energy reading leading 3 OUTPUT directions of each L0 OV:
L1H0 query [0.043 0.021 0.035 0.022 0.061 0.015] key [0.272 0.012 0.012 0.051 0.342 0.012]
L1H1 query [0.071 0.008 0.008 0.024 0.107 0.011] key [0.521 0.002 0.017 0.07  0.627 0.008]
L1H2 query [0.063 0.009 0.009 0.027 0.108 0.014] key [0.489 0.003 0.017 0.072 0.631 0.009]
L1H3 query [0.08  0.004 0.012 0.041 0.137 0.008] key [0.546 0.003 0.023 0.085 0.678 0.007]
L1H4 query [0.075 0.006 0.008 0.031 0.119 0.019] key [0.516 0.003 0.024 0.079 0.645 0.013]
L1H5 query [0.05  0.007 0.016 0.019 0.084 0.032] key [0.459 0.003 0.014 0.051 0.525 0.017]
L2H0 query [0.009 0.008 0.009 0.008 0.01  0.01 ] key [0.046 0.009 0.011 0.012 0.051 0.014]
L2H1 query [0.008 0.007 0.012 0.006 0.005 0.019] key [0.007 0.012 0.011 0.005 0.005 0.02 ]
L2H2 query [0.013 0.007 0.006 0.01  0.014 0.023] key [0.057 0.006 0.01  0.015 0.07  0.019]
L2H3 query [0.018 0.009 0.009 0.01  0.022 0.008] key [0.077 0.008 0.01  0.02  0.095 0.013]
L2H4 query [0.018 0.008 0.006 0.012 0.023 0.009] key [0.079 0.006 0.027 0.02  0.095 0.011]
L2H5 query [0.009 0.007 0.016 0.007 0.006 0.008] key [0.039 0.007 0.012 0.009 0.041 0.009]
L3H0 query [0.004 0.006 0.005 0.005 0.002 0.01 ] key [0.016 0.007 0.011 0.006 0.017 0.01 ]
L3H1 query [0.005 0.005 0.008 0.007 0.003 0.011] key [0.01  0.008 0.01  0.006 0.009 0.016]
L3H2 query [0.004 0.007 0.01  0.006 0.003 0.01 ] key [0.022 0.007 0.01  0.007 0.025 0.01 ]
L3H3 query [0.006 0.007 0.013 0.008 0.005 0.005] key [0.01  0.006 0.01  0.006 0.01  0.008]
L3H4 query [0.01  0.007 0.012 0.006 0.004 0.006] key [0.025 0.008 0.01  0.007 0.025 0.008]
L3H5 query [0.005 0.01  0.007 0.006 0.003 0.022] key [0.006 0.015 0.01  0.006 0.005 0.023]

Selected key-composition spectra, on weight products only:
(2, 1) (0, 5) trace/Frob/nuclear [0.5595 0.549  2.6366] sym_positive_fraction 0.6084 eig_top_real [0.0559 0.0559 0.0505 0.0501 0.0501 0.0379 0.0379 0.0331 0.0331 0.0242 0.0242 0.0225]
(3, 1) (2, 2) trace/Frob/nuclear [ 3.1409  1.7435 10.9188] sym_positive_fraction 0.648 eig_top_real [0.2974 0.1824 0.1824 0.1799 0.1725 0.1725 0.116  0.1148 0.1148 0.1126 0.1126 0.0976]
(3, 5) (2, 2) trace/Frob/nuclear [ 6.521   2.6779 16.7035] sym_positive_fraction 0.7003 eig_top_real [0.4    0.4    
```

## Call 4

```python
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
T=lambda a: torch.as_tensor(np.asarray(a),dtype=torch.float64)
P=torch.eye(384)-torch.ones((384,384))/384
M,B={},{}

for l in range(4):
    g=T(W[f"ln1_w.{l}"])
    for j in range(6):
        s=slice(64*j,64*(j+1))
        q,k,v=[(T(W[f"{n}.{l}"])[s]*g)@P for n in ["W_q","W_k","W_v"]]
        M[l,j]=q.T@k/8
        B[l,j]=P@T(W[f"W_o.{l}"])[:,s]@v

print("QK spectral structure:")
for l in range(4):
    for j in range(6):
        a=M[l,j]
        u,s,vh=torch.linalg.svd(a)
        ev=torch.linalg.eigvals(a)
        ev=ev[torch.argsort(ev.abs(),descending=True)[:8]]
        overlaps=torch.linalg.svdvals(u[:,:6].T@vh[:6].T)
        print(f"L{l}H{j}",
              "symmetric energy",round(float(((a+a.T)/2).norm().square()/a.norm().square()),3),
              "top6 Q/K-subspace overlaps",np.round(overlaps.numpy(),2),
              "eigenvalues",[(round(float(z.real),3),round(float(z.imag),3)) for z in ev])

u,s,vh=torch.linalg.svd(B[0,4])
r=vh[:3].T
print("\nL0H4 OV top-3 INPUT subspace and other L0 QK maps:")
for j in range(6):
    a=M[0,j]
    print(j,"query/key energy",
          np.round([float((r.T@a).norm().square()/a.norm().square()),
                    float((a@r).norm().square()/a.norm().square())],4),
          "3x3 restriction",np.round((r.T@a@r).numpy(),4))

print("\nDominant effective L0 OV input-subspace similarities (3 dims):")
rs=[torch.linalg.svd(B[0,j]).Vh[:3].T for j in range(6)]
print(np.round([[float((x.T@y).square().sum()/3) for y in rs] for x in rs],3))

print("\nJoint QK covariance spectra:")
for l in range(4):
    c=sum(a.T@a+a@a.T for (ll,j),a in M.items() if ll==l)
    e=torch.linalg.eigvalsh(c).flip(0)
    print(l,np.round((e[:45]/e.sum()).numpy(),4))

print("\nL0 FC input and projection output mean directions:")
f=(T(W["W_fc.0"])*T(W["ln2_w.0"]))@P
o=P@T(W["W_proj.0"])
fn=f/f.norm(dim=1,keepdim=True)
on=o/o.norm(dim=0,keepdim=True)
print("FC row norm quantiles",np.round(np.quantile(f.norm(dim=1).numpy(),[0,.1,.25,.5,.75,.9,1]),3))
print("FC nearest-other-row cosine quantiles",
      np.round(np.quantile((fn@fn.T-torch.eye(1536)*2).max(dim=1).values.numpy(),
                           [0,.1,.25,.5,.75,.9,1]),3))
print("Per-neuron paired input/output cosine quantiles",
      np.round(np.quantile((fn*on.T).sum(1).numpy(),[0,.1,.25,.5,.75,.9,1]),3))

```

Output:

```
QK spectral structure:
L0H0 symmetric energy 0.591 top6 Q/K-subspace overlaps [0.69 0.43 0.38 0.34 0.26 0.14] eigenvalues [(-0.049, 0.0), (0.006, 0.009), (0.006, -0.009), (-0.003, 0.0), (-0.001, -0.002), (-0.001, 0.002), (-0.002, -0.0), (-0.002, 0.0)]
L0H1 symmetric energy 0.45 top6 Q/K-subspace overlaps [0.97 0.96 0.95 0.91 0.82 0.63] eigenvalues [(0.218, -0.345), (0.218, 0.345), (-0.284, 0.0), (-0.123, 0.15), (-0.123, -0.15), (-0.097, 0.098), (-0.097, -0.098), (-0.032, 0.041)]
L0H2 symmetric energy 0.701 top6 Q/K-subspace overlaps [0.9  0.82 0.65 0.59 0.2  0.05] eigenvalues [(0.091, 0.0), (0.053, 0.0), (-0.022, 0.0), (-0.013, 0.001), (-0.013, -0.001), (0.011, 0.0), (-0.01, 0.0), (-0.006, 0.004)]
L0H3 symmetric energy 0.605 top6 Q/K-subspace overlaps [0.82 0.78 0.61 0.43 0.38 0.1 ] eigenvalues [(-0.078, -0.005), (-0.078, 0.005), (-0.02, 0.0), (0.006, 0.017), (0.006, -0.017), (0.007, 0.0), (-0.006, -0.002), (-0.006, 0.002)]
L0H4 symmetric energy 0.512 top6 Q/K-subspace overlaps [0.5  0.35 0.29 0.2  0.11 0.01] eigenvalues [(-0.001, -0.001), (-0.001, 0.001), (-0.001, 0.0), (-0.001, -0.0), (0.0, 0.001), (0.0, -0.001), (-0.001, 0.0), (0.001, 0.0)]
L0H5 symmetric energy 0.386 top6 Q/K-subspace overlaps [0.93 0.91 0.85 0.79 0.67 0.52] eigenvalues [(0.207, -0.275), (0.207, 0.275), (0.114, -0.172), (0.114, 0.172), (0.178, 0.0), (0.155, 0.084), (0.155, -0.084), (0.035, 0.166)]
L1H0 symmetric energy 0.51 top6 Q/K-subspace overlaps [0.43 0.38 0.27 0.14 0.08 0.04] eigenvalues [(0.028, -0.051), (0.028, 0.051), (-0.051, -0.023), (-0.051, 0.023), (-0.051, 0.0), (-0.041, 0.013), (-0.041, -0.013), (-0.042, 0.0)]
L1H1 symmetric energy 0.498 top6 Q/K-subspace overlaps [0.69 0.54 0.43 0.37 0.29 0.21] eigenvalues [(0.08, -0.26), (0.08, 0.26), (-0.144, -0.009), (-0.144, 0.009), (-0.092, 0.079), (-0.092, -0.079), (-0.102, 0.0), (-0.099, 0.019)]
L1H2 symmetric energy 0.492 top6 Q/K-subspace overlaps [0.4  0.36 0.32 0.27 0.17 0.15] eigenvalues [(0.057, -0.202), (0.057, 0.202), (-0.108, 0.0), (-0.07, 0.05), (-0.07, -0.05), (-0.079, 0.004), (-0.079, -0.004), (-0.064, 0.0)]
L1H3 symmetric energy 0.478 top6 Q/K-subspace overlaps [0.51 0.4  0.36 0.33 0.22 0.06] eigenvalues [(0.076, -0.272), (0.076, 0.272), (-0.094, 0.0), (0.092, 0.0), (-0.08, 0.024), (-0.08, -0.024), (-0.077, 0.0), (-0.072, 0.0)]
L1H4 symmetric energy 0.481 top6 Q/K-subspace overlaps [0.52 0.38 0.37 0.22 0.12 0.  ] eigenvalues [(0.071, -0.239), (0.071, 0.239), (-0.085, -0.014), (-0.085, 0.014), (-0.078, 0.0), (-0.065, 0.028), (-0.065, -0.028), (-0.07, 0.0)]
L1H5 symmetric energy 0.505 top6 Q/K-subspace overlaps [0.53 0.44 0.41 0.29 0.14 0.11] eigenvalues [(0.072, -0.204), (0.072, 0.204), (-0.156, 0.0), (-0.124, 0.0), (-0.083, -0.053), (-0.083, 0.053), (-0.08, -0.03), (-0.08, 0.03)]
L2H0 symmetric energy 0.499 top6 Q/K-subspace overlaps [0.29 0.19 0.1  0.08 0.06 0.02] eigenvalues [(0.033, -0.132), (0.033, 0.132), (-0.117, 0.0), (0.091, -0.074), (0.091, 0.074), (0.037, 0.104), (0.037, -0.104), (-0.103, -0.031)]
L2H1 symmetric energy 0.52 top6 Q/K-subspace overlaps [0.34 0.32 0.26 0.17 0.12 0.02] eigenvalues [(0.142, 0.036), (0.142, -0.036), (0.142, 0.0), (0.141, 0.011), (0.141, -0.011), (0.131, 0.022), (0.131, -0.022), (0.103, 0.079)]
L2H2 symmetric energy 0.516 top6 Q/K-subspace overlaps [0.28 0.26 0.24 0.2  0.09 0.04] eigenvalues [(0.259, 0.0), (0.08, 0.216), (0.08, -0.216), (0.193, 0.058), (0.193, -0.058), (0.195, -0.026), (0.195, 0.026), (-0.172, 0.022)]
L2H3 symmetric energy 0.494 top6 Q/K-subspace overlaps [0.43 0.34 0.17 0.12 0.05 0.  ] eigenvalues [(0.058, -0.185), (0.058, 0.185), (0.002, 0.106), (0.002, -0.106), (-0.055, -0.085), (-0.055, 0.085), (0.097, -0.026), (0.097, 0.026)]
L2H4 symmetric energy 0.5 top6 Q/K-subspace overlaps [0.32 0.26 0.22 0.17 0.07 0.01] eigenvalues [(0.08, -0.22), (0.08, 0.22), (0.154, 0.0), (-0.121, 0.0), (0.086, 0.077), (0.086, -0.077), (-0.111, 0.028), (-0.111, -0.028)]
L2H5 symmetric energy 0.497 top6 Q/K-subspace overlaps [0.23 0.17 0.14 0.09 0.04 0.02
```

## Call 5

```python
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
T=lambda a: torch.as_tensor(np.asarray(a),dtype=torch.float64)
P=torch.eye(384)-torch.ones((384,384))/384

print("Output-side singular spectra; possible alphabet-sized bottlenecks:")
for l in range(4):
    for name in ["W_proj","W_o"]:
        a=P@T(W[f"{name}.{l}"])
        s=torch.linalg.svdvals(a)
        print(name,l,"first 65",np.round(s[:65].numpy(),3))
        ratios=s[:-1]/s[1:]
        ids=torch.argsort(ratios[:80],descending=True)[:8]
        print("largest early gaps (rank, ratio)",
              [(int(i)+1,round(float(ratios[i]),3)) for i in ids])

a=torch.cat([P@T(W["W_o.3"]),P@T(W["W_proj.3"])],dim=1)
u,s,vh=torch.linalg.svd(a,full_matrices=False)
print("\nCombined final-block outputs first 65 SVs",np.round(s[:65].numpy(),3))

print("\nMLP weight-row norm distributions and low-norm fractions:")
for l in range(4):
    f=T(W[f"W_fc.{l}"])
    o=T(W[f"W_proj.{l}"])
    fn=f.norm(dim=1).numpy()
    on=o.norm(dim=0).numpy()
    print(l,"FC norms",np.round(np.quantile(fn,[0,.1,.25,.5,.75,.9,.99,1]),3),
          "Proj norms",np.round(np.quantile(on,[0,.1,.25,.5,.75,.9,.99,1]),3),
          "FC < .6",round(float((fn<.6).mean()),3))

print("\nL0 positional-versus-value subspace separation:")
g=T(W["ln1_w.0"])
q=(T(W["W_q.0"])*g)@P
k=(T(W["W_k.0"])*g)@P
v=(T(W["W_v.0"])*g)@P
uq,sq,vq=torch.linalg.svd(torch.cat([q,k],dim=0),full_matrices=False)
uv,sv,vv=torch.linalg.svd(v,full_matrices=False)
print("QK input singular spectrum",np.round(sq[:60].numpy(),3))
print("V input singular spectrum",np.round(sv[:60].numpy(),3))
for r in [2,4,8,16,24,28,32,48]:
    z=torch.linalg.svdvals(vq[:r]@vv[:r].T)
    print("rank",r,"subspace overlaps",np.round(z.numpy(),3))

```

Output:

```
Output-side singular spectra; possible alphabet-sized bottlenecks:
W_proj 0 first 65 [10.931  7.576  6.628  6.293  6.166  5.924  5.886  5.768  5.706  5.61   5.495  5.462  5.404  5.313  5.193  5.161  5.119  5.07   5.003  4.937  4.877
  4.793  4.692  4.664  4.651  4.637  4.548  4.526  4.487  4.404  4.329  4.283  4.246  4.206  4.178  4.141  4.12   4.105  4.061  4.023  3.997  3.958
  3.934  3.904  3.857  3.85   3.797  3.758  3.709  3.694  3.668  3.622  3.618  3.591  3.573  3.516  3.476  3.445  3.421  3.387  3.363  3.343  3.335
  3.268  3.248]
largest early gaps (rank, ratio) [(1, 1.443), (2, 1.143), (3, 1.053), (5, 1.041), (77, 1.025), (14, 1.023), (72, 1.022), (22, 1.021)]
W_o 0 first 65 [6.676 5.235 2.834 2.679 2.619 2.581 2.549 2.515 2.503 2.477 2.449 2.443 2.424 2.404 2.365 2.361 2.319 2.291 2.269 2.259 2.243 2.217 2.2   2.196
 2.173 2.14  2.13  2.116 2.099 2.086 2.039 2.029 2.012 1.995 1.979 1.972 1.934 1.923 1.885 1.876 1.829 1.823 1.816 1.8   1.751 1.736 1.711 1.692
 1.654 1.642 1.613 1.608 1.589 1.572 1.551 1.536 1.498 1.483 1.474 1.437 1.415 1.384 1.358 1.355 1.346]
largest early gaps (rank, ratio) [(2, 1.847), (1, 1.275), (3, 1.058), (44, 1.028), (80, 1.028), (65, 1.027), (40, 1.026), (59, 1.025)]
W_proj 1 first 65 [7.625 7.359 6.907 6.489 6.344 6.223 6.192 6.02  5.949 5.902 5.82  5.756 5.728 5.692 5.674 5.65  5.633 5.608 5.581 5.567 5.544 5.523 5.479 5.457
 5.45  5.408 5.404 5.39  5.374 5.341 5.338 5.309 5.285 5.267 5.257 5.223 5.197 5.193 5.179 5.159 5.154 5.127 5.125 5.112 5.094 5.083 5.082 5.071
 5.055 5.045 5.022 5.014 5.007 4.985 4.975 4.958 4.944 4.929 4.923 4.908 4.894 4.885 4.878 4.863 4.848]
largest early gaps (rank, ratio) [(2, 1.065), (3, 1.064), (1, 1.036), (7, 1.029), (4, 1.023), (5, 1.019), (10, 1.014), (8, 1.012)]
W_o 1 first 65 [4.19  3.651 3.385 3.352 3.24  3.216 3.141 3.115 3.109 3.086 3.031 3.005 2.981 2.95  2.917 2.902 2.877 2.862 2.85  2.829 2.805 2.797 2.78  2.762
 2.723 2.71  2.704 2.691 2.678 2.661 2.656 2.651 2.624 2.612 2.61  2.575 2.559 2.534 2.527 2.52  2.506 2.491 2.472 2.448 2.438 2.428 2.417 2.388
 2.359 2.341 2.327 2.319 2.279 2.268 2.246 2.238 2.228 2.215 2.181 2.157 2.135 2.126 2.099 2.09  2.058]
largest early gaps (rank, ratio) [(1, 1.148), (2, 1.079), (4, 1.034), (6, 1.024), (74, 1.021), (10, 1.018), (52, 1.018), (64, 1.015)]
W_proj 2 first 65 [10.095  8.985  8.075  7.909  7.791  7.656  7.588  7.375  7.212  7.144  7.089  7.034  7.003  6.951  6.73   6.709  6.63   6.572  6.452  6.385  6.341
  6.321  6.252  6.207  6.087  6.     5.905  5.842  5.789  5.765  5.718  5.716  5.679  5.636  5.599  5.562  5.505  5.458  5.415  5.393  5.37   5.36
  5.305  5.281  5.267  5.252  5.238  5.213  5.172  5.164  5.141  5.106  5.081  5.078  5.052  5.037  5.024  5.003  4.989  4.968  4.947  4.943  4.935
  4.924  4.908]
largest early gaps (rank, ratio) [(1, 1.124), (2, 1.113), (14, 1.033), (7, 1.029), (8, 1.023), (3, 1.021), (24, 1.02), (18, 1.019)]
W_o 2 first 65 [4.442 4.073 3.768 3.455 3.44  3.365 3.35  3.297 3.254 3.233 3.178 3.174 3.165 3.149 3.122 3.11  3.08  3.052 3.047 3.012 2.997 2.988 2.971 2.959
 2.956 2.94  2.922 2.91  2.903 2.892 2.879 2.875 2.849 2.833 2.83  2.822 2.811 2.802 2.792 2.775 2.756 2.748 2.739 2.736 2.721 2.703 2.693 2.684
 2.682 2.668 2.667 2.647 2.632 2.63  2.613 2.608 2.599 2.595 2.578 2.576 2.557 2.553 2.534 2.52  2.512]
largest early gaps (rank, ratio) [(1, 1.091), (3, 1.09), (2, 1.081), (5, 1.022), (10, 1.017), (7, 1.016), (8, 1.013), (19, 1.011)]
W_proj 3 first 65 [28.175 20.293 19.292 19.135 18.708 18.57  18.139 17.776 17.484 16.54  15.913 15.61  14.975 14.872 14.417 14.313 13.327 12.971 12.593 12.37  12.224
 12.019 11.322 11.115 10.768  9.226  8.573  7.929  6.499  5.271  3.966  3.184  2.791  2.697  2.653  2.554  2.51   2.47   2.378  2.29   2.25   2.215
  2.117  2.081  2.035  2.01   1.984  1.969  1.951  1.926  1.888  1.868  1.851  1.833  1.816  1.795  1.791  1.781  1.763  1.735  1.721  1.712  1.704
  1.692  1.681]
largest early gaps (rank, ratio) [(1, 1.388), (30, 1.329), (31, 
```

## Call 6

```python
import numpy as np
import torch
from sklearn.cluster import KMeans
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
T=lambda a: torch.as_tensor(np.asarray(a),dtype=torch.float64)
P=torch.eye(384)-torch.ones((384,384))/384
g=T(W["ln1_w.0"])
vv={}
for j in [1,5]:
    vv[j]=(T(W["W_v.0"])[64*j:64*(j+1)]*g)@P
    print("V effective head",j,"singular values",np.round(torch.linalg.svdvals(vv[j])[:45].numpy(),3))
a=torch.linalg.svd(vv[1],full_matrices=False).Vh
b=torch.linalg.svd(vv[5],full_matrices=False).Vh
print("\nCommon L0H1/L0H5 value-input subspaces:")
for r in [12,20,24,28,32,40,48,64]:
    print(r,np.round(torch.linalg.svdvals(a[:r]@b[:r].T).numpy(),3))

# Analyze trained first-MLP rows after restriction to a weight-derived subspace.
f=(T(W["W_fc.0"])*T(W["ln2_w.0"]))@P
active=np.linalg.norm(W["W_fc.0"],axis=1)>.8
c=vv[1].T@vv[1]+vv[5].T@vv[5]
e,u=torch.linalg.eigh(c)
u=u.flip(1)
print("\nJoint value-input eigenvalues",np.round(e.flip(0)[:50].numpy(),3))
for r in [20,24,28,32]:
    z=(f@u[:,:r]).numpy()[active]
    norms=np.linalg.norm(z,axis=1)
    z=z/(norms[:,None]+1e-12)
    sim=z@z.T
    np.fill_diagonal(sim,-1)
    print("\nProjected FC0 rank",r,"active",len(z),
          "nearest-row cosine quantiles",
          np.round(np.quantile(sim.max(1),[.1,.25,.5,.75,.9]),3))
    for k in [20,24,28,32]:
        km=KMeans(n_clusters=k,n_init=10,random_state=1).fit(z)
        counts=np.bincount(km.labels_,minlength=k)
        coherence=np.linalg.norm(km.cluster_centers_,axis=1)
        print("k",k,"average coherence",round(float(np.average(coherence,weights=counts)),3),
              "cluster sizes",sorted(counts.tolist(),reverse=True),
              "coherence",np.round(np.sort(coherence),2))

print("\nFinal MLP output-column geometry in top 28 output directions:")
o=P@T(W["W_proj.3"])
u,s,vh=torch.linalg.svd(o,full_matrices=False)
z=vh[:28].T.numpy()
z=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12)
for k in [20,24,28,32,40]:
    km=KMeans(n_clusters=k,n_init=10,random_state=1).fit(z)
    counts=np.bincount(km.labels_,minlength=k)
    coherence=np.linalg.norm(km.cluster_centers_,axis=1)
    print("k",k,"average coherence",round(float(np.average(coherence,weights=counts)),3),
          "cluster sizes",sorted(counts.tolist(),reverse=True),
          "coherence",np.round(np.sort(coherence),2))

```

Output:

```
V effective head 1 singular values [0.48  0.455 0.448 0.429 0.421 0.41  0.403 0.385 0.379 0.373 0.37  0.36  0.353 0.341 0.33  0.327 0.322 0.313 0.309 0.303 0.295 0.287 0.281 0.278
 0.27  0.255 0.253 0.241 0.226 0.216 0.209 0.201 0.198 0.171 0.164 0.159 0.155 0.148 0.145 0.143 0.137 0.134 0.126 0.124 0.117]
V effective head 5 singular values [0.493 0.473 0.446 0.437 0.426 0.408 0.404 0.393 0.386 0.376 0.363 0.361 0.356 0.345 0.333 0.329 0.322 0.316 0.31  0.304 0.299 0.289 0.285 0.277
 0.274 0.267 0.252 0.245 0.241 0.235 0.233 0.221 0.216 0.212 0.208 0.203 0.199 0.194 0.188 0.179 0.169 0.164 0.157 0.143 0.138]

Common L0H1/L0H5 value-input subspaces:
12 [0.923 0.908 0.879 0.832 0.787 0.707 0.659 0.531 0.416 0.281 0.231 0.023]
20 [0.963 0.943 0.935 0.924 0.921 0.907 0.891 0.883 0.87  0.868 0.856 0.84  0.798 0.773 0.581 0.559 0.506 0.353 0.134 0.034]
24 [0.966 0.958 0.957 0.946 0.936 0.932 0.923 0.912 0.909 0.895 0.891 0.874 0.862 0.843 0.839 0.785 0.675 0.639 0.587 0.549 0.425 0.287 0.156 0.005]
28 [0.972 0.966 0.959 0.957 0.946 0.94  0.938 0.934 0.931 0.918 0.913 0.9   0.889 0.883 0.872 0.849 0.836 0.81  0.8   0.788 0.702 0.615 0.55  0.482
 0.427 0.356 0.193 0.048]
32 [0.977 0.969 0.961 0.958 0.951 0.946 0.945 0.943 0.939 0.931 0.925 0.917 0.912 0.905 0.885 0.875 0.872 0.858 0.839 0.82  0.787 0.727 0.708 0.666
 0.61  0.567 0.487 0.385 0.262 0.133 0.083 0.013]
40 [0.981 0.975 0.971 0.966 0.963 0.958 0.956 0.952 0.948 0.947 0.946 0.936 0.933 0.927 0.918 0.913 0.898 0.894 0.886 0.883 0.864 0.839 0.825 0.804
 0.792 0.756 0.74  0.694 0.671 0.633 0.548 0.531 0.502 0.471 0.344 0.304 0.232 0.088 0.057 0.017]
48 [0.984 0.98  0.974 0.973 0.97  0.967 0.964 0.962 0.956 0.955 0.949 0.946 0.942 0.939 0.929 0.927 0.925 0.911 0.899 0.896 0.885 0.866 0.866 0.854
 0.835 0.82  0.802 0.786 0.762 0.747 0.729 0.706 0.693 0.647 0.614 0.6   0.518 0.476 0.445 0.418 0.319 0.317 0.289 0.258 0.15  0.134 0.059 0.049]
64 [0.987 0.984 0.98  0.98  0.977 0.974 0.972 0.97  0.97  0.966 0.961 0.96  0.958 0.953 0.95  0.945 0.938 0.935 0.928 0.924 0.917 0.902 0.897 0.892
 0.881 0.879 0.856 0.853 0.833 0.828 0.812 0.795 0.785 0.781 0.743 0.727 0.71  0.69  0.657 0.65  0.632 0.609 0.571 0.564 0.555 0.504 0.478 0.461
 0.426 0.415 0.395 0.382 0.36  0.339 0.306 0.253 0.241 0.209 0.194 0.12  0.073 0.067 0.053 0.023]

Joint value-input eigenvalues [0.408 0.386 0.353 0.348 0.326 0.295 0.289 0.28  0.266 0.264 0.244 0.239 0.225 0.222 0.218 0.205 0.201 0.198 0.19  0.178 0.173 0.157 0.145 0.145
 0.131 0.126 0.125 0.112 0.107 0.095 0.089 0.083 0.082 0.076 0.075 0.067 0.066 0.064 0.059 0.056 0.053 0.051 0.05  0.049 0.041 0.037 0.033 0.032
 0.03  0.029]

Projected FC0 rank 20 active 531 nearest-row cosine quantiles [0.649 0.71  0.796 0.922 0.991]
k 20 average coherence 0.692 cluster sizes [42, 38, 37, 37, 36, 30, 29, 28, 26, 26, 24, 24, 23, 21, 21, 20, 19, 18, 18, 14] coherence [0.58 0.59 0.62 0.63 0.64 0.66 0.66 0.67 0.7  0.7  0.71 0.71 0.72 0.73 0.74 0.76 0.76 0.76 0.79 0.84]
k 24 average coherence 0.723 cluster sizes [36, 35, 34, 27, 26, 25, 25, 24, 24, 22, 22, 21, 21, 20, 20, 19, 19, 19, 17, 17, 15, 15, 15, 13] coherence [0.63 0.64 0.65 0.66 0.66 0.68 0.68 0.68 0.7  0.7  0.71 0.73 0.73 0.73 0.75 0.75 0.76 0.76 0.78 0.79 0.79 0.81 0.81 0.85]
k 28 average coherence 0.746 cluster sizes [37, 32, 26, 25, 23, 21, 20, 20, 20, 20, 19, 19, 19, 18, 18, 18, 18, 18, 17, 17, 16, 16, 16, 14, 13, 12, 12, 7] coherence [0.66 0.67 0.68 0.69 0.7  0.7  0.7  0.7  0.72 0.72 0.73 0.73 0.74 0.75 0.75 0.75 0.76 0.76 0.77 0.77 0.77 0.77 0.78 0.8  0.8  0.81 0.84 0.89]
k 32 average coherence 0.759 cluster sizes [35, 33, 22, 22, 21, 20, 20, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 14, 13, 13, 13, 12, 11, 11, 11, 11, 10, 9, 8] coherence [0.62 0.67 0.69 0.7  0.7  0.7  0.7  0.7  0.71 0.72 0.72 0.73 0.74 0.74 0.75 0.76 0.76 0.77 0.77 0.78 0.78 0.78 0.78 0.8  0.81 0.82 0.83 0.83 0.84
 0.85 0.87 0.88]

Projected FC0 rank 24 active 531 nearest-row cosine quantiles [0.629 0.698 0.792 0.928 0.993]
k 20 a
```

## Call 7

```python
import numpy as np
import torch
from scipy.linalg import eig
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
T=lambda a: torch.as_tensor(np.asarray(a),dtype=torch.float64)
P=torch.eye(384)-torch.ones((384,384))/384

print("Final MLP output weights after linear regression on input weights.")
print("This is weight-space regression only, not evaluation of an MLP.")
f=T(W["W_fc.3"])
o=P@T(W["W_proj.3"])
alpha=-float((o*f.T).sum()/f.square().sum())
print("Best scalar cancellation coefficient",round(alpha,5))
u,s,vh=torch.linalg.svd(f,full_matrices=False)
for method in ["scalar","regress"]:
    if method=="scalar":
        r=o+alpha*(P@f.T)
    else:
        r=o-(o@u)@u.T
    sv=torch.linalg.svdvals(r)
    print(method,"first 55 singular values",np.round(sv[:55].numpy(),3))
    ratios=sv[:-1]/sv[1:]
    ids=torch.argsort(ratios[15:45],descending=True)[:8]+15
    print("gaps",[(int(i)+1,round(float(ratios[i]),3)) for i in ids])

g=T(W["ln1_w.0"])
m={}
for j in [1,5]:
    sl=slice(64*j,64*(j+1))
    q=(T(W["W_q.0"])[sl]*g)@P
    k=(T(W["W_k.0"])[sl]*g)@P
    m[j]=q.T@k/8
c=sum(a@a.T+a.T@a for a in m.values())
ee,uu=torch.linalg.eigh(c)
uu=uu.flip(1)

print("\nShared QK subspace: congruence-invariant phase diagnostics.")
for r in [8,12,16,20,24,28]:
    z=uu[:,:r]
    a=(z.T@m[1]@z).numpy()
    b=(z.T@m[5]@z).numpy()
    print("rank",r)
    for label,x,y in [("H1 versus transpose",a,a.T),
                       ("H5 versus transpose",b,b.T),
                       ("H1 versus H5",a,b)]:
        ev=eig(x,y,right=False)
        good=ev[np.isfinite(ev)]
        positive=good[good.imag>1e-5]
        positive=positive[np.argsort(np.angle(positive))]
        print(label,[(round(float(np.angle(v)),3),round(float(abs(v)),3)) for v in positive])

```

Output:

```
Final MLP output weights after linear regression on input weights.
This is weight-space regression only, not evaluation of an MLP.
Best scalar cancellation coefficient 0.1226
scalar first 55 singular values [28.283 20.308 19.323 19.114 18.692 18.549 18.137 17.786 17.483 16.527 15.894 15.6   14.962 14.885 14.406 14.312 13.34  12.985 12.591 12.374 12.256
 12.037 11.345 11.107 10.768  9.222  8.598  7.927  6.518  5.299  3.919  3.262  2.783  2.656  2.402  2.166  2.125  1.959  1.933  1.864  1.809  1.786
  1.757  1.74   1.704  1.659  1.646  1.626  1.579  1.576  1.56   1.535  1.513  1.502  1.49 ]
gaps [(30, 1.352), (29, 1.23), (28, 1.216), (31, 1.201), (32, 1.172), (25, 1.168), (35, 1.109), (34, 1.106)]
regress first 55 singular values [18.909 18.247 17.385 16.993 16.749 16.475 16.143 15.907 15.758 14.826 14.425 14.017 13.553 13.176 12.916 12.74  11.888 11.44  11.079 10.939 10.762
 10.254  9.966  9.931  9.492  8.276  7.586  7.012  4.48   3.376  2.688  2.222  1.946  1.805  1.713  1.473  1.43   1.362  1.343  1.332  1.314  1.303
  1.276  1.239  1.228  1.205  1.187  1.181  1.172  1.163  1.155  1.15   1.141  1.13   1.119]
gaps [(28, 1.565), (29, 1.327), (30, 1.256), (31, 1.21), (35, 1.163), (25, 1.147), (32, 1.141), (26, 1.091)]

Shared QK subspace: congruence-invariant phase diagnostics.
rank 8
H1 versus transpose [(1.721, 1.0), (2.465, 1.0)]
H5 versus transpose [(1.281, 1.0), (1.896, 1.0), (2.669, 1.0)]
H1 versus H5 [(1.138, 0.121), (2.248, 1.244)]
rank 12
H1 versus transpose [(1.166, 1.0), (1.595, 1.0), (1.88, 0.795), (1.88, 1.257)]
H5 versus transpose [(1.119, 1.0), (1.909, 1.0), (1.957, 1.0), (2.47, 1.135), (2.47, 0.881)]
H1 versus H5 [(0.748, 0.1), (1.525, 0.156), (2.197, 1.26), (2.834, 0.57)]
rank 16
H1 versus transpose [(0.134, 1.0), (0.828, 1.932), (0.828, 0.518), (1.476, 1.0), (1.8, 1.0), (1.993, 1.258), (1.993, 0.795)]
H5 versus transpose [(0.944, 1.0), (1.503, 1.0), (2.092, 1.0), (2.489, 0.779), (2.489, 1.284)]
H1 versus H5 [(0.478, 0.176), (0.823, 0.201), (1.357, 0.155), (1.611, 0.223), (2.133, 1.253), (2.637, 0.951)]
rank 20
H1 versus transpose [(0.067, 1.0), (0.819, 1.0), (1.214, 1.0), (1.627, 1.0), (1.937, 0.792), (1.937, 1.263), (1.94, 1.0)]
H5 versus transpose [(0.753, 1.0), (1.356, 1.0), (1.898, 1.0), (2.226, 1.038), (2.226, 0.963), (2.437, 1.0), (2.79, 1.0), (3.023, 1.0)]
H1 versus H5 [(0.678, 1.291), (0.716, 0.193), (0.925, 0.186), (1.349, 0.164), (1.654, 0.19), (1.995, 1.047), (2.123, 1.336), (2.171, 0.186)]
rank 24
H1 versus transpose [(0.024, 1.0), (0.333, 1.0), (0.958, 4.242), (0.958, 0.236), (1.368, 1.0), (1.563, 1.0), (1.854, 1.0), (1.982, 1.0), (2.652, 1.702), (2.652, 0.587), (2.924, 1.0)]
H5 versus transpose [(1.055, 1.0), (1.713, 1.0), (1.88, 1.0), (2.232, 1.227), (2.232, 0.815), (2.394, 1.0), (2.661, 0.978), (2.661, 1.023)]
H1 versus H5 [(0.114, 1.404), (0.126, 4.66), (0.299, 0.459), (0.823, 0.164), (0.892, 0.169), (1.455, 0.155), (1.584, 0.19), (1.899, 0.212), (2.021, 0.969), (2.132, 1.328), (2.209, 0.431)]
rank 28
H1 versus transpose [(0.084, 1.0), (0.426, 1.0), (0.978, 1.0), (1.517, 1.0), (1.714, 1.0), (1.926, 1.0), (2.355, 1.0), (2.82, 0.144), (2.82, 6.929), (2.933, 1.0)]
H5 versus transpose [(1.13, 1.0), (1.523, 0.136), (1.523, 7.341), (1.6, 1.0), (2.216, 0.603), (2.216, 1.658), (2.37, 1.0), (2.559, 0.908), (2.559, 1.101), (2.735, 0.868), (2.735, 1.152)]
H1 versus H5 [(0.671, 0.373), (0.693, 0.209), (0.915, 0.33), (1.347, 0.142), (1.374, 0.102), (1.58, 0.244), (1.644, 0.135), (2.013, 1.369), (2.014, 0.208), (2.112, 0.49), (2.469, 1.38)]

```

## Call 8

```python
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
T=lambda a: torch.as_tensor(np.asarray(a),dtype=torch.float64)
P=torch.eye(384)-torch.ones((384,384))/384
M,B={},{}

for l in range(4):
    g=T(W[f"ln1_w.{l}"])
    for j in range(6):
        sl=slice(64*j,64*(j+1))
        q,k,v=[(T(W[f"{n}.{l}"])[sl]*g)@P for n in ["W_q","W_k","W_v"]]
        M[l,j]=q.T@k/8
        B[l,j]=P@T(W[f"W_o.{l}"])[:,sl]@v

print("Directional composition: key trace(M B), query trace(M B.T), each divided by ||M||||B||.")
print("A substantially key-specific positive match is more suggestive of an induction circuit.")
for l in range(1,4):
    for j in range(6):
        a=M[l,j]
        rows=[]
        for p in range(l):
            for h in range(6):
                b=B[p,h]
                den=a.norm()*b.norm()
                key=float(torch.trace(a@b)/den)
                query=float(torch.trace(a@b.T)/den)
                rows.append((key-query,key,query,p,h))
        rows.sort(reverse=True)
        print(f"L{l}H{j}",[(f"L{p}H{h}",round(k,4),round(q,4),round(d,4))
                            for d,k,q,p,h in rows[:3]])

print("\nL0 QK energy in a value-defined candidate character subspace:")
g=T(W["ln1_w.0"])
v1=(T(W["W_v.0"])[64:128]*g)@P
v5=(T(W["W_v.0"])[320:384]*g)@P
c=v1.T@v1+v5.T@v5
ev,u=torch.linalg.eigh(c)
u=u.flip(1)
for r in [16,24,32]:
    z=u[:,:r]
    print("subspace dimension",r)
    for j in range(6):
        a=M[0,j]
        den=a.norm().square()
        qe=float((z.T@a).norm().square()/den)
        ke=float((a@z).norm().square()/den)
        both=float((z.T@a@z).norm().square()/den)
        print(j,"query/key/both",np.round([qe,ke,both],4))

print("\nL1 attention query-bias key directions: pairwise cosine.")
kb=[]
for j in range(6):
    sl=slice(64*j,64*(j+1))
    q=T(W["W_q.1"])[sl]
    k=(T(W["W_k.1"])[sl]*T(W["ln1_w.1"]))@P
    v=k.T@(q@T(W["ln1_b.1"]))/8
    kb.append(v/v.norm())
print(np.round(torch.stack(kb).numpy()@torch.stack(kb).numpy().T,3))

```

Output:

```
Directional composition: key trace(M B), query trace(M B.T), each divided by ||M||||B||.
A substantially key-specific positive match is more suggestive of an induction circuit.
L1H0 [('L0H5', -0.0034, -0.0471, 0.0437), ('L0H4', 0.0207, -0.0068, 0.0276), ('L0H1', -0.0001, -0.0222, 0.0221)]
L1H1 [('L0H4', 0.0191, -0.0246, 0.0437), ('L0H3', 0.0055, -0.0119, 0.0174), ('L0H5', 0.0154, 0.0072, 0.0082)]
L1H2 [('L0H4', 0.0326, -0.0233, 0.056), ('L0H5', -0.0027, -0.0332, 0.0305), ('L0H3', 0.0092, -0.0086, 0.0178)]
L1H3 [('L0H4', 0.0421, -0.0282, 0.0703), ('L0H3', 0.0063, -0.0094, 0.0157), ('L0H5', 0.0124, 0.0092, 0.0032)]
L1H4 [('L0H4', 0.0439, -0.0262, 0.0701), ('L0H5', 0.0094, -0.0234, 0.0328), ('L0H3', 0.0114, -0.0045, 0.0159)]
L1H5 [('L0H5', 0.0128, -0.0122, 0.025), ('L0H4', -0.0036, -0.0203, 0.0167), ('L0H3', -0.0004, -0.0077, 0.0073)]
L2H0 [('L0H4', 0.0099, -0.0047, 0.0146), ('L1H2', -0.0003, -0.0129, 0.0127), ('L0H3', 0.0078, 0.0008, 0.007)]
L2H1 [('L0H5', 0.0776, -0.0398, 0.1174), ('L1H2', 0.0482, -0.0636, 0.1118), ('L1H4', 0.0476, -0.0495, 0.0971)]
L2H2 [('L0H4', 0.02, -0.0077, 0.0276), ('L0H0', 0.0114, 0.0063, 0.005), ('L0H3', 0.0015, -0.0021, 0.0035)]
L2H3 [('L0H4', 0.0396, -0.009, 0.0486), ('L0H0', 0.0177, 0.0052, 0.0126), ('L0H3', 0.0066, -0.0018, 0.0084)]
L2H4 [('L0H4', 0.0246, -0.0048, 0.0294), ('L0H3', 0.0101, -0.0014, 0.0115), ('L0H5', 0.0018, -0.0006, 0.0024)]
L2H5 [('L1H2', 0.0095, -0.012, 0.0215), ('L0H4', 0.0188, -0.0023, 0.0211), ('L1H5', 0.0068, -0.0092, 0.016)]
L3H0 [('L2H4', 0.0044, -0.0182, 0.0225), ('L0H5', 0.019, 0.0037, 0.0153), ('L0H4', 0.0098, -0.0003, 0.0101)]
L3H1 [('L2H2', 0.1104, -0.0164, 0.1268), ('L0H5', 0.061, -0.0198, 0.0808), ('L1H2', 0.038, -0.0323, 0.0703)]
L3H2 [('L0H4', 0.013, -0.0025, 0.0155), ('L0H0', 0.0062, 0.0017, 0.0045), ('L0H3', -0.0008, -0.0014, 0.0006)]
L3H3 [('L2H5', 0.0045, -0.0134, 0.0179), ('L2H3', 0.0103, -0.0069, 0.0172), ('L2H4', 0.0148, 0.0019, 0.0129)]
L3H4 [('L0H5', 0.0101, -0.0033, 0.0134), ('L0H4', 0.0119, 0.0004, 0.0115), ('L1H1', 0.0069, -0.0039, 0.0108)]
L3H5 [('L2H2', 0.1836, -0.0354, 0.2189), ('L2H3', 0.055, -0.0604, 0.1154), ('L1H3', 0.0683, -0.0422, 0.1104)]

L0 QK energy in a value-defined candidate character subspace:
subspace dimension 16
0 query/key/both [0.0224 0.2603 0.0044]
1 query/key/both [0.0286 0.0327 0.0015]
2 query/key/both [0.0605 0.1709 0.0173]
3 query/key/both [0.0436 0.1638 0.0092]
4 query/key/both [0.0547 0.0596 0.0032]
5 query/key/both [0.0308 0.0319 0.0011]
subspace dimension 24
0 query/key/both [0.03   0.3361 0.0076]
1 query/key/both [0.0649 0.0706 0.0128]
2 query/key/both [0.0876 0.2359 0.0348]
3 query/key/both [0.0651 0.2143 0.0168]
4 query/key/both [0.0781 0.091  0.0077]
5 query/key/both [0.0536 0.057  0.0038]
subspace dimension 32
0 query/key/both [0.0502 0.3927 0.0154]
1 query/key/both [0.1885 0.2035 0.0842]
2 query/key/both [0.1574 0.3177 0.0624]
3 query/key/both [0.1216 0.3235 0.0365]
4 query/key/both [0.113  0.1314 0.0162]
5 query/key/both [0.1712 0.1751 0.0323]

L1 attention query-bias key directions: pairwise cosine.
[[1.    0.798 0.808 0.801 0.81  0.757]
 [0.798 1.    0.923 0.924 0.921 0.897]
 [0.808 0.923 1.    0.899 0.9   0.868]
 [0.801 0.924 0.899 1.    0.926 0.899]
 [0.81  0.921 0.9   0.926 1.    0.883]
 [0.757 0.897 0.868 0.899 0.883 1.   ]]

```

## Call 9

```python
import numpy as np
import torch
from scipy.linalg import eig
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
T=lambda a: torch.as_tensor(np.asarray(a),dtype=torch.float64)
P=torch.eye(384)-torch.ones((384,384))/384
g=T(W["ln1_w.0"])
m={}
for j in [1,5]:
    sl=slice(64*j,64*(j+1))
    q=(T(W["W_q.0"])[sl]*g)@P
    k=(T(W["W_k.0"])[sl]*g)@P
    m[j]=q.T@k/8
c=sum(a@a.T+a.T@a for a in m.values())
_,u=torch.linalg.eigh(c)
u=u.flip(1)

print("Relative-shift diagnostic using only generalized eigenvectors of QK weight products.")
print("For shared rotational modes, phase(H1) approximately k*phase(H5) would suggest a lag ratio k.")
for r in [12,16,20,24,28,32,40]:
    z=u[:,:r]
    a=(z.T@m[1]@z).numpy()
    b=(z.T@m[5]@z).numpy()
    ev,vec=eig(b.T,b)
    rows=[]
    for i,x in enumerate(ev):
        if not np.isfinite(x) or x.imag<=1e-5 or abs(abs(x)-1)>.12:
            continue
        v=vec[:,i]
        av=a@v
        atv=a.T@v
        overlap=np.vdot(av,atv)/(np.linalg.norm(av)*np.linalg.norm(atv)+1e-15)
        rows.append((float(np.angle(x)),float(np.angle(overlap)),float(abs(overlap))))
    print("\nrank",r,"(H5 phase,H1 phase,mode agreement)",np.round(rows,3).tolist())
    if rows:
        p,q,w=np.array(rows).T
        print("phase agreement for H1/H5 lag ratios -3,-2,-1,1,2,3,4:",
              [(k,round(float(np.average(np.cos(q-k*p),weights=w**4)),3))
               for k in [-3,-2,-1,1,2,3,4]])

```

Output:

```
Relative-shift diagnostic using only generalized eigenvectors of QK weight products.
For shared rotational modes, phase(H1) approximately k*phase(H5) would suggest a lag ratio k.

rank 12 (H5 phase,H1 phase,mode agreement) [[1.119, 0.703, 0.559], [1.909, 2.094, 0.746], [1.957, 2.244, 0.67], [2.47, -1.649, 0.722]]
phase agreement for H1/H5 lag ratios -3,-2,-1,1,2,3,4: [(-3, 0.154), (-2, 0.141), (-1, -0.158), (1, 0.493), (2, 0.222), (3, -0.899), (4, 0.502)]

rank 16 (H5 phase,H1 phase,mode agreement) [[0.944, -1.064, 0.252], [1.503, 0.037, 0.209], [2.092, 2.497, 0.759]]
phase agreement for H1/H5 lag ratios -3,-2,-1,1,2,3,4: [(-3, -0.784), (-2, 0.908), (-1, -0.109), (1, 0.898), (2, -0.131), (3, -0.8), (4, 0.907)]

rank 20 (H5 phase,H1 phase,mode agreement) [[0.753, 0.182, 0.552], [1.356, 0.74, 0.57], [1.898, 2.17, 0.748], [2.226, -1.942, 0.736], [2.226, -1.956, 0.642], [2.437, -1.628, 0.898], [3.023, -2.646, 0.413], [2.79, -1.664, 0.348]]
phase agreement for H1/H5 lag ratios -3,-2,-1,1,2,3,4: [(-3, 0.313), (-2, -0.522), (-1, 0.445), (1, -0.086), (2, 0.627), (3, -0.797), (4, 0.171)]

rank 24 (H5 phase,H1 phase,mode agreement) [[1.055, -3.122, 0.399], [1.713, 2.482, 0.595], [1.88, -2.537, 0.09], [2.394, -1.529, 0.91], [2.661, -1.741, 0.721], [2.661, -1.943, 0.626]]
phase agreement for H1/H5 lag ratios -3,-2,-1,1,2,3,4: [(-3, 0.815), (-2, -0.75), (-1, 0.516), (1, -0.404), (2, 0.833), (3, -0.791), (4, 0.369)]

rank 28 (H5 phase,H1 phase,mode agreement) [[1.13, -2.598, 0.491], [1.6, 0.918, 0.418], [2.37, -1.723, 0.899], [2.559, -1.791, 0.762], [2.559, -2.104, 0.367]]
phase agreement for H1/H5 lag ratios -3,-2,-1,1,2,3,4: [(-3, 0.729), (-2, -0.875), (-1, 0.694), (1, -0.478), (2, 0.834), (3, -0.787), (4, 0.458)]

rank 32 (H5 phase,H1 phase,mode agreement) [[1.081, -2.056, 0.212], [2.508, -1.578, 0.689], [2.937, -0.866, 0.295], [2.777, -1.781, 0.33], [2.508, -1.675, 0.617]]
phase agreement for H1/H5 lag ratios -3,-2,-1,1,2,3,4: [(-3, 0.907), (-2, -0.926), (-1, 0.603), (1, -0.549), (2, 0.917), (3, -0.945), (4, 0.626)]

rank 40 (H5 phase,H1 phase,mode agreement) [[1.714, -2.585, 0.567], [2.52, -1.388, 0.773]]
phase agreement for H1/H5 lag ratios -3,-2,-1,1,2,3,4: [(-3, 0.582), (-2, -0.527), (-1, 0.475), (1, -0.648), (2, 0.984), (3, -0.66), (4, 0.127)]

```

## Call 10

```python
import numpy as np
import torch
from sklearn.cluster import DBSCAN
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
T=lambda a: torch.as_tensor(np.asarray(a),dtype=torch.float64)
P=torch.eye(384)-torch.ones((384,384))/384
g=T(W["ln1_w.0"])
vs={}
bs={}
for j in [1,5]:
    sl=slice(64*j,64*(j+1))
    vs[j]=(T(W["W_v.0"])[sl]*g)@P
    bs[j]=P@T(W["W_o.0"])[:,sl]@vs[j]
_,u=torch.linalg.eigh(sum(v.T@v for v in vs.values()))
u=u.flip(1)
f=(T(W["W_fc.0"])*T(W["ln2_w.0"]))@P
active=np.linalg.norm(W["W_fc.0"],axis=1)>.8
ids=np.flatnonzero(active)

print("Static MLP read-direction alignments for current / H5-value / H1-value channels.")
print("These are products of weight matrices, with no inputs or nonlinearities.")
for r in [20,28,40]:
    z=u[:,:r]
    arr=[(a@z).numpy()[active] for a in [f,f@bs[5],f@bs[1]]]
    norms=[np.linalg.norm(a,axis=1) for a in arr]
    nn=[a/(n[:,None]+1e-12) for a,n in zip(arr,norms)]
    cs=np.stack([(nn[0]*nn[1]).sum(1),(nn[0]*nn[2]).sum(1),
                 (nn[1]*nn[2]).sum(1)],1)
    print("\nrank",r)
    print("Channel norm quantiles",np.round([np.quantile(n,[.1,.5,.9]) for n in norms],3))
    print("Pair cosine quantiles",np.round(np.quantile(cs,[.01,.1,.5,.9,.99],axis=0),3))
    for threshold in [.5,.65,.8]:
        mask=cs.min(1)>threshold
        print("all-three pair cosines >",threshold,"count",int(mask.sum()))
    order=np.argsort(cs.min(1))[::-1][:15]
    print("best same-direction neurons (index, 3 cosines, 3 norms):")
    for i in order:
        print(int(ids[i]),np.round(cs[i],3).tolist(),
              np.round([n[i] for n in norms],3).tolist())
    if r==28:
        selected=np.flatnonzero(cs.min(1)>.5)
        if len(selected):
            sim=nn[0][selected]@nn[0][selected].T
            print("Similarities among strongest repeated-direction candidates:")
            print(np.round(sim[:20,:20],2))
        for a,b,label in [(0,1,"current/H5"),(0,2,"current/H1"),(1,2,"H5/H1")]:
            # Permutation baseline for row correspondence.
            rng=np.random.default_rng(0)
            baseline=np.sum(nn[a]*nn[b][rng.permutation(len(ids))],axis=1)
            actual=np.sum(nn[a]*nn[b],axis=1)
            print(label,"actual/random mean",
                  round(float(actual.mean()),4),round(float(baseline.mean()),4),
                  "actual/random fraction > .65",
                  round(float((actual>.65).mean()),4),round(float((baseline>.65).mean()),4))

```

Output:

```
Static MLP read-direction alignments for current / H5-value / H1-value channels.
These are products of weight matrices, with no inputs or nonlinearities.

rank 20
Channel norm quantiles [[0.353 0.706 1.196]
 [0.179 0.448 0.697]
 [0.116 0.361 0.605]]
Pair cosine quantiles [[-0.918 -0.776 -0.57 ]
 [-0.663 -0.443 -0.222]
 [-0.082  0.15   0.252]
 [ 0.494  0.589  0.608]
 [ 0.747  0.744  0.79 ]]
all-three pair cosines > 0.5 count 17
all-three pair cosines > 0.65 count 2
all-three pair cosines > 0.8 count 0
best same-direction neurons (index, 3 cosines, 3 norms):
14 [0.758, 0.755, 0.801] [0.386, 0.398, 0.262]
643 [0.703, 0.679, 0.746] [1.141, 0.647, 0.374]
795 [0.612, 0.631, 0.673] [0.473, 0.452, 0.392]
1065 [0.622, 0.609, 0.638] [0.402, 0.341, 0.338]
310 [0.574, 0.59, 0.705] [0.39, 0.569, 0.259]
799 [0.693, 0.573, 0.692] [0.442, 0.5, 0.417]
129 [0.583, 0.567, 0.598] [0.518, 0.477, 0.371]
581 [0.761, 0.566, 0.87] [0.304, 0.511, 0.32]
1265 [0.562, 0.7, 0.62] [0.491, 0.392, 0.387]
85 [0.555, 0.677, 0.573] [0.271, 0.344, 0.2]
26 [0.547, 0.563, 0.676] [0.361, 0.392, 0.383]
325 [0.54, 0.653, 0.677] [0.616, 0.291, 0.333]
1446 [0.54, 0.641, 0.735] [0.558, 0.288, 0.351]
251 [0.669, 0.533, 0.601] [0.38, 0.374, 0.225]
1354 [0.55, 0.518, 0.616] [0.449, 0.275, 0.295]

rank 28
Channel norm quantiles [[0.426 0.856 1.341]
 [0.189 0.471 0.731]
 [0.134 0.386 0.642]]
Pair cosine quantiles [[-0.898 -0.722 -0.509]
 [-0.629 -0.432 -0.183]
 [-0.084  0.151  0.22 ]
 [ 0.471  0.558  0.586]
 [ 0.711  0.697  0.743]]
all-three pair cosines > 0.5 count 13
all-three pair cosines > 0.65 count 1
all-three pair cosines > 0.8 count 0
best same-direction neurons (index, 3 cosines, 3 norms):
14 [0.737, 0.682, 0.745] [0.415, 0.407, 0.268]
795 [0.652, 0.623, 0.64] [0.524, 0.498, 0.412]
251 [0.662, 0.636, 0.605] [0.474, 0.392, 0.257]
1065 [0.668, 0.589, 0.639] [0.484, 0.363, 0.349]
1265 [0.62, 0.647, 0.588] [0.59, 0.44, 0.403]
799 [0.644, 0.582, 0.678] [0.492, 0.525, 0.43]
310 [0.572, 0.601, 0.669] [0.478, 0.593, 0.276]
129 [0.557, 0.603, 0.607] [0.618, 0.495, 0.398]
753 [0.665, 0.555, 0.62] [0.384, 0.417, 0.363]
1206 [0.542, 0.732, 0.565] [0.572, 0.446, 0.359]
581 [0.725, 0.524, 0.78] [0.426, 0.553, 0.348]
85 [0.521, 0.55, 0.533] [0.315, 0.356, 0.22]
26 [0.511, 0.55, 0.661] [0.387, 0.404, 0.395]
325 [0.497, 0.652, 0.622] [0.641, 0.323, 0.34]
1354 [0.489, 0.566, 0.521] [0.525, 0.305, 0.32]
Similarities among strongest repeated-direction candidates:
[[ 1.   -0.01  0.09  0.04  0.31  0.31 -0.01  0.45  0.33 -0.2  -0.07  0.28  0.57]
 [-0.01  1.    0.35  0.03  0.05  0.27  0.05 -0.06  0.23  0.31  0.18 -0.33  0.03]
 [ 0.09  0.35  1.   -0.02  0.17  0.2   0.61  0.08  0.16  0.   -0.09 -0.15  0.26]
 [ 0.04  0.03 -0.02  1.    0.22  0.07 -0.4   0.14 -0.25 -0.06 -0.59  0.13 -0.24]
 [ 0.31  0.05  0.17  0.22  1.    0.54  0.21  0.58  0.18 -0.1   0.22  0.04  0.34]
 [ 0.31  0.27  0.2   0.07  0.54  1.    0.22  0.36  0.32 -0.06  0.2  -0.18  0.21]
 [-0.01  0.05  0.61 -0.4   0.21  0.22  1.    0.21  0.04 -0.05  0.13 -0.29  0.33]
 [ 0.45 -0.06  0.08  0.14  0.58  0.36  0.21  1.    0.22 -0.19  0.06 -0.08  0.67]
 [ 0.33  0.23  0.16 -0.25  0.18  0.32  0.04  0.22  1.    0.48  0.3  -0.05  0.46]
 [-0.2   0.31  0.   -0.06 -0.1  -0.06 -0.05 -0.19  0.48  1.    0.28  0.18 -0.11]
 [-0.07  0.18 -0.09 -0.59  0.22  0.2   0.13  0.06  0.3   0.28  1.   -0.26  0.1 ]
 [ 0.28 -0.33 -0.15  0.13  0.04 -0.18 -0.29 -0.08 -0.05  0.18 -0.26  1.    0.01]
 [ 0.57  0.03  0.26 -0.24  0.34  0.21  0.33  0.67  0.46 -0.11  0.1   0.01  1.  ]]
current/H5 actual/random mean -0.0771 -0.0008 actual/random fraction > .65 0.032 0.0038
current/H1 actual/random mean 0.1083 -0.0077 actual/random fraction > .65 0.0339 0.0
H5/H1 actual/random mean 0.2086 0.0146 actual/random fraction > .65 0.0452 0.0

rank 40
Channel norm quantiles [[0.484 0.898 1.405]
 [0.194 0.479 0.737]
 [0.137 0.39  0.646]]
Pair cosine quantiles [[-0.886 -0.708 -0.508]
 [-0.619 -0.439 -0.174]
 [-0.07   0.146  0.214]
 [ 0.426  0.526  0.57 ]
 [ 0.669  0.673  0.729]]
all-thre
```

## Call 11

```python
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
T=lambda a: torch.as_tensor(np.asarray(a),dtype=torch.float64)
P=torch.eye(384)-torch.ones((384,384))/384
g=T(W["ln1_w.0"])
v,b={},{}
for j in [1,5]:
    sl=slice(64*j,64*(j+1))
    v[j]=(T(W["W_v.0"])[sl]*g)@P
    b[j]=P@T(W["W_o.0"])[:,sl]@v[j]
_,u=torch.linalg.eigh(sum(x.T@x for x in v.values()))
z=u.flip(1)[:,:28]
f=(T(W["W_fc.0"])*T(W["ln2_w.0"]))@P
active=T(np.linalg.norm(W["W_fc.0"],axis=1)>.8).bool()
a=[(x@z)[active] for x in [f,f@b[5],f@b[1]]]
a=[x/x.norm(dim=1,keepdim=True) for x in a]
c=[]
print("Cross-covariance spectra of normalized MLP weight read-directions:")
for i,j in [(0,1),(0,2),(1,2)]:
    m=a[i].T@a[j]/len(a[i])
    c.append(m)
    se=torch.linalg.eigvalsh((m+m.T)/2)
    sv=torch.linalg.svdvals(m)
    print((i,j),"trace",round(float(torch.trace(m)),4),
          "SV",np.round(sv[:12].numpy(),4),
          "sym eigenvalues",np.round(se.numpy(),4))
print("\nCosines between the three symmetric cross-covariance matrices:")
cs=[(x+x.T)/2 for x in c]
print(np.round([[float((x*y).sum()/(x.norm()*y.norm())) for y in cs] for x in cs],3))
alt=cs[0]+cs[2]-cs[1]
e,u=torch.linalg.eigh(alt)
print("Alternation-contrast eigenvalues",np.round(e.numpy(),4))
for i in range(4):
    d=u[:,i]
    print("mode",i,"per-pair quadratic coefficients",
          np.round([float(d@m@d) for m in cs],4))

```

Output:

```
Cross-covariance spectra of normalized MLP weight read-directions:
(0, 1) trace -0.0771 SV [0.0481 0.0233 0.0208 0.0199 0.0177 0.0174 0.016  0.0146 0.0123 0.0113 0.01   0.0095] sym eigenvalues [-0.0304 -0.0183 -0.0138 -0.0122 -0.0117 -0.0105 -0.0088 -0.0078 -0.0069 -0.0059 -0.005  -0.0047 -0.0043 -0.0034 -0.0029 -0.0021 -0.001  -0.0004
  0.0011  0.0023  0.0025  0.0032  0.0041  0.0053  0.0099  0.0114  0.0136  0.0196]
(0, 2) trace 0.1083 SV [0.0493 0.028  0.0259 0.024  0.0205 0.02   0.0165 0.0162 0.0148 0.0135 0.0128 0.0119] sym eigenvalues [-0.0305 -0.0131 -0.011  -0.0071 -0.0061 -0.0038 -0.0031 -0.0023 -0.0018 -0.0007  0.0012  0.0017  0.0024  0.0039  0.0049  0.0051  0.0061  0.008
  0.0089  0.0093  0.0114  0.0121  0.0147  0.0158  0.0169  0.0178  0.0224  0.0253]
(1, 2) trace 0.2086 SV [0.0418 0.0349 0.0305 0.0278 0.0249 0.0216 0.0203 0.0179 0.0163 0.0147 0.0129 0.0125] sym eigenvalues [-0.0169 -0.0072 -0.0054 -0.0033 -0.0023 -0.0009 -0.0009 -0.0008  0.0004  0.0011  0.0016  0.0023  0.003   0.0048  0.0066  0.0075  0.0085  0.009
  0.0108  0.0126  0.0144  0.0164  0.0175  0.0202  0.0221  0.0272  0.0287  0.0319]

Cosines between the three symmetric cross-covariance matrices:
[[ 1.     0.017 -0.166]
 [ 0.017  1.     0.27 ]
 [-0.166  0.27   1.   ]]
Alternation-contrast eigenvalues [-0.038  -0.0277 -0.0191 -0.0177 -0.0166 -0.0134 -0.0119 -0.0089 -0.0087 -0.0078 -0.0056 -0.0044 -0.0027  0.      0.0004  0.0026  0.0048  0.0064
  0.0075  0.0096  0.0104  0.0149  0.015   0.0182  0.0207  0.0232  0.0322  0.0397]
mode 0 per-pair quadratic coefficients [-0.0194  0.0108 -0.0078]
mode 1 per-pair quadratic coefficients [-0.0131  0.0108 -0.0038]
mode 2 per-pair quadratic coefficients [-0.0063  0.0136  0.0008]
mode 3 per-pair quadratic coefficients [-0.0072  0.0135  0.003 ]

```
