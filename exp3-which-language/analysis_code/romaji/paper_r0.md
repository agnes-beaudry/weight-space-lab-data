## Call 1

```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=4):
    print("LayerNorm parameters: mean, std, min, max, norm")
    for l in range(4):
        for name in ("ln1_w", "ln1_b", "ln2_w", "ln2_b"):
            a = W[f"{name}.{l}"].astype(np.float64)
            print(f"{name}.{l}", np.round([a.mean(), a.std(), a.min(), a.max(), np.linalg.norm(a)], 4))
    print("\nMatrix summaries: Frobenius norm, top singular value, stable rank")
    for l in range(4):
        for name in ("W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"):
            a = W[f"{name}.{l}"].astype(np.float64)
            s = svdvals(a)
            print(f"{name}.{l}", np.round([np.linalg.norm(a), s[0], (s*s).sum()/s[0]**2], 4))
    print("\nHead algebra (no activations):")
    print("head | QK Frobenius, top SV, stable rank, trace, symmetric eig min/max"
          " | OV Frobenius, top SV, stable rank, trace | ||Wq b||, ||Wk b||")
    for l in range(4):
        b = W[f"ln1_b.{l}"].astype(np.float64)
        for j in range(6):
            s = slice(j*64, (j+1)*64)
            q = W[f"W_q.{l}"][s].astype(np.float64)
            k = W[f"W_k.{l}"][s].astype(np.float64)
            v = W[f"W_v.{l}"][s].astype(np.float64)
            o = W[f"W_o.{l}"][:, s].astype(np.float64)
            qk = q.T @ k / 8
            ov = o @ v
            sq, so = svdvals(qk), svdvals(ov)
            eq = np.linalg.eigvalsh((qk+qk.T)/2)
            a = [np.linalg.norm(qk), sq[0], (sq*sq).sum()/sq[0]**2,
                 np.trace(qk), eq[0], eq[-1]]
            c = [np.linalg.norm(ov), so[0], (so*so).sum()/so[0]**2, np.trace(ov)]
            print(f"L{l}H{j}", np.round(a, 3), "|", np.round(c, 3), "|",
                  np.round([np.linalg.norm(q@b), np.linalg.norm(k@b)], 3))

```

Output:

```
LayerNorm parameters: mean, std, min, max, norm
ln1_w.0 [0.2965 0.1884 0.0352 0.9785 6.8836]
ln1_b.0 [ 0.0004  0.0311 -0.0954  0.0906  0.6101]
ln2_w.0 [ 0.8718  0.3382  0.042   1.775  18.3232]
ln2_b.0 [ 0.0134  0.0691 -0.6236  0.4636  1.3799]
ln1_w.1 [ 0.5058  0.0892  0.197   1.0846 10.0651]
ln1_b.1 [ 0.011   0.1546 -1.5676  1.3651  3.0369]
ln2_w.1 [ 1.1315  0.1741  0.0019  1.5196 22.4333]
ln2_b.1 [ 0.0065  0.1435 -0.3892  0.417   2.8153]
ln1_w.2 [ 0.6809  0.0687  0.3175  1.0927 13.4102]
ln1_b.2 [ 0.0131  0.1565 -1.6589  1.3318  3.0765]
ln2_w.2 [ 1.3104  0.1148  0.012   1.5078 25.777 ]
ln2_b.2 [ 0.0123  0.1737 -0.517   0.4489  3.4127]
ln1_w.3 [ 0.7087  0.0494  0.4078  0.8155 13.9223]
ln1_b.3 [ 0.0296  0.0994 -0.7639  0.8241  2.0324]
ln2_w.3 [ 1.2668  0.0894  0.2328  1.4532 24.886 ]
ln2_b.3 [ 0.0128  0.1413 -0.5402  0.3932  2.7798]

Matrix summaries: Frobenius norm, top singular value, stable rank
W_q.0 [17.6693  4.3676 16.3664]
W_k.0 [18.5161  4.1635 19.7784]
W_v.0 [14.9367  5.2084  8.2242]
W_o.0 [16.5495  7.3608  5.0551]
W_fc.0 [55.3764 18.1182  9.3416]
W_proj.0 [50.007  10.899  21.0517]
W_q.1 [38.9293 10.4064 13.9942]
W_k.1 [34.8926 14.4767  5.8093]
W_v.1 [28.7445  4.7281 36.9599]
W_o.1 [28.479   4.0451 49.5664]
W_fc.1 [80.3794 19.8292 16.4317]
W_proj.1 [77.3844  9.0375 73.3173]
W_q.2 [38.9025  5.9663 42.5157]
W_k.2 [36.75    9.8633 13.8827]
W_v.2 [33.1825  3.9143 71.8631]
W_o.2 [32.4345  4.3982 54.3828]
W_fc.2 [79.8282 15.7338 25.7422]
W_proj.2 [78.0482 10.1974 58.5793]
W_q.3 [33.2388  4.0862 66.1679]
W_k.3 [32.4633  6.3618 26.039 ]
W_v.3 [39.4218  5.6601 48.5088]
W_o.3 [40.1331 10.4767 14.6743]
W_fc.3 [78.2595 15.1141 26.8106]
W_proj.3 [85.1194 28.4613  8.9443]

Head algebra (no activations):
head | QK Frobenius, top SV, stable rank, trace, symmetric eig min/max | OV Frobenius, top SV, stable rank, trace | ||Wq b||, ||Wk b||
L0H0 [ 1.382  0.853  2.623 -1.619 -0.425  0.501] | [ 7.879  3.611  4.76  -0.676] | [0.504 0.725]
L0H1 [ 0.95   0.838  1.284 -0.847 -0.599  0.243] | [6.871 5.721 1.443 0.527] | [0.228 0.261]
L0H2 [ 0.634  0.543  1.362 -0.598 -0.388  0.16 ] | [4.9   3.943 1.545 0.277] | [0.146 0.314]
L0H3 [ 4.305  1.114 14.933  7.641 -0.478  0.899] | [14.071  4.426 10.107 -7.452] | [0.467 0.543]
L0H4 [ 0.636  0.51   1.56  -0.545 -0.338  0.176] | [ 3.116  2.449  1.62  -0.41 ] | [0.27 0.55]
L0H5 [ 0.21   0.18   1.348 -0.051 -0.104  0.077] | [29.436 24.799  1.409  2.348] | [0.143 0.171]
L1H0 [ 6.605  3.084  4.587 -6.95  -1.73   1.612] | [ 22.128   6.118  13.083 -30.069] | [ 6.688 13.795]
L1H1 [ 6.033  3.082  3.832 -5.844 -1.703  1.507] | [ 21.591   6.051  12.731 -29.585] | [ 6.079 13.929]
L1H2 [ 5.712  2.848  4.023 -7.262 -1.622  1.395] | [ 22.746   6.72   11.455 -33.608] | [ 5.564 13.162]
L1H3 [ 5.738  3.106  3.412 -5.817 -1.714  1.533] | [ 23.885   6.544  13.32  -36.07 ] | [ 5.665 13.742]
L1H4 [ 5.304  2.62   4.099 -7.654 -1.345  1.364] | [ 25.845   7.908  10.681 -37.328] | [ 5.347 10.986]
L1H5 [ 6.123  3.024  4.1   -6.208 -1.85   1.487] | [ 20.966   5.849  12.848 -28.92 ] | [ 6.224 14.146]
L2H0 [ 3.354  1.229  7.451 -3.371 -0.57   0.681] | [ 23.754   7.576   9.831 -24.508] | [ 4.265 10.635]
L2H1 [ 4.549  1.189 14.639 -0.512 -0.629  0.665] | [ 25.597   5.978  18.336 -50.868] | [ 3.787 10.117]
L2H2 [ 3.612  0.963 14.072 -2.835 -0.528  0.522] | [ 27.059   7.257  13.902 -43.194] | [3.784 9.886]
L2H3 [ 3.414  0.896 14.503 -1.229 -0.453  0.509] | [ 27.609   7.041  15.375 -43.815] | [3.082 7.454]
L2H4 [ 4.594  1.537  8.928 -2.083 -0.752  0.842] | [ 21.682   5.217  17.274 -45.57 ] | [ 4.319 11.042]
L2H5 [ 4.752  1.229 14.942 -0.662 -0.671  0.711] | [ 24.543   5.432  20.412 -50.098] | [ 3.751 10.382]
L3H0 [ 4.017  0.839 22.904  4.356 -0.395  0.532] | [ 29.023   8.117  12.783 -64.616] | [1.761 3.665]
L3H1 [ 2.412  0.658 13.446  0.9   -0.333  0.379] | [ 37.053  11.496  10.389 -58.624] | [1.46  3.019]
L3H2 [ 2.848  0.734 15.049  0.674 -0.362  0.377] | [ 32.882  10.376  10.042 -52.39 ] | [1.94 3.36]
L3H3 [ 3.627  0.645 31.642 11.586 
```

## Call 2

```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

def cosine(a, b):
    return float(np.vdot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)))

with threadpool_limits(limits=4):
    print("Layer 0 singular spectra (first 60, then indices 80, 120, 200, 300, 383):")
    for name in ("W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"):
        s = svdvals(W[f"{name}.0"].astype(float))
        print(name, np.round(s[:60], 3))
        print("tail", np.round(s[[80,120,200,300,383]], 4))

    qks, ovs = [], []
    print("\nLayer 0 individual-head spectra (top 12):")
    for j in range(6):
        sl = slice(64*j, 64*(j+1))
        q = W["W_q.0"][sl].astype(float)
        k = W["W_k.0"][sl].astype(float)
        v = W["W_v.0"][sl].astype(float)
        o = W["W_o.0"][:,sl].astype(float)
        qks.append(q.T @ k / 8)
        ovs.append(o @ v)
        print("H", j, "QK", np.round(svdvals(qks[-1])[:12], 4),
              "OV", np.round(svdvals(ovs[-1])[:12], 4))
    print("\nLayer 0 QK Frobenius cosines")
    print(np.round([[cosine(a,b) for b in qks] for a in qks], 3))
    print("Layer 0 OV Frobenius cosines")
    print(np.round([[cosine(a,b) for b in ovs] for a in ovs], 3))

    print("\nLayerNorm largest absolute bias coordinates:")
    for l in range(4):
        for name in ("ln1_b", "ln2_b"):
            a = W[f"{name}.{l}"]
            ix = np.argsort(np.abs(a))[-8:][::-1]
            print(f"{name}.{l}", list(zip(ix.tolist(), np.round(a[ix],4).tolist())))

    print("\nAffine QK algebra using learned LN scale and bias:")
    print("head | ||A||, top SV, stable rank, tr(A), ||key-linear coefficient||,"
          " coefficient fraction in top key singular direction")
    for l in range(4):
        g = W[f"ln1_w.{l}"].astype(float)
        b = W[f"ln1_b.{l}"].astype(float)
        for j in range(6):
            sl = slice(64*j, 64*(j+1))
            q = W[f"W_q.{l}"][sl].astype(float)
            k = W[f"W_k.{l}"][sl].astype(float)
            qg, kg = q*g, k*g
            qg -= qg.mean(axis=1, keepdims=True)
            kg -= kg.mean(axis=1, keepdims=True)
            a = qg.T @ kg / 8
            u, s, vt = np.linalg.svd(a, full_matrices=False)
            c = (q@b) @ kg / 8
            frac = abs(c@vt[0])/(np.linalg.norm(c)+1e-30)
            print(f"L{l}H{j}", np.round([np.linalg.norm(a),s[0],
                  (s*s).sum()/s[0]**2,np.trace(a),np.linalg.norm(c),frac],4))

```

Output:

```
Layer 0 singular spectra (first 60, then indices 80, 120, 200, 300, 383):
W_q [4.368 3.131 3.099 3.042 2.994 2.951 2.926 2.881 2.859 2.857 2.804 2.766 2.762 2.692 2.683 2.661 2.637 2.588 2.539 2.509 2.458 2.433 2.403 2.299
 2.272 2.173 2.098 2.048 1.954 1.887 1.877 1.865 1.738 1.725 1.665 1.605 1.552 1.535 1.476 1.424 1.378 1.341 1.306 1.264 1.24  1.213 1.154 1.122
 1.109 1.068 1.053 1.031 1.017 1.007 0.985 0.957 0.944 0.913 0.888 0.879]
tail [0.6556 0.4899 0.3045 0.1303 0.0006]
W_k [4.163 3.952 3.467 3.427 3.364 3.335 3.279 3.256 3.214 3.163 3.115 3.029 2.977 2.901 2.881 2.832 2.798 2.76  2.658 2.584 2.528 2.449 2.353 2.301
 2.281 2.206 2.121 2.074 1.917 1.896 1.813 1.753 1.713 1.676 1.573 1.53  1.47  1.431 1.415 1.355 1.3   1.26  1.187 1.169 1.161 1.116 1.108 1.104
 1.072 1.032 1.003 0.988 0.965 0.953 0.931 0.911 0.907 0.885 0.873 0.866]
tail [0.6608 0.5097 0.317  0.1399 0.0003]
W_v [5.208 3.364 3.006 2.962 2.815 2.639 2.527 2.46  2.37  2.301 2.25  2.23  2.151 2.135 2.044 2.039 2.001 1.981 1.91  1.875 1.822 1.773 1.749 1.708
 1.634 1.611 1.556 1.487 1.453 1.402 1.268 1.213 1.187 1.13  1.109 1.08  1.026 0.992 0.987 0.959 0.933 0.931 0.908 0.892 0.887 0.866 0.864 0.853
 0.843 0.827 0.811 0.793 0.783 0.781 0.759 0.748 0.721 0.715 0.71  0.701]
tail [0.5718 0.4517 0.2931 0.1256 0.0007]
W_o [7.361 6.099 3.167 2.452 2.321 2.202 2.171 2.16  2.105 2.072 2.021 1.995 1.975 1.967 1.935 1.922 1.905 1.885 1.857 1.84  1.815 1.784 1.759 1.744
 1.712 1.694 1.634 1.611 1.585 1.553 1.511 1.466 1.456 1.446 1.397 1.392 1.37  1.336 1.321 1.305 1.294 1.27  1.248 1.218 1.197 1.182 1.167 1.148
 1.136 1.099 1.095 1.077 1.049 1.044 1.011 1.005 0.998 0.985 0.966 0.941]
tail [0.6103 0.3674 0.1734 0.059  0.0002]
W_fc [18.118 12.76  11.337  9.323  9.267  9.021  8.898  8.817  8.609  8.4    8.189  8.06   7.819  7.724  7.62   7.543  7.226  7.081  6.979  6.833  6.779
  6.682  6.592  6.402  6.39   6.306  6.207  6.18   5.974  5.864  5.789  5.741  5.646  5.579  5.512  5.353  5.196  5.193  5.122  5.053  5.029  4.849
  4.646  4.573  4.452  4.163  4.133  3.899  3.888  3.557  3.455  3.347  3.241  3.008  2.921  2.833  2.796  2.657  2.612  2.555]
tail [1.7122 1.2795 0.9228 0.6587 0.4311]
W_proj [10.899  7.702  7.173  6.638  6.541  6.288  6.201  6.133  5.941  5.917  5.758  5.641  5.552  5.476  5.393  5.363  5.259  5.206  5.171  5.153  5.057
  5.018  4.96   4.88   4.821  4.788  4.77   4.73   4.643  4.579  4.546  4.529  4.505  4.493  4.44   4.415  4.369  4.315  4.293  4.261  4.243  4.185
  4.134  4.104  4.064  4.049  4.026  3.988  3.967  3.95   3.933  3.901  3.857  3.804  3.778  3.754  3.74   3.713  3.692  3.683]
tail [3.1846 2.4346 1.3916 0.6756 0.2813]

Layer 0 individual-head spectra (top 12):
H 0 QK [0.8532 0.6126 0.3681 0.2867 0.2786 0.2441 0.2317 0.2099 0.2016 0.1885 0.1818 0.1728] OV [3.6115 3.3924 2.8811 2.3143 2.1276 1.8367 1.6785 1.5294 1.377  1.3026 1.1515 1.1013]
H 1 QK [0.838  0.3105 0.2193 0.1245 0.0908 0.0863 0.0659 0.0389 0.0378 0.0336 0.0332 0.0297] OV [5.7206 3.3585 1.1458 0.5449 0.4435 0.3867 0.3839 0.3541 0.331  0.3111 0.2885 0.2674]
H 2 QK [0.5432 0.1831 0.1574 0.1329 0.0717 0.0659 0.0464 0.0456 0.0362 0.0332 0.0321 0.0292] OV [3.9425 2.3569 1.0061 0.5213 0.5    0.4496 0.3863 0.3309 0.3101 0.3051 0.2774 0.2548]
H 3 QK [1.114  1.0924 1.0705 1.064  1.0301 0.9924 0.9677 0.9537 0.9464 0.9002 0.8883 0.8685] OV [4.4261 3.911  3.6812 3.5972 3.5958 3.4449 3.343  3.2829 3.0733 3.0135 2.9723 2.9144]
H 4 QK [0.5096 0.2267 0.1852 0.1237 0.1013 0.0658 0.0655 0.0489 0.0484 0.0454 0.0401 0.035 ] OV [2.4485 1.3166 0.666  0.4611 0.4377 0.3631 0.3568 0.3142 0.2959 0.2618 0.2484 0.2372]
H 5 QK [0.1805 0.0481 0.0241 0.0205 0.0202 0.0191 0.0188 0.0181 0.0179 0.0172 0.0169 0.0164] OV [24.7988 12.8996  5.8492  4.2294  3.6654  2.3878  1.919   1.6636  1.3942  1.1336  0.8624  0.7865]

Layer 0 QK Frobenius cosines
[[ 1.     0.032  0.067 -0.132  0.114  0.017]
 [ 0.032  1.     0.58  -0.022  0.419 -0.008]
 [ 0.067  0.58   1.    -0.028  0.451 -0.053]
 [-0.132 -0.022 -0.028  1.    
```

## Call 3

```python
import numpy as np
from scipy.linalg import svd, eigvals
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=4):
    g = W["ln1_w.0"].astype(float)
    A, B, decomps = [], [], []
    for j in range(6):
        sl = slice(64*j,64*(j+1))
        q = W["W_q.0"][sl].astype(float)*g
        k = W["W_k.0"][sl].astype(float)*g
        v = W["W_v.0"][sl].astype(float)*g
        q -= q.mean(1,keepdims=True)
        k -= k.mean(1,keepdims=True)
        v -= v.mean(1,keepdims=True)
        A.append(q.T@k/8)
        B.append(W["W_o.0"][:,sl].astype(float)@v)
        decomps.append(svd(A[-1], full_matrices=False))
    print("Layer 0 scaled QK: symmetric-energy fraction, positive/negative eigenvalue sums of symmetric part")
    for j,a in enumerate(A):
        sym = (a+a.T)/2
        ev = np.linalg.eigvalsh(sym)
        z = eigvals(a)
        z = z[np.argsort(abs(z))[::-1]][:24]
        print("H", j, np.round([np.linalg.norm(sym)**2/np.linalg.norm(a)**2,
                               ev[ev>0].sum(),ev[ev<0].sum()],4))
        print("eigenvalues:", np.round(z,4))

    print("\nFraction of each OV's squared norm captured by QK H3 right/left subspaces")
    u,s,vt = decomps[3]
    for r in (4,8,16,24,32,48):
        print("rank", r,
              "key", np.round([np.linalg.norm(b@vt[:r].T)**2/np.linalg.norm(b)**2 for b in B],3),
              "query", np.round([np.linalg.norm(b@u[:,:r])**2/np.linalg.norm(b)**2 for b in B],3))

    print("\nOwn-head QK/OV subspace overlaps (r=1,4,8,16,24):")
    for j in range(6):
        u,s,vt=decomps[j]
        _,sb,vb=svd(B[j],full_matrices=False)
        vals=[]
        for r in (1,4,8,16,24):
            vals.append([np.linalg.norm(u[:,:r].T@vb[:r].T)**2/r,
                         np.linalg.norm(vt[:r]@vb[:r].T)**2/r])
        print("H",j,np.round(vals,3))

    print("\nLayer 0 cross-head subspace overlap: top 8 query and key directions")
    for which in ("query","key"):
        bases=[z[0][:,:8] if which=="query" else z[2][:8].T for z in decomps]
        print(which, np.round([[np.linalg.norm(a.T@b)**2/8 for b in bases] for a in bases],3))

    print("\nQK H3 singular-vector pair cosines:")
    u,s,vt=decomps[3]
    print(np.round(np.diag(vt[:32]@u[:,:32]),3))
    print("SV",np.round(s[:40],4))
    print("\nL0 LN scale largest and smallest coordinates:")
    for ix in (np.argsort(g)[-24:][::-1],np.argsort(g)[:24]):
        print(list(zip(ix.tolist(),np.round(g[ix],4).tolist())))

```

Output:

```
Layer 0 scaled QK: symmetric-energy fraction, positive/negative eigenvalue sums of symmetric part
H 0 [ 0.5704  0.3783 -0.483 ]
eigenvalues: [ 0.0796+0.j      0.0483+0.j     -0.024 +0.0057j -0.024 -0.0057j -0.0226+0.j      0.0176+0.0134j  0.0176-0.0134j -0.0201+0.j     -0.0138-0.0121j
 -0.0138+0.0121j -0.0139-0.0031j -0.0139+0.0031j -0.0137+0.j     -0.011 -0.0039j -0.011 +0.0039j -0.0111+0.0023j -0.0111-0.0023j -0.006 -0.0062j
 -0.006 +0.0062j -0.0063+0.j     -0.003 -0.0049j -0.003 +0.0049j -0.0052-0.0023j -0.0052+0.0023j]
H 1 [ 0.5805  0.1152 -0.2136]
eigenvalues: [-0.0619+0.j     -0.01  +0.j     -0.0089+0.j      0.0043+0.j     -0.0039+0.j     -0.0033-0.0018j -0.0033+0.0018j -0.0017+0.0004j -0.0017-0.0004j
 -0.0016-0.0007j -0.0016+0.0007j -0.0011+0.0003j -0.0011-0.0003j -0.0007+0.0008j -0.0007-0.0008j  0.001 +0.j     -0.0009+0.j      0.0004+0.0007j
  0.0004-0.0007j -0.0007-0.0005j -0.0007+0.0005j -0.0005-0.0007j -0.0005+0.0007j -0.0002-0.0007j]
H 2 [ 0.6443  0.0809 -0.1682]
eigenvalues: [-0.0566+0.j     -0.0074-0.0027j -0.0074+0.0027j -0.0034-0.0008j -0.0034+0.0008j -0.0025+0.j     -0.0013-0.0003j -0.0013+0.0003j -0.0006-0.0012j
 -0.0006+0.0012j  0.0006+0.001j   0.0006-0.001j  -0.0003-0.001j  -0.0003+0.001j  -0.001 +0.j      0.0001+0.001j   0.0001-0.001j  -0.0005-0.0008j
 -0.0005+0.0008j -0.0007+0.0006j -0.0007-0.0006j  0.0008+0.j     -0.0006-0.0004j -0.0006+0.0004j]
H 3 [ 0.4664  1.9585 -0.617 ]
eigenvalues: [ 0.1366+0.1327j  0.1366-0.1327j  0.1099+0.0891j  0.1099-0.0891j  0.095 +0.0979j  0.095 -0.0979j  0.0841+0.1041j  0.0841-0.1041j  0.0644+0.1142j
  0.0644-0.1142j  0.0957-0.0687j  0.0957+0.0687j  0.0114+0.1166j  0.0114-0.1166j  0.1116+0.j      0.0559+0.0696j  0.0559-0.0696j  0.0749+0.0468j
  0.0749-0.0468j  0.0135-0.0747j  0.0135+0.0747j -0.0348-0.0526j -0.0348+0.0526j -0.0451+0.0173j]
H 4 [ 0.5251  0.103  -0.1489]
eigenvalues: [-0.0152+0.j     -0.0095+0.j     -0.0077+0.j     -0.0026-0.0003j -0.0026+0.0003j -0.0022-0.0014j -0.0022+0.0014j -0.0002+0.0023j -0.0002-0.0023j
 -0.0014-0.0012j -0.0014+0.0012j -0.0015+0.j     -0.0006-0.0011j -0.0006+0.0011j  0.0008-0.0009j  0.0008+0.0009j  0.0012+0.j     -0.0003-0.0011j
 -0.0003+0.0011j  0.0005-0.0009j  0.0005+0.0009j -0.001 +0.j      0.001 +0.j     -0.0009-0.0001j]
H 5 [ 0.5122  0.0451 -0.0536]
eigenvalues: [-0.0049+0.j      0.0013+0.j     -0.0012+0.j     -0.0004-0.0007j -0.0004+0.0007j -0.0008+0.j     -0.0007+0.0002j -0.0007-0.0002j  0.0002-0.0007j
  0.0002+0.0007j -0.0001-0.0006j -0.0001+0.0006j -0.0003-0.0005j -0.0003+0.0005j  0.0002-0.0006j  0.0002+0.0006j -0.0005-0.0003j -0.0005+0.0003j
  0.0006+0.j     -0.0005+0.0001j -0.0005-0.0001j  0.0005+0.0001j  0.0005-0.0001j -0.0004+0.0004j]

Fraction of each OV's squared norm captured by QK H3 right/left subspaces
rank 4 key [0.007 0.441 0.562 0.012 0.362 0.826] query [0.025 0.484 0.585 0.011 0.354 0.697]
rank 8 key [0.014 0.462 0.584 0.027 0.374 0.853] query [0.033 0.503 0.612 0.026 0.382 0.732]
rank 16 key [0.025 0.48  0.608 0.06  0.413 0.882] query [0.043 0.528 0.646 0.054 0.422 0.767]
rank 24 key [0.037 0.503 0.625 0.085 0.445 0.898] query [0.058 0.577 0.679 0.074 0.487 0.782]
rank 32 key [0.071 0.525 0.64  0.112 0.475 0.907] query [0.095 0.679 0.727 0.089 0.645 0.811]
rank 48 key [0.263 0.605 0.707 0.29  0.551 0.924] query [0.288 0.831 0.844 0.25  0.787 0.887]

Own-head QK/OV subspace overlaps (r=1,4,8,16,24):
H 0 [[0.003 0.   ]
 [0.081 0.117]
 [0.073 0.093]
 [0.092 0.113]
 [0.15  0.168]]
H 1 [[0.175 0.078]
 [0.267 0.106]
 [0.188 0.106]
 [0.153 0.142]
 [0.162 0.162]]
H 2 [[0.011 0.071]
 [0.263 0.112]
 [0.206 0.118]
 [0.159 0.135]
 [0.175 0.166]]
H 3 [[0.    0.002]
 [0.005 0.004]
 [0.01  0.011]
 [0.02  0.024]
 [0.116 0.133]]
H 4 [[0.437 0.049]
 [0.191 0.058]
 [0.144 0.073]
 [0.136 0.121]
 [0.153 0.166]]
H 5 [[0.356 0.   ]
 [0.248 0.057]
 [0.171 0.093]
 [0.142 0.106]
 [0.152 0.132]]

Layer 0 cross-head subspace overlap: top 8 query and key directions
query [[1.    0.217 0.215 0.332 0.189 0.157]
 [0.217 1.    0.488 0.12  0.419 0.
```

## Call 4

```python
import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=4):
    A, B = {}, {}
    for l in range(4):
        g = W[f"ln1_w.{l}"].astype(float)
        for j in range(6):
            sl = slice(j*64,(j+1)*64)
            q = W[f"W_q.{l}"][sl].astype(float)*g
            k = W[f"W_k.{l}"][sl].astype(float)*g
            v = W[f"W_v.{l}"][sl].astype(float)*g
            q -= q.mean(1,keepdims=True)
            k -= k.mean(1,keepdims=True)
            v -= v.mean(1,keepdims=True)
            A[l,j] = q.T@k/8
            B[l,j] = W[f"W_o.{l}"][:,sl].astype(float)@v
    print("Key-path matching: tr(QK_l @ OV_0)/(||QK_l||F ||OV_0||F)")
    for l in range(1,4):
        for j in range(6):
            a=A[l,j]
            print(f"L{l}H{j}",np.round([np.trace(a@B[0,h])/
                   (np.linalg.norm(a)*np.linalg.norm(B[0,h])) for h in range(6)],4))
    print("\nLargest matching compositions across all preceding heads:")
    for l in range(1,4):
        for j in range(6):
            a=A[l,j]
            scores=[]
            for m in range(l):
                for h in range(6):
                    b=B[m,h]
                    z=np.trace(a@b)/(np.linalg.norm(a)*np.linalg.norm(b))
                    scores.append((z,f"L{m}H{h}"))
            print(f"L{l}H{j}",[(h,round(z,4)) for z,h in sorted(scores,reverse=True)[:4]])

    print("\nL0H3 OV spectrum after LN scale/centering")
    u,s,vt=svd(B[0,3],full_matrices=False)
    print(np.round(s[:64],4))
    print("L0H0 OV spectrum")
    print(np.round(svd(B[0,0],compute_uv=False)[:48],4))

    print("\nMLP0 input-weight Gram coupling between current-token and H3-written subspaces")
    f=W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float)
    f-=f.mean(1,keepdims=True)
    gram=f.T@f
    for r in (12,16,20,24,28,32):
        basis=vt[:r].T
        t=basis.T@gram@B[0,3]@basis
        z=np.linalg.eigvals(t)
        z=z[np.argsort(abs(z))[::-1]][:10]
        print("rank",r,"eigs",np.round(z,3))

    print("\nLater-head QK on H3's read-to-write circuit (r=24)")
    basis=vt[:24].T
    for l in range(1,4):
        for j in range(6):
            t=basis.T@A[l,j]@B[0,3]@basis
            sy=(t+t.T)/2
            ev=np.linalg.eigvalsh(sy)
            print(f"L{l}H{j}",np.round([
                  np.linalg.norm(t),np.trace(t),
                  np.linalg.norm(sy)**2/(np.linalg.norm(t)**2+1e-30),
                  ev[-1],ev[0]],4))

```

Output:

```
Key-path matching: tr(QK_l @ OV_0)/(||QK_l||F ||OV_0||F)
L1H0 [0.0018 0.0341 0.0502 0.0058 0.0292 0.0479]
L1H1 [0.0004 0.0065 0.025  0.0141 0.0042 0.0212]
L1H2 [-0.0018  0.0296  0.0418  0.0121  0.0312  0.0467]
L1H3 [0.0012 0.0283 0.0453 0.0051 0.0242 0.0429]
L1H4 [0.0007 0.0167 0.0297 0.001  0.0233 0.0368]
L1H5 [-0.0012  0.0195  0.0396  0.0112  0.0194  0.0307]
L2H0 [-0.0004  0.0385  0.0584  0.0038  0.0229  0.0667]
L2H1 [0.0014 0.021  0.0334 0.0005 0.0129 0.0425]
L2H2 [0.0025 0.0257 0.0391 0.0053 0.0136 0.0461]
L2H3 [0.0031 0.0164 0.0226 0.0047 0.0122 0.0247]
L2H4 [-0.0014  0.0299  0.0451  0.0088  0.022   0.0512]
L2H5 [0.0008 0.0211 0.0362 0.0006 0.0135 0.0475]
L3H0 [-0.0025  0.0081  0.0128 -0.0104  0.0049  0.013 ]
L3H1 [ 0.0024  0.0098  0.0134 -0.0017  0.0037  0.0139]
L3H2 [-0.0004  0.0069  0.0119  0.0018  0.0006  0.0036]
L3H3 [-0.008   0.0019  0.0021  0.0626  0.0024 -0.0016]
L3H4 [0.0052 0.0093 0.0092 0.0064 0.0104 0.0089]
L3H5 [-0.0004  0.0024  0.0073 -0.0032  0.      0.0088]

Largest matching compositions across all preceding heads:
L1H0 [('L0H2', np.float64(0.0502)), ('L0H5', np.float64(0.0479)), ('L0H1', np.float64(0.0341)), ('L0H4', np.float64(0.0292))]
L1H1 [('L0H2', np.float64(0.025)), ('L0H5', np.float64(0.0212)), ('L0H3', np.float64(0.0141)), ('L0H1', np.float64(0.0065))]
L1H2 [('L0H5', np.float64(0.0467)), ('L0H2', np.float64(0.0418)), ('L0H4', np.float64(0.0312)), ('L0H1', np.float64(0.0296))]
L1H3 [('L0H2', np.float64(0.0453)), ('L0H5', np.float64(0.0429)), ('L0H1', np.float64(0.0283)), ('L0H4', np.float64(0.0242))]
L1H4 [('L0H5', np.float64(0.0368)), ('L0H2', np.float64(0.0297)), ('L0H4', np.float64(0.0233)), ('L0H1', np.float64(0.0167))]
L1H5 [('L0H2', np.float64(0.0396)), ('L0H5', np.float64(0.0307)), ('L0H1', np.float64(0.0195)), ('L0H4', np.float64(0.0194))]
L2H0 [('L0H5', np.float64(0.0667)), ('L0H2', np.float64(0.0584)), ('L0H1', np.float64(0.0385)), ('L0H4', np.float64(0.0229))]
L2H1 [('L0H5', np.float64(0.0425)), ('L0H2', np.float64(0.0334)), ('L0H1', np.float64(0.021)), ('L0H4', np.float64(0.0129))]
L2H2 [('L0H5', np.float64(0.0461)), ('L0H2', np.float64(0.0391)), ('L0H1', np.float64(0.0257)), ('L0H4', np.float64(0.0136))]
L2H3 [('L0H5', np.float64(0.0247)), ('L0H2', np.float64(0.0226)), ('L0H1', np.float64(0.0164)), ('L0H4', np.float64(0.0122))]
L2H4 [('L0H5', np.float64(0.0512)), ('L0H2', np.float64(0.0451)), ('L0H1', np.float64(0.0299)), ('L0H4', np.float64(0.022))]
L2H5 [('L0H5', np.float64(0.0475)), ('L0H2', np.float64(0.0362)), ('L0H1', np.float64(0.0211)), ('L0H4', np.float64(0.0135))]
L3H0 [('L0H5', np.float64(0.013)), ('L0H2', np.float64(0.0128)), ('L0H1', np.float64(0.0081)), ('L0H4', np.float64(0.0049))]
L3H1 [('L0H5', np.float64(0.0139)), ('L0H2', np.float64(0.0134)), ('L0H1', np.float64(0.0098)), ('L0H4', np.float64(0.0037))]
L3H2 [('L0H2', np.float64(0.0119)), ('L0H1', np.float64(0.0069)), ('L0H5', np.float64(0.0036)), ('L0H3', np.float64(0.0018))]
L3H3 [('L0H3', np.float64(0.0626)), ('L2H5', np.float64(0.0482)), ('L2H1', np.float64(0.0328)), ('L1H0', np.float64(0.0305))]
L3H4 [('L2H2', np.float64(0.0143)), ('L2H4', np.float64(0.0122)), ('L0H4', np.float64(0.0104)), ('L2H5', np.float64(0.0094))]
L3H5 [('L0H5', np.float64(0.0088)), ('L0H2', np.float64(0.0073)), ('L0H1', np.float64(0.0024)), ('L0H4', np.float64(0.0))]

L0H3 OV spectrum after LN scale/centering
[1.126  0.9554 0.9495 0.8895 0.8632 0.8537 0.8418 0.7799 0.7586 0.7228 0.7078 0.6727 0.6691 0.6516 0.6339 0.6167 0.5652 0.528  0.4384 0.3824 0.3491
 0.3158 0.2968 0.2947 0.2815 0.2755 0.2646 0.246  0.2403 0.233  0.2243 0.2131 0.2093 0.1972 0.1892 0.1722 0.1586 0.1483 0.1368 0.1327 0.1111 0.0978
 0.083  0.0703 0.0613 0.0516 0.0467 0.0461 0.0386 0.0335 0.0318 0.0305 0.0265 0.0261 0.0251 0.0226 0.0221 0.0213 0.0206 0.0184 0.0178 0.0146 0.0127
 0.0108]
L0H0 OV spectrum
[0.9574 0.898  0.8182 0.6413 0.5544 0.4742 0.4388 0.4058 0.3721 0.3476 0.3294 0.3082 0.2579 0.2439 0.2052 0.1953 0.1708 0.1431 0.1304 0.1084 0.096
 0.0926 0.0869 0.0828 0.069
```

## Call 5

```python
import numpy as np
from scipy.linalg import svd, eigh
from threadpoolctl import threadpool_limits

def orth(a, tol=1e-8):
    u,s,_=svd(a,full_matrices=False)
    return u[:,s>s[0]*tol]

with threadpool_limits(limits=4):
    A,B,C={},{},{}
    for l in range(4):
        g=W[f"ln1_w.{l}"].astype(float)
        b=W[f"ln1_b.{l}"].astype(float)
        for j in range(6):
            sl=slice(j*64,(j+1)*64)
            q=W[f"W_q.{l}"][sl].astype(float)
            k=W[f"W_k.{l}"][sl].astype(float)
            v=W[f"W_v.{l}"][sl].astype(float)*g
            qg=q*g; kg=k*g
            qg-=qg.mean(1,keepdims=True)
            kg-=kg.mean(1,keepdims=True)
            v-=v.mean(1,keepdims=True)
            A[l,j]=qg.T@kg/8
            B[l,j]=W[f"W_o.{l}"][:,sl].astype(float)@v
            C[l,j]=(q@b)@kg/8

    up,sp,vp=svd(A[0,3],full_matrices=False)
    ut,st,vt=svd(B[0,3],full_matrices=False)
    pos=orth(np.column_stack([up[:,:24],vp[:24].T]))
    tok=orth((np.eye(384)-pos@pos.T)@vt[:20].T)
    writes=orth(np.column_stack([svd(B[0,j],full_matrices=False)[0][:,:3]
                                for j in (1,2,4,5)]))
    print("Basis dimensions: position candidate, token candidate, low-rank L0 writes",
          pos.shape[1],tok.shape[1],writes.shape[1])
    print("QK squared-energy fractions in query/key subspaces; then bias-key fraction")
    for l in range(4):
        for j in range(6):
            a=A[l,j]; c=C[l,j]
            vals=[]
            for basis in (pos,tok,writes):
                vals.extend([np.linalg.norm(basis.T@a)**2/np.linalg.norm(a)**2,
                             np.linalg.norm(a@basis)**2/np.linalg.norm(a)**2,
                             np.linalg.norm(c@basis)**2/(np.linalg.norm(c)**2+1e-30)])
            print(f"L{l}H{j}",np.round(vals,3))

    print("\nLayer 1 and 2 key-bias coefficient cosines")
    cs=np.array([C[l,j]/np.linalg.norm(C[l,j]) for l in (1,2) for j in range(6)])
    print(np.round(cs@cs.T,3))
    print("Largest coordinates of mean normalized L1 key-bias coefficient")
    c=cs[:6].mean(0)
    ix=np.argsort(abs(c))[-16:][::-1]
    print(list(zip(ix.tolist(),np.round(c[ix],4).tolist())))

    print("\nPosition-subspace operator correlations with powers of L0H3")
    p=pos
    a=p.T@A[0,3]@p
    metric=(a@a.T+a.T@a)/2
    ev,evc=eigh(metric)
    keep=ev>ev.max()*0.01
    z=evc[:,keep]/ev[keep]**0.25
    a=z.T@a@z
    powers=[np.eye(a.shape[0])]
    for n in range(1,7):
        powers.append(powers[-1]@a)
    for j in range(6):
        aj=z.T@(p.T@A[0,j]@p)@z
        scores=[np.vdot(aj,x)/(np.linalg.norm(aj)*np.linalg.norm(x)) for x in powers]
        print("H",j,np.round(scores,3))

    print("\nMLP0 coefficient coupling in H3 read subspace (generalized eigenvalues)")
    f=W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float)
    f-=f.mean(1,keepdims=True)
    for r in (16,20,24,28):
        basis=vt[:r].T
        f0=f@basis
        f1=f@B[0,3]@basis
        for centered in (False,True):
            x=f0-f0.mean(0) if centered else f0
            y=f1-f1.mean(0) if centered else f1
            reg=np.linalg.solve(x.T@x+0.01*np.eye(r),x.T@y)
            ev=np.linalg.eigvals(reg)
            ev=ev[np.argsort(abs(ev))[::-1]][:8]
            print("r",r,"centered",centered,np.round(ev,3))

```

Output:

```
Basis dimensions: position candidate, token candidate, low-rank L0 writes 48 20 12
QK squared-energy fractions in query/key subspaces; then bias-key fraction
L0H0 [0.791 0.818 0.528 0.058 0.073 0.247 0.169 0.083 0.062]
L0H1 [0.702 0.544 0.402 0.051 0.041 0.154 0.097 0.066 0.042]
L0H2 [0.777 0.56  0.513 0.036 0.118 0.145 0.046 0.067 0.067]
L0H3 [0.984 0.983 0.914 0.001 0.002 0.051 0.079 0.098 0.151]
L0H4 [0.676 0.493 0.499 0.078 0.176 0.206 0.095 0.058 0.054]
L0H5 [0.816 0.338 0.321 0.017 0.414 0.429 0.238 0.056 0.051]
L1H0 [0.128 0.23  0.281 0.098 0.172 0.165 0.17  0.441 0.858]
L1H1 [0.141 0.234 0.252 0.098 0.188 0.191 0.18  0.476 0.856]
L1H2 [0.131 0.233 0.244 0.086 0.171 0.232 0.152 0.498 0.857]
L1H3 [0.153 0.245 0.267 0.07  0.155 0.173 0.171 0.509 0.86 ]
L1H4 [0.124 0.221 0.223 0.072 0.141 0.182 0.108 0.459 0.808]
L1H5 [0.149 0.236 0.277 0.106 0.18  0.192 0.179 0.463 0.863]
L2H0 [0.151 0.203 0.281 0.052 0.082 0.166 0.132 0.272 0.848]
L2H1 [0.134 0.149 0.277 0.055 0.075 0.153 0.067 0.121 0.833]
L2H2 [0.131 0.157 0.261 0.046 0.073 0.138 0.079 0.15  0.807]
L2H3 [0.118 0.134 0.25  0.044 0.065 0.116 0.048 0.114 0.641]
L2H4 [0.133 0.169 0.248 0.053 0.073 0.114 0.088 0.189 0.84 ]
L2H5 [0.123 0.152 0.272 0.055 0.08  0.121 0.068 0.126 0.861]
L3H0 [0.115 0.128 0.274 0.049 0.06  0.083 0.034 0.066 0.72 ]
L3H1 [0.125 0.131 0.247 0.048 0.059 0.086 0.037 0.087 0.528]
L3H2 [0.121 0.141 0.281 0.048 0.062 0.127 0.044 0.088 0.666]
L3H3 [0.112 0.117 0.114 0.07  0.049 0.046 0.027 0.029 0.031]
L3H4 [0.123 0.139 0.172 0.057 0.058 0.055 0.046 0.043 0.079]
L3H5 [0.118 0.138 0.297 0.049 0.062 0.102 0.035 0.087 0.754]

Layer 1 and 2 key-bias coefficient cosines
[[1.    0.829 0.853 0.823 0.825 0.86  0.871 0.831 0.837 0.732 0.841 0.838]
 [0.829 1.    0.851 0.864 0.821 0.861 0.82  0.879 0.811 0.738 0.838 0.86 ]
 [0.853 0.851 1.    0.83  0.834 0.833 0.849 0.853 0.83  0.688 0.875 0.854]
 [0.823 0.864 0.83  1.    0.834 0.873 0.812 0.853 0.847 0.736 0.844 0.856]
 [0.825 0.821 0.834 0.834 1.    0.828 0.873 0.808 0.761 0.753 0.816 0.812]
 [0.86  0.861 0.833 0.873 0.828 1.    0.846 0.856 0.814 0.733 0.82  0.868]
 [0.871 0.82  0.849 0.812 0.873 0.846 1.    0.807 0.766 0.703 0.82  0.835]
 [0.831 0.879 0.853 0.853 0.808 0.856 0.807 1.    0.811 0.719 0.824 0.837]
 [0.837 0.811 0.83  0.847 0.761 0.814 0.766 0.811 1.    0.699 0.832 0.827]
 [0.732 0.738 0.688 0.736 0.753 0.733 0.703 0.719 0.699 1.    0.715 0.73 ]
 [0.841 0.838 0.875 0.844 0.816 0.82  0.82  0.824 0.832 0.715 1.    0.845]
 [0.838 0.86  0.854 0.856 0.812 0.868 0.835 0.837 0.827 0.73  0.845 1.   ]]
Largest coordinates of mean normalized L1 key-bias coefficient
[(21, 0.9013), (58, 0.1345), (39, -0.1269), (99, -0.0247), (284, 0.0205), (144, -0.0203), (231, -0.0196), (30, -0.0192), (171, -0.0191), (185, -0.019), (210, -0.0187), (315, -0.0182), (246, -0.0175), (97, -0.0174), (351, -0.0173), (308, -0.0169)]

Position-subspace operator correlations with powers of L0H3
H 0 [-0.12   0.049 -0.03   0.015  0.044  0.04   0.015]
H 1 [-0.132 -0.035 -0.001  0.006 -0.012 -0.01  -0.004]
H 2 [-0.158 -0.012  0.011  0.02   0.007  0.001 -0.015]
H 3 [ 0.239  1.     0.347 -0.175 -0.319 -0.254 -0.101]
H 4 [-0.109 -0.072  0.012  0.039 -0.001 -0.022 -0.03 ]
H 5 [-0.084 -0.044  0.022  0.013 -0.027 -0.024 -0.011]

MLP0 coefficient coupling in H3 read subspace (generalized eigenvalues)
r 16 centered False [ 0.318+0.j     0.19 -0.148j  0.19 +0.148j -0.052-0.173j -0.052+0.173j -0.177+0.j    -0.143+0.048j -0.143-0.048j]
r 16 centered True [ 0.394+0.j     0.191-0.116j  0.191+0.116j -0.055-0.167j -0.055+0.167j -0.153-0.068j -0.153+0.068j -0.148+0.j   ]
r 20 centered False [-0.205-0.138j -0.205+0.138j  0.245-0.007j  0.245+0.007j  0.189+0.126j  0.189-0.126j  0.142-0.153j  0.142+0.153j]
r 20 centered True [ 0.303+0.j    -0.202-0.135j -0.202+0.135j  0.159+0.148j  0.159-0.148j -0.203+0.j     0.183+0.065j  0.183-0.065j]
r 24 centered False [ 0.282-0.172j  0.282+0.172j  0.271+0.j    -0.161-0.202j -0.161+0.202j  0.16 -0.172j  0.16 +0.172j -0.219
```

## Call 6

```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=4):
    print("Final-layer projection spectrum")
    up,sp,vp=svd(W["W_proj.3"].astype(float),full_matrices=False)
    print(np.round(sp[:40],3))
    print("Leading output directions: participation ratio, max absolute coordinate, mean-direction cosine")
    for j in range(6):
        u=up[:,j]
        ix=np.argsort(abs(u))[-8:][::-1]
        print(j,np.round([1/(u**4).sum(),abs(u).max(),u.sum()/np.sqrt(384)],4),
              list(zip(ix.tolist(),np.round(u[ix],3).tolist())))
    print("\nFinal-layer output-column cosine quantiles with leading output direction")
    p=W["W_proj.3"].astype(float)
    for j in range(3):
        c=(up[:,j]@p)/np.linalg.norm(p,axis=0)
        print(j,np.round(np.quantile(c,[0,.01,.1,.25,.5,.75,.9,.99,1]),3))

    print("\nFirst-layer MLP coefficient clustering in H3's leading input-weight subspace")
    g=W["ln1_w.0"].astype(float)
    v=W["W_v.0"][192:256].astype(float)*g
    v-=v.mean(1,keepdims=True)
    b=W["W_o.0"][:,192:256].astype(float)@v
    u,s,vt=svd(b,full_matrices=False)
    f=W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float)
    f-=f.mean(1,keepdims=True)
    x=f@vt[:20].T
    x-=x.mean(0)
    ux,sx,vx=svd(x,full_matrices=False)
    x=ux*np.sqrt(len(x))
    x/=np.linalg.norm(x,axis=1,keepdims=True)
    for k in (8,12,16,20,24,28,32):
        fit=KMeans(n_clusters=k,n_init=10,random_state=42).fit(x)
        sizes=np.bincount(fit.labels_,minlength=k)
        sil=silhouette_score(x,fit.labels_,sample_size=1000,random_state=42)
        print("k",k,"silhouette",round(sil,4),"sizes",np.sort(sizes)[::-1])
    print("\nFirst-layer MLP coefficient row-norm quantiles in current/prior subspaces")
    for r in (16,20,24):
        basis=vt[:r].T
        cur=f@basis
        prev=f@b@basis
        for name,z in (("current",cur),("prior",prev)):
            print(r,name,np.round(np.quantile(np.linalg.norm(z,axis=1),[0,.1,.5,.9,1]),3))

    print("\nL0 output-row norms for low-rank heads and MLP0")
    for j in (1,2,4,5):
        o=W["W_o.0"][:,64*j:64*(j+1)].astype(float)
        rn=np.linalg.norm(o,axis=1)
        ix=np.argsort(rn)[-10:][::-1]
        print("H",j,list(zip(ix.tolist(),np.round(rn[ix],3).tolist())))
    p=W["W_proj.0"].astype(float)
    rn=np.linalg.norm(p,axis=1)
    ix=np.argsort(rn)[-10:][::-1]
    print("MLP0",list(zip(ix.tolist(),np.round(rn[ix],3).tolist())))

```

Output:

```
Final-layer projection spectrum
[28.461 22.214 21.34  20.708 19.943 19.662 18.986 18.744 18.295 17.457 17.132 16.561 15.329 14.874 14.328 13.438 12.667 12.455 12.317 11.531 11.047
  9.084  8.353  8.24   6.37   5.836  4.626  4.583  3.777  3.61   3.252  3.183  3.086  2.941  2.865  2.801  2.639  2.564  2.538  2.466]
Leading output directions: participation ratio, max absolute coordinate, mean-direction cosine
0 [12.9537  0.428  -0.0207] [(184, -0.428), (39, -0.342), (5, 0.321), (383, -0.254), (132, -0.231), (219, -0.217), (21, -0.216), (89, -0.215)]
1 [117.4406   0.1726   0.0018] [(351, -0.173), (316, 0.161), (283, -0.151), (43, 0.139), (194, -0.138), (366, 0.136), (147, 0.134), (247, 0.126)]
2 [137.8838   0.1599  -0.0017] [(323, 0.16), (219, -0.126), (95, 0.123), (61, -0.123), (201, -0.122), (184, -0.117), (49, 0.116), (367, -0.115)]
3 [124.7351   0.1654  -0.0019] [(90, -0.165), (200, -0.16), (8, -0.149), (355, 0.134), (383, -0.131), (53, 0.129), (174, -0.127), (128, 0.121)]
4 [140.5561   0.1374  -0.0141] [(161, 0.137), (135, 0.131), (10, 0.131), (302, -0.124), (182, -0.124), (284, -0.123), (252, 0.122), (114, -0.121)]
5 [142.0652   0.1433   0.0124] [(36, 0.143), (300, 0.143), (26, -0.142), (124, 0.128), (234, -0.122), (110, 0.12), (269, 0.118), (199, -0.115)]

Final-layer output-column cosine quantiles with leading output direction
0 [-0.922 -0.815 -0.525 -0.312 -0.067  0.135  0.319  0.565  0.753]
1 [-0.792 -0.588 -0.359 -0.181 -0.001  0.175  0.338  0.534  0.774]
2 [-0.669 -0.571 -0.333 -0.188 -0.013  0.171  0.325  0.534  0.683]

First-layer MLP coefficient clustering in H3's leading input-weight subspace
k 8 silhouette 0.0444 sizes [343 292 238 158 152 151 123  79]
k 12 silhouette 0.0565 sizes [288 209 160 138 124 108  98  97  89  89  71  65]
k 16 silhouette 0.068 sizes [274 186 124 105  99  84  83  82  78  76  61  61  60  59  59  45]
k 20 silhouette 0.0727 sizes [166 142 129 127  94  78  74  70  67  64  63  60  60  56  55  51  48  47  44  41]
k 24 silhouette 0.0837 sizes [119 111 111 108 106  99  91  63  58  57  56  55  55  55  54  52  48  40  37  36  36  32  29  28]
k 28 silhouette 0.083 sizes [124 115 105  99  85  74  61  61  53  52  52  51  46  45  44  43  43  41  40  39  39  38  36  34  33  28  28  27]
k 32 silhouette 0.0883 sizes [93 93 89 84 70 66 65 64 57 52 49 48 48 43 43 43 43 41 40 39 39 39 37 35 32 31 30 27 27 27 22 20]

First-layer MLP coefficient row-norm quantiles in current/prior subspaces
16 current [0.044 0.08  0.16  1.216 2.123]
16 prior [0.03  0.047 0.088 1.004 2.316]
20 current [0.05  0.092 0.174 1.321 2.274]
20 prior [0.032 0.05  0.092 1.022 2.331]
24 current [0.056 0.096 0.179 1.337 2.308]
24 prior [0.034 0.051 0.093 1.022 2.332]

L0 output-row norms for low-rank heads and MLP0
H 1 [(39, 2.401), (21, 1.884), (58, 1.773), (99, 1.239), (184, 0.788), (163, 0.328), (153, 0.304), (336, 0.263), (290, 0.258), (377, 0.247)]
H 2 [(39, 1.984), (21, 1.667), (58, 1.495), (99, 1.285), (184, 0.745), (163, 0.261), (217, 0.257), (165, 0.248), (52, 0.246), (98, 0.244)]
H 4 [(39, 1.613), (21, 1.244), (58, 1.135), (99, 0.784), (184, 0.555), (165, 0.24), (52, 0.226), (109, 0.223), (163, 0.22), (153, 0.213)]
H 5 [(21, 5.347), (39, 3.769), (99, 3.356), (58, 3.004), (184, 1.533), (163, 0.493), (354, 0.481), (93, 0.475), (217, 0.462), (81, 0.458)]
MLP0 [(184, 8.562), (21, 5.174), (266, 3.414), (383, 3.25), (294, 3.099), (37, 2.864), (39, 2.858), (60, 2.85), (86, 2.842), (133, 2.8)]

```

## Call 7

```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=4):
    g=W["ln1_w.0"].astype(float)
    v=W["W_v.0"][192:256].astype(float)*g
    v-=v.mean(1,keepdims=True)
    b=W["W_o.0"][:,192:256].astype(float)@v
    u,s,vt=svd(b,full_matrices=False)
    basis=vt[:20].T
    f=W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float)
    f-=f.mean(1,keepdims=True)
    cur=f@basis
    prev=f@b@basis
    nc=np.linalg.norm(cur,axis=1)
    npv=np.linalg.norm(prev,axis=1)
    print("Numbers of neurons above current/prior coefficient-norm thresholds")
    for t in (.2,.4,.6,.8,1.0):
        print(t,int((nc>t).sum()),int((npv>t).sum()),int(((nc>t)&(npv>t)).sum()))
    print("\nClustering high-signal current coefficient rows, cosine geometry")
    mask=nc>.5
    x=cur[mask]
    x=x/np.linalg.norm(x,axis=1,keepdims=True)
    for k in (8,12,16,20,24,28,32):
        km=KMeans(n_clusters=k,n_init=20,random_state=0).fit(x)
        print(k,"silhouette",round(silhouette_score(x,km.labels_),4),
              "sizes",np.sort(np.bincount(km.labels_))[::-1])

    print("\nJoint current/prior coefficient clustering")
    mc=nc>.5
    mp=npv>.4
    xc=cur[mc]/nc[mc,None]
    xp=prev[mp]/npv[mp,None]
    xx=np.vstack([xc,xp])
    for k in (16,20,24,28):
        km=KMeans(n_clusters=k,n_init=20,random_state=0).fit(xx)
        centers=km.cluster_centers_
        centers=centers/np.linalg.norm(centers,axis=1,keepdims=True)
        lc=np.argmax((cur/nc[:,None])@centers.T,axis=1)
        lp=np.argmax((prev/npv[:,None])@centers.T,axis=1)
        keep=mc&mp
        counts=np.zeros((k,k))
        np.add.at(counts,(lp[keep],lc[keep]),1)
        adj=counts+counts.T
        deg=adj.sum(1)
        normal=adj/np.sqrt(np.maximum(deg[:,None]*deg[None,:],1))
        ev,vec=np.linalg.eigh(normal)
        signs=vec[:,0]>0
        cross=adj[signs][:,~signs].sum()*2/(adj.sum()+1e-30)
        print("k",k,"silhouette",round(silhouette_score(xx,km.labels_),4),
              "min eigs",np.round(ev[:4],3),"bipartite-cut fraction",round(cross,3),
              "partition sizes",int(signs.sum()),int((~signs).sum()))
        if k==24:
            print("current cluster sizes",np.bincount(lc[mc],minlength=k))
            print("prior cluster sizes",np.bincount(lp[mp],minlength=k))
            print("joint counts, rows=prior coefficient cluster, columns=current coefficient cluster")
            print(counts.astype(int))

```

Output:

```
Numbers of neurons above current/prior coefficient-norm thresholds
0.2 719 558 558
0.4 596 510 505
0.6 533 416 391
0.8 428 291 257
1.0 330 166 142

Clustering high-signal current coefficient rows, cosine geometry
8 silhouette 0.1066 sizes [146 110  84  58  57  47  42  33]
12 silhouette 0.1362 sizes [101  70  66  54  43  41  40  36  35  34  33  24]
16 silhouette 0.1605 sizes [69 54 52 49 45 42 40 31 29 28 27 26 26 25 21 13]
20 silhouette 0.1766 sizes [60 44 39 36 34 32 31 30 29 28 27 26 26 23 21 21 20 20 16 14]
24 silhouette 0.1859 sizes [48 46 36 30 29 28 27 26 26 25 24 23 22 22 22 21 20 19 19 17 14 13 12  8]
28 silhouette 0.1936 sizes [37 36 28 28 27 25 24 23 22 22 21 21 21 21 21 20 20 19 19 18 16 16 14 13 13 12 12  8]
32 silhouette 0.1832 sizes [34 29 26 26 25 22 22 21 21 21 21 20 19 19 19 19 18 18 17 15 15 14 14 14 14 13 13 13 12 12  6  5]

Joint current/prior coefficient clustering
k 16 silhouette 0.1093 min eigs [-0.152 -0.104 -0.084 -0.057] bipartite-cut fraction 0.534 partition sizes 8 8
k 20 silhouette 0.1251 min eigs [-0.218 -0.137 -0.078 -0.056] bipartite-cut fraction 0.514 partition sizes 11 9
k 24 silhouette 0.1235 min eigs [-0.253 -0.147 -0.123 -0.108] bipartite-cut fraction 0.528 partition sizes 9 15
current cluster sizes [22 16 14 17 33 23 57 18 44 11  6 21 24 26 16 21 11 36 21 33 32 14 49 12]
prior cluster sizes [20 25 16 59 30 12 18 20 10  8  9 38 23  9 11 20 34 24 30 19 20 38 11  6]
joint counts, rows=prior coefficient cluster, columns=current coefficient cluster
[[ 3  0  0  1  1  1  1  0  2  0  0  1  0  1  1  0  1  1  0  1  2  0  3  0]
 [ 1  2  0  0  2  0  3  0  6  1  1  0  0  2  0  2  1  1  0  1  1  0  1  0]
 [ 0  1  4  0  0  2  2  0  1  0  0  0  2  0  0  1  0  0  0  1  1  0  1  0]
 [ 1  0  0  7  5  1  2  0  2  2  1  2  3  2  0  0  1  7  3  6  3  2  4  1]
 [ 3  0  1  1  3  1  4  1  2  0  0  0  2  2  1  0  0  0  1  0  3  1  1  1]
 [ 0  0  0  0  1  2  1  0  1  0  0  1  1  0  1  0  0  1  1  1  0  0  0  0]
 [ 0  2  0  2  0  0  5  0  1  1  0  0  0  1  0  2  1  0  0  0  0  0  0  0]
 [ 0  0  1  0  4  1  0  2  1  0  0  0  1  1  0  2  0  0  1  2  1  2  1  0]
 [ 0  0  0  0  0  0  0  0  5  0  0  1  0  1  0  0  0  1  0  0  0  0  2  0]
 [ 0  1  0  0  0  0  0  0  0  3  0  0  1  0  0  0  0  0  0  1  1  0  1  0]
 [ 0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  1  1  0  5  0]
 [ 0  0  0  1  4  0  3  1  0  0  0  6  1  2  2  0  0  1  0  1  0  3 12  0]
 [ 1  0  1  0  2  1  3  1  2  0  0  0  1  1  1  1  0  2  0  1  1  0  2  0]
 [ 0  1  0  2  0  0  2  0  0  0  1  0  0  1  0  0  0  1  0  0  0  0  0  0]
 [ 1  0  0  0  0  1  2  0  1  0  0  0  0  0  3  0  0  2  0  1  0  0  0  0]
 [ 0  0  1  0  0  0  2  0  0  0  0  2  2  1  0  2  0  1  1  3  2  1  1  1]
 [ 1  1  0  0  0  3  5  1  3  0  0  1  3  3  1  1  1  2  2  2  1  0  2  1]
 [ 2  0  0  0  3  0  3  2  1  0  0  0  0  0  1  2  1  4  2  1  2  0  0  0]
 [ 2  1  0  1  1  3  2  2  3  1  0  1  1  0  0  2  0  1  1  3  2  0  3  0]
 [ 3  0  1  1  0  0  3  0  2  0  0  0  2  1  0  1  0  1  0  2  0  0  2  0]
 [ 1  0  1  0  2  0  1  2  0  0  0  2  0  1  0  1  1  1  1  0  5  0  1  0]
 [ 0  0  1  0  2  2  4  3  2  0  0  1  2  3  1  0  1  4  2  2  3  3  1  1]
 [ 0  2  0  0  1  0  1  0  1  0  0  0  0  0  1  1  0  0  0  1  0  0  2  1]
 [ 0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  4]]
k 28 silhouette 0.1149 min eigs [-0.281 -0.227 -0.185 -0.152] bipartite-cut fraction 0.54 partition sizes 16 12

```

## Call 8

```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=4):
    p=W["W_proj.3"].astype(float)
    u,s,vt=svd(p,full_matrices=False)
    print("Final MLP projection-column clustering (weight vectors only)")
    for r in (20,24,28):
        z=(u[:,:r].T@p).T
        zn=np.linalg.norm(z,axis=1)
        keep=zn>np.quantile(zn,.25)
        z=z[keep]/zn[keep,None]
        for k in (20,24,28):
            km=KMeans(n_clusters=k,n_init=20,random_state=42).fit(z)
            print("r",r,"k",k,"silhouette",
                  round(silhouette_score(z,km.labels_,sample_size=1000,random_state=42),4),
                  "sizes",np.sort(np.bincount(km.labels_))[::-1])

    print("\nL3H3 induction composition versus other heads, with rotated-weight null controls")
    def qk(l,j):
        g=W[f"ln1_w.{l}"].astype(float)
        sl=slice(64*j,64*(j+1))
        q=W[f"W_q.{l}"][sl].astype(float)*g
        k=W[f"W_k.{l}"][sl].astype(float)*g
        q-=q.mean(1,keepdims=True)
        k-=k.mean(1,keepdims=True)
        return q.T@k/8
    def ov(l,j):
        g=W[f"ln1_w.{l}"].astype(float)
        sl=slice(64*j,64*(j+1))
        v=W[f"W_v.{l}"][sl].astype(float)*g
        v-=v.mean(1,keepdims=True)
        return W[f"W_o.{l}"][:,sl].astype(float)@v
    b=ov(0,3)
    _,_,vb=svd(b,full_matrices=False)
    bas=vb[:24].T
    rng=np.random.default_rng(42)
    for j in range(6):
        a=qk(3,j)
        t=bas.T@a@b@bas
        sy=(t+t.T)/2
        ev=np.linalg.eigvalsh(sy)
        signed=np.trace(t)/np.linalg.norm(t)
        c=bas.T@a
        d=b@bas
        null=[]
        for _ in range(300):
            perm=rng.permutation(384)
            signs=rng.choice([-1.,1.],384)
            tn=(c[:,perm]*signs)@d
            null.append(np.trace(tn)/(np.linalg.norm(tn)+1e-30))
        print("L3H"+str(j),"trace/norm",round(signed,3),
              "symmetric positive/negative eigenvalue sums",
              np.round([ev[ev>0].sum(),ev[ev<0].sum()],4),
              "null mean/sd",np.round([np.mean(null),np.std(null)],3),
              "null max",round(max(null),3))

```

Output:

```
Final MLP projection-column clustering (weight vectors only)
r 20 k 20 silhouette 0.0655 sizes [106  78  70  63  61  60  60  60  58  57  57  55  55  51  50  50  48  42  38  33]
r 20 k 24 silhouette 0.0637 sizes [96 63 57 57 55 54 52 52 51 50 49 49 47 45 45 43 41 39 38 38 38 37 32 24]
r 20 k 28 silhouette 0.0656 sizes [81 53 50 48 47 47 47 45 45 44 44 41 40 40 40 39 37 37 36 36 36 35 34 33 32 31 31 23]
r 24 k 20 silhouette 0.0558 sizes [87 76 71 67 66 66 65 65 57 56 53 52 52 51 51 50 49 46 37 35]
r 24 k 24 silhouette 0.0595 sizes [103  63  62  59  59  59  54  51  50  49  49  48  47  42  42  41  40  39  39  37  35  29  29  26]
r 24 k 28 silhouette 0.0575 sizes [74 62 53 52 51 51 50 48 45 45 45 45 44 43 41 39 39 37 36 35 35 30 28 28 26 26 23 21]
r 28 k 20 silhouette 0.0583 sizes [84 67 65 65 63 62 61 60 59 59 55 55 55 55 53 50 50 47 45 42]
r 28 k 24 silhouette 0.0604 sizes [104  65  57  57  55  53  53  50  47  47  47  47  47  46  45  43  41  40  39  38  36  32  32  31]
r 28 k 28 silhouette 0.0609 sizes [89 55 51 47 47 47 47 47 46 44 44 43 42 42 41 41 37 37 36 36 36 34 32 29 27 27 25 23]

L3H3 induction composition versus other heads, with rotated-weight null controls
L3H0 trace/norm -0.597 symmetric positive/negative eigenvalue sums [ 0.1106 -0.174 ] null mean/sd [0.009 0.214] null max 0.765
L3H1 trace/norm -0.09 symmetric positive/negative eigenvalue sums [ 0.0713 -0.076 ] null mean/sd [-0.015  0.193] null max 0.607
L3H2 trace/norm 0.184 symmetric positive/negative eigenvalue sums [ 0.0927 -0.0821] null mean/sd [-0.004  0.205] null max 0.514
L3H3 trace/norm 2.73 symmetric positive/negative eigenvalue sums [ 0.4513 -0.0362] null mean/sd [-0.029  0.23 ] null max 0.622
L3H4 trace/norm 0.551 symmetric positive/negative eigenvalue sums [ 0.0426 -0.0287] null mean/sd [-0.007  0.199] null max 0.504
L3H5 trace/norm -0.152 symmetric positive/negative eigenvalue sums [ 0.0969 -0.1077] null mean/sd [0.011 0.202] null max 0.639

```
