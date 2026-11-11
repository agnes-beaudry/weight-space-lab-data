## Call 1

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)

np.set_printoptions(precision=4, suppress=True)
print("Tensor names and shapes:")
print([(k, v.shape) for k, v in W.items()])

def describe(a):
    a = np.asarray(a, dtype=np.float64)
    return np.array([a.mean(), a.std(), a.min(), np.median(a), a.max()])

print("\nLayerNorm parameters: mean, std, min, median, max")
for l in range(4):
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        print(f"{name}.{l}", describe(W[f"{name}.{l}"]))

print("\nAttention heads (raw weights, no network evaluation)")
print("QK_s: top 5 singular values; QK_er: entropy effective rank")
print("QK_sym: symmetric-part squared-norm fraction; QK_tr: trace/Frobenius norm")
print("OV_s: top 5 singular values; OV_tr: trace/Frobenius norm")
print("OV_eig: trace / sum(abs(eigenvalues)); biasQ: ||ln1_b @ Q.T||")
for l in range(4):
    b = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s = slice(64*j, 64*(j+1))
        Q = W[f"W_q.{l}"][s].astype(float)
        K = W[f"W_k.{l}"][s].astype(float)
        V = W[f"W_v.{l}"][s].astype(float)
        O = W[f"W_o.{l}"][:, s].astype(float)
        A = Q.T @ K / 8
        B = O @ V
        sa = np.linalg.svd(A, compute_uv=False)[:64]
        sb = np.linalg.svd(B, compute_uv=False)[:64]
        pa = sa / sa.sum()
        er = np.exp(-(pa * np.log(pa + 1e-30)).sum())
        ea = np.linalg.eigvalsh((A + A.T)/2)
        eb = np.linalg.eigvals(V @ O)
        print(f"L{l}H{j}",
              "normQKVO", np.round([np.linalg.norm(t) for t in [Q,K,V,O]], 3),
              "QK_s", sa[:5].round(3), "QK_er", round(er,2),
              "QK_sym", round(np.linalg.norm((A+A.T)/2)**2/np.linalg.norm(A)**2,3),
              "QK_tr", round(np.trace(A)/np.linalg.norm(A),3),
              "QK_sym_ext", np.round([ea[0],ea[-1]],3),
              "OV_s", sb[:5].round(3),
              "OV_tr", round(np.trace(B)/np.linalg.norm(B),3),
              "OV_eig", round(np.trace(B)/(np.abs(eb).sum()+1e-30),3),
              "biasQK", np.round([np.linalg.norm(Q@b),np.linalg.norm(K@b)],3))

print("\nMLP weight-only summaries")
for l in range(4):
    F = W[f"W_fc.{l}"].astype(float)
    P = W[f"W_proj.{l}"].astype(float)
    nf = np.linalg.norm(F,axis=1)
    np_ = np.linalg.norm(P,axis=0)
    cos = np.sum(F*P.T,axis=1)/(nf*np_+1e-30)
    product = P@F
    print("L",l,"F row norm",describe(nf),"P col norm",describe(np_),
          "paired cosine quantiles",np.quantile(cos,[0,.01,.1,.5,.9,.99,1]),
          "PF norm",np.linalg.norm(product),"PF trace",np.trace(product))

```

Output:

```
Tensor names and shapes:
[('ln1_w.0', (384,)), ('ln1_b.0', (384,)), ('W_q.0', (384, 384)), ('W_k.0', (384, 384)), ('W_v.0', (384, 384)), ('W_o.0', (384, 384)), ('ln2_w.0', (384,)), ('ln2_b.0', (384,)), ('W_fc.0', (1536, 384)), ('W_proj.0', (384, 1536)), ('ln1_w.1', (384,)), ('ln1_b.1', (384,)), ('W_q.1', (384, 384)), ('W_k.1', (384, 384)), ('W_v.1', (384, 384)), ('W_o.1', (384, 384)), ('ln2_w.1', (384,)), ('ln2_b.1', (384,)), ('W_fc.1', (1536, 384)), ('W_proj.1', (384, 1536)), ('ln1_w.2', (384,)), ('ln1_b.2', (384,)), ('W_q.2', (384, 384)), ('W_k.2', (384, 384)), ('W_v.2', (384, 384)), ('W_o.2', (384, 384)), ('ln2_w.2', (384,)), ('ln2_b.2', (384,)), ('W_fc.2', (1536, 384)), ('W_proj.2', (384, 1536)), ('ln1_w.3', (384,)), ('ln1_b.3', (384,)), ('W_q.3', (384, 384)), ('W_k.3', (384, 384)), ('W_v.3', (384, 384)), ('W_o.3', (384, 384)), ('ln2_w.3', (384,)), ('ln2_b.3', (384,)), ('W_fc.3', (1536, 384)), ('W_proj.3', (384, 1536))]

LayerNorm parameters: mean, std, min, median, max
ln1_w.0 [0.2965 0.1884 0.0352 0.2422 0.9785]
ln1_b.0 [ 0.0004  0.0311 -0.0954 -0.0007  0.0906]
ln2_w.0 [0.8718 0.3382 0.042  0.8492 1.775 ]
ln2_b.0 [ 0.0134  0.0691 -0.6236  0.0121  0.4636]
ln1_w.1 [0.5058 0.0892 0.197  0.5041 1.0846]
ln1_b.1 [ 0.011   0.1546 -1.5676  0.0142  1.3651]
ln2_w.1 [1.1315 0.1741 0.0019 1.1552 1.5196]
ln2_b.1 [ 0.0065  0.1435 -0.3892  0.0066  0.417 ]
ln1_w.2 [0.6809 0.0687 0.3175 0.6837 1.0927]
ln1_b.2 [ 0.0131  0.1565 -1.6589  0.0174  1.3318]
ln2_w.2 [1.3104 0.1148 0.012  1.3222 1.5078]
ln2_b.2 [ 0.0123  0.1737 -0.517   0.0155  0.4489]
ln1_w.3 [0.7087 0.0494 0.4078 0.7108 0.8155]
ln1_b.3 [ 0.0296  0.0994 -0.7639  0.0276  0.8241]
ln2_w.3 [1.2668 0.0894 0.2328 1.2705 1.4532]
ln2_b.3 [ 0.0128  0.1413 -0.5402  0.0149  0.3932]

Attention heads (raw weights, no network evaluation)
QK_s: top 5 singular values; QK_er: entropy effective rank
QK_sym: symmetric-part squared-norm fraction; QK_tr: trace/Frobenius norm
OV_s: top 5 singular values; OV_tr: trace/Frobenius norm
OV_eig: trace / sum(abs(eigenvalues)); biasQ: ||ln1_b @ Q.T||
L0H0 normQKVO [7.14  7.395 5.916 6.127] QK_s [0.853 0.613 0.368 0.287 0.279] QK_er 34.19 QK_sym 0.527 QK_tr -1.172 QK_sym_ext [-0.425  0.501] OV_s [3.611 3.392 2.881 2.314 2.128] OV_tr -0.086 OV_eig -0.126 biasQK [0.504 0.725]
L0H1 normQKVO [4.486 4.927 4.323 4.825] QK_s [0.838 0.311 0.219 0.125 0.091] QK_er 20.99 QK_sym 0.598 QK_tr -0.892 QK_sym_ext [-0.599  0.243] OV_s [5.721 3.359 1.146 0.545 0.444] OV_tr 0.077 OV_eig 0.116 biasQK [0.228 0.261]
L0H2 normQKVO [4.065 4.362 4.017 4.36 ] QK_s [0.543 0.183 0.157 0.133 0.072] QK_er 27.69 QK_sym 0.596 QK_tr -0.944 QK_sym_ext [-0.388  0.16 ] OV_s [3.943 2.357 1.006 0.521 0.5  ] OV_tr 0.057 OV_eig 0.075 biasQK [0.146 0.314]
L0H3 normQKVO [14.141 14.661  8.259  9.445] QK_s [1.114 1.092 1.07  1.064 1.03 ] QK_er 40.48 QK_sym 0.434 QK_tr 1.775 QK_sym_ext [-0.478  0.899] OV_s [4.426 3.911 3.681 3.597 3.596] OV_tr -0.53 OV_eig -0.419 biasQK [0.467 0.543]
L0H4 normQKVO [4.205 4.614 3.689 3.657] QK_s [0.51  0.227 0.185 0.124 0.101] QK_er 29.61 QK_sym 0.567 QK_tr -0.856 QK_sym_ext [-0.338  0.176] OV_s [2.449 1.317 0.666 0.461 0.438] OV_tr -0.132 OV_eig -0.136 biasQK [0.27 0.55]
L0H5 normQKVO [2.632 2.936 8.453 9.565] QK_s [0.18  0.048 0.024 0.02  0.02 ] QK_er 41.84 QK_sym 0.514 QK_tr -0.242 QK_sym_ext [-0.104  0.077] OV_s [24.799 12.9    5.849  4.229  3.665] OV_tr 0.08 OV_eig 0.19 biasQK [0.143 0.171]
L1H0 normQKVO [16.628 14.95  11.763 11.711] QK_s [3.084 2.605 2.082 1.854 1.729] QK_er 25.35 QK_sym 0.539 QK_tr -1.052 QK_sym_ext [-1.73   1.612] OV_s [6.118 5.692 5.46  5.236 5.209] OV_tr -1.359 OV_eig -0.802 biasQK [ 6.688 13.795]
L1H1 normQKVO [16.019 14.34  11.237 11.401] QK_s [3.082 2.497 1.812 1.767 1.429] QK_er 25.71 QK_sym 0.526 QK_tr -0.969 QK_sym_ext [-1.703  1.507] OV_s [6.051 5.716 5.492 5.23  5.099] OV_tr -1.37 OV_eig -0.838 biasQK [ 6.079 13.929]
L1H2 normQKVO [15.576 14.136 11.739 11.38 ] QK_s [2.848 2.211 1.794 1.583 1.438] QK_er 27.12 QK_sym 0.537 QK_tr -1.271 QK_sym_ext [-1.622  1.395] OV_s [6.72  6.39  6.182 5.782 5.597] OV_tr -1.478 OV_eig -0.866 biasQK [ 5.564 13.162]
L1H3 normQKVO [15.745 13.949 11.801 11.885] QK_s [3.106 2.415 1.629 1.446 1.323] QK_er 26.43 QK_sym 0.522 QK_tr -1.014 QK_sym_ext [-1.714  1.533] OV_s [6.544 6.21  6.074 5.823 5.568] OV_tr -1.51 OV_eig -0.887 biasQK [ 5.665 13.742]
L1H4 normQKVO [15.205 13.642 12.731 12.113] QK_s [2.62  1.795 1.743 1.336 1.248] QK_er 28.72 QK_sym 0.536 QK_tr -1.443 QK_sym_ext [-1.345  1.364] OV_s [7.908 7.206 6.838 6.499 6.413] OV_tr -1.444 OV_eig -0.826 biasQK [ 5.347 10.986]
L1H5 normQKVO [16.146 14.418 11.065 11.244] QK_s [3.024 2.555 1.908 1.631 1.608] QK_er 24.97 QK_sym 0.528 QK_tr -1.014 QK_sym_ext [-1.85   1.487] OV_s [5.849 5.644 5.5   5.382 5.204] OV_tr -1.379 OV_eig -0.822 biasQK [ 6.224 14.146]
L2H0 normQKVO [14.489 13.572 13.047 12.69 ] QK_s [1.229 1.058 0.753 0.673 0.657] QK_er 54.58 QK_sym 0.496 QK_tr -1.005 QK_sym_ext [-0.57   0.681] OV_s [7.576 6.429 6.378 5.428 5.292] OV_tr -1.032 OV_eig -0.542 biasQK [ 4.265 10.635]
L2H1 normQKVO [16.803 15.934 13.859 13.478] QK_s [1.189 1.084 1.07  0.971 0.965] QK_er 55.17 QK_sym 0.498 QK_tr -0.112 QK_sym_ext [-0.629  0.665] OV_s [5.978 5.58  5.491 5.339 5.145] OV_tr -1.987 OV_eig -0.806 biasQK [ 3.787 10.117]
L2H2 normQKVO [15.082 14.471 14.057 13.611] QK_s [0.963 0.854 0.834 0.794 0.708] QK_er 57.3 QK_sym 0.505 QK_tr -0.785 QK_sym_ext [-0.528  0.522] OV_s [7.257 7.015 6.63  6.291 5.817] OV_tr -1.596 OV_eig -0.717 biasQK [3.784 9.886]
L2H3 normQKVO [14.704 13.851 14.29  13.833] QK_s [0.896 0.857 0.782 0.716 0.705] QK_er 56.94 QK_sym 0.498 QK_tr -0.36 QK_sym_ext [-0.453  0.509] OV_s [7.041 6.885 5.901 5.851 5.831] OV_tr -1.587 OV_eig -0.735 biasQK [3.082 7.454]
L2H4 normQKVO [16.878 15.77  12.388 12.522] QK_s [1.537 1.198 1.06  1.013 0.939] QK_er 53.86 QK_sym 0.494 QK_tr -0.453 QK_sym_ext [-0.752  0.842] OV_s [5.217 4.892 4.719 4.609 4.265] OV_tr -2.102 OV_eig -0.861 biasQK [ 4.319 11.042]
L2H5 normQKVO [17.106 16.203 13
```

## Call 2

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True)

d = 384
A, B, RQ, RK = {}, {}, {}, {}
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s = slice(64*j,64*(j+1))
        q = W[f"W_q.{l}"][s].astype(float)*g
        k = W[f"W_k.{l}"][s].astype(float)*g
        v = W[f"W_v.{l}"][s].astype(float)*g
        q -= q.mean(axis=1,keepdims=True)
        k -= k.mean(axis=1,keepdims=True)
        v -= v.mean(axis=1,keepdims=True)
        o = W[f"W_o.{l}"][:,s].astype(float)
        o -= o.mean(axis=0,keepdims=True)
        name = f"L{l}H{j}"
        RQ[name], RK[name] = q, k
        A[name], B[name] = q.T@k/8, o@v

print("LayerNorm-scale-adjusted, centered matrix summaries (no LayerNorm evaluation)")
print("Columns: QK Frobenius norm, top 6 singular values, trace/norm; OV norm, top 3 singular values")
for name in A:
    sa = np.linalg.svd(A[name],compute_uv=False)
    sb = np.linalg.svd(B[name],compute_uv=False)
    print(name, round(np.linalg.norm(A[name]),3), sa[:6].round(3),
          round(np.trace(A[name])/np.linalg.norm(A[name]),3),
          round(np.linalg.norm(B[name]),3), sb[:3].round(3))

print("\nEigenvalues of the first-layer QK products, largest magnitudes first")
for j in range(6):
    name = f"L0H{j}"
    ev = np.linalg.eigvals(RK[name]@RQ[name].T/8)
    ev = ev[np.argsort(-np.abs(ev))]
    print(name, ev[:24])

print("\nCross-layer key-composition signatures")
print("For each receiving head, strongest earlier OV sources by signed trace(A @ B)/||A @ B||.")
for l in range(1,4):
    for j in range(6):
        target = f"L{l}H{j}"
        rows = []
        for m in range(l):
            for h in range(6):
                source = f"L{m}H{h}"
                C = A[target]@B[source]
                score = np.trace(C)/(np.linalg.norm(C)+1e-30)
                overlap = np.trace(C)/(np.linalg.norm(A[target])*np.linalg.norm(B[source]))
                rows.append((score,source,overlap,np.linalg.norm(C)))
        rows.sort(reverse=True)
        print(target, [(s,round(v,3),round(o,3),round(n,3)) for v,s,o,n in rows[:5]])

print("\nFirst-layer head alignment with other heads")
for l in range(1,4):
    for j in range(6):
        target = f"L{l}H{j}"
        keys, queries = [], []
        for h in range(6):
            source=f"L0H{h}"
            C=A[target]@B[source]
            D=B[source].T@A[target]
            keys.append(np.trace(C)/(np.linalg.norm(C)+1e-30))
            queries.append(np.trace(D)/(np.linalg.norm(D)+1e-30))
        print(target,"key",np.round(keys,3),"query",np.round(queries,3))

print("\nLargest LayerNorm bias coordinates (index:value)")
for l in range(4):
    for typ in ["ln1_b","ln2_b"]:
        b=W[f"{typ}.{l}"]
        ix=np.argsort(-abs(b))[:8]
        print(f"{typ}.{l}",[(int(i),round(float(b[i]),4)) for i in ix])

```

Output:

```
LayerNorm-scale-adjusted, centered matrix summaries (no LayerNorm evaluation)
Columns: QK Frobenius norm, top 6 singular values, trace/norm; OV norm, top 3 singular values
L0H0 0.293 [0.248 0.102 0.047 0.043 0.037 0.036] -0.357 2.126 [0.957 0.898 0.818]
L0H1 0.159 [0.15  0.042 0.021 0.012 0.01  0.007] -0.619 3.979 [3.379 2.022 0.344]
L0H2 0.107 [0.102 0.018 0.016 0.012 0.008 0.007] -0.813 3.033 [2.643 1.362 0.399]
L0H3 0.698 [0.244 0.222 0.201 0.182 0.179 0.157] 1.923 3.533 [1.126 0.955 0.949]
L0H4 0.088 [0.08  0.022 0.019 0.013 0.011 0.008] -0.519 1.683 [1.453 0.723 0.213]
L0H5 0.034 [0.032 0.004 0.003 0.003 0.002 0.002] -0.253 18.804 [16.269  8.778  2.832]
L1H0 1.978 [1.185 0.723 0.557 0.534 0.472 0.432] -0.884 11.479 [3.189 2.991 2.858]
L1H1 1.843 [1.154 0.708 0.582 0.465 0.398 0.368] -0.845 11.263 [3.147 3.02  2.872]
L1H2 1.781 [1.176 0.616 0.485 0.436 0.385 0.354] -1.06 11.806 [3.55  3.303 3.182]
L1H3 1.787 [1.187 0.671 0.532 0.388 0.366 0.329] -0.846 12.398 [3.408 3.269 3.119]
L1H4 1.629 [1.059 0.504 0.476 0.351 0.334 0.31 ] -1.234 13.398 [4.131 3.768 3.609]
L1H5 1.863 [1.146 0.71  0.556 0.499 0.433 0.379] -0.875 10.917 [3.083 2.946 2.891]
L2H0 1.631 [0.689 0.553 0.35  0.327 0.307 0.3  ] -0.953 16.379 [5.289 4.504 4.391]
L2H1 2.154 [0.578 0.513 0.509 0.465 0.458 0.444] -0.1 17.615 [4.136 3.837 3.791]
L2H2 1.72 [0.518 0.41  0.399 0.374 0.355 0.333] -0.754 18.662 [5.045 4.937 4.561]
L2H3 1.623 [0.502 0.384 0.364 0.334 0.333 0.312] -0.367 18.995 [4.832 4.752 4.087]
L2H4 2.215 [0.813 0.627 0.506 0.471 0.448 0.423] -0.442 14.875 [3.533 3.372 3.227]
L2H5 2.264 [0.615 0.564 0.539 0.512 0.498 0.477] -0.118 16.857 [3.735 3.601 3.518]
L3H0 2.029 [0.435 0.414 0.39  0.38  0.373 0.36 ] 1.073 20.709 [5.784 5.652 5.063]
L3H1 1.216 [0.345 0.308 0.285 0.253 0.244 0.227] 0.355 26.503 [8.226 7.821 6.561]
L3H2 1.437 [0.383 0.304 0.294 0.277 0.266 0.258] 0.243 23.503 [7.459 6.107 5.569]
L3H3 1.815 [0.322 0.311 0.307 0.304 0.297 0.294] 3.212 37.523 [10.809  9.984  9.82 ]
L3H4 0.622 [0.236 0.202 0.17  0.149 0.132 0.124] 0.003 49.232 [21.266 17.672 16.977]
L3H5 1.665 [0.454 0.321 0.315 0.299 0.289 0.288] 0.863 22.562 [6.3   5.976 5.473]

Eigenvalues of the first-layer QK products, largest magnitudes first
L0H0 [ 0.0796+0.j      0.0483+0.j     -0.024 +0.0057j -0.024 -0.0057j -0.0226+0.j      0.0176+0.0134j  0.0176-0.0134j -0.0201+0.j     -0.0138+0.0121j
 -0.0138-0.0121j -0.0139+0.0031j -0.0139-0.0031j -0.0137+0.j     -0.011 +0.0039j -0.011 -0.0039j -0.0111-0.0023j -0.0111+0.0023j -0.006 +0.0062j
 -0.006 -0.0062j -0.0063+0.j     -0.003 +0.0049j -0.003 -0.0049j -0.0052+0.0023j -0.0052-0.0023j]
L0H1 [-0.0619+0.j     -0.01  +0.j     -0.0089+0.j      0.0043+0.j     -0.0039+0.j     -0.0033+0.0018j -0.0033-0.0018j -0.0017+0.0004j -0.0017-0.0004j
 -0.0016+0.0007j -0.0016-0.0007j -0.0011+0.0003j -0.0011-0.0003j -0.0007+0.0008j -0.0007-0.0008j  0.001 +0.j     -0.0009+0.j      0.0004+0.0007j
  0.0004-0.0007j -0.0007+0.0005j -0.0007-0.0005j -0.0005+0.0007j -0.0005-0.0007j -0.0002+0.0007j]
L0H2 [-0.0566+0.j     -0.0074+0.0027j -0.0074-0.0027j -0.0034+0.0008j -0.0034-0.0008j -0.0025+0.j     -0.0013+0.0003j -0.0013-0.0003j -0.0006+0.0012j
 -0.0006-0.0012j  0.0006+0.001j   0.0006-0.001j  -0.0003-0.001j  -0.0003+0.001j  -0.001 +0.j      0.0001+0.001j   0.0001-0.001j  -0.0005+0.0008j
 -0.0005-0.0008j -0.0007+0.0006j -0.0007-0.0006j  0.0008+0.j     -0.0006-0.0004j -0.0006+0.0004j]
L0H3 [ 0.1366+0.1327j  0.1366-0.1327j  0.1099+0.0891j  0.1099-0.0891j  0.095 +0.0979j  0.095 -0.0979j  0.0841+0.1041j  0.0841-0.1041j  0.0644+0.1142j
  0.0644-0.1142j  0.0957+0.0687j  0.0957-0.0687j  0.0114+0.1166j  0.0114-0.1166j  0.1116+0.j      0.0559-0.0696j  0.0559+0.0696j  0.0749+0.0468j
  0.0749-0.0468j  0.0135+0.0747j  0.0135-0.0747j -0.0348+0.0526j -0.0348-0.0526j -0.0451-0.0173j]
L0H4 [-0.0152+0.j     -0.0095+0.j     -0.0077+0.j     -0.0026+0.0003j -0.0026-0.0003j -0.0022+0.0014j -0.0022-0.0014j -0.0002+0.0023j -0.0002-0.0023j
 -0.0014+0.0012j -0.0014-0.0012j -0.0015+0.j     -0.0006-0.0011j -0.0006+0.0011j  0.0008+0.0009j  0.0008-0.0009j  0.0012+0.j     -0.0003+0.0011j
 -0.0003-0.0011j  0.0005+0.0009j  0.0005-0.0009j -0.001 +0.j      0.001 +0.j     -0.0009-0.0001j]
L0H5 [-0.0049+0.j      0.0013+0.j     -0.0012+0.j     -0.0004+0.0007j -0.0004-0.0007j -0.0008+0.j     -0.0007+0.0002j -0.0007-0.0002j  0.0002+0.0007j
  0.0002-0.0007j -0.0001+0.0006j -0.0001-0.0006j -0.0003+0.0005j -0.0003-0.0005j  0.0002-0.0006j  0.0002+0.0006j -0.0005+0.0003j -0.0005-0.0003j
  0.0006+0.j     -0.0005+0.0001j -0.0005-0.0001j  0.0005+0.0001j  0.0005-0.0001j -0.0004+0.0004j]

Cross-layer key-composition signatures
For each receiving head, strongest earlier OV sources by signed trace(A @ B)/||A @ B||.
L1H0 [('L0H2', np.float64(0.134), np.float64(0.05), np.float64(2.256)), ('L0H5', np.float64(0.103), np.float64(0.048), np.float64(17.364)), ('L0H3', np.float64(0.102), np.float64(0.006), np.float64(0.399)), ('L0H4', np.float64(0.096), np.float64(0.029), np.float64(1.012)), ('L0H1', np.float64(0.095), np.float64(0.034), np.float64(2.837))]
L1H1 [('L0H3', np.float64(0.218), np.float64(0.014), np.float64(0.421)), ('L0H2', np.float64(0.065), np.float64(0.025), np.float64(2.162)), ('L0H5', np.float64(0.043), np.float64(0.021), np.float64(16.962)), ('L0H1', np.float64(0.017), np.float64(0.006), np.float64(2.722)), ('L0H0', np.float64(0.015), np.float64(0.0), np.float64(0.114))]
L1H2 [('L0H3', np.float64(0.168), np.float64(0.012), np.float64(0.455)), ('L0H2', np.float64(0.106), np.float64(0.042), np.float64(2.138)), ('L0H4', np.float64(0.1), np.float64(0.031), np.float64(0.938)), ('L0H5', np.float64(0.092), np.float64(0.047), np.float64(17.043)), ('L0H1', np.float64(0.078), np.float64(0.03), np.float64(2.688))]
L1H3 [('L0H2', np.float64(0.113), np.float64(0.045), np.float64(2.171)), ('L0H5', np.float64(0.085), np.float64(0.043), np.float64(17.066)), ('L0H3', np.float64(0.076), np.float64(0.005), np.float64(0.428)), ('L0H4', np.float64(0.075), np.float64(0.024), np.float64(0
```

## Call 3

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

def centered_read(M, g):
    R = M.astype(float) * g
    return R - R.mean(axis=1, keepdims=True)

g = W["ln1_w.0"].astype(float)
Vs, Os, Qs, Ks = [], [], [], []
for j in range(6):
    s=slice(64*j,64*(j+1))
    Vs.append(centered_read(W["W_v.0"][s],g))
    Os.append(W["W_o.0"][:,s].astype(float))
    Qs.append(centered_read(W["W_q.0"][s],g))
    Ks.append(centered_read(W["W_k.0"][s],g))
    print(f"L0H{j} Q singular values",np.linalg.svd(Qs[-1],compute_uv=False)[:40])
    print(f"L0H{j} K singular values",np.linalg.svd(Ks[-1],compute_uv=False)[:40])
    print(f"L0H{j} V singular values",np.linalg.svd(Vs[-1],compute_uv=False)[:40])
    print(f"L0H{j} O singular values",np.linalg.svd(Os[-1],compute_uv=False)[:40])

_,sv,vt=np.linalg.svd(np.vstack(Vs),full_matrices=False)
T=vt[:27].T
_,sp,pt=np.linalg.svd(np.vstack([Qs[3],Ks[3]]),full_matrices=False)
P=pt[:32].T
print("\nCombined first-layer value singular values",sv[:70])
print("Combined L0H3 Q,K singular values",sp[:70])
print("Singular values of overlap between these candidate subspaces",np.linalg.svd(T.T@P,compute_uv=False))

print("\nL0 effective QK energy fractions in candidate value and positional subspaces")
for j in range(6):
    A=Qs[j].T@Ks[j]/8
    n=np.linalg.norm(A)**2
    print(j,"value-left/right/both",
          np.array([np.linalg.norm(T.T@A)**2,np.linalg.norm(A@T)**2,np.linalg.norm(T.T@A@T)**2])/n,
          "pos-left/right/both",
          np.array([np.linalg.norm(P.T@A)**2,np.linalg.norm(A@P)**2,np.linalg.norm(P.T@A@P)**2])/n)

print("\nFirst MLP: norm histogram and read/write singular spectra")
F=W["W_fc.0"].astype(float)
G=centered_read(W["W_fc.0"], W["ln2_w.0"].astype(float))
O=W["W_proj.0"].astype(float)
nf=np.linalg.norm(F,axis=1)
print("Row-norm bins",np.histogram(nf,bins=[0,.4,.5,.75,1,1.5,2,2.5,3,4,5]))
print("F singular values",np.linalg.svd(F,compute_uv=False)[:90])
print("P singular values",np.linalg.svd(O,compute_uv=False)[:90])

print("\nFirst MLP read energy on current-value candidate subspace and first-head output subspaces")
for j in range(6):
    U,s,_=np.linalg.svd(Os[j],full_matrices=False)
    r=int(np.sum(s>.7))
    U=U[:,:max(1,r)]
    en=np.linalg.norm(G@U,axis=1)**2/(np.linalg.norm(G,axis=1)**2+1e-30)
    print("head",j,"output rank(s>.7)",r,
          "MLP fraction by F norm group",
          [round(float(en[(nf>=a)&(nf<b)].mean()),3) for a,b in [(0,.75),(.75,1.5),(1.5,2.5),(2.5,5)]])

print("\nShared L0 OV directions: pairwise Frobenius cosines")
Bs=[o@v for o,v in zip(Os,Vs)]
print(np.array([[np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)) for b in Bs] for a in Bs]))
print("\nFirst-layer top OV output coordinates")
for j,B in enumerate(Bs):
    U,s,Vt=np.linalg.svd(B,full_matrices=False)
    for r in range(2):
        ix=np.argsort(-np.abs(U[:,r]))[:7]
        print(j,r,round(s[r],3),[(int(i),round(float(U[i,r]),3)) for i in ix])

```

Output:

```
L0H0 Q singular values [1.279 1.188 0.592 0.542 0.514 0.492 0.482 0.454 0.438 0.424 0.415 0.404 0.389 0.377 0.359 0.356 0.346 0.32  0.31  0.302 0.29  0.288 0.274 0.267 0.253 0.243 0.235 0.23  0.225
 0.207 0.198 0.196 0.194 0.191 0.183 0.174 0.173 0.167 0.16  0.153]
L0H0 K singular values [1.658 0.852 0.672 0.631 0.617 0.604 0.574 0.561 0.504 0.488 0.46  0.443 0.432 0.426 0.409 0.389 0.371 0.357 0.345 0.325 0.321 0.309 0.293 0.293 0.267 0.259 0.243 0.221 0.218
 0.204 0.203 0.198 0.192 0.191 0.184 0.179 0.173 0.171 0.162 0.16 ]
L0H0 V singular values [0.535 0.506 0.445 0.384 0.371 0.341 0.337 0.327 0.312 0.299 0.288 0.278 0.269 0.259 0.239 0.223 0.211 0.202 0.195 0.19  0.185 0.183 0.175 0.168 0.167 0.164 0.162 0.16  0.157
 0.15  0.145 0.143 0.139 0.138 0.129 0.127 0.126 0.125 0.122 0.12 ]
L0H0 O singular values [2.087 1.915 1.852 1.672 1.522 1.482 1.43  1.312 1.259 1.155 1.13  1.049 1.037 1.017 0.918 0.809 0.731 0.718 0.709 0.608 0.581 0.564 0.546 0.519 0.475 0.445 0.407 0.402 0.399
 0.371 0.368 0.355 0.34  0.334 0.321 0.317 0.308 0.295 0.278 0.273]
L0H1 Q singular values [1.308 0.791 0.458 0.335 0.325 0.261 0.232 0.197 0.193 0.185 0.18  0.168 0.166 0.159 0.158 0.156 0.148 0.144 0.141 0.139 0.137 0.13  0.128 0.126 0.124 0.121 0.118 0.115 0.114
 0.11  0.109 0.107 0.106 0.104 0.103 0.101 0.098 0.098 0.096 0.094]
L0H1 K singular values [0.937 0.43  0.381 0.309 0.261 0.251 0.238 0.208 0.198 0.191 0.186 0.179 0.173 0.169 0.161 0.159 0.154 0.153 0.149 0.148 0.148 0.145 0.144 0.141 0.138 0.135 0.132 0.131 0.127
 0.126 0.123 0.118 0.116 0.115 0.114 0.113 0.108 0.106 0.104 0.103]
L0H1 V singular values [1.167 0.925 0.356 0.299 0.264 0.242 0.221 0.216 0.214 0.208 0.202 0.198 0.189 0.178 0.175 0.173 0.169 0.164 0.162 0.155 0.154 0.151 0.148 0.146 0.141 0.14  0.138 0.135 0.132
 0.131 0.128 0.126 0.122 0.121 0.117 0.115 0.115 0.109 0.108 0.107]
L0H1 O singular values [3.37  1.92  1.167 0.779 0.692 0.659 0.635 0.58  0.556 0.52  0.504 0.479 0.46  0.439 0.431 0.421 0.418 0.403 0.386 0.368 0.362 0.355 0.337 0.326 0.316 0.308 0.298 0.288 0.285
 0.275 0.267 0.256 0.249 0.246 0.239 0.234 0.229 0.22  0.218 0.214]
L0H2 Q singular values [1.128 0.428 0.41  0.347 0.291 0.26  0.214 0.192 0.19  0.178 0.176 0.17  0.159 0.155 0.154 0.152 0.147 0.144 0.14  0.134 0.131 0.129 0.125 0.124 0.121 0.12  0.117 0.115 0.113
 0.11  0.107 0.105 0.103 0.101 0.1   0.097 0.094 0.094 0.092 0.091]
L0H2 K singular values [0.734 0.37  0.321 0.292 0.26  0.24  0.23  0.214 0.199 0.194 0.193 0.187 0.184 0.175 0.171 0.167 0.16  0.156 0.153 0.153 0.147 0.144 0.142 0.14  0.135 0.133 0.13  0.13  0.128
 0.125 0.122 0.119 0.117 0.113 0.111 0.11  0.109 0.106 0.104 0.104]
L0H2 V singular values [1.014 0.816 0.383 0.289 0.248 0.228 0.225 0.216 0.205 0.199 0.189 0.182 0.18  0.177 0.172 0.168 0.162 0.159 0.158 0.152 0.15  0.147 0.144 0.142 0.138 0.135 0.131 0.13  0.129
 0.124 0.123 0.121 0.121 0.118 0.116 0.114 0.11  0.109 0.107 0.104]
L0H2 O singular values [2.837 1.665 1.114 0.758 0.714 0.671 0.617 0.579 0.574 0.544 0.525 0.505 0.493 0.457 0.438 0.423 0.408 0.404 0.383 0.372 0.359 0.351 0.331 0.321 0.317 0.308 0.306 0.294 0.284
 0.276 0.269 0.261 0.259 0.246 0.236 0.229 0.22  0.218 0.207 0.206]
L0H3 Q singular values [1.212 1.159 1.114 1.08  1.063 1.051 1.017 0.997 0.972 0.956 0.948 0.928 0.91  0.902 0.88  0.873 0.866 0.837 0.815 0.805 0.79  0.761 0.753 0.733 0.709 0.687 0.643 0.623 0.579
 0.517 0.502 0.48  0.472 0.443 0.412 0.39  0.365 0.362 0.342 0.313]
L0H3 K singular values [2.378 1.965 1.646 1.472 1.448 1.339 1.293 1.224 1.182 1.143 1.084 1.059 1.055 1.03  0.982 0.958 0.939 0.906 0.848 0.836 0.807 0.775 0.741 0.698 0.671 0.647 0.602 0.572 0.553
 0.49  0.459 0.436 0.41  0.384 0.365 0.352 0.343 0.319 0.3   0.289]
L0H3 V singular values [0.57  0.513 0.504 0.472 0.47  0.45  0.44  0.434 0.425 0.411 0.39  0.383 0.379 0.367 0.359 0.342 0.339 0.336 0.324 0.297 0.286 0.276 0.273 0.266 0.254 0.248 0.245 0.241 0.237
 0.231 0.225 0.223 0.22  0.21  0.206 0.204 0.197 0.194 0.183 0.18 ]
L0H3 O singular values [2.133 2.11  2.076 1.975 1.963 1.937 1.913 1.891 1.878 1.845 1.823 1.813 1.78  1.737 1.713 1.678 1.632 1.457 1.395 1.326 1.286 1.272 1.214 1.201 1.19  1.131 1.101 1.078 1.065
 1.011 0.994 0.956 0.935 0.916 0.864 0.817 0.756 0.74  0.713 0.688]
L0H4 Q singular values [1.028 0.444 0.403 0.346 0.326 0.275 0.244 0.231 0.219 0.196 0.189 0.183 0.178 0.168 0.166 0.161 0.159 0.156 0.151 0.142 0.14  0.138 0.132 0.131 0.128 0.127 0.125 0.124 0.12
 0.115 0.113 0.11  0.108 0.104 0.103 0.101 0.099 0.096 0.095 0.094]
L0H4 K singular values [0.632 0.415 0.399 0.313 0.299 0.258 0.256 0.232 0.222 0.213 0.197 0.196 0.191 0.186 0.183 0.178 0.175 0.171 0.167 0.16  0.155 0.153 0.148 0.147 0.146 0.142 0.14  0.138 0.134
 0.131 0.13  0.127 0.125 0.123 0.121 0.115 0.114 0.111 0.11  0.106]
L0H4 V singular values [0.699 0.636 0.274 0.249 0.229 0.222 0.208 0.203 0.19  0.185 0.179 0.174 0.174 0.169 0.167 0.164 0.161 0.159 0.155 0.152 0.148 0.143 0.141 0.138 0.135 0.134 0.128 0.127 0.125
 0.122 0.119 0.119 0.115 0.112 0.11  0.108 0.107 0.106 0.103 0.102]
L0H4 O singular values [2.229 1.184 0.907 0.696 0.655 0.637 0.589 0.553 0.528 0.502 0.493 0.48  0.457 0.435 0.422 0.406 0.389 0.376 0.363 0.348 0.336 0.331 0.323 0.307 0.305 0.285 0.282 0.275 0.268
 0.262 0.255 0.248 0.245 0.24  0.227 0.224 0.216 0.21  0.205 0.204]
L0H5 Q singular values [0.799 0.201 0.152 0.145 0.141 0.138 0.132 0.127 0.127 0.124 0.12  0.115 0.114 0.111 0.11  0.107 0.104 0.104 0.102 0.099 0.099 0.097 0.096 0.092 0.09  0.089 0.088 0.088 0.085
 0.083 0.081 0.081 0.079 0.078 0.076 0.075 0.074 0.073 0.072 0.07 ]
L0H5 K singular values [0.326 0.203 0.19  0.184 0.175 0.169 0.167 0.161 0.157 0.154 0.15  0.144 0.139 0.135 0.133 0.13  0.128 0.126 0.126 0.121 0.12  0.117 0.117 0.114 0.112 0.11  0.108 0.105 0.104
 0.103 0.102 0.1   0.098 0.095 0.094 0.093 0.089 0.088 0.087 0.086]
L0H5 V singular values [2.988 1.914 1.213 0.638 0.545 0.506 0.483 0.439 0.378 0.346 0.315 0.31  0.275 0.263 0.245 0.24  0.232 0.227 0.214 0.21  0.201
```

## Call 4

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=170)

def read(M,g):
    Z=M.astype(float)*g
    return Z-Z.mean(axis=1,keepdims=True)

F=W["W_fc.0"].astype(float)
G=read(W["W_fc.0"],W["ln2_w.0"])
s=slice(192,256)
O=W["W_o.0"][:,s].astype(float)
O-=O.mean(axis=0,keepdims=True)
V=read(W["W_v.0"][s],W["ln1_w.0"])
B=O@V

print("Raw V0H3 singular values:",np.linalg.svd(W["W_v.0"][s],compute_uv=False))
print("Raw Q0H3 singular values:",np.linalg.svd(W["W_q.0"][s],compute_uv=False))
print("Raw K0H3 singular values:",np.linalg.svd(W["W_k.0"][s],compute_uv=False))

U,_,_=np.linalg.svd(O,full_matrices=False)
Gcur=G-(G@U)@U.T
Gprev=G@B
# Remove a small, explicitly weight-derived control-coordinate subspace.
ctrl=np.array([21,39,58,99,184])
Gcur[:,ctrl]=0
Gprev[:,ctrl]=0
mask=np.linalg.norm(F,axis=1)>1.3
ids=np.where(mask)[0]
print("\nNumber of selected MLP weight rows:",len(ids))

for label,X in [("current-read residual",Gcur),("H3-composed read",Gprev)]:
    X=X[mask]
    X=X-X.mean(axis=0,keepdims=True)
    u,ss,vt=np.linalg.svd(X,full_matrices=False)
    print("\n",label,"singular values:",ss[:45])
    for drop in [0,1,2]:
        Z=u[:,drop:27]*ss[drop:27]
        Z/=np.linalg.norm(Z,axis=1,keepdims=True)+1e-20
        results=[]
        for k in [8,12,16,20,24,28,32,36]:
            km=KMeans(n_clusters=k,n_init=5,random_state=0).fit(Z)
            sil=silhouette_score(Z,km.labels_)
            results.append((k,round(float(sil),3),round(float(km.inertia_/len(Z)),3)))
        print("drop",drop,"k,silhouette,mean inertia",results)

print("\nWeight-only composition strengths: first MLP reads from each L0 OV product")
for j in range(6):
    s=slice(j*64,(j+1)*64)
    o=W["W_o.0"][:,s].astype(float)
    v=read(W["W_v.0"][s],W["ln1_w.0"])
    C=G@o@v
    print(j,"all",round(np.linalg.norm(C),3),
          "selected",round(np.linalg.norm(C[mask]),3),
          "top singular values",np.linalg.svd(C,compute_uv=False)[:12])

```

Output:

```
Raw V0H3 singular values: [2.102 1.948 1.903 1.866 1.843 1.819 1.769 1.755 1.674 1.651 1.629 1.621 1.589 1.553 1.52  1.465 1.417 1.303 1.16  1.101 0.945 0.93  0.848 0.818 0.79  0.779 0.745 0.739
 0.73  0.701 0.688 0.658 0.642 0.622 0.618 0.605 0.575 0.565 0.543 0.522 0.515 0.491 0.481 0.466 0.446 0.43  0.41  0.4   0.378 0.366 0.359 0.349 0.345 0.342 0.336 0.329
 0.318 0.313 0.313 0.298 0.287 0.279 0.271 0.262]
Raw Q0H3 singular values: [2.962 2.894 2.873 2.845 2.806 2.792 2.766 2.756 2.703 2.634 2.588 2.561 2.544 2.512 2.447 2.438 2.426 2.336 2.33  2.259 2.143 2.098 1.968 1.946 1.838 1.756 1.721 1.685
 1.575 1.54  1.495 1.463 1.423 1.378 1.283 1.136 1.116 1.094 1.06  1.023 0.991 0.969 0.962 0.897 0.864 0.824 0.778 0.77  0.731 0.725 0.698 0.688 0.663 0.637 0.614 0.589
 0.573 0.556 0.532 0.519 0.497 0.474 0.443 0.352]
Raw K0H3 singular values: [3.362 3.238 3.209 3.181 3.099 3.086 3.024 2.999 2.951 2.923 2.856 2.765 2.731 2.654 2.622 2.539 2.478 2.407 2.31  2.217 2.134 2.057 1.976 1.806 1.76  1.737 1.61  1.6
 1.536 1.42  1.296 1.229 1.198 1.172 1.117 1.058 1.016 0.988 0.942 0.878 0.85  0.812 0.785 0.751 0.72  0.711 0.704 0.685 0.641 0.61  0.601 0.578 0.562 0.542 0.532 0.526
 0.505 0.492 0.484 0.453 0.433 0.421 0.396 0.366]

Number of selected MLP weight rows: 561

 current-read residual singular values: [12.763  9.854  9.002  8.865  8.573  8.492  8.244  8.009  7.798  7.631  7.352  7.111  7.006  6.745  6.61   6.292  6.169  6.022  5.936  5.767  5.748  5.422  5.237  5.123
  4.918  4.752  4.319  4.181  4.017  3.457  3.446  3.274  3.176  3.053  2.984  2.941  2.775  2.702  2.693  2.666  2.578  2.514  2.455  2.409  2.365]
drop 0 k,silhouette,mean inertia [(8, 0.112, 0.726), (12, 0.139, 0.64), (16, 0.141, 0.597), (20, 0.171, 0.543), (24, 0.182, 0.499), (28, 0.176, 0.462), (32, 0.179, 0.441), (36, 0.174, 0.424)]
drop 1 k,silhouette,mean inertia [(8, 0.086, 0.771), (12, 0.115, 0.689), (16, 0.137, 0.626), (20, 0.16, 0.565), (24, 0.178, 0.518), (28, 0.182, 0.482), (32, 0.193, 0.452), (36, 0.194, 0.436)]
drop 2 k,silhouette,mean inertia [(8, 0.083, 0.771), (12, 0.104, 0.697), (16, 0.133, 0.626), (20, 0.156, 0.569), (24, 0.175, 0.518), (28, 0.178, 0.489), (32, 0.183, 0.462), (36, 0.191, 0.439)]

 H3-composed read singular values: [9.434 6.321 5.995 5.718 5.37  5.236 5.064 4.907 4.609 4.513 4.196 3.899 3.611 3.514 3.34  2.623 2.258 1.659 1.231 0.861 0.63  0.522 0.39  0.376 0.331 0.303 0.277 0.257
 0.237 0.227 0.22  0.209 0.188 0.183 0.178 0.166 0.157 0.144 0.139 0.133 0.114 0.105 0.092 0.091 0.075]
drop 0 k,silhouette,mean inertia [(8, 0.071, 0.73), (12, 0.072, 0.679), (16, 0.079, 0.633), (20, 0.077, 0.605), (24, 0.092, 0.572), (28, 0.087, 0.549), (32, 0.094, 0.528), (36, 0.091, 0.513)]
drop 1 k,silhouette,mean inertia [(8, 0.07, 0.748), (12, 0.072, 0.693), (16, 0.086, 0.637), (20, 0.086, 0.608), (24, 0.088, 0.579), (28, 0.087, 0.559), (32, 0.095, 0.531), (36, 0.09, 0.519)]
drop 2 k,silhouette,mean inertia [(8, 0.074, 0.739), (12, 0.08, 0.674), (16, 0.088, 0.631), (20, 0.094, 0.59), (24, 0.092, 0.569), (28, 0.095, 0.539), (32, 0.093, 0.524), (36, 0.098, 0.506)]

Weight-only composition strengths: first MLP reads from each L0 OV product
0 all 5.314 selected 5.176 top singular values [2.958 2.548 2.002 1.523 1.187 1.    0.944 0.799 0.751 0.657 0.568 0.52 ]
1 all 13.507 selected 12.514 top singular values [13.339  1.707  0.798  0.598  0.379  0.318  0.244  0.213  0.193  0.176  0.15   0.136]
2 all 8.765 selected 8.131 top singular values [8.623 1.097 0.708 0.577 0.3   0.213 0.204 0.185 0.171 0.15  0.144 0.133]
3 all 21.37 selected 21.116 top singular values [9.586 6.442 6.146 5.847 5.562 5.355 5.256 5.073 4.694 4.623 4.258 4.025]
4 all 6.648 selected 6.13 top singular values [6.566 0.701 0.489 0.26  0.211 0.193 0.178 0.161 0.141 0.124 0.119 0.111]
5 all 23.977 selected 22.249 top singular values [19.925  9.497  6.016  4.302  3.756  2.986  1.956  1.461  1.176  0.718  0.687  0.551]

```

## Call 5

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=170)

d=384
def read(M,g):
    Z=M.astype(float)*g
    return Z-Z.mean(axis=1,keepdims=True)
def cos(a,b):
    return float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))

A={}; B={}; Q={}; K={}; bias={}
for l in range(4):
    for h in range(6):
        s=slice(64*h,64*(h+1)); key=(l,h)
        Q[key]=read(W[f"W_q.{l}"][s],W[f"ln1_w.{l}"])
        K[key]=read(W[f"W_k.{l}"][s],W[f"ln1_w.{l}"])
        A[key]=Q[key].T@K[key]/8
        o=W[f"W_o.{l}"][:,s].astype(float)
        o-=o.mean(axis=0,keepdims=True)
        B[key]=o@read(W[f"W_v.{l}"][s],W[f"ln1_w.{l}"])
        bias[key]=K[key].T@(W[f"W_q.{l}"][s].astype(float)@W[f"ln1_b.{l}"])/8

C=np.eye(d)[:,[21,39,58,99,184]]
R=np.eye(d)-C@C.T
_,_,vt=np.linalg.svd(np.vstack([Q[0,3],K[0,3]]),full_matrices=False)
P=vt[:32].T
print("Within-layer QK matrix cosine similarities")
for l in range(4):
    print("Layer",l)
    print(np.array([[cos(A[l,h],A[l,j]) for j in range(6)] for h in range(6)]))

print("\nPer-head control-coordinate and positional-subspace fractions")
print("Columns: control query energy, control key energy; positional QK both-axis energy; bias-term norm; top 5 control-removed QK singular values")
for l in range(4):
    for h in range(6):
        a=A[l,h]; n=np.linalg.norm(a)**2
        ca=np.linalg.norm(C.T@a)**2/n
        ck=np.linalg.norm(a@C)**2/n
        pp=np.linalg.norm(P.T@a@P)**2/n
        s=np.linalg.svd(R@a@R,compute_uv=False)
        print(f"L{l}H{h}",np.round([ca,ck,pp,np.linalg.norm(bias[l,h])],3),s[:5])

print("\nBias-key direction similarities, within each layer")
for l in range(4):
    print(l,np.array([[cos(bias[l,h],bias[l,j]) for j in range(6)] for h in range(6)]))

print("\nLinear MLP-product key compositions (not nonlinear MLP evaluations)")
M={}
for l in range(3):
    p=W[f"W_proj.{l}"].astype(float)
    p-=p.mean(axis=0,keepdims=True)
    M[l]=p@read(W[f"W_fc.{l}"],W[f"ln2_w.{l}"])
for l in range(1,4):
    for h in range(6):
        scores=[]
        for k in range(l):
            z=A[l,h]@M[k]
            scores.append((k,round(np.trace(z)/(np.linalg.norm(z)+1e-30),3),round(np.linalg.norm(z),3)))
        print(f"L{l}H{h}",scores)

print("\nL0H3-aligned QK matrix power overlaps, using its rank-24 polar factor")
u,s,vt=np.linalg.svd(A[0,3],full_matrices=False)
T=u[:,:24]@vt[:24]
powers={0:u[:,:24]@u[:,:24].T}
for i in range(1,7):
    powers[i]=np.linalg.matrix_power(T,i)
    powers[-i]=powers[i].T
for l in range(4):
    for h in range(6):
        vals=[(i,cos(A[l,h],powers[i])) for i in range(-6,7)]
        vals.sort(key=lambda z:abs(z[1]),reverse=True)
        print(f"L{l}H{h}",[(i,round(v,3)) for i,v in vals[:3]])

```

Output:

```
Within-layer QK matrix cosine similarities
Layer 0
[[ 1.     0.012  0.008  0.137  0.013  0.044]
 [ 0.012  1.     0.691 -0.026  0.57   0.026]
 [ 0.008  0.691  1.    -0.063  0.493 -0.075]
 [ 0.137 -0.026 -0.063  1.    -0.044  0.01 ]
 [ 0.013  0.57   0.493 -0.044  1.     0.033]
 [ 0.044  0.026 -0.075  0.01   0.033  1.   ]]
Layer 1
[[1.    0.357 0.372 0.345 0.334 0.376]
 [0.357 1.    0.371 0.361 0.302 0.398]
 [0.372 0.371 1.    0.368 0.375 0.351]
 [0.345 0.361 0.368 1.    0.32  0.394]
 [0.334 0.302 0.375 0.32  1.    0.288]
 [0.376 0.398 0.351 0.394 0.288 1.   ]]
Layer 2
[[1.    0.128 0.161 0.117 0.179 0.134]
 [0.128 1.    0.116 0.086 0.121 0.116]
 [0.161 0.116 1.    0.101 0.144 0.123]
 [0.117 0.086 0.101 1.    0.092 0.093]
 [0.179 0.121 0.144 0.092 1.    0.134]
 [0.134 0.116 0.123 0.093 0.134 1.   ]]
Layer 3
[[1.    0.058 0.082 0.06  0.007 0.108]
 [0.058 1.    0.069 0.034 0.041 0.075]
 [0.082 0.069 1.    0.031 0.036 0.094]
 [0.06  0.034 0.031 1.    0.036 0.058]
 [0.007 0.041 0.036 0.036 1.    0.027]
 [0.108 0.075 0.094 0.058 0.027 1.   ]]

Per-head control-coordinate and positional-subspace fractions
Columns: control query energy, control key energy; positional QK both-axis energy; bias-term norm; top 5 control-removed QK singular values
L0H0 [0.15  0.085 0.624 0.043] [0.214 0.092 0.044 0.039 0.036]
L0H1 [0.113 0.01  0.288 0.013] [0.141 0.038 0.02  0.011 0.008]
L0H2 [0.067 0.016 0.369 0.007] [0.098 0.017 0.014 0.011 0.008]
L0H3 [0.062 0.11  0.933 0.043] [0.226 0.204 0.18  0.161 0.156]
L0H4 [0.121 0.014 0.252 0.017] [0.073 0.022 0.018 0.012 0.01 ]
L0H5 [0.243 0.023 0.101 0.005] [0.027 0.004 0.003 0.002 0.002]
L1H0 [0.053 0.433 0.016 3.975] [0.672 0.544 0.474 0.434 0.407]
L1H1 [0.064 0.473 0.019 3.689] [0.699 0.466 0.412 0.368 0.319]
L1H2 [0.044 0.491 0.017 3.051] [0.573 0.473 0.396 0.363 0.319]
L1H3 [0.065 0.501 0.023 3.39 ] [0.69  0.39  0.376 0.331 0.304]
L1H4 [0.031 0.455 0.014 2.329] [0.506 0.416 0.347 0.323 0.305]
L1H5 [0.073 0.462 0.022 3.895] [0.677 0.503 0.434 0.385 0.354]
L2H0 [0.071 0.262 0.021 2.02 ] [0.389 0.346 0.317 0.304 0.298]
L2H1 [0.035 0.104 0.013 1.88 ] [0.526 0.493 0.474 0.455 0.44 ]
L2H2 [0.036 0.134 0.015 1.608] [0.405 0.388 0.356 0.334 0.325]
L2H3 [0.013 0.094 0.008 1.06 ] [0.389 0.367 0.356 0.329 0.327]
L2H4 [0.045 0.174 0.015 2.305] [0.585 0.494 0.471 0.427 0.424]
L2H5 [0.035 0.111 0.013 2.023] [0.56  0.536 0.501 0.484 0.462]
L3H0 [0.012 0.046 0.007 0.5  ] [0.411 0.394 0.38  0.378 0.365]
L3H1 [0.016 0.064 0.008 0.28 ] [0.311 0.29  0.253 0.245 0.231]
L3H2 [0.021 0.067 0.009 0.495] [0.304 0.296 0.285 0.268 0.265]
L3H3 [0.003 0.007 0.007 0.22 ] [0.321 0.31  0.307 0.304 0.297]
L3H4 [0.015 0.015 0.007 0.144] [0.235 0.196 0.167 0.143 0.13 ]
L3H5 [0.013 0.065 0.008 0.493] [0.332 0.318 0.306 0.298 0.289]

Bias-key direction similarities, within each layer
0 [[ 1.    -0.054 -0.104 -0.041  0.29  -0.095]
 [-0.054  1.     0.347 -0.093  0.41  -0.226]
 [-0.104  0.347  1.    -0.057  0.567  0.524]
 [-0.041 -0.093 -0.057  1.     0.022  0.074]
 [ 0.29   0.41   0.567  0.022  1.     0.373]
 [-0.095 -0.226  0.524  0.074  0.373  1.   ]]
1 [[1.    0.829 0.853 0.823 0.825 0.86 ]
 [0.829 1.    0.851 0.864 0.821 0.861]
 [0.853 0.851 1.    0.83  0.834 0.833]
 [0.823 0.864 0.83  1.    0.834 0.873]
 [0.825 0.821 0.834 0.834 1.    0.828]
 [0.86  0.861 0.833 0.873 0.828 1.   ]]
2 [[1.    0.807 0.766 0.703 0.82  0.835]
 [0.807 1.    0.811 0.719 0.824 0.837]
 [0.766 0.811 1.    0.699 0.832 0.827]
 [0.703 0.719 0.699 1.    0.715 0.73 ]
 [0.82  0.824 0.832 0.715 1.    0.845]
 [0.835 0.837 0.827 0.73  0.845 1.   ]]
3 [[ 1.     0.6    0.747  0.013  0.159  0.776]
 [ 0.6    1.     0.553  0.015  0.289  0.669]
 [ 0.747  0.553  1.     0.011  0.208  0.736]
 [ 0.013  0.015  0.011  1.     0.026 -0.046]
 [ 0.159  0.289  0.208  0.026  1.     0.254]
 [ 0.776  0.669  0.736 -0.046  0.254  1.   ]]

Linear MLP-product key compositions (not nonlinear MLP evaluations)
L1H0 [(0, np.float64(-0.142), np.float64(66.888))]
L1H1 [(0, np.float64(-0.174), np.float64(64.557))]
L1H2 [(0, np.float64(-0.051), np.float64(69.488))]
L1H3 [(0, np.float64(-0.093), np.float64(65.227))]
L1H4 [(0, np.float64(-0.039), np.float64(64.165))]
L1H5 [(0, np.float64(-0.243), np.float64(63.871))]
L2H0 [(0, np.float64(0.036), np.float64(37.687)), (1, np.float64(-0.015), np.float64(15.571))]
L2H1 [(0, np.float64(-0.105), np.float64(34.302)), (1, np.float64(0.268), np.float64(21.007))]
L2H2 [(0, np.float64(-0.014), np.float64(30.113)), (1, np.float64(0.31), np.float64(17.141))]
L2H3 [(0, np.float64(-0.06), np.float64(30.689)), (1, np.float64(0.295), np.float64(16.353))]
L2H4 [(0, np.float64(-0.044), np.float64(43.71)), (1, np.float64(0.005), np.float64(21.369))]
L2H5 [(0, np.float64(-0.048), np.float64(37.101)), (1, np.float64(0.213), np.float64(21.717))]
L3H0 [(0, np.float64(-0.08), np.float64(27.331)), (1, np.float64(-0.32), np.float64(18.878)), (2, np.float64(0.217), np.float64(21.839))]
L3H1 [(0, np.float64(-0.047), np.float64(19.264)), (1, np.float64(0.009), np.float64(11.815)), (2, np.float64(0.11), np.float64(13.853))]
L3H2 [(0, np.float64(-0.003), np.float64(21.311)), (1, np.float64(0.018), np.float64(13.679)), (2, np.float64(0.176), np.float64(15.897))]
L3H3 [(0, np.float64(-0.264), np.float64(19.662)), (1, np.float64(-0.463), np.float64(16.07)), (2, np.float64(-0.324), np.float64(15.718))]
L3H4 [(0, np.float64(-0.009), np.float64(8.827)), (1, np.float64(-0.154), np.float64(6.055)), (2, np.float64(0.011), np.float64(7.361))]
L3H5 [(0, np.float64(-0.072), np.float64(26.089)), (1, np.float64(-0.106), np.float64(16.224)), (2, np.float64(0.316), np.float64(17.792))]

L0H3-aligned QK matrix power overlaps, using its rank-24 polar factor
L0H0 [(-2, -0.07), (-1, -0.063), (-3, -0.056)]
L0H1 [(-1, 0.046), (-5, -0.028), (0, -0.025)]
L0H2 [(0, -0.059), (-2, 0.039), (1, -0.038)]
L0H3 [(1, 0.926), (2, 0.482), (4, -0.433)]
L0H4 [(1, -0.032), (0, -0.028), (-1, 0.027)]
L0H5 [(-1, -0.05), (0, -0.045), (-5, 0.037)]
L1H0 [(2, 0.012), (-6, 0.01), (-2, -0.01)]
L1H
```

## Call 6

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=2,suppress=True,linewidth=200)

def rd(M,g):
    X=M.astype(float)*g
    return X-X.mean(axis=1,keepdims=True)

F=W["W_fc.0"].astype(float)
G=rd(F,W["ln2_w.0"])
s=slice(192,256)
O=W["W_o.0"][:,s].astype(float)
V=rd(W["W_v.0"][s],W["ln1_w.0"])
Q=rd(W["W_q.0"][s],W["ln1_w.0"])
K=rd(W["W_k.0"][s],W["ln1_w.0"])
uo,so,_=np.linalg.svd(O,full_matrices=False)
_,sp,pt=np.linalg.svd(np.vstack([Q,K]),full_matrices=False)
ctrl=np.eye(384)[:,[21,39,58,99,184]]
mask=np.linalg.norm(F,axis=1)>1.3
G=G[mask]
GP=G@O@V
variants={}
for name,blocks in [
    ("O64", [uo,ctrl]),
    ("O32_position24", [uo[:,:32],pt[:24].T,ctrl]),
    ("O64_position32", [uo,pt[:32].T,ctrl]),
]:
    U,sv,_=np.linalg.svd(np.hstack(blocks),full_matrices=False)
    U=U[:,sv>1e-6]
    C=G-(G@U)@U.T
    variants[name]=C

results=[]
for name,C in variants.items():
    C=C-C.mean(axis=0,keepdims=True)
    u,ss,vt=np.linalg.svd(C,full_matrices=False)
    print(name,"spectrum",ss[:36].round(2))
    for r in [18,22,26,30]:
        X=u[:,:r]*ss[:r]
        X/=np.linalg.norm(X,axis=1,keepdims=True)+1e-20
        for k in [20,24,28]:
            km=KMeans(n_clusters=k,n_init=8,random_state=7).fit(X)
            score=float(silhouette_score(X,km.labels_))
            results.append((score,name,r,k,km.labels_))
results.sort(key=lambda x:x[0],reverse=True)
print("\nBest cluster fits (silhouette, construction, rank, number)")
print([(round(s,3),n,r,k) for s,n,r,k,_ in results[:12]])

score,name,r,k,lab=results[0]
C=variants[name]
cent=np.array([C[lab==i].mean(axis=0) for i in range(k)])
prev=np.array([GP[lab==i].mean(axis=0) for i in range(k)])
cent-=cent.mean(axis=0,keepdims=True)
prev-=prev.mean(axis=0,keepdims=True)
gram=cent@cent.T
reg=.08*np.trace(gram)/k
J=(prev@cent.T)@np.linalg.inv(gram+reg*np.eye(k))
J-=J.mean(axis=0,keepdims=True)
J-=J.mean(axis=1,keepdims=True)
J/=np.sqrt(np.mean(J**2))
print("\nBest-fit cluster counts",np.bincount(lab,minlength=k))
print("Regularized cross-channel cluster coefficient matrix (weight algebra only)")
print(J.round(1))
print("\nSymmetrized coefficient eigenvalues")
ev,U=np.linalg.eigh((J+J.T)/2)
print(ev.round(2))
vec=U[:,0]
part=vec>0
print("Most-negative-eigenvector partition:",np.where(part)[0],np.where(~part)[0])
print("Four block means:",[[round(float(J[np.ix_(a,b)].mean()),3) for b in [part,~part]] for a in [part,~part]])

print("\nTop-3 previous-channel coefficient clusters for each current-read cluster")
for i in range(k):
    ix=np.argsort(-J[i])[:3]
    print(i,int(np.sum(lab==i)),[(int(j),round(float(J[i,j]),2)) for j in ix])

```

Output:

```
O64 spectrum [12.76  9.85  9.    8.86  8.57  8.49  8.24  8.01  7.8   7.63  7.35  7.11  7.    6.75  6.61  6.29  6.17  6.02  5.93  5.77  5.75  5.42  5.24  5.12  4.92  4.75  4.31  4.18  4.01  3.45  3.44  3.27  3.18
  3.05  2.98  2.94]
O32_position24 spectrum [12.9  10.01  9.34  8.99  8.66  8.6   8.25  8.12  7.88  7.75  7.52  7.26  7.11  6.8   6.71  6.52  6.34  6.11  6.    5.9   5.82  5.59  5.37  5.3   5.19  4.91  4.6   4.19  4.11  3.62  3.46  3.31  3.28
  3.09  3.07  3.  ]
O64_position32 spectrum [12.45  9.63  8.88  8.81  8.53  8.41  8.18  7.96  7.73  7.53  7.29  7.01  6.91  6.68  6.56  6.22  6.07  5.98  5.83  5.71  5.63  5.37  5.09  5.05  4.83  4.57  4.13  3.95  3.93  3.36  3.3   3.11  3.06
  2.93  2.9   2.85]

Best cluster fits (silhouette, construction, rank, number)
[(0.2, 'O64_position32', 26, 28), (0.199, 'O64_position32', 22, 24), (0.199, 'O64_position32', 18, 20), (0.195, 'O64', 18, 20), (0.191, 'O64', 26, 24), (0.188, 'O64', 22, 20), (0.188, 'O64_position32', 22, 28), (0.187, 'O64_position32', 18, 28), (0.185, 'O64', 22, 28), (0.185, 'O32_position24', 26, 28), (0.184, 'O64', 18, 28), (0.182, 'O32_position24', 18, 28)]

Best-fit cluster counts [15  8 25 14 92 23 25 15 29 20 28 18 13 13 21 17 12 28 11 21 13 13 16 17 14 13 10 17]
Regularized cross-channel cluster coefficient matrix (weight algebra only)
[[-0.  -1.1 -0.9  0.   0.2 -1.8 -0.6  0.8  0.8  0.3 -0.1  0.6 -0.4 -0.1 -0.7  0.7  1.5  0.4 -0.2 -1.  -0.9  0.3  0.8 -0.1  2.   0.4  0.2 -1. ]
 [-0.4  2.8  0.2  0.1 -0.4  1.8 -0.3  0.4 -0.7 -0.2 -0.6 -1.3  0.4  1.2  1.6 -0.5 -4.  -1.2  0.3  0.2 -0.2  0.2 -0.8  0.3 -0.5  0.9 -1.2  1.5]
 [ 0.3 -0.1  2.4 -1.5  0.2  2.9 -0.1 -0.8 -1.3  0.3  0.2 -1.1  0.4  1.4 -1.6 -1.5  1.3 -0.7 -0.8  1.2  0.  -0.2  1.2 -1.4 -2.4  1.3  0.9 -0.6]
 [ 0.4  0.2 -1.1  2.3 -0.1  0.1  0.9 -1.7 -0.4 -0.1 -0.5  0.3  0.1 -2.  -0.1  0.3  1.2  0.8 -0.6  1.2 -0.   0.5 -1.6  0.  -0.6  0.8  0.1 -0.3]
 [ 0.4 -0.2  0.4 -1.2  1.2  0.6  0.5 -0.6 -0.2  0.3  1.8  0.7 -0.2  0.4  0.5  0.5 -0.5  0.  -0.7  0.2 -0.1 -0.9  0.2 -1.7 -0.1  0.7 -0.3 -1.7]
 [ 1.5 -0.5  1.9 -0.5  1.3  2.3  0.4 -2.1 -2.6  0.3  0.7 -0.9  0.6 -1.5  1.5  0.2 -0.7  2.7 -0.5  3.3 -2.1 -0.9 -0.9 -1.2 -1.1  0.6  0.  -2. ]
 [ 0.5  0.4  1.2  0.1  0.3 -1.7  2.3 -0.6 -0.8 -0.4  0.  -0.6  0.2 -0.6  0.8 -0.4 -1.1  1.1 -0.3  1.2 -1.4 -0.2 -1.8  0.6 -0.4  1.4  0.4 -0.2]
 [ 0.1  0.3 -0.4  0.2 -0.2  0.5  0.4 -0.6  0.1  0.3  0.4  0.4  0.2  0.2 -0.8 -0.1  0.1 -0.5  0.1 -0.2 -0.9 -0.9 -1.2  0.5  0.3  1.4 -0.2  0.3]
 [-1.5  0.2 -2.9  1.3 -1.9 -2.6 -1.4  1.3  1.8 -0.5 -1.5  3.1  0.5  1.1 -0.9  2.4  1.4 -2.9  0.5 -1.9  0.4  2.  -2.1  1.6  2.4 -0.7 -0.6  1.5]
 [ 0.6 -0.3 -0.3 -0.9  0.3  0.3 -0.  -0.7  0.5  2.1  1.2  0.3 -0.9  0.5  0.5  0.   0.4 -1.1  0.  -0.3 -1.5 -0.3  0.6 -1.1 -0.3  0.1  0.9 -0.6]
 [-0.1 -0.2 -0.6 -0.  -0.6 -0.5  0.3  0.7  1.5  0.5  0.6  0.6 -0.4  0.1 -1.8 -0.6  1.9 -2.1  0.3 -1.5  0.5  0.4  1.2  0.2  0.  -1.5  0.8  0.6]
 [-0.3 -0.2  0.1  0.3 -0.2 -1.6  0.2  0.1  0.7  0.2  0.1  0.9 -0.2  0.1 -0.3  0.6  0.2 -0.6  0.  -0.5 -0.1  0.2 -0.7  0.3  0.2  0.3  0.   0.3]
 [ 0.3 -0.1  1.  -1.7  0.3  0.8  0.1 -0.2 -0.1 -0.4  0.2 -0.4  2.6  0.5  0.8  0.3  0.4 -0.3 -1.1  0.7 -0.3  0.   0.6 -2.8 -0.7  0.8 -0.3 -1.1]
 [-1.7 -0.1  0.9 -0.8 -0.5 -2.2 -2.2  2.  -0.6 -1.2 -2.7  1.2 -1.   2.4 -2.   3.8  0.9  0.9 -0.4 -0.7  1.1  1.   2.3  1.2 -1.2  0.3 -0.2 -0.3]
 [ 0.4 -0.4 -0.  -0.   0.2  0.   0.9  0.2 -0.7 -0.2 -0.8 -0.1 -0.4  0.1 -0.1 -0.4  0.   1.1 -0.2  0.5 -0.4  0.7  0.1 -0.6  0.5 -0.3  0.1 -0. ]
 [-0.1 -0.  -1.5  0.7 -0.3 -0.6  0.3  0.7  0.3 -0.5 -0.6 -1.   0.  -0.4  1.7 -0.8  0.1  0.6  0.3 -0.5 -0.2  0.3  0.1  0.4  0.9 -1.4 -0.3  1.6]
 [ 0.1 -0.2  1.3 -0.1  0.3 -0.4 -0.5  0.2 -0.2 -0.1  0.3  0.3 -0.2 -0.4 -0.5 -0.2  0.   0.6  0.4  0.  -0.1 -0.6  0.9  0.6 -0.  -0.4 -0.2 -0.8]
 [ 0.6 -0.7  0.6 -0.4  0.9  0.2  0.4  0.  -0.2  0.3  0.6 -1.  -0.1 -1.   1.5 -1.4 -1.8  1.9  0.2  0.4 -0.4 -0.6  0.6 -0.8  0.6 -0.5 -0.2 -0.1]
 [ 0.7  0.6 -0.7 -0.5  0.4  1.   0.  -0.  -1.1  0.2  1.1  0.1 -0.7  0.3  0.8 -1.3 -2.4  2.3  1.8 -0.3  0.4 -0.2  1.1  0.5 -0.4 -2.2 -0.9 -0.5]
 [ 0.6 -0.1  0.2 -0.5  0.3  2.5 -0.5 -0.6 -0.6  0.8  1.   0.8  0.1  0.2 -1.  -0.6 -1.6  0.1  0.4  0.4 -1.2 -1.2  0.3 -0.2  0.3  0.8 -0.3 -0.5]
 [-0.7  0.  -1.6  0.7 -0.7  1.8 -1.7 -0.2  0.9 -0.1 -0.1 -0.2 -0.5 -0.5  0.6  0.2  0.1 -1.8  0.6 -0.3  3.5  0.8  1.5  0.4 -1.4 -1.7  0.4  0. ]
 [ 0.4  0.4  1.9  0.6  0.6  0.8  0.3  0.2 -0.5 -0.5 -0.6 -3.3  0.3 -1.9  0.7 -0.  -0.8  1.2 -0.6  1.3  0.1  1.7 -0.1 -0.3 -1.2 -1.1 -0.4  0.7]
 [ 0.  -0.3  0.4 -0.6  0.7  0.4 -0.4  0.1  0.8  0.3  0.9 -0.4  0.8 -0.5  2.  -1.  -1.8 -0.1  0.3 -0.2  1.1 -0.4  0.4 -1.8  0.7 -0.9 -0.7 -0. ]
 [-0.  -0.3 -0.4  0.  -0.  -0.4 -0.1 -0.2 -0.2 -0.1 -0.3 -0.1 -0.6 -0.2 -0.7  1.1  1.3  0.6 -0.1 -0.1 -0.4 -0.1  0.1  1.3 -0.4  0.9  0.  -0.8]
 [-0.3  0.1 -1.1  0.4 -0.3 -0.4  1.2  0.5  0.8  0.  -0.1  2.1 -0.2  0.2 -1.4  0.1 -0.4 -1.3  0.5 -1.2  0.7 -0.6 -1.   0.8  1.7 -1.  -0.5  0.6]
 [ 0.3  0.3 -0.5  1.2 -0.1  0.2  0.4 -0.7 -0.6 -0.5 -0.9 -1.9  0.1 -0.6 -0.6 -0.6  2.3  2.1 -0.   0.1 -0.   0.6 -2.3  0.7  0.4  0.2 -0.7  1.1]
 [-0.3 -0.6  0.9  0.4 -0.2 -1.6  0.4 -0.2  0.7 -0.3  0.9 -0.5 -0.2 -0.8 -0.2  0.3 -0.  -1.  -0.8  0.4  1.4 -0.9 -0.3  0.1 -0.2  0.6  2.7 -0.6]
 [-1.9  0.  -1.3  0.4 -1.6 -2.7 -1.1  1.8  1.9 -0.5 -1.   1.3 -0.5  2.  -0.4 -0.8  1.9 -2.7  0.7 -2.8  0.9 -0.6  0.9  2.5  0.8 -0.1  0.1  2.9]]

Symmetrized coefficient eigenvalues
[-4.85 -4.19 -3.33 -2.75 -2.19 -1.99 -0.97 -0.68 -0.6  -0.18 -0.    0.15  0.26  0.73  0.99  1.35  1.41  1.74  2.36  2.78  3.4   3.94  4.3   5.41  6.1   6.36  7.41 13.64]
Most-negative-eigenvector partition: [ 1  5  6  7 11 15 16 17 18 21 22 26] [ 0  2  3  4  8  9 10 12 13 14 19 20 23 24 25 27]
Four block means: [[-0.116, 0.087], [0.087, -0.065]]

Top-3 previous-channel coefficient clusters for each current-read cluster
0 15 [(24, 2.02), (16, 1.45), (7, 0.83)]
1 8 [(1, 2.81), (5, 1.82), (14, 1.64)]
2 25 [(5, 2.91), (2, 2.42), (13, 1.38)]
3 14 [(3, 2.25), (16, 1.25), (19, 1.21)]
4 92 [(10, 1.76), (4, 1.2), (25, 0.74)]
5 23 [
```

## Call 7

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)

print("MLP read and write singular spectra; first 64 and selected tail quantiles")
for l in range(4):
    for typ in ["W_fc","W_proj"]:
        M=W[f"{typ}.{l}"].astype(float)
        s=np.linalg.svd(M,compute_uv=False)
        print(f"{typ}.{l}",s[:64])
        print("tail indices 80,100,128,192,256,320,383:",
              s[[80,100,128,192,256,320,383]])
        print("largest adjacent spectral ratios among first 60:",
              sorted([(i+1,float(s[i]/s[i+1])) for i in range(60)],
                     key=lambda z:z[1],reverse=True)[:7])

print("\nLast-layer projection covariance eigenvalue concentration")
P=W["W_proj.3"].astype(float)
U,s,vt=np.linalg.svd(P,full_matrices=False)
for r in [1,2,5,10,15,18,20,22,24,26,27,28,32,40,64]:
    print(r,round(float(np.sum(s[:r]**2)/np.sum(s**2)),4))

print("\nFirst-layer MLP read spectrum restricted to last-layer principal write subspaces")
F=W["W_fc.0"].astype(float)*W["ln2_w.0"]
for r in [20,24,28,32]:
    vals=np.linalg.svd(F@U[:,:r],compute_uv=False)
    print(r,vals)

print("\nSelected-coordinate LayerNorm weights and biases")
ix=[21,39,58,99,184]
for l in range(4):
    for name in ["ln1_w","ln1_b","ln2_w","ln2_b"]:
        print(f"{name}.{l}",W[f"{name}.{l}"][ix])

print("\nSpectra of last-layer OV eigenvalues, strongest magnitudes")
for j in range(6):
    sl=slice(64*j,64*(j+1))
    V=W["W_v.3"][sl].astype(float)*W["ln1_w.3"]
    O=W["W_o.3"][:,sl].astype(float)
    ev=np.linalg.eigvals(V@O)
    ev=ev[np.argsort(-abs(ev))]
    print(j,ev[:16])

```

Output:

```
MLP read and write singular spectra; first 64 and selected tail quantiles
W_fc.0 [18.118 12.76  11.337  9.323  9.267  9.021  8.898  8.817  8.609  8.4    8.189  8.06   7.819  7.724  7.62   7.543  7.226  7.081  6.979  6.833  6.779  6.682  6.592  6.402  6.39
  6.306  6.207  6.18   5.974  5.864  5.789  5.741  5.646  5.579  5.512  5.353  5.196  5.193  5.122  5.053  5.029  4.849  4.646  4.573  4.452  4.163  4.133  3.899  3.888  3.557
  3.455  3.347  3.241  3.008  2.921  2.833  2.796  2.657  2.612  2.555  2.418  2.381  2.33   2.3  ]
tail indices 80,100,128,192,256,320,383: [1.712 1.441 1.234 0.948 0.765 0.611 0.431]
largest adjacent spectral ratios among first 60: [(1, 1.4198725705756245), (3, 1.2160754874309363), (2, 1.1255283212765597), (49, 1.093109016954483), (53, 1.0773536623360322), (45, 1.0694199934184154), (47, 1.0600723588821415)]
W_proj.0 [10.899  7.702  7.173  6.638  6.541  6.288  6.201  6.133  5.941  5.917  5.758  5.641  5.552  5.476  5.393  5.363  5.259  5.206  5.171  5.153  5.057  5.018  4.96   4.88   4.821
  4.788  4.77   4.73   4.643  4.579  4.546  4.529  4.505  4.493  4.44   4.415  4.369  4.315  4.293  4.261  4.243  4.185  4.134  4.104  4.064  4.049  4.026  3.988  3.967  3.95
  3.933  3.901  3.857  3.804  3.778  3.754  3.74   3.713  3.692  3.683  3.652  3.612  3.607  3.575]
tail indices 80,100,128,192,256,320,383: [3.185 2.802 2.308 1.473 0.932 0.579 0.281]
largest adjacent spectral ratios among first 60: [(1, 1.4150739590359884), (3, 1.0806116248625277), (2, 1.073793176148483), (5, 1.0402703479810216), (8, 1.0322999269423645), (10, 1.0276747651284515), (11, 1.0206948609130797)]
W_fc.1 [19.829 11.575 11.522 11.262 11.059 10.875 10.632 10.497 10.363 10.3   10.174 10.05   9.875  9.759  9.703  9.671  9.42   9.191  9.065  8.835  8.81   8.705  8.595  8.58   8.435
  8.206  8.123  8.078  7.899  7.826  7.758  7.676  7.635  7.592  7.463  7.395  7.33   7.244  7.185  7.132  7.062  6.82   6.764  6.645  6.591  6.568  6.514  6.48   6.426  6.346
  6.271  6.216  6.192  6.148  6.1    6.035  5.947  5.81   5.766  5.762  5.715  5.68   5.646  5.578]
tail indices 80,100,128,192,256,320,383: [4.785 4.08  3.238 1.938 1.254 0.908 0.434]
largest adjacent spectral ratios among first 60: [(1, 1.7131055119046008), (41, 1.035362687097689), (25, 1.0280141733399553), (16, 1.0266769278521082), (19, 1.026007879360779), (17, 1.0248351633616963), (57, 1.0236424008356602)]
W_proj.1 [9.038 8.457 8.103 7.333 7.318 7.088 6.772 6.666 6.566 6.429 6.372 6.278 6.246 6.232 6.126 6.053 6.012 5.96  5.914 5.901 5.845 5.833 5.816 5.764 5.744 5.705 5.68  5.656 5.651
 5.594 5.563 5.549 5.526 5.508 5.489 5.439 5.431 5.41  5.396 5.392 5.358 5.344 5.325 5.303 5.296 5.278 5.261 5.241 5.235 5.228 5.207 5.195 5.175 5.142 5.137 5.112 5.1   5.084
 5.068 5.05  5.031 5.022 5.004 4.986]
tail indices 80,100,128,192,256,320,383: [4.749 4.529 4.21  3.596 3.019 2.449 0.359]
largest adjacent spectral ratios among first 60: [(3, 1.1050228014290797), (1, 1.0686433120577044), (6, 1.0466270391408243), (2, 1.0437325032616935), (5, 1.0325808016710536), (9, 1.0213116863152258), (14, 1.017271856715043)]
W_fc.2 [15.734  9.249  8.718  8.694  8.47   7.865  7.739  7.482  7.456  7.36   7.336  7.236  7.21   7.163  7.095  7.027  6.917  6.876  6.815  6.772  6.717  6.701  6.681  6.591  6.564
  6.544  6.497  6.456  6.439  6.417  6.335  6.289  6.24   6.202  6.181  6.13   6.09   6.046  6.014  5.985  5.9    5.876  5.857  5.844  5.827  5.762  5.732  5.707  5.701  5.666
  5.651  5.608  5.573  5.535  5.519  5.495  5.461  5.445  5.404  5.382  5.35   5.323  5.318  5.307]
tail indices 80,100,128,192,256,320,383: [4.933 4.573 4.123 3.374 2.724 2.114 0.407]
largest adjacent spectral ratios among first 60: [(1, 1.7011480999518707), (5, 1.0769823878275124), (2, 1.0609065360521581), (7, 1.0343999740124776), (4, 1.026357401180896), (6, 1.0162378463756006), (16, 1.015972278930189)]
W_proj.2 [10.197  9.035  8.882  8.411  8.224  8.088  7.988  7.827  7.679  7.491  7.372  7.26   7.174  7.069  7.031  6.803  6.736  6.659  6.459  6.431  6.34   6.319  6.253  6.231  6.158
  6.073  6.017  6.     5.963  5.904  5.886  5.792  5.762  5.741  5.731  5.672  5.655  5.635  5.569  5.546  5.526  5.514  5.471  5.442  5.429  5.419  5.395  5.362  5.343  5.322
  5.283  5.278  5.251  5.246  5.216  5.184  5.14   5.135  5.122  5.108  5.093  5.066  5.049  5.028]
tail indices 80,100,128,192,256,320,383: [4.735 4.455 4.109 3.446 2.852 2.296 0.357]
largest adjacent spectral ratios among first 60: [(1, 1.1286962341082047), (3, 1.0560361908211944), (15, 1.0334718792845088), (18, 1.030848324793185), (9, 1.0250751035959011), (4, 1.0226910191088667), (7, 1.0204814000742461)]
W_fc.3 [15.114  8.611  8.486  8.073  7.769  7.529  7.449  7.334  7.094  6.971  6.891  6.808  6.601  6.537  6.454  6.331  6.294  6.232  6.162  6.115  6.058  6.034  5.926  5.878  5.85
  5.803  5.786  5.751  5.723  5.713  5.669  5.644  5.591  5.547  5.529  5.512  5.494  5.466  5.454  5.444  5.4    5.348  5.344  5.325  5.294  5.275  5.253  5.237  5.214  5.193
  5.186  5.17   5.139  5.13   5.098  5.091  5.061  5.059  5.046  5.036  5.014  5.001  4.972  4.962]
tail indices 80,100,128,192,256,320,383: [4.694 4.429 4.12  3.516 2.967 2.434 0.658]
largest adjacent spectral ratios among first 60: [(1, 1.7551767924559971), (3, 1.0511342304741367), (4, 1.0391679854024989), (8, 1.0339285450842044), (5, 1.031841540738613), (12, 1.0312650373142738), (15, 1.0194762651845115)]
W_proj.3 [28.461 22.214 21.34  20.708 19.943 19.662 18.986 18.744 18.295 17.457 17.132 16.561 15.329 14.874 14.328 13.438 12.667 12.455 12.317 11.531 11.047  9.084  8.353  8.24   6.37
  5.836  4.626  4.583  3.777  3.61   3.252  3.183  3.086  2.941  2.865  2.801  2.639  2.564  2.538  2.466  2.419  2.404  2.346  2.31   2.246  2.227  2.155  2.147  2.089  2.069
  2.065  2.041  1.975  1.936  1.921  1.917  1.857  1.848  1.83   1.82   1.798  1.779  1.764  1.737]
tail indices 80,100,128,192,256,320,383: [1.515 1.321 1.11  0.814 0.621 0.472 0.315]
largest adjacent spectral ratios among first 60: [
```

## Call 8

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)

def read(M,g):
    X=M.astype(float)*g
    return X-X.mean(axis=1,keepdims=True)

A={}; B={}
for l in range(4):
    for h in range(6):
        sl=slice(64*h,64*(h+1))
        Q=read(W[f"W_q.{l}"][sl],W[f"ln1_w.{l}"])
        K=read(W[f"W_k.{l}"][sl],W[f"ln1_w.{l}"])
        O=W[f"W_o.{l}"][:,sl].astype(float)
        O-=O.mean(axis=0,keepdims=True)
        V=read(W[f"W_v.{l}"][sl],W[f"ln1_w.{l}"])
        A[l,h]=Q.T@K/8
        B[l,h]=O@V

sl=slice(192,256)
O=W["W_o.0"][:,sl].astype(float)
Q=read(W["W_q.0"][sl],W["ln1_w.0"])
K=read(W["W_k.0"][sl],W["ln1_w.0"])
V=read(W["W_v.0"][sl],W["ln1_w.0"])
Uo,_,_=np.linalg.svd(O,full_matrices=False)
_,_,vp=np.linalg.svd(np.vstack([Q,K]),full_matrices=False)
ctrl=np.eye(384)[:,[21,39,58,99,184]]
U,s,_=np.linalg.svd(np.hstack([Uo,vp[:32].T,ctrl]),full_matrices=False)
U=U[:,s>1e-6]
G=read(W["W_fc.0"],W["ln2_w.0"])
Gc=G-(G@U)@U.T
_,_,tc=np.linalg.svd(Gc,full_matrices=False)
_,_,tv=np.linalg.svd(V,full_matrices=False)
bases={"MLP-current-20":tc[:20].T,"MLP-current-27":tc[:27].T,
       "value-20":tv[:20].T,"value-27":tv[:27].T}

print("Restricted key-composition trace scores for source L0H3")
print("Each pair is trace/||product||, trace/sum(abs(eigenvalues)); not attention or MLP evaluations.")
for l in range(1,4):
    for h in range(6):
        entries=[]
        for name,T in bases.items():
            C=T.T@A[l,h]@B[0,3]@T
            ev=np.linalg.eigvals(C)
            entries.append((name,round(np.trace(C)/(np.linalg.norm(C)+1e-30),3),
                            round(np.trace(C)/(abs(ev).sum()+1e-30),3)))
        print(f"L{l}H{h}",entries)

print("\nL3H3 restricted composition eigenvalues")
for name,T in bases.items():
    C=T.T@A[3,3]@B[0,3]@T
    ev=np.linalg.eigvals(C)
    print(name,ev[np.argsort(-abs(ev))][:16])

print("\nL0H3 direct QK energy in the four candidate content subspaces")
for name,T in bases.items():
    C=T.T@A[0,3]@T
    print(name,"fraction",round(np.linalg.norm(C)**2/np.linalg.norm(A[0,3])**2,4),
          "trace/norm",round(np.trace(C)/(np.linalg.norm(C)+1e-30),3))

print("\nFinal projection spectra after subtracting simple paired-weight correlations")
P=W["W_proj.3"].astype(float)
F=W["W_fc.3"].astype(float)
for alpha in [0,.1,.2,.3,.4]:
    s=np.linalg.svd(P+alpha*F.T,compute_uv=False)
    print("alpha",alpha,"singular values 16..36",s[15:36],
          "tail squared norm after rank 28",round(float(np.sum(s[28:]**2)),3))

```

Output:

```
Restricted key-composition trace scores for source L0H3
Each pair is trace/||product||, trace/sum(abs(eigenvalues)); not attention or MLP evaluations.
L1H0 [('MLP-current-20', np.float64(0.548), np.float64(0.25)), ('MLP-current-27', np.float64(0.396), np.float64(0.205)), ('value-20', np.float64(0.329), np.float64(0.141)), ('value-27', np.float64(0.32), np.float64(0.136))]
L1H1 [('MLP-current-20', np.float64(0.366), np.float64(0.191)), ('MLP-current-27', np.float64(0.327), np.float64(0.182)), ('value-20', np.float64(0.632), np.float64(0.283)), ('value-27', np.float64(0.576), np.float64(0.258))]
L1H2 [('MLP-current-20', np.float64(0.237), np.float64(0.119)), ('MLP-current-27', np.float64(0.226), np.float64(0.125)), ('value-20', np.float64(0.482), np.float64(0.222)), ('value-27', np.float64(0.495), np.float64(0.242))]
L1H3 [('MLP-current-20', np.float64(0.022), np.float64(0.011)), ('MLP-current-27', np.float64(0.015), np.float64(0.008)), ('value-20', np.float64(0.164), np.float64(0.076)), ('value-27', np.float64(0.205), np.float64(0.094))]
L1H4 [('MLP-current-20', np.float64(-0.264), np.float64(-0.121)), ('MLP-current-27', np.float64(-0.159), np.float64(-0.076)), ('value-20', np.float64(0.157), np.float64(0.07)), ('value-27', np.float64(0.058), np.float64(0.027))]
L1H5 [('MLP-current-20', np.float64(0.443), np.float64(0.217)), ('MLP-current-27', np.float64(0.454), np.float64(0.235)), ('value-20', np.float64(0.518), np.float64(0.255)), ('value-27', np.float64(0.488), np.float64(0.238))]
L2H0 [('MLP-current-20', np.float64(0.076), np.float64(0.032)), ('MLP-current-27', np.float64(0.039), np.float64(0.017)), ('value-20', np.float64(0.282), np.float64(0.109)), ('value-27', np.float64(0.211), np.float64(0.077))]
L2H1 [('MLP-current-20', np.float64(-0.045), np.float64(-0.019)), ('MLP-current-27', np.float64(0.035), np.float64(0.015)), ('value-20', np.float64(0.107), np.float64(0.042)), ('value-27', np.float64(0.029), np.float64(0.011))]
L2H2 [('MLP-current-20', np.float64(0.174), np.float64(0.076)), ('MLP-current-27', np.float64(0.109), np.float64(0.047)), ('value-20', np.float64(0.349), np.float64(0.132)), ('value-27', np.float64(0.364), np.float64(0.137))]
L2H3 [('MLP-current-20', np.float64(-0.112), np.float64(-0.048)), ('MLP-current-27', np.float64(-0.082), np.float64(-0.037)), ('value-20', np.float64(0.434), np.float64(0.169)), ('value-27', np.float64(0.353), np.float64(0.135))]
L2H4 [('MLP-current-20', np.float64(0.128), np.float64(0.057)), ('MLP-current-27', np.float64(0.263), np.float64(0.12)), ('value-20', np.float64(0.708), np.float64(0.273)), ('value-27', np.float64(0.654), np.float64(0.228))]
L2H5 [('MLP-current-20', np.float64(0.135), np.float64(0.061)), ('MLP-current-27', np.float64(0.04), np.float64(0.018)), ('value-20', np.float64(-0.03), np.float64(-0.012)), ('value-27', np.float64(0.008), np.float64(0.003))]
L3H0 [('MLP-current-20', np.float64(-0.464), np.float64(-0.198)), ('MLP-current-27', np.float64(-0.436), np.float64(-0.19)), ('value-20', np.float64(-0.629), np.float64(-0.243)), ('value-27', np.float64(-0.568), np.float64(-0.217))]
L3H1 [('MLP-current-20', np.float64(-0.191), np.float64(-0.081)), ('MLP-current-27', np.float64(-0.086), np.float64(-0.036)), ('value-20', np.float64(-0.214), np.float64(-0.08)), ('value-27', np.float64(-0.113), np.float64(-0.04))]
L3H2 [('MLP-current-20', np.float64(0.022), np.float64(0.009)), ('MLP-current-27', np.float64(0.027), np.float64(0.011)), ('value-20', np.float64(0.202), np.float64(0.071)), ('value-27', np.float64(0.111), np.float64(0.038))]
L3H3 [('MLP-current-20', np.float64(2.563), np.float64(0.913)), ('MLP-current-27', np.float64(2.405), np.float64(0.876)), ('value-20', np.float64(2.858), np.float64(0.912)), ('value-27', np.float64(2.638), np.float64(0.844))]
L3H4 [('MLP-current-20', np.float64(0.688), np.float64(0.309)), ('MLP-current-27', np.float64(0.605), np.float64(0.259)), ('value-20', np.float64(0.682), np.float64(0.269)), ('value-27', np.float64(0.519), np.float64(0.189))]
L3H5 [('MLP-current-20', np.float64(-0.08), np.float64(-0.034)), ('MLP-current-27', np.float64(-0.14), np.float64(-0.058)), ('value-20', np.float64(-0.126), np.float64(-0.046)), ('value-27', np.float64(-0.178), np.float64(-0.063))]

L3H3 restricted composition eigenvalues
MLP-current-20 [ 0.037+0.005j  0.037-0.005j  0.026+0.007j  0.026-0.007j  0.016+0.01j   0.016-0.01j   0.018+0.003j  0.018-0.003j  0.014+0.j     0.012+0.j     0.01 -0.007j  0.01 +0.007j
  0.008+0.003j  0.008-0.003j -0.004+0.j     0.003+0.j   ]
MLP-current-27 [ 0.041+0.j     0.034+0.006j  0.034-0.006j  0.023+0.006j  0.023-0.006j  0.022+0.j     0.015+0.011j  0.015-0.011j  0.017+0.j     0.012+0.j     0.008+0.008j  0.008-0.008j
  0.011+0.j     0.008-0.002j  0.008+0.002j -0.005+0.j   ]
value-20 [0.053+0.j    0.039+0.018j 0.039-0.018j 0.038+0.01j  0.038-0.01j  0.038+0.j    0.025+0.014j 0.025-0.014j 0.027+0.j    0.023-0.002j 0.023+0.002j 0.011+0.007j 0.011-0.007j
 0.008+0.j    0.004+0.004j 0.004-0.004j]
value-27 [0.054+0.j    0.039+0.019j 0.039-0.019j 0.037+0.011j 0.037-0.011j 0.036+0.j    0.033+0.j    0.026+0.013j 0.026-0.013j 0.025+0.j    0.017+0.002j 0.017-0.002j 0.008-0.007j
 0.008+0.007j 0.006+0.004j 0.006-0.004j]

L0H3 direct QK energy in the four candidate content subspaces
MLP-current-20 fraction 0.0001 trace/norm -0.97
MLP-current-27 fraction 0.0001 trace/norm -1.013
value-20 fraction 0.004 trace/norm 0.572
value-27 fraction 0.0474 trace/norm 0.923

Final projection spectra after subtracting simple paired-weight correlations
alpha 0 singular values 16..36 [13.438 12.667 12.455 12.317 11.531 11.047  9.084  8.353  8.24   6.37   5.836  4.626  4.583  3.777  3.61   3.252  3.183  3.086  2.941  2.865  2.801] tail squared norm after rank 28 452.724
alpha 0.1 singular values 16..36 [13.423 12.683 12.475 12.334 11.549 11.066  9.157  8.375  8.242  6.354  5.869  4.779  4.595  3.711  3.647  3.278  3.157  2.991  2.953  2.768  2.757] tail squared norm after rank 28 393.569
alpha 0.2 singular values 16
```

## Call 9

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4,suppress=True)

def read(M,g):
    X=M.astype(float)*g
    return X-X.mean(axis=1,keepdims=True)

A={}; B={}
for l in range(4):
    for h in range(6):
        sl=slice(64*h,64*(h+1))
        q=read(W[f"W_q.{l}"][sl],W[f"ln1_w.{l}"])
        k=read(W[f"W_k.{l}"][sl],W[f"ln1_w.{l}"])
        o=W[f"W_o.{l}"][:,sl].astype(float)
        o-=o.mean(axis=0,keepdims=True)
        v=read(W[f"W_v.{l}"][sl],W[f"ln1_w.{l}"])
        A[l,h]=q.T@k/8
        B[l,h]=o@v

# A content-read basis derived solely from the first MLP's weights,
# orthogonalized against the main positional/read and head-output spaces.
sl=slice(192,256)
O=W["W_o.0"][:,sl].astype(float)
q=read(W["W_q.0"][sl],W["ln1_w.0"])
k=read(W["W_k.0"][sl],W["ln1_w.0"])
uo,_,_=np.linalg.svd(O,full_matrices=False)
_,_,vp=np.linalg.svd(np.vstack([q,k]),full_matrices=False)
ctrl=np.eye(384)[:,[21,39,58,99,184]]
U,ss,_=np.linalg.svd(np.hstack([uo,vp[:32].T,ctrl]),full_matrices=False)
U=U[:,ss>1e-6]
G=read(W["W_fc.0"],W["ln2_w.0"])
Gc=G-(G@U)@U.T
_,_,vt=np.linalg.svd(Gc,full_matrices=False)
T=vt[:20].T

def score(C):
    return float(np.trace(C)/(np.linalg.norm(C)+1e-30))

print("Induction orientation check, restricted to the weight-derived content basis")
for l,h in [(1,0),(1,1),(1,5),(2,4),(3,3),(3,4)]:
    key=T.T@A[l,h]@B[0,3]@T
    query=T.T@B[0,3].T@A[l,h]@T
    direct=T.T@A[l,h]@T
    print(f"L{l}H{h}","key",round(score(key),4),
          "query",round(score(query),4),"direct",round(score(direct),4))

# Null comparisons randomize only the pairing of learned OV head dimensions.
# This operates on weight matrices, not on network inputs.
rng=np.random.default_rng(2026)
o=W["W_o.0"][:,sl].astype(float)
o-=o.mean(axis=0,keepdims=True)
v=read(W["W_v.0"][sl],W["ln1_w.0"])
L=T.T@A[3,3]@o
R=v@T
real=score(L@R)
null=[]
for _ in range(1000):
    perm=rng.permutation(64)
    signs=rng.choice([-1.,1.],64)
    null.append(score((L[:,perm]*signs)@R))
null=np.array(null)
print("\nL3H3/L0H3 key-composition pairing test:")
print("observed",real,"null mean/std",null.mean(),null.std(),
      "null quantiles",np.quantile(null,[.001,.01,.5,.99,.999]),
      "null >= observed",np.sum(null>=real))

print("\nCompact final head diagnostics")
print("head: adjusted QK norm; rank-one energy fraction; adjusted OV norm; key-composition score with L0H3")
for l in range(4):
    for h in range(6):
        a=A[l,h]; s=np.linalg.svd(a,compute_uv=False)
        kc=score(T.T@a@B[0,3]@T) if l else np.nan
        print(f"L{l}H{h}: {np.linalg.norm(a):.3f}; {s[0]**2/np.sum(s**2):.3f}; "
              f"{np.linalg.norm(B[l,h]):.3f}; {kc:.3f}")

```

Output:

```
Induction orientation check, restricted to the weight-derived content basis
L1H0 key 0.5483 query -0.2766 direct -2.594
L1H1 key 0.3657 query 0.7412 direct -2.2268
L1H5 key 0.4426 query 0.5168 direct -2.4333
L2H4 key 0.1276 query -0.1981 direct -0.6714
L3H3 key 2.5634 query -0.9712 direct 1.5839
L3H4 key 0.6884 query -0.2219 direct -0.1726

L3H3/L0H3 key-composition pairing test:
observed 2.563396480228847 null mean/std -0.017657194738876187 0.23775823892951933 null quantiles [-0.7543 -0.5865 -0.023   0.5103  0.6962] null >= observed 0

Compact final head diagnostics
head: adjusted QK norm; rank-one energy fraction; adjusted OV norm; key-composition score with L0H3
L0H0: 0.293; 0.715; 2.126; nan
L0H1: 0.159; 0.892; 3.979; nan
L0H2: 0.107; 0.909; 3.033; nan
L0H3: 0.698; 0.122; 3.533; nan
L0H4: 0.088; 0.809; 1.683; nan
L0H5: 0.034; 0.902; 18.804; nan
L1H0: 1.978; 0.359; 11.479; 0.548
L1H1: 1.843; 0.392; 11.263; 0.366
L1H2: 1.781; 0.436; 11.806; 0.237
L1H3: 1.787; 0.441; 12.398; 0.022
L1H4: 1.629; 0.422; 13.398; -0.264
L1H5: 1.863; 0.378; 10.917; 0.443
L2H0: 1.631; 0.178; 16.379; 0.076
L2H1: 2.154; 0.072; 17.615; -0.045
L2H2: 1.720; 0.091; 18.662; 0.174
L2H3: 1.623; 0.096; 18.995; -0.112
L2H4: 2.215; 0.135; 14.875; 0.128
L2H5: 2.264; 0.074; 16.857; 0.135
L3H0: 2.029; 0.046; 20.709; -0.464
L3H1: 1.216; 0.081; 26.503; -0.191
L3H2: 1.437; 0.071; 23.503; 0.022
L3H3: 1.815; 0.031; 37.523; 2.563
L3H4: 0.622; 0.144; 49.232; 0.688
L3H5: 1.665; 0.074; 22.562; -0.080

```
