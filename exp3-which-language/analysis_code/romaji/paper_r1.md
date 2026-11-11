## Call 1

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)

def summary(a):
    return "norm=%.3f mean=%.5f std=%.5f min=%.4f max=%.4f" % (
        np.linalg.norm(a), a.mean(), a.std(), a.min(), a.max())

for l in range(4):
    print("\nLAYER", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b", "W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        print(name, summary(W[f"{name}.{l}"]))
    print("head: Qnorm Knorm Vnorm Onorm | QK sv(top6), top1_energy, symmetry, trace/norm | OV sv(top4), trace/norm")
    for j in range(6):
        s = slice(j*64, (j+1)*64)
        q = W[f"W_q.{l}"][s].astype(float)
        k = W[f"W_k.{l}"][s].astype(float)
        v = W[f"W_v.{l}"][s].astype(float)
        o = W[f"W_o.{l}"][:, s].astype(float)
        g = W[f"ln1_w.{l}"]
        A = (q*g).T @ (k*g) / 8
        B = o @ (v*g)
        sa = np.linalg.svd(A, compute_uv=False)[:64]
        sb = np.linalg.svd(B, compute_uv=False)[:64]
        print("L%dH%d" % (l,j),
              np.round([np.linalg.norm(z) for z in [q,k,v,o]], 3),
              "|", np.round(sa[:6],3), "E1=%.3f sym=%.3f tr=%.3f" %
              (sa[0]**2/(sa@sa), np.sum(A*A.T)/np.sum(A*A), np.trace(A)/np.linalg.norm(A)),
              "|", np.round(sb[:4],3), "tr=%.3f" % (np.trace(B)/np.linalg.norm(B)))

```

Output:

```

LAYER 0
ln1_w norm=6.884 mean=0.29650 std=0.18837 min=0.0352 max=0.9785
ln1_b norm=0.610 mean=0.00039 std=0.03113 min=-0.0954 max=0.0906
ln2_w norm=18.323 mean=0.87176 std=0.33816 min=0.0420 max=1.7750
ln2_b norm=1.380 mean=0.01340 std=0.06913 min=-0.6236 max=0.4636
W_q norm=17.669 mean=0.00005 std=0.04601 min=-0.4639 max=0.4214
W_k norm=18.516 mean=0.00001 std=0.04822 min=-0.5163 max=0.5037
W_v norm=14.937 mean=0.00002 std=0.03890 min=-0.3922 max=0.3895
W_o norm=16.550 mean=-0.00012 std=0.04310 min=-1.1239 max=1.0823
W_fc norm=55.376 mean=0.00075 std=0.07210 min=-0.8384 max=1.1226
W_proj norm=50.007 mean=-0.00000 std=0.06511 min=-0.6883 max=1.4636
head: Qnorm Knorm Vnorm Onorm | QK sv(top6), top1_energy, symmetry, trace/norm | OV sv(top4), trace/norm
L0H0 [7.14  7.395 5.916 6.127] | [0.248 0.102 0.047 0.043 0.037 0.036] E1=0.714 sym=0.140 tr=-0.357 | [0.958 0.898 0.818 0.642] tr=-0.067
L0H1 [4.486 4.927 4.323 4.825] | [0.15  0.042 0.021 0.012 0.01  0.007] E1=0.892 sym=0.161 tr=-0.620 | [3.38  2.023 0.345 0.18 ] tr=0.231
L0H2 [4.065 4.362 4.017 4.36 ] | [0.102 0.018 0.016 0.012 0.008 0.007] E1=0.908 sym=0.289 tr=-0.814 | [2.643 1.363 0.4   0.176] tr=0.229
L0H3 [14.141 14.661  8.259  9.445] | [0.244 0.223 0.201 0.182 0.179 0.157] E1=0.122 sym=-0.067 tr=1.923 | [1.127 0.955 0.95  0.89 ] tr=-0.570
L0H4 [4.205 4.614 3.689 3.657] | [0.08  0.022 0.019 0.013 0.011 0.008] E1=0.808 sym=0.050 tr=-0.519 | [1.454 0.724 0.213 0.141] tr=0.102
L0H5 [2.632 2.936 8.453 9.565] | [0.032 0.004 0.003 0.003 0.002 0.002] E1=0.901 sym=0.024 tr=-0.254 | [16.282  8.786  2.835  1.117] tr=0.111

LAYER 1
ln1_w norm=10.065 mean=0.50583 std=0.08918 min=0.1970 max=1.0846
ln1_b norm=3.037 mean=0.01103 std=0.15458 min=-1.5676 max=1.3651
ln2_w norm=22.433 mean=1.13147 std=0.17415 min=0.0019 max=1.5196
ln2_b norm=2.815 mean=0.00647 std=0.14352 min=-0.3892 max=0.4170
W_q norm=38.929 mean=-0.00027 std=0.10138 min=-0.4410 max=0.4642
W_k norm=34.893 mean=0.00005 std=0.09087 min=-1.0865 max=1.2757
W_v norm=28.744 mean=0.00003 std=0.07486 min=-0.3539 max=0.3314
W_o norm=28.479 mean=0.00002 std=0.07416 min=-0.4175 max=0.4550
W_fc norm=80.379 mean=0.00139 std=0.10465 min=-0.6542 max=0.6450
W_proj norm=77.384 mean=-0.00008 std=0.10076 min=-0.5072 max=2.6601
head: Qnorm Knorm Vnorm Onorm | QK sv(top6), top1_energy, symmetry, trace/norm | OV sv(top4), trace/norm
L1H0 [16.628 14.95  11.763 11.711] | [1.185 0.729 0.557 0.534 0.472 0.432] E1=0.358 sym=0.042 tr=-0.883 | [3.189 2.991 2.858 2.707] tr=-1.346
L1H1 [16.019 14.34  11.237 11.401] | [1.155 0.715 0.583 0.465 0.398 0.368] E1=0.391 sym=0.018 tr=-0.845 | [3.148 3.021 2.872 2.793] tr=-1.351
L1H2 [15.576 14.136 11.739 11.38 ] | [1.176 0.62  0.485 0.436 0.385 0.355] E1=0.435 sym=0.037 tr=-1.059 | [3.55  3.305 3.182 2.996] tr=-1.467
L1H3 [15.745 13.949 11.801 11.885] | [1.188 0.678 0.532 0.388 0.366 0.329] E1=0.440 sym=0.006 tr=-0.844 | [3.408 3.269 3.119 3.022] tr=-1.481
L1H4 [15.205 13.642 12.731 12.113] | [1.059 0.505 0.477 0.351 0.334 0.31 ] E1=0.422 sym=0.047 tr=-1.234 | [4.132 3.769 3.609 3.358] tr=-1.429
L1H5 [16.146 14.418 11.065 11.244] | [1.147 0.719 0.556 0.499 0.433 0.379] E1=0.377 sym=0.017 tr=-0.874 | [3.085 2.949 2.892 2.805] tr=-1.369

LAYER 2
ln1_w norm=13.410 mean=0.68087 std=0.06874 min=0.3175 max=1.0927
ln1_b norm=3.077 mean=0.01305 std=0.15645 min=-1.6589 max=1.3318
ln2_w norm=25.777 mean=1.31041 std=0.11485 min=0.0120 max=1.5078
ln2_b norm=3.413 mean=0.01229 std=0.17372 min=-0.5170 max=0.4489
W_q norm=38.902 mean=0.00008 std=0.10131 min=-0.4865 max=0.4916
W_k norm=36.750 mean=0.00001 std=0.09570 min=-1.2235 max=1.1936
W_v norm=33.183 mean=-0.00003 std=0.08641 min=-0.4009 max=0.3919
W_o norm=32.435 mean=-0.00000 std=0.08446 min=-0.5312 max=0.5185
W_fc norm=79.828 mean=0.00039 std=0.10394 min=-0.5244 max=0.5953
W_proj norm=78.048 mean=-0.00007 std=0.10163 min=-0.5107 max=0.8014
head: Qnorm Knorm Vnorm Onorm | QK sv(top6), top1_energy, symmetry, trace/norm | OV sv(top4), trace/norm
L2H0 [14.489 13.572 13.0
```

## Call 2

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)

for l in range(4):
    print("\nLAYER",l)
    for name in ["W_q","W_k","W_v","W_o","W_fc","W_proj"]:
        a=W[f"{name}.{l}"].astype(float)
        sv=np.linalg.svd(a,compute_uv=False)
        print(name,"sv12",np.round(sv[:12],3),
              "energy ranks 1,5,10,28,64",np.round([np.sum(sv[:r]**2)/np.sum(sv**2) for r in [1,5,10,28,64]],3))
    g=W[f"ln1_w.{l}"].astype(float)
    b=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s=slice(64*j,64*(j+1))
        q=W[f"W_q.{l}"][s].astype(float)
        k=W[f"W_k.{l}"][s].astype(float)
        v=W[f"W_v.{l}"][s].astype(float)
        o=W[f"W_o.{l}"][:,s].astype(float)
        e=np.linalg.eigvals((k*g)@(q*g).T/8)
        e=e[np.argsort(-abs(e))]
        A=(q*g).T@(k*g)/8
        keybias=(q@b)@(k*g)/8
        u,sv,vt=np.linalg.svd(A,full_matrices=False)
        print("L%dH%d"%(l,j),"QK eig12",np.round(e[:12],3),
              "keybias_norm=%.3f"%np.linalg.norm(keybias),
              "keybias_topright=%.3f"%(keybias@vt[0]/(np.linalg.norm(keybias)+1e-12)))
    print("LN1 largest bias coords", sorted(enumerate(b),key=lambda z:-abs(z[1]))[:8])

print("\nFIRST-LAYER HEAD QK AND OV COSINES")
As=[];Bs=[]
for j in range(6):
    s=slice(64*j,64*(j+1)); g=W["ln1_w.0"]
    As.append((W["W_q.0"][s]*g).T@(W["W_k.0"][s]*g)/8)
    Bs.append(W["W_o.0"][:,s]@(W["W_v.0"][s]*g))
for label,mat in [("QK",As),("OV",Bs)]:
    print(label)
    print(np.round([[np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)) for b in mat] for a in mat],3))

```

Output:

```

LAYER 0
W_q sv12 [4.368 3.131 3.099 3.042 2.994 2.951 2.926 2.881 2.859 2.857 2.804 2.766] energy ranks 1,5,10,28,64 [0.061 0.182 0.316 0.676 0.875]
W_k sv12 [4.163 3.952 3.467 3.427 3.364 3.335 3.279 3.256 3.214 3.163 3.115 3.029] energy ranks 1,5,10,28,64 [0.051 0.198 0.352 0.713 0.88 ]
W_v sv12 [5.208 3.364 3.006 2.962 2.815 2.639 2.527 2.46  2.37  2.301 2.25  2.23 ] energy ranks 1,5,10,28,64 [0.122 0.288 0.424 0.715 0.856]
W_o sv12 [7.361 6.099 3.167 2.452 2.321 2.202 2.171 2.16  2.105 2.072 2.021 1.995] energy ranks 1,5,10,28,64 [0.198 0.412 0.496 0.718 0.91 ]
W_fc sv12 [18.118 12.76  11.337  9.323  9.267  9.021  8.898  8.817  8.609  8.4    8.189  8.06 ] energy ranks 1,5,10,28,64 [0.107 0.258 0.383 0.676 0.892]
W_proj sv12 [10.899  7.702  7.173  6.638  6.541  6.288  6.201  6.133  5.941  5.917  5.758  5.641] energy ranks 1,5,10,28,64 [0.048 0.127 0.201 0.394 0.633]
L0H0 QK eig12 [ 0.079+0.j     0.049+0.j    -0.024-0.006j -0.024+0.006j -0.023+0.j     0.018+0.013j  0.018-0.013j -0.02 +0.j    -0.014+0.012j -0.014-0.012j
 -0.014+0.003j -0.014-0.003j] keybias_norm=0.043 keybias_topright=-0.694
L0H1 QK eig12 [-0.062+0.j    -0.01 +0.j    -0.009+0.j     0.004+0.j    -0.004+0.j    -0.003+0.002j -0.003-0.002j -0.002+0.j    -0.002-0.j    -0.002+0.001j
 -0.002-0.001j -0.001+0.j   ] keybias_norm=0.013 keybias_topright=-0.607
L0H2 QK eig12 [-0.057+0.j    -0.007+0.003j -0.007-0.003j -0.003+0.001j -0.003-0.001j -0.003+0.j    -0.001-0.j    -0.001+0.j    -0.001-0.001j -0.001+0.001j
  0.001+0.001j  0.001-0.001j] keybias_norm=0.007 keybias_topright=-0.876
L0H3 QK eig12 [0.137+0.133j 0.137-0.133j 0.11 -0.089j 0.11 +0.089j 0.095-0.098j 0.095+0.098j 0.084-0.104j 0.084+0.104j 0.064-0.114j 0.064+0.114j 0.096+0.068j
 0.096-0.068j] keybias_norm=0.043 keybias_topright=0.152
L0H4 QK eig12 [-0.015+0.j    -0.01 +0.j    -0.008+0.j    -0.003+0.j    -0.003-0.j    -0.002+0.001j -0.002-0.001j -0.   +0.002j -0.   -0.002j -0.001+0.001j
 -0.001-0.001j -0.002+0.j   ] keybias_norm=0.017 keybias_topright=-0.942
L0H5 QK eig12 [-0.005+0.j     0.001+0.j    -0.001+0.j    -0.   +0.001j -0.   -0.001j -0.001+0.j    -0.001+0.j    -0.001-0.j     0.   +0.001j  0.   -0.001j
 -0.   +0.001j -0.   -0.001j] keybias_norm=0.005 keybias_topright=0.971
LN1 largest bias coords [(312, np.float64(-0.09542398899793625)), (146, np.float64(0.09060516208410263)), (12, np.float64(0.08561607450246811)), (104, np.float64(-0.08411034941673279)), (289, np.float64(0.08255884051322937)), (156, np.float64(-0.0779787078499794)), (303, np.float64(-0.07763753086328506)), (328, np.float64(0.07621065527200699))]

LAYER 1
W_q sv12 [10.406  8.239  7.475  7.033  6.622  6.395  6.275  6.13   6.065  5.901  5.819  5.716] energy ranks 1,5,10,28,64 [0.071 0.215 0.34  0.646 0.893]
W_k sv12 [14.477 11.787  6.37   6.054  5.523  5.407  5.268  5.167  5.091  4.924  4.776  4.529] energy ranks 1,5,10,28,64 [0.172 0.375 0.485 0.716 0.905]
W_v sv12 [4.728 4.547 4.459 4.395 4.361 4.332 4.291 4.276 4.244 4.133 4.079 3.991] energy ranks 1,5,10,28,64 [0.027 0.123 0.232 0.514 0.801]
W_o sv12 [4.045 3.267 3.188 3.179 3.163 3.101 3.071 3.046 3.046 3.036 3.009 2.985] energy ranks 1,5,10,28,64 [0.02  0.071 0.128 0.312 0.596]
W_fc sv12 [19.829 11.575 11.522 11.262 11.059 10.875 10.632 10.497 10.363 10.3   10.174 10.05 ] energy ranks 1,5,10,28,64 [0.061 0.141 0.227 0.457 0.706]
W_proj sv12 [9.038 8.457 8.103 7.333 7.318 7.088 6.772 6.666 6.566 6.429 6.372 6.278] energy ranks 1,5,10,28,64 [0.014 0.054 0.092 0.199 0.367]
L1H0 QK eig12 [-0.264+0.j     0.115+0.23j   0.115-0.23j  -0.221+0.j    -0.199+0.05j  -0.199-0.05j  -0.1  +0.121j -0.1  -0.121j -0.143+0.j    -0.121+0.04j
 -0.121-0.04j  -0.039+0.1j  ] keybias_norm=3.975 keybias_topright=0.986
L1H1 QK eig12 [ 0.093+0.245j  0.093-0.245j -0.202+0.j    -0.164+0.018j -0.164-0.018j -0.145+0.j    -0.119-0.07j  -0.119+0.07j  -0.056-0.108j -0.056+0.108j
 -0.1  +0.033j -0.1  -0.033j] keybias_norm=3.689 keybias_topright=-0.975
L1H2 QK eig12 [-0.249+0.j    -0.201+0.j     0.069-0.182j  0.069+0.1
```

## Call 3

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)

print("Last MLP output singular values:")
for l in [0,3]:
    sv=np.linalg.svd(W[f"W_proj.{l}"].astype(float),compute_uv=False)
    print("L",l,np.round(sv[:48],4))

print("\nL0 head singular-vector coordinate structure")
As=[];Bs=[]
g=W["ln1_w.0"].astype(float)
for j in range(6):
    s=slice(64*j,64*(j+1))
    q=W["W_q.0"][s].astype(float); k=W["W_k.0"][s].astype(float)
    A=(q*g).T@(k*g)/8
    B=W["W_o.0"][:,s].astype(float)@(W["W_v.0"][s].astype(float)*g)
    As.append(A); Bs.append(B)
    ua,sa,va=np.linalg.svd(A,full_matrices=False)
    ub,sb,vb=np.linalg.svd(B,full_matrices=False)
    print("L0H%d"%j)
    for label,z in [("QK left1",ua[:,0]),("QK right1",va[0]),("OV left1",ub[:,0]),("OV right1",vb[0]),("OV left2",ub[:,1])]:
        ix=np.argsort(-abs(z))[:8]
        print(label,list(zip(ix.tolist(),np.round(z[ix],3).tolist())))
    print("unscaled QK sv",np.round(np.linalg.svd(q.T@k/8,compute_uv=False)[:12],3))

print("\nKey-bias similarities within layers (weight-only affine coefficients)")
for l in range(1,4):
    g=W[f"ln1_w.{l}"].astype(float); b=W[f"ln1_b.{l}"].astype(float)
    R=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        r=(W[f"W_q.{l}"][s]@b)@(W[f"W_k.{l}"][s]*g)/8
        R.append(r/np.linalg.norm(r))
    print("L",l,np.round(np.array(R)@np.array(R).T,2))
    for j,r in enumerate(R):
        ix=np.argsort(-abs(r))[:6]
        print("H",j,"top coords",list(zip(ix.tolist(),np.round(r[ix],3).tolist())),
              "OV0 output align",np.round([r@np.linalg.svd(B,full_matrices=False)[0][:,0] for B in Bs],3))

```

Output:

```
Last MLP output singular values:
L 0 [10.899   7.7021  7.1728  6.6377  6.5415  6.2882  6.2009  6.1328  5.9409  5.9174  5.758   5.6413  5.5516  5.4755  5.3925  5.3625  5.2591  5.2062
  5.1709  5.153   5.0573  5.0182  4.9601  4.8805  4.8211  4.7883  4.7697  4.7303  4.6429  4.5787  4.5457  4.5291  4.5053  4.4926  4.4404  4.4154
  4.369   4.3154  4.2935  4.2606  4.2432  4.1849  4.1339  4.1036  4.0637  4.0493  4.0262  3.9879]
L 3 [28.4613 22.2137 21.3397 20.708  19.9429 19.6618 18.9864 18.744  18.2945 17.4567 17.1321 16.5609 15.3286 14.8739 14.3285 13.4378 12.667  12.4546
 12.317  11.5309 11.0466  9.0836  8.3531  8.2404  6.3701  5.8357  4.6262  4.5827  3.7775  3.6102  3.2518  3.1826  3.0856  2.9413  2.8655  2.8013
  2.6391  2.5637  2.538   2.4655  2.4188  2.4037  2.346   2.3102  2.2461  2.227   2.1548  2.1472]

L0 head singular-vector coordinate structure
L0H0
QK left1 [(317, -0.307), (184, 0.284), (39, 0.269), (239, -0.251), (180, 0.229), (349, 0.199), (99, 0.199), (228, 0.19)]
QK right1 [(211, -0.322), (256, 0.26), (65, -0.23), (173, 0.216), (223, 0.213), (132, -0.212), (4, 0.212), (39, 0.209)]
OV left1 [(336, 0.194), (96, 0.158), (291, -0.15), (327, 0.149), (203, -0.14), (202, -0.128), (330, 0.12), (68, -0.117)]
OV right1 [(373, 0.174), (225, -0.164), (134, -0.158), (158, -0.155), (7, 0.152), (51, 0.148), (96, -0.142), (173, -0.132)]
OV left2 [(39, 0.262), (6, 0.192), (282, -0.186), (156, -0.148), (163, -0.136), (152, -0.133), (342, -0.128), (219, 0.127)]
unscaled QK sv [0.853 0.613 0.368 0.287 0.279 0.244 0.232 0.21  0.202 0.188 0.182 0.173]
L0H1
QK left1 [(256, 0.372), (184, -0.3), (354, -0.232), (173, 0.225), (4, 0.223), (56, 0.2), (69, -0.183), (100, 0.173)]
QK right1 [(100, -0.387), (349, 0.272), (317, -0.252), (235, 0.208), (271, -0.194), (258, 0.181), (69, 0.167), (368, 0.152)]
OV left1 [(39, 0.692), (58, -0.51), (99, 0.358), (21, -0.212), (184, 0.201), (153, -0.054), (336, -0.038), (63, 0.037)]
OV right1 [(184, 0.347), (180, 0.315), (99, 0.271), (39, 0.267), (56, -0.254), (317, -0.239), (69, 0.235), (256, -0.214)]
OV left2 [(21, 0.961), (39, 0.158), (58, -0.129), (184, 0.072), (306, -0.032), (173, -0.029), (225, -0.028), (232, -0.026)]
unscaled QK sv [0.838 0.311 0.219 0.125 0.091 0.086 0.066 0.039 0.038 0.034 0.033 0.03 ]
L0H2
QK left1 [(256, 0.394), (173, 0.255), (211, -0.251), (184, -0.238), (4, 0.222), (228, 0.205), (65, -0.188), (124, 0.171)]
QK right1 [(100, -0.346), (349, 0.254), (256, -0.201), (317, -0.173), (56, -0.167), (271, -0.154), (175, 0.153), (240, 0.149)]
OV left1 [(39, -0.659), (58, 0.494), (21, 0.373), (99, -0.355), (184, -0.162), (153, 0.045), (383, -0.035), (165, -0.029)]
OV right1 [(184, -0.365), (180, -0.304), (99, -0.298), (317, 0.248), (56, 0.234), (39, -0.232), (69, -0.208), (301, -0.195)]
OV left2 [(21, 0.91), (39, 0.221), (99, 0.207), (58, -0.201), (184, 0.053), (252, -0.03), (351, -0.03), (153, -0.029)]
unscaled QK sv [0.543 0.183 0.157 0.133 0.072 0.066 0.046 0.046 0.036 0.033 0.032 0.029]
L0H3
QK left1 [(354, 0.309), (162, -0.3), (65, -0.252), (223, 0.24), (256, 0.193), (180, 0.192), (300, -0.192), (211, -0.182)]
QK right1 [(162, -0.563), (184, -0.359), (354, 0.251), (99, -0.25), (65, -0.225), (256, 0.197), (132, -0.149), (223, 0.141)]
OV left1 [(92, -0.178), (86, 0.169), (188, -0.163), (18, 0.16), (131, 0.158), (303, 0.156), (215, 0.148), (173, 0.139)]
OV right1 [(223, -0.227), (26, -0.198), (200, -0.172), (48, -0.168), (359, 0.144), (124, 0.141), (178, -0.137), (354, 0.133)]
OV left2 [(65, -0.191), (125, -0.18), (320, 0.174), (202, 0.168), (86, 0.158), (287, 0.152), (20, -0.142), (350, -0.136)]
unscaled QK sv [1.114 1.092 1.07  1.064 1.03  0.992 0.968 0.954 0.946 0.9   0.888 0.868]
L0H4
QK left1 [(256, 0.43), (184, -0.307), (173, 0.225), (228, 0.22), (56, 0.219), (69, -0.193), (21, 0.18), (180, -0.16)]
QK right1 [(100, -0.282), (260, 0.188), (258, 0.181), (160, -0.178), (175, 0.168), (271, -0.154), (240, 0.153), (317, -0.151)]
OV left1 [(39, -0.696), (58, 0.511), (99, -0.32), (184, -0.227
```

## Call 4

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)

As={};Bs={}
for l in range(4):
    g=W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        s=slice(64*j,64*(j+1))
        As[l,j]=(W[f"W_q.{l}"][s]*g).T@(W[f"W_k.{l}"][s]*g)/8
        Bs[l,j]=W[f"W_o.{l}"][:,s].astype(float)@(W[f"W_v.{l}"][s]*g)
print("Trace(A_lh B_0j)/||A_lh B_0j||, for j=0..5 (positive matching composition)")
for l in range(1,4):
    for h in range(6):
        vals=[]
        for j in range(6):
            C=As[l,h]@Bs[0,j]
            vals.append(np.trace(C)/np.linalg.norm(C))
        print("L%dH%d"%(l,h),np.round(vals,3))

print("\nStrongest matching compositions over ALL preceding heads")
for l in range(1,4):
    for h in range(6):
        vals=[]
        for p in range(l):
            for j in range(6):
                C=As[l,h]@Bs[p,j]
                vals.append((np.trace(C)/np.linalg.norm(C),p,j))
        print("L%dH%d"%(l,h),[(round(x,3),p,j) for x,p,j in sorted(vals,reverse=True)[:4]])

print("\nL0H3 Q,K,V singular spectra")
for name in ["W_q","W_k","W_v"]:
    a=W[f"{name}.0"][192:256].astype(float)
    print(name,np.round(np.linalg.svd(a,compute_uv=False),3))
print("\nMLP0 input spectrum")
a=W["W_fc.0"].astype(float)*W["ln2_w.0"]
print(np.round(np.linalg.svd(a,compute_uv=False)[:100],3))

print("\nPositional-like QK L0H3 versus value subspaces: squared overlap")
_,_,qv=np.linalg.svd(W["W_q.0"][192:256],full_matrices=False)
_,_,kv=np.linalg.svd(W["W_k.0"][192:256],full_matrices=False)
_,_,vv=np.linalg.svd(W["W_v.0"][192:256],full_matrices=False)
for r in [16,24,28,32,40]:
    print(r, "Q/K",np.linalg.norm(qv[:r]@kv[:r].T)**2/r,
          "Q/V",np.linalg.norm(qv[:r]@vv[:r].T)**2/r,
          "K/V",np.linalg.norm(kv[:r]@vv[:r].T)**2/r)

```

Output:

```
Trace(A_lh B_0j)/||A_lh B_0j||, for j=0..5 (positive matching composition)
L1H0 [0.051 0.094 0.134 0.102 0.096 0.1  ]
L1H1 [0.018 0.017 0.065 0.216 0.013 0.041]
L1H2 [-0.06   0.078  0.106  0.167  0.1    0.09 ]
L1H3 [0.042 0.073 0.113 0.076 0.075 0.082]
L1H4 [0.02  0.046 0.08  0.015 0.078 0.074]
L1H5 [-0.043  0.052  0.102  0.198  0.061  0.062]
L2H0 [-0.009  0.133  0.196  0.079  0.089  0.189]
L2H1 [0.035 0.116 0.179 0.009 0.08  0.197]
L2H2 [0.054 0.123 0.182 0.099 0.073 0.184]
L2H3 [0.064 0.096 0.129 0.082 0.084 0.11 ]
L2H4 [-0.031  0.128  0.186  0.159  0.109  0.18 ]
L2H5 [0.018 0.112 0.187 0.01  0.081 0.21 ]
L3H0 [-0.049  0.069  0.107 -0.177  0.047  0.085]
L3H1 [ 0.034  0.069  0.093 -0.033  0.029  0.075]
L3H2 [-0.006  0.046  0.081  0.033  0.002  0.016]
L3H3 [-0.155  0.069  0.079  0.922  0.079 -0.055]
L3H4 [0.069 0.172 0.173 0.124 0.19  0.127]
L3H5 [-0.008  0.016  0.05  -0.062 -0.001  0.047]

Strongest matching compositions over ALL preceding heads
L1H0 [(np.float64(0.134), 0, 2), (np.float64(0.102), 0, 3), (np.float64(0.1), 0, 5), (np.float64(0.096), 0, 4)]
L1H1 [(np.float64(0.216), 0, 3), (np.float64(0.065), 0, 2), (np.float64(0.041), 0, 5), (np.float64(0.018), 0, 0)]
L1H2 [(np.float64(0.167), 0, 3), (np.float64(0.106), 0, 2), (np.float64(0.1), 0, 4), (np.float64(0.09), 0, 5)]
L1H3 [(np.float64(0.113), 0, 2), (np.float64(0.082), 0, 5), (np.float64(0.076), 0, 3), (np.float64(0.075), 0, 4)]
L1H4 [(np.float64(0.08), 0, 2), (np.float64(0.078), 0, 4), (np.float64(0.074), 0, 5), (np.float64(0.046), 0, 1)]
L1H5 [(np.float64(0.198), 0, 3), (np.float64(0.102), 0, 2), (np.float64(0.062), 0, 5), (np.float64(0.061), 0, 4)]
L2H0 [(np.float64(0.208), 1, 3), (np.float64(0.196), 0, 2), (np.float64(0.189), 0, 5), (np.float64(0.164), 1, 1)]
L2H1 [(np.float64(0.197), 0, 5), (np.float64(0.179), 0, 2), (np.float64(0.116), 0, 1), (np.float64(0.08), 0, 4)]
L2H2 [(np.float64(0.184), 0, 5), (np.float64(0.182), 0, 2), (np.float64(0.123), 0, 1), (np.float64(0.099), 0, 3)]
L2H3 [(np.float64(0.129), 0, 2), (np.float64(0.11), 0, 5), (np.float64(0.096), 0, 1), (np.float64(0.084), 0, 4)]
L2H4 [(np.float64(0.186), 0, 2), (np.float64(0.18), 0, 5), (np.float64(0.159), 0, 3), (np.float64(0.128), 0, 1)]
L2H5 [(np.float64(0.21), 0, 5), (np.float64(0.187), 0, 2), (np.float64(0.112), 0, 1), (np.float64(0.081), 0, 4)]
L3H0 [(np.float64(0.107), 0, 2), (np.float64(0.085), 0, 5), (np.float64(0.069), 0, 1), (np.float64(0.047), 0, 4)]
L3H1 [(np.float64(0.093), 0, 2), (np.float64(0.075), 0, 5), (np.float64(0.072), 1, 2), (np.float64(0.069), 0, 1)]
L3H2 [(np.float64(0.081), 0, 2), (np.float64(0.046), 0, 1), (np.float64(0.033), 0, 3), (np.float64(0.016), 0, 5)]
L3H3 [(np.float64(0.922), 0, 3), (np.float64(0.806), 2, 5), (np.float64(0.562), 2, 1), (np.float64(0.514), 1, 0)]
L3H4 [(np.float64(0.276), 2, 2), (np.float64(0.272), 2, 4), (np.float64(0.202), 2, 5), (np.float64(0.19), 0, 4)]
L3H5 [(np.float64(0.05), 0, 2), (np.float64(0.047), 0, 5), (np.float64(0.016), 0, 1), (np.float64(-0.001), 0, 4)]

L0H3 Q,K,V singular spectra
W_q [2.962 2.894 2.873 2.845 2.806 2.792 2.766 2.756 2.703 2.634 2.588 2.561 2.544 2.512 2.447 2.438 2.426 2.336 2.33  2.259 2.143 2.098 1.968 1.946
 1.838 1.756 1.721 1.685 1.575 1.54  1.495 1.463 1.423 1.378 1.283 1.136 1.116 1.094 1.06  1.023 0.991 0.969 0.962 0.897 0.864 0.824 0.778 0.77
 0.731 0.725 0.698 0.688 0.663 0.637 0.614 0.589 0.573 0.556 0.532 0.519 0.497 0.474 0.443 0.352]
W_k [3.362 3.238 3.209 3.181 3.099 3.086 3.024 2.999 2.951 2.923 2.856 2.765 2.731 2.654 2.622 2.539 2.478 2.407 2.31  2.217 2.134 2.057 1.976 1.806
 1.76  1.737 1.61  1.6   1.536 1.42  1.296 1.229 1.198 1.172 1.117 1.058 1.016 0.988 0.942 0.878 0.85  0.812 0.785 0.751 0.72  0.711 0.704 0.685
 0.641 0.61  0.601 0.578 0.562 0.542 0.532 0.526 0.505 0.492 0.484 0.453 0.433 0.421 0.396 0.366]
W_v [2.102 1.948 1.903 1.866 1.843 1.819 1.769 1.755 1.674 1.651 1.629 1.621 1.589 1.553 1.52  1.465 1.417 1.303 1.16  1.101 0.945 0.93  0.848 0.818
 0.79  0.779 0.745 0.739 0.73  0.701 
```

## Call 5

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)

g0=W["ln1_w.0"].astype(float)
q=W["W_q.0"][192:256].astype(float)
k=W["W_k.0"][192:256].astype(float)
v=W["W_v.0"][192:256].astype(float)
o=W["W_o.0"][:,192:256].astype(float)
_,_,qp=np.linalg.svd(q*g0,full_matrices=False)
_,_,vp=np.linalg.svd(v*g0,full_matrices=False)
P=qp[:20].T
T=vp[:20].T
B=o@(v*g0)
U=np.linalg.svd(B,full_matrices=False)[0][:,:20]
print("QK energy captured by candidate position, token and previous-token subspaces")
print("head | P-query P-key | T-query T-key | U-query U-key")
for l in range(4):
    g=W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        s=slice(64*j,64*(j+1))
        A=(W[f"W_q.{l}"][s]*g).T@(W[f"W_k.{l}"][s]*g)/8
        den=np.linalg.norm(A)**2
        vals=[]
        for Z in [P,T,U]:
            vals += [np.linalg.norm(Z.T@A)**2/den,np.linalg.norm(A@Z)**2/den]
        print("L%dH%d"%(l,j),np.round(vals,3))

print("\nL0H3 raw QK nonzero eigenvalues, descending magnitude")
e=np.linalg.eigvals(k@q.T/8)
e=e[np.argsort(-abs(e))]
print(np.round(e[:40],4))
print("positive-imag phases",np.round(np.sort(np.angle(e[(e.imag>0.01)&(abs(e)>.2)])),4))

F=W["W_fc.0"].astype(float)*W["ln2_w.0"]
print("\nMLP0 current/previous weight coupling (no activations)")
for r in [16,20,24,28]:
    T=vp[:r].T
    X=F@T
    Y=F@B@T
    X-=X.mean(0)
    Y-=Y.mean(0)
    C=X.T@Y
    ss=np.linalg.svd(C,compute_uv=False)
    ev=np.linalg.eigvalsh((C+C.T)/2)
    print("r",r,"sv",np.round(ss[:10],2),
          "sym eig extremes",np.round(np.r_[ev[:5],ev[-5:]],2),
          "trace/norm",round(np.trace(C)/np.linalg.norm(C),3))

```

Output:

```
QK energy captured by candidate position, token and previous-token subspaces
head | P-query P-key | T-query T-key | U-query U-key
L0H0 [0.339 0.363 0.101 0.115 0.058 0.054]
L0H1 [0.378 0.144 0.033 0.092 0.055 0.052]
L0H2 [0.466 0.328 0.036 0.17  0.047 0.041]
L0H3 [0.834 0.745 0.053 0.062 0.102 0.097]
L0H4 [0.318 0.163 0.057 0.204 0.048 0.079]
L0H5 [0.259 0.099 0.035 0.498 0.056 0.028]
L1H0 [0.046 0.077 0.087 0.171 0.133 0.07 ]
L1H1 [0.052 0.082 0.099 0.198 0.139 0.083]
L1H2 [0.046 0.077 0.08  0.177 0.178 0.106]
L1H3 [0.057 0.086 0.071 0.164 0.141 0.087]
L1H4 [0.042 0.074 0.059 0.148 0.201 0.086]
L1H5 [0.063 0.08  0.105 0.186 0.163 0.063]
L2H0 [0.058 0.079 0.052 0.092 0.055 0.05 ]
L2H1 [0.052 0.059 0.056 0.077 0.076 0.063]
L2H2 [0.054 0.06  0.046 0.076 0.061 0.056]
L2H3 [0.049 0.05  0.043 0.064 0.05  0.064]
L2H4 [0.051 0.068 0.053 0.075 0.071 0.06 ]
L2H5 [0.052 0.06  0.055 0.084 0.073 0.062]
L3H0 [0.047 0.054 0.048 0.063 0.06  0.067]
L3H1 [0.052 0.052 0.05  0.057 0.045 0.059]
L3H2 [0.049 0.057 0.05  0.064 0.047 0.051]
L3H3 [0.046 0.048 0.07  0.048 0.079 0.092]
L3H4 [0.048 0.06  0.057 0.059 0.042 0.05 ]
L3H5 [0.048 0.054 0.051 0.065 0.052 0.057]

L0H3 raw QK nonzero eigenvalues, descending magnitude
[ 0.7237+0.6913j  0.7237-0.6913j  0.4463+0.8419j  0.4463-0.8419j  0.5721+0.7221j  0.5721-0.7221j  0.6425+0.6136j  0.6425-0.6136j  0.676 +0.4626j
  0.676 -0.4626j  0.1094+0.7623j  0.1094-0.7623j  0.4629+0.6042j  0.4629-0.6042j  0.6213+0.j      0.4872+0.3765j  0.4872-0.3765j -0.0013+0.6066j
 -0.0013-0.6066j  0.3503+0.2828j  0.3503-0.2828j -0.2554+0.3536j -0.2554-0.3536j  0.0404+0.3357j  0.0404-0.3357j -0.2878+0.j     -0.1337+0.2534j
 -0.1337-0.2534j -0.2357+0.j      0.157 +0.j     -0.1513+0.0352j -0.1513-0.0352j -0.1234+0.0288j -0.1234-0.0288j  0.1132+0.j     -0.0773+0.j
  0.0001-0.0576j  0.0001+0.0576j -0.0294-0.0488j -0.0294+0.0488j]
positive-imag phases [0.6001 0.6579 0.6792 0.7624 0.7625 0.9007 0.917  1.0833 1.4283 1.4509 1.5729 2.0562 2.1963]

MLP0 current/previous weight coupling (no activations)
r 16 sv [44.1  17.18 12.62 11.3  10.6   7.71  7.09  5.68  5.15  4.62] sym eig extremes [-8.47 -6.45 -5.87 -5.44 -3.33  4.47  7.16  8.74 12.96 40.31] trace/norm 0.878
r 20 sv [54.57 20.6  14.08 13.3  12.07 10.32  8.74  8.01  7.47  6.15] sym eig extremes [-10.31  -9.43  -6.73  -6.42  -5.89   5.1    8.09   8.96  14.32  46.97] trace/norm 0.747
r 24 sv [58.11 21.8  15.33 14.05 13.63 12.33 10.91  9.14  8.13  6.36] sym eig extremes [-11.89 -10.16  -8.09  -7.55  -6.38   5.71   9.1    9.75  14.6   49.13] trace/norm 0.722
r 28 sv [58.41 22.03 16.73 14.78 13.93 13.1  11.42  9.69  8.34  6.93] sym eig extremes [-12.34 -10.37  -8.3   -7.88  -6.38   6.26   9.92  10.21  14.79  49.31] trace/norm 0.712

```

## Call 6

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)

g0=W["ln1_w.0"].astype(float)
v=W["W_v.0"][192:256].astype(float)*g0
o=W["W_o.0"][:,192:256].astype(float)
B=o@v
T=np.linalg.svd(v,full_matrices=False)[2][:20].T
coords=np.array([21,39,58,99,184])
print("Counter-coordinate blocks and induction-path diagnostics")
for l in range(1,4):
    g=W[f"ln1_w.{l}"].astype(float)
    b=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s=slice(64*j,64*(j+1))
        q=W[f"W_q.{l}"][s].astype(float)
        k=W[f"W_k.{l}"][s].astype(float)
        A=(q*g).T@(k*g)/8
        r=(q@b)@(k*g)/8
        C=T.T@A@B@T
        eig=np.linalg.eigvals(C)
        sym=np.linalg.eigvalsh((C+C.T)/2)
        print("L%dH%d"%(l,j),
              "A21,21=%.3f col21energy=%.3f row21energy=%.3f r21=%.3f"%
              (A[21,21],np.sum(A[:,21]**2)/np.sum(A*A),np.sum(A[21]**2)/np.sum(A*A),r[21]),
              "token-path tr/norm=%.3f positive_sym=%d/20"%
              (np.trace(C)/(np.linalg.norm(C)+1e-12),sum(sym>0)),
              "sym_extrema",np.round([sym[0],sym[-1]],3))
        if j in [0,3,4]:
            print("A block rows/cols",coords.tolist())
            print(np.round(A[np.ix_(coords,coords)],3))

print("\nL0 QK/OV effective ranks")
for j in range(6):
    s=slice(64*j,64*(j+1))
    A=(W["W_q.0"][s]*g0).T@(W["W_k.0"][s]*g0)/8
    B=W["W_o.0"][:,s]@(W["W_v.0"][s]*g0)
    for name,M in [("QK",A),("OV",B)]:
        sv=np.linalg.svd(M,compute_uv=False)
        energy=np.cumsum(sv**2)/np.sum(sv**2)
        print("L0H%d %s"%(j,name),"rank90/95/99",np.searchsorted(energy,[.90,.95,.99])+1)

```

Output:

```
Counter-coordinate blocks and induction-path diagnostics
L1H0 A21,21=0.087 col21energy=0.301 row21energy=0.012 r21=3.627 token-path tr/norm=0.329 positive_sym=12/20 sym_extrema [-0.038  0.039]
A block rows/cols [21, 39, 58, 99, 184]
[[ 0.087  0.05  -0.058  0.184 -0.002]
 [-0.123  0.016 -0.012  0.001 -0.001]
 [ 0.159 -0.02   0.015 -0.001  0.001]
 [-0.265  0.058 -0.053  0.073 -0.002]
 [-0.105  0.003  0.001 -0.029 -0.001]]
L1H1 A21,21=0.090 col21energy=0.325 row21energy=0.015 r21=3.352 token-path tr/norm=0.630 positive_sym=10/20 sym_extrema [-0.034  0.056]
L1H2 A21,21=0.077 col21energy=0.364 row21energy=0.011 r21=2.769 token-path tr/norm=0.479 positive_sym=10/20 sym_extrema [-0.047  0.041]
L1H3 A21,21=0.089 col21energy=0.364 row21energy=0.016 r21=3.079 token-path tr/norm=0.163 positive_sym=10/20 sym_extrema [-0.049  0.037]
A block rows/cols [21, 39, 58, 99, 184]
[[ 0.089  0.043 -0.062  0.188  0.001]
 [-0.122  0.015 -0.016 -0.     0.   ]
 [ 0.15  -0.018  0.018  0.004 -0.   ]
 [-0.267  0.054 -0.063  0.075 -0.001]
 [-0.106  0.007 -0.005 -0.023  0.   ]]
L1H4 A21,21=0.080 col21energy=0.348 row21energy=0.008 r21=2.008 token-path tr/norm=0.152 positive_sym=11/20 sym_extrema [-0.027  0.027]
A block rows/cols [21, 39, 58, 99, 184]
[[ 0.08   0.022 -0.032  0.112  0.002]
 [-0.072  0.013 -0.019  0.009 -0.   ]
 [ 0.084 -0.016  0.023 -0.014  0.   ]
 [-0.164  0.037 -0.053  0.047 -0.001]
 [-0.049  0.004 -0.009 -0.009  0.   ]]
L1H5 A21,21=0.092 col21energy=0.319 row21energy=0.018 r21=3.563 token-path tr/norm=0.515 positive_sym=11/20 sym_extrema [-0.05   0.055]
L2H0 A21,21=0.062 col21energy=0.154 row21energy=0.021 r21=1.815 token-path tr/norm=0.284 positive_sym=10/20 sym_extrema [-0.015  0.022]
A block rows/cols [21, 39, 58, 99, 184]
[[ 0.062  0.067 -0.067  0.202  0.   ]
 [-0.164  0.017 -0.014 -0.01   0.006]
 [ 0.15  -0.015  0.012  0.003 -0.004]
 [-0.228  0.058 -0.051  0.096  0.004]
 [-0.051 -0.002  0.003 -0.028  0.003]]
L2H1 A21,21=0.061 col21energy=0.056 row21energy=0.010 r21=1.682 token-path tr/norm=0.105 positive_sym=11/20 sym_extrema [-0.028  0.03 ]
L2H2 A21,21=0.057 col21energy=0.079 row21energy=0.010 r21=1.411 token-path tr/norm=0.345 positive_sym=10/20 sym_extrema [-0.016  0.02 ]
L2H3 A21,21=0.039 col21energy=0.071 row21energy=0.004 r21=0.803 token-path tr/norm=0.435 positive_sym=10/20 sym_extrema [-0.018  0.025]
A block rows/cols [21, 39, 58, 99, 184]
[[ 0.039  0.026 -0.025  0.083  0.002]
 [-0.061  0.008 -0.011  0.004  0.003]
 [ 0.049 -0.007  0.009 -0.007 -0.002]
 [-0.098  0.023 -0.021  0.031  0.   ]
 [-0.011 -0.002 -0.002 -0.005  0.003]]
L2H4 A21,21=0.059 col21energy=0.102 row21energy=0.013 r21=2.069 token-path tr/norm=0.700 positive_sym=11/20 sym_extrema [-0.028  0.034]
A block rows/cols [21, 39, 58, 99, 184]
[[ 0.059  0.072 -0.076  0.22  -0.   ]
 [-0.171  0.014 -0.016 -0.029  0.004]
 [ 0.141 -0.011  0.014  0.022 -0.002]
 [-0.264  0.062 -0.058  0.092 -0.002]
 [-0.053 -0.005  0.002 -0.045  0.005]]
L2H5 A21,21=0.061 col21energy=0.062 row21energy=0.010 r21=1.859 token-path tr/norm=-0.035 positive_sym=10/20 sym_extrema [-0.028  0.028]
L3H0 A21,21=0.019 col21energy=0.031 row21energy=0.005 r21=0.408 token-path tr/norm=-0.629 positive_sym=8/20 sym_extrema [-0.032  0.029]
A block rows/cols [21, 39, 58, 99, 184]
[[ 0.019  0.053 -0.046  0.104  0.003]
 [-0.071  0.006 -0.    -0.002  0.006]
 [ 0.04  -0.004  0.006  0.002 -0.006]
 [-0.096  0.017 -0.003  0.028  0.006]
 [-0.032  0.003  0.001 -0.006  0.007]]
L3H1 A21,21=0.031 col21energy=0.052 row21energy=0.006 r21=0.181 token-path tr/norm=-0.222 positive_sym=9/20 sym_extrema [-0.014  0.014]
L3H2 A21,21=0.024 col21energy=0.053 row21energy=0.009 r21=0.382 token-path tr/norm=0.193 positive_sym=11/20 sym_extrema [-0.014  0.018]
L3H3 A21,21=0.007 col21energy=0.000 row21energy=0.000 r21=-0.011 token-path tr/norm=2.860 positive_sym=16/20 sym_extrema [-0.007  0.06 ]
A block rows/cols [21, 39, 58, 99, 184]
[[ 0.007  0.002  0.     0.005  0.003]
 [-0.     0.004 -0.003  0.005  0.019]
 [-0.002  0.003  0.01   0.005  0
```

## Call 7

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from threadpoolctl import threadpool_limits
threadpool_limits(4)

g=W["ln1_w.0"].astype(float)
V=W["W_v.0"][192:256].astype(float)*g
B=W["W_o.0"][:,192:256].astype(float)@V
T=np.linalg.svd(V,full_matrices=False)[2][:24].T
F=W["W_fc.0"].astype(float)*W["ln2_w.0"]
X=F@T
Y=F@B@T
X-=X.mean(0); Y-=Y.mean(0)
Z=np.vstack([X,Y])
cov=Z.T@Z/len(Z)
ev,U=np.linalg.eigh(cov)
wh=U@np.diag(1/np.sqrt(ev+1e-6))@U.T
Z=Z@wh
Z/=np.linalg.norm(Z,axis=1,keepdims=True)
print("Clustering ONLY first-MLP weight projections onto current/previous value subspaces")
for nc in [16,20,24,28]:
    km=KMeans(n_clusters=nc,n_init=20,random_state=7).fit(Z)
    centers=km.cluster_centers_
    print("clusters",nc,"mean within-cluster cosine",round(np.mean(np.linalg.norm(centers,axis=1)),3),
          "counts",np.sort(np.bincount(km.labels_,minlength=nc))[::-1])
    if nc==24:
        tab=np.zeros((nc,nc),int)
        for a,b in zip(km.labels_[:1536],km.labels_[1536:]):
            tab[a,b]+=1
        expected=tab.sum(1)[:,None]*tab.sum(0)[None,:]/tab.sum()
        R=(tab-expected)/np.sqrt(expected+1)
        print("pair-table standardized residual singular values",np.round(np.linalg.svd(R,compute_uv=False)[:12],2))
        ij=np.dstack(np.unravel_index(np.argsort(R.ravel())[-20:][::-1],R.shape))[0]
        print("largest pair enrichments (arbitrary cluster labels):",
              [(int(a),int(b),int(tab[a,b]),round(float(expected[a,b]),1),round(float(R[a,b]),2)) for a,b in ij])

print("\nLast-MLP output spectrum: adjacent ratios")
sv=np.linalg.svd(W["W_proj.3"].astype(float),compute_uv=False)
print([(i+1,round(sv[i],3),round(sv[i]/sv[i+1],3)) for i in range(18,32)])
print("top24 energy",np.sum(sv[:24]**2)/np.sum(sv**2),"top28 energy",np.sum(sv[:28]**2)/np.sum(sv**2))

```

Output:

```
Clustering ONLY first-MLP weight projections onto current/previous value subspaces
clusters 16 mean within-cluster cosine 0.506 counts [281 253 227 226 215 210 207 202 200 187 156 156 150 142 141 119]
clusters 20 mean within-cluster cosine 0.53 counts [229 221 206 204 201 195 180 160 152 141 140 135 131 126 121 120 116 108  97  89]
clusters 24 mean within-cluster cosine 0.549 counts [227 219 210 196 191 161 148 131 127 121 117 112 105 104 103 101 101 100  93  93  87  78  77  70]
pair-table standardized residual singular values [21.65 10.39  9.11  8.02  7.06  6.58  6.12  5.33  5.06  4.9   4.21  4.09]
largest pair enrichments (arbitrary cluster labels): [(1, 5, 52, 17.8, 7.88), (22, 13, 17, 3.6, 6.26), (6, 0, 42, 17.3, 5.76), (17, 17, 12, 2.1, 5.58), (23, 23, 8, 1.1, 4.68), (9, 9, 10, 2.0, 4.57), (14, 14, 8, 1.3, 4.36), (18, 18, 9, 1.8, 4.28), (22, 7, 10, 2.4, 4.15), (6, 21, 41, 21.7, 4.04), (8, 21, 47, 26.1, 4.01), (10, 2, 9, 2.1, 3.88), (4, 4, 7, 1.4, 3.67), (8, 15, 13, 4.5, 3.65), (8, 12, 13, 4.5, 3.65), (8, 0, 37, 20.9, 3.45), (8, 11, 26, 13.1, 3.42), (7, 10, 7, 1.9, 3.04), (19, 9, 7, 1.9, 2.97), (9, 14, 7, 2.0, 2.92)]
clusters 28 mean within-cluster cosine 0.567 counts [183 172 162 158 150 145 143 143 140 114 112 110 107 107  94  92  91  88  85  85  82  81  78  77  73  72  66  62]

Last-MLP output spectrum: adjacent ratios
[(19, np.float64(12.317), np.float64(1.068)), (20, np.float64(11.531), np.float64(1.044)), (21, np.float64(11.047), np.float64(1.216)), (22, np.float64(9.084), np.float64(1.087)), (23, np.float64(8.353), np.float64(1.014)), (24, np.float64(8.24), np.float64(1.294)), (25, np.float64(6.37), np.float64(1.092)), (26, np.float64(5.836), np.float64(1.261)), (27, np.float64(4.626), np.float64(1.009)), (28, np.float64(4.583), np.float64(1.213)), (29, np.float64(3.777), np.float64(1.046)), (30, np.float64(3.61), np.float64(1.11)), (31, np.float64(3.252), np.float64(1.022)), (32, np.float64(3.183), np.float64(1.031))]
top24 energy 0.9213614809422224 top28 energy 0.937515023916908

```

## Call 8

```python
import numpy as np
from scipy.linalg import fractional_matrix_power
from threadpoolctl import threadpool_limits
threadpool_limits(4)

for scaled in [False,True]:
    g=W["ln1_w.0"].astype(float) if scaled else np.ones(384)
    As=[]
    for j in range(6):
        s=slice(64*j,64*(j+1))
        As.append((W["W_q.0"][s]*g).T@(W["W_k.0"][s]*g)/8)
    print("\nScaled" if scaled else "\nUnscaled","QK kernel power comparisons")
    ref=As[3]/np.linalg.norm(As[3],2)
    powers=[np.eye(384)]
    for z in range(8):
        powers.append(powers[-1]@ref)
    for j in range(6):
        print("L0H%d cos powers 0..8"%j,np.round([
            np.sum(As[j]*P)/(np.linalg.norm(As[j])*np.linalg.norm(P)) for P in powers],3))
    q=W["W_q.0"][192:256].astype(float)*g
    k=W["W_k.0"][192:256].astype(float)*g
    _,s,P=np.linalg.svd(np.vstack([q,k]),full_matrices=False)
    for r in [16,24,32]:
        Z=P[:r].T
        # Whitening of a weight Gram matrix, not a network normalization.
        H=(Z.T@(q.T@q+k.T@k)@Z)/16
        ev,U=np.linalg.eigh(H)
        wh=U@np.diag(1/np.sqrt(ev))@U.T
        M=[wh@Z.T@A@Z@wh for A in As]
        u,sv,vt=np.linalg.svd(M[3])
        R=u@vt
        print("rank",r,"polar power comparison")
        for j in [0,1,2,4,5]:
            vals=[]
            for lag in range(9):
                z=np.linalg.matrix_power(R,lag)
                vals.append(np.sum(M[j]*z)/(np.linalg.norm(M[j])*np.linalg.norm(z)))
            print("H",j,np.round(vals,3))

```

Output:

```

Unscaled QK kernel power comparisons
L0H0 cos powers 0..8 [-0.06  -0.132 -0.041  0.097  0.162  0.122  0.023 -0.062 -0.09 ]
L0H1 cos powers 0..8 [-0.046 -0.022 -0.005  0.006  0.006 -0.001 -0.006 -0.006 -0.001]
L0H2 cos powers 0..8 [-0.048 -0.028 -0.005  0.005  0.005 -0.    -0.003 -0.003 -0.001]
L0H3 cos powers 0..8 [ 0.091  1.     0.523 -0.126 -0.505 -0.488 -0.211  0.106  0.286]
L0H4 cos powers 0..8 [-0.044 -0.032  0.021  0.036  0.016 -0.01  -0.02  -0.013  0.001]
L0H5 cos powers 0..8 [-0.012 -0.013  0.008  0.009  0.003 -0.005 -0.008 -0.006 -0.001]
rank 16 polar power comparison
H 0 [-0.398 -0.337 -0.009  0.348  0.492  0.372  0.131 -0.049 -0.089]
H 1 [-0.073 -0.038  0.126  0.236  0.185  0.055 -0.014  0.005  0.058]
H 2 [-0.202 -0.208 -0.127 -0.05  -0.049 -0.099 -0.138 -0.153 -0.159]
H 4 [-0.323 -0.209  0.134  0.351  0.234 -0.051 -0.212 -0.161 -0.023]
H 5 [-0.1   -0.037  0.046  0.061 -0.002 -0.072 -0.097 -0.079 -0.037]
rank 24 polar power comparison
H 0 [-0.314 -0.2   -0.022  0.18   0.182  0.113 -0.106 -0.131 -0.117]
H 1 [-0.095 -0.064 -0.065  0.001 -0.022  0.032  0.001  0.064 -0.002]
H 2 [-0.122 -0.095 -0.076 -0.019  0.031  0.065  0.022  0.046 -0.015]
H 4 [-0.162 -0.05   0.1    0.136  0.08  -0.052 -0.163 -0.076 -0.013]
H 5 [-0.149 -0.035  0.108  0.165  0.138  0.075 -0.044 -0.109 -0.124]
rank 32 polar power comparison
H 0 [-0.282 -0.155 -0.032  0.127  0.151  0.08  -0.002 -0.036 -0.023]
H 1 [-0.174 -0.08  -0.043  0.022  0.038 -0.007 -0.017 -0.005  0.023]
H 2 [-0.176 -0.114 -0.067  0.     0.054  0.02  -0.007 -0.002  0.014]
H 4 [-0.165 -0.097  0.068  0.142  0.02  -0.052 -0.071  0.023  0.027]
H 5 [-0.083 -0.062  0.077  0.1    0.095 -0.003  0.009 -0.066 -0.03 ]

Scaled QK kernel power comparisons
L0H0 cos powers 0..8 [-0.018  0.137  0.111 -0.066 -0.29  -0.366 -0.206  0.099  0.368]
L0H1 cos powers 0..8 [-0.032 -0.026 -0.013 -0.01  -0.001  0.013  0.021  0.016  0.001]
L0H2 cos powers 0..8 [-0.042 -0.063 -0.019  0.048  0.103  0.099  0.031 -0.06  -0.121]
L0H3 cos powers 0..8 [ 0.098  1.     0.595  0.018 -0.409 -0.535 -0.377 -0.056  0.27 ]
L0H4 cos powers 0..8 [-0.026 -0.044 -0.016 -0.008  0.004  0.021  0.03   0.022 -0.001]
L0H5 cos powers 0..8 [-0.013  0.01   0.061  0.067  0.027 -0.038 -0.083 -0.081 -0.032]
rank 16 polar power comparison
H 0 [-0.17  -0.052  0.038  0.031 -0.056 -0.131 -0.122 -0.045  0.014]
H 1 [-0.013 -0.106 -0.126 -0.065  0.036  0.114  0.114  0.031 -0.078]
H 2 [-0.168 -0.176 -0.084  0.046  0.129  0.116  0.022 -0.084 -0.135]
H 4 [ 0.024 -0.095 -0.123 -0.061  0.011  0.058  0.093  0.097  0.035]
H 5 [-0.146 -0.005  0.151  0.193  0.082 -0.092 -0.191 -0.153 -0.017]
rank 24 polar power comparison
H 0 [-0.256 -0.13   0.015  0.103  0.084  0.043 -0.032 -0.012 -0.003]
H 1 [-0.062 -0.069  0.023 -0.069 -0.051  0.112  0.028 -0.002 -0.003]
H 2 [-0.155 -0.099  0.004 -0.004  0.072  0.128 -0.014 -0.059 -0.083]
H 4 [-0.075 -0.08   0.063  0.027  0.001  0.019 -0.053 -0.024  0.037]
H 5 [-0.094  0.024  0.093  0.14   0.048 -0.054 -0.109 -0.075 -0.008]
rank 32 polar power comparison
H 0 [-0.126  0.047 -0.031  0.086  0.024  0.093 -0.041  0.04  -0.013]
H 1 [-0.133 -0.004 -0.059  0.039 -0.073  0.045 -0.059  0.047 -0.031]
H 2 [-0.16   0.028 -0.084  0.057 -0.086  0.08  -0.091  0.059 -0.061]
H 4 [-0.104 -0.062 -0.021  0.008 -0.014 -0.02  -0.047 -0.013  0.005]
H 5 [-0.131 -0.036  0.025  0.058 -0.05   0.018 -0.053 -0.008 -0.048]

```

## Call 9

```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)

g0=W["ln1_w.0"].astype(float)
V0=W["W_v.0"][192:256].astype(float)*g0
T=np.linalg.svd(V0,full_matrices=False)[2][:20].T
B0=W["W_o.0"][:,192:256].astype(float)@V0
Uout=np.linalg.svd(W["W_proj.3"].astype(float),full_matrices=False)[0][:,:24]

print("Final heads: weight-only token-to-output and matching diagnostics")
for j in range(6):
    s=slice(j*64,(j+1)*64)
    g=W["ln1_w.3"].astype(float)
    A=(W["W_q.3"][s]*g).T@(W["W_k.3"][s]*g)/8
    B=W["W_o.3"][:,s].astype(float)@(W["W_v.3"][s]*g)
    D=B@T
    C=T.T@A@B0@T
    print("L3H%d"%j,
          "OV token-map norm %.3f output-subspace energy %.3f"%
          (np.linalg.norm(D),np.linalg.norm(Uout.T@D)**2/np.linalg.norm(D)**2),
          "matching trace/norm %.3f"%(np.trace(C)/np.linalg.norm(C)),
          "matching eig real",np.round(np.sort(np.linalg.eigvals(C).real)[-6:],4))

print("\nCoordinate concentration of L0 OV output (rows 21,39,58,99,184)")
ix=[21,39,58,99,184]
for j in range(6):
    s=slice(j*64,(j+1)*64)
    B=W["W_o.0"][:,s].astype(float)@(W["W_v.0"][s]*g0)
    print("L0H%d"%j,round(np.sum(B[ix]**2)/np.sum(B**2),4))

print("\nInduction matching control: signed/permuted head-output coordinates")
rng=np.random.default_rng(1022)
s=slice(192,256);g=W["ln1_w.3"].astype(float)
A=(W["W_q.3"][s]*g).T@(W["W_k.3"][s]*g)/8
left=T.T@A
right=B0@T
actual=left@right
null=[]
for _ in range(1000):
    perm=rng.permutation(384)
    sign=rng.choice([-1.,1.],384)
    C=(left*sign)@right[perm]
    null.append(np.trace(C)/np.linalg.norm(C))
print("observed",np.trace(actual)/np.linalg.norm(actual),
      "null mean/std",np.mean(null),np.std(null),
      "null 1/99 percentiles",np.percentile(null,[1,99]),
      "null max",max(null))

```

Output:

```
Final heads: weight-only token-to-output and matching diagnostics
L3H0 OV token-map norm 4.999 output-subspace energy 0.087 matching trace/norm -0.629 matching eig real [0.0012 0.0015 0.0015 0.0071 0.0071 0.0183]
L3H1 OV token-map norm 6.302 output-subspace energy 0.098 matching trace/norm -0.222 matching eig real [0.0035 0.0035 0.0072 0.0072 0.0074 0.0074]
L3H2 OV token-map norm 5.569 output-subspace energy 0.092 matching trace/norm 0.193 matching eig real [0.0041 0.0041 0.0074 0.0094 0.0094 0.0117]
L3H3 OV token-map norm 13.422 output-subspace energy 0.348 matching trace/norm 2.860 matching eig real [0.0375 0.0382 0.0382 0.0386 0.0386 0.0533]
L3H4 OV token-map norm 10.491 output-subspace energy 0.119 matching trace/norm 0.662 matching eig real [0.0025 0.0025 0.0029 0.0029 0.0038 0.0064]
L3H5 OV token-map norm 5.152 output-subspace energy 0.077 matching trace/norm -0.132 matching eig real [0.0042 0.0042 0.008  0.008  0.0107 0.0107]

Coordinate concentration of L0 OV output (rows 21,39,58,99,184)
L0H0 0.0344
L0H1 0.9395
L0H2 0.942
L0H3 0.0109
L0H4 0.8777
L0H5 0.97

Induction matching control: signed/permuted head-output coordinates
observed 2.8602269991267235 null mean/std -0.0002919724439507183 0.23419073556467282 null 1/99 percentiles [-0.5479  0.5497] null max 0.6820147810277253

```
