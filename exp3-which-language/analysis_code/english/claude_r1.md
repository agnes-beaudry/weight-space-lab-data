## Call 1

```python
import numpy as np
print(sorted(W.keys()))
for k in sorted(W.keys()):
    a=W[k]
    print(k,a.shape,a.dtype,'norm',np.linalg.norm(a).round(3),'mean',a.mean().round(4),'std',a.std().round(4))
for l in range(4):
    for name in ['W_q','W_k','W_v']:
        M=W[f'{name}.{l}']
        print(l,name,'row-block fro:',[round(float(np.linalg.norm(M[64*j:64*j+64])),2) for j in range(6)])
    M=W[f'W_o.{l}']
    print(l,'W_o col-block fro:',[round(float(np.linalg.norm(M[:,64*j:64*j+64])),2) for j in range(6)])
    for name in ['ln1','ln2']:
        w=W[f'{name}_w.{l}'];b=W[f'{name}_b.{l}']
        print(l,name,'w mean/std/min/max',w.mean().round(3),w.std().round(3),w.min().round(3),w.max().round(3),'b mean/std',b.mean().round(3),b.std().round(3))

```

Output:

```
['W_fc.0', 'W_fc.1', 'W_fc.2', 'W_fc.3', 'W_k.0', 'W_k.1', 'W_k.2', 'W_k.3', 'W_o.0', 'W_o.1', 'W_o.2', 'W_o.3', 'W_proj.0', 'W_proj.1', 'W_proj.2', 'W_proj.3', 'W_q.0', 'W_q.1', 'W_q.2', 'W_q.3', 'W_v.0', 'W_v.1', 'W_v.2', 'W_v.3', 'ln1_b.0', 'ln1_b.1', 'ln1_b.2', 'ln1_b.3', 'ln1_w.0', 'ln1_w.1', 'ln1_w.2', 'ln1_w.3', 'ln2_b.0', 'ln2_b.1', 'ln2_b.2', 'ln2_b.3', 'ln2_w.0', 'ln2_w.1', 'ln2_w.2', 'ln2_w.3']
W_fc.0 (1536, 384) float32 norm 46.249 mean -0.0012 std 0.0602
W_fc.1 (1536, 384) float32 norm 79.127 mean -0.0004 std 0.103
W_fc.2 (1536, 384) float32 norm 78.967 mean -0.0003 std 0.1028
W_fc.3 (1536, 384) float32 norm 77.776 mean -1e-04 std 0.1013
W_k.0 (384, 384) float32 norm 30.391 mean -1e-04 std 0.0791
W_k.1 (384, 384) float32 norm 29.251 mean 0.0 std 0.0762
W_k.2 (384, 384) float32 norm 37.769 mean 0.0 std 0.0984
W_k.3 (384, 384) float32 norm 32.209 mean -1e-04 std 0.0839
W_o.0 (384, 384) float32 norm 16.371 mean -1e-04 std 0.0426
W_o.1 (384, 384) float32 norm 25.923 mean -0.0 std 0.0675
W_o.2 (384, 384) float32 norm 32.84 mean 0.0 std 0.0855
W_o.3 (384, 384) float32 norm 40.455 mean 1e-04 std 0.1054
W_proj.0 (384, 1536) float32 norm 39.308 mean -1e-04 std 0.0512
W_proj.1 (384, 1536) float32 norm 75.71 mean 0.0 std 0.0986
W_proj.2 (384, 1536) float32 norm 76.831 mean -1e-04 std 0.1
W_proj.3 (384, 1536) float32 norm 82.395 mean 0.0002 std 0.1073
W_q.0 (384, 384) float32 norm 32.291 mean 0.0003 std 0.0841
W_q.1 (384, 384) float32 norm 32.313 mean -0.0 std 0.0841
W_q.2 (384, 384) float32 norm 39.346 mean 0.0 std 0.1025
W_q.3 (384, 384) float32 norm 32.948 mean -0.0 std 0.0858
W_v.0 (384, 384) float32 norm 14.836 mean 1e-04 std 0.0386
W_v.1 (384, 384) float32 norm 27.353 mean -0.0 std 0.0712
W_v.2 (384, 384) float32 norm 33.305 mean 1e-04 std 0.0867
W_v.3 (384, 384) float32 norm 39.794 mean 1e-04 std 0.1036
ln1_b.0 (384,) float32 norm 0.563 mean 0.0008 std 0.0287
ln1_b.1 (384,) float32 norm 3.277 mean -0.005 std 0.1672
ln1_b.2 (384,) float32 norm 2.989 mean -0.001 std 0.1525
ln1_b.3 (384,) float32 norm 1.68 mean 0.0121 std 0.0849
ln1_w.0 (384,) float32 norm 6.705 mean 0.2403 std 0.2436
ln1_w.1 (384,) float32 norm 9.187 mean 0.4575 std 0.1022
ln1_w.2 (384,) float32 norm 13.897 mean 0.7052 std 0.0749
ln1_w.3 (384,) float32 norm 14.825 mean 0.7546 std 0.0542
ln2_b.0 (384,) float32 norm 1.339 mean 0.0022 std 0.0683
ln2_b.1 (384,) float32 norm 2.673 mean 0.0005 std 0.1364
ln2_b.2 (384,) float32 norm 3.013 mean 0.005 std 0.1537
ln2_b.3 (384,) float32 norm 2.502 mean 0.0043 std 0.1276
ln2_w.0 (384,) float32 norm 16.965 mean 0.8006 std 0.3295
ln2_w.1 (384,) float32 norm 20.809 mean 1.0429 std 0.2003
ln2_w.2 (384,) float32 norm 25.833 mean 1.3123 std 0.1255
ln2_w.3 (384,) float32 norm 25.089 mean 1.2778 std 0.0807
0 W_q row-block fro: [3.31, 3.09, 30.82, 6.82, 3.97, 3.19]
0 W_k row-block fro: [4.18, 3.66, 28.3, 7.24, 5.0, 3.8]
0 W_v row-block fro: [6.04, 5.38, 7.76, 5.43, 5.21, 6.14]
0 W_o col-block fro: [6.67, 6.02, 8.92, 5.03, 5.59, 7.16]
0 ln1 w mean/std/min/max 0.24 0.244 -0.001 1.034 b mean/std 0.001 0.029
0 ln2 w mean/std/min/max 0.801 0.33 0.004 1.507 b mean/std 0.002 0.068
1 W_q row-block fro: [10.72, 13.29, 14.48, 13.36, 13.52, 13.49]
1 W_k row-block fro: [9.83, 12.07, 13.09, 12.15, 12.13, 12.13]
1 W_v row-block fro: [11.16, 11.72, 10.67, 10.9, 11.15, 11.38]
1 W_o col-block fro: [9.98, 11.3, 10.27, 9.93, 10.79, 11.14]
1 ln1 w mean/std/min/max 0.458 0.102 0.136 1.124 b mean/std -0.005 0.167
1 ln2 w mean/std/min/max 1.043 0.2 -0.002 1.486 b mean/std 0.001 0.136
2 W_q row-block fro: [16.82, 15.96, 13.97, 16.12, 16.88, 16.45]
2 W_k row-block fro: [16.21, 15.29, 13.24, 15.55, 16.19, 15.84]
2 W_v row-block fro: [13.11, 13.77, 14.57, 13.5, 13.24, 13.34]
2 W_o col-block fro: [12.94, 13.53, 14.24, 13.39, 13.14, 13.16]
2 ln1 w mean/std/min/max 0.705 0.075 0.208 1.014 b mean/std -0.001 0.153
2 ln2 w mean/std/min/max 1.312 0.126 0.001 1.465 b mean/std 0.005 0.154
3 W_q row-block fro: [15.75, 14.22, 12.74, 13.28, 14.36, 9.51]
3 W_k row-block fro: [15.26, 14.2, 12.45, 13.06, 13.87, 9.23]
3 W_v row-block fro: [14.87, 17.88, 15.52, 15.73, 15.17, 17.99]
3 W_o col-block fro: [14.87, 18.33, 15.59, 16.12, 15.3, 18.51]
3 ln1 w mean/std/min/max 0.755 0.054 0.294 0.87 b mean/std 0.012 0.085
3 ln2 w mean/std/min/max 1.278 0.081 0.436 1.431 b mean/std 0.004 0.128

```

## Call 2

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=180)
def sv(M): return np.linalg.svd(M,compute_uv=False)
for l in range(4):
    for name in ['W_q','W_k','W_v']:
        M=W[f'{name}.{l}']
        for j in range(6):
            s=sv(M[64*j:64*j+64])
            print(l,name,j,'top',s[:12].round(2),'#>0.25top',(s>0.25*s[0]).sum(),'#>0.05top',(s>0.05*s[0]).sum(),'min',s[-1].round(3))
    M=W[f'W_o.{l}']
    for j in range(6):
        s=sv(M[:,64*j:64*j+64])
        print(l,'W_o',j,'top',s[:12].round(2),'#>0.25top',(s>0.25*s[0]).sum())
    print(l,'fc',sv(W[f'W_fc.{l}'])[:40].round(1))
    print(l,'proj',sv(W[f'W_proj.{l}'])[:40].round(1))

```

Output:

```
0 W_q 0 top [1.49 0.9  0.81 0.66 0.57 0.54 0.49 0.47 0.46 0.45 0.44 0.44] #>0.25top 21 #>0.05top 64 min 0.183
0 W_q 1 top [1.12 0.88 0.7  0.54 0.54 0.51 0.5  0.48 0.46 0.46 0.45 0.44] #>0.25top 41 #>0.05top 64 min 0.191
0 W_q 2 top [8.23 8.2  7.94 7.8  7.66 7.62 7.49 7.18 6.92 6.79 6.37 6.1 ] #>0.25top 28 #>0.05top 63 min 0.396
0 W_q 3 top [2.52 2.33 1.98 1.78 1.65 1.46 1.39 1.33 1.27 1.18 1.14 1.01] #>0.25top 22 #>0.05top 64 min 0.233
0 W_q 4 top [2.01 1.52 1.05 0.74 0.68 0.57 0.53 0.52 0.51 0.5  0.48 0.48] #>0.25top 9 #>0.05top 64 min 0.191
0 W_q 5 top [1.41 0.9  0.59 0.55 0.52 0.51 0.5  0.48 0.47 0.46 0.45 0.44] #>0.25top 25 #>0.05top 64 min 0.183
0 W_k 0 top [1.71 1.07 0.93 0.77 0.73 0.68 0.67 0.65 0.65 0.63 0.61 0.61] #>0.25top 31 #>0.05top 64 min 0.231
0 W_k 1 top [1.12 0.97 0.79 0.67 0.64 0.63 0.62 0.59 0.58 0.56 0.54 0.54] #>0.25top 53 #>0.05top 64 min 0.229
0 W_k 2 top [7.6  7.4  7.28 6.87 6.78 6.61 6.43 6.4  6.33 6.08 5.96 5.88] #>0.25top 29 #>0.05top 63 min 0.364
0 W_k 3 top [2.79 2.7  1.84 1.65 1.62 1.49 1.39 1.36 1.28 1.25 1.17 1.12] #>0.25top 23 #>0.05top 64 min 0.279
0 W_k 4 top [3.02 1.49 1.09 0.81 0.73 0.7  0.69 0.65 0.64 0.62 0.59 0.57] #>0.25top 4 #>0.05top 64 min 0.242
0 W_k 5 top [1.6  0.82 0.73 0.69 0.68 0.64 0.63 0.6  0.59 0.59 0.57 0.55] #>0.25top 30 #>0.05top 64 min 0.218
0 W_v 0 top [2.81 2.15 1.67 1.64 1.41 1.37 1.13 1.03 0.93 0.82 0.77 0.73] #>0.25top 13 #>0.05top 64 min 0.242
0 W_v 1 top [2.38 1.7  1.61 1.48 1.27 1.14 0.98 0.97 0.9  0.75 0.72 0.7 ] #>0.25top 16 #>0.05top 64 min 0.232
0 W_v 2 top [1.95 1.88 1.85 1.8  1.77 1.73 1.72 1.67 1.65 1.57 1.55 1.52] #>0.25top 36 #>0.05top 64 min 0.23
0 W_v 3 top [2.2  2.01 1.6  1.53 1.16 1.02 0.97 0.89 0.87 0.83 0.78 0.73] #>0.25top 20 #>0.05top 64 min 0.226
0 W_v 4 top [1.86 1.8  1.45 1.27 1.22 1.1  1.01 1.   0.95 0.89 0.84 0.76] #>0.25top 24 #>0.05top 64 min 0.223
0 W_v 5 top [2.55 2.21 1.68 1.57 1.45 1.4  1.21 1.11 1.09 1.04 0.91 0.82] #>0.25top 18 #>0.05top 64 min 0.233
0 W_o 0 top [3.94 3.24 2.31 1.7  1.38 1.11 1.   0.78 0.72 0.65 0.63 0.57] #>0.25top 7
0 W_o 1 top [3.44 3.17 1.85 1.46 1.39 1.01 0.81 0.77 0.61 0.58 0.53 0.52] #>0.25top 6
0 W_o 2 top [2.25 2.16 2.13 2.1  2.05 2.01 1.97 1.96 1.89 1.86 1.8  1.78] #>0.25top 35
0 W_o 3 top [2.3  2.06 1.79 1.57 1.33 0.97 0.94 0.88 0.83 0.74 0.68 0.63] #>0.25top 12
0 W_o 4 top [2.9  2.55 1.76 1.53 1.3  1.25 1.06 0.8  0.79 0.71 0.68 0.63] #>0.25top 9
0 W_o 5 top [4.39 3.   2.57 1.76 1.55 1.26 1.15 1.12 0.92 0.78 0.7  0.62] #>0.25top 8
0 fc [18.2 11.   8.7  7.5  7.2  7.1  7.   6.6  6.6  6.4  6.3  6.3  6.1  6.   6.   5.9  5.8  5.8  5.6  5.5  5.5  5.4  5.2  5.1  5.1  5.   4.9  4.9  4.8  4.7  4.6  4.5  4.5  4.4  4.3
  4.2  4.   4.   3.9  3.8]
0 proj [9.8 7.4 6.3 5.9 5.8 5.7 5.5 5.5 5.3 5.2 5.2 5.1 5.  4.9 4.9 4.8 4.8 4.7 4.6 4.6 4.6 4.5 4.4 4.4 4.3 4.2 4.1 4.1 4.  4.  3.9 3.8 3.8 3.8 3.8 3.7 3.7 3.6 3.6 3.5]
1 W_q 0 top [3.63 3.24 3.09 2.96 2.63 2.38 2.33 2.21 2.09 2.01 1.99 1.9 ] #>0.25top 26 #>0.05top 64 min 0.295
1 W_q 1 top [4.37 3.69 3.64 3.31 3.22 3.11 3.05 2.86 2.82 2.62 2.5  2.44] #>0.25top 26 #>0.05top 64 min 0.27
1 W_q 2 top [4.61 4.11 3.71 3.59 3.49 3.42 3.25 3.15 3.04 2.97 2.9  2.75] #>0.25top 27 #>0.05top 64 min 0.269
1 W_q 3 top [4.38 3.9  3.52 3.31 3.14 3.09 3.06 2.95 2.86 2.77 2.69 2.53] #>0.25top 24 #>0.05top 64 min 0.278
1 W_q 4 top [4.62 3.75 3.42 3.37 3.29 3.13 3.09 3.05 2.79 2.67 2.58 2.4 ] #>0.25top 26 #>0.05top 64 min 0.277
1 W_q 5 top [4.61 3.8  3.46 3.32 3.12 3.07 2.99 2.86 2.85 2.76 2.62 2.48] #>0.25top 26 #>0.05top 64 min 0.265
1 W_k 0 top [4.43 3.51 2.68 2.49 2.26 2.06 1.85 1.73 1.69 1.65 1.56 1.49] #>0.25top 18 #>0.05top 64 min 0.251
1 W_k 1 top [5.8  4.46 3.42 2.91 2.54 2.39 2.21 2.09 2.07 1.95 1.87 1.82] #>0.25top 18 #>0.05top 60 min 0.266
1 W_k 2 top [6.43 4.87 3.45 3.25 2.82 2.6  2.54 2.32 2.3  2.13 2.08 1.97] #>0.25top 18 #>0.05top 56 min 0.252
1 W_k 3 top [5.92 4.4  3.61 3.11 2.63 2.4  2.31 2.17 1.99 1.96 1.89 1.84] #>0.25top 16 #>0.05top 58 min 0.263
1 W_k 4 top [6.16 4.61 3.27 2.92 2.44 2.21 2.13 2.04 1.97 1.95 1.85 1.76] #>0.25top 16 #>0.05top 56 min 0.249
1 W_k 5 top [6.45 4.7  3.26 2.79 2.51 2.2  2.14 2.07 1.98 1.9  1.77 1.72] #>0.25top 13 #>0.05top 54 min 0.256
1 W_v 0 top [3.01 2.92 2.79 2.74 2.59 2.43 2.36 2.35 2.31 2.2  2.19 2.  ] #>0.25top 35 #>0.05top 64 min 0.307
1 W_v 1 top [2.68 2.57 2.54 2.44 2.4  2.3  2.27 2.2  2.18 2.15 2.13 2.06] #>0.25top 45 #>0.05top 64 min 0.336
1 W_v 2 top [2.55 2.49 2.35 2.29 2.19 2.17 2.13 2.13 2.07 2.05 2.01 1.96] #>0.25top 43 #>0.05top 64 min 0.344
1 W_v 3 top [2.89 2.82 2.68 2.58 2.5  2.47 2.43 2.28 2.23 2.14 2.07 2.03] #>0.25top 34 #>0.05top 64 min 0.31
1 W_v 4 top [2.71 2.63 2.55 2.48 2.42 2.3  2.29 2.23 2.15 2.12 2.1  2.06] #>0.25top 40 #>0.05top 64 min 0.317
1 W_v 5 top [2.75 2.56 2.47 2.44 2.42 2.31 2.27 2.24 2.21 2.17 2.12 2.06] #>0.25top 40 #>0.05top 64 min 0.312
1 W_o 0 top [3.48 2.78 2.58 2.49 2.49 2.27 2.21 2.16 2.07 2.01 1.9  1.88] #>0.25top 26
1 W_o 1 top [2.83 2.56 2.53 2.46 2.41 2.36 2.31 2.29 2.22 2.18 2.15 2.08] #>0.25top 39
1 W_o 2 top [2.58 2.5  2.41 2.36 2.31 2.29 2.23 2.17 2.14 2.11 2.07 2.01] #>0.25top 35
1 W_o 3 top [2.84 2.61 2.56 2.47 2.46 2.38 2.36 2.29 2.25 2.16 2.13 1.98] #>0.25top 27
1 W_o 4 top [2.62 2.59 2.52 2.48 2.43 2.4  2.37 2.34 2.33 2.24 2.2  2.15] #>0.25top 32
1 W_o 5 top [2.56 2.55 2.52 2.48 2.42 2.4  2.37 2.29 2.28 2.24 2.22 2.2 ] #>0.25top 34
1 fc [21.5 13.1 12.8 12.6 12.3 12.1 11.9 11.7 11.4 11.3 11.  10.8 10.6 10.6 10.3  9.9  9.8  9.7  9.6  9.4  9.2  9.2  9.1  8.9  8.8  8.7  8.5  8.4  8.4  8.2  8.1  7.9  7.7  7.7  7.6
  7.5  7.4  7.3  7.1  7.1]
1 proj [8.2 7.9 7.2 7.  6.7 6.4 6.3 6.1 6.1 6.  5.9 5.9 5.8 5.8 5.7 5.7 5.6 5.6 5.6 5.6 5.6 5.5 5.5 5.5 5.5 5.4 5.4 5.4 5.4 5.4 5.4 5.4 5.3 5.3 5.3 5.3 5.3 5.2 5.2 5.2]
2 W_q 0 top [3.76 3.4  3.22 3.18 3.08 2.97 2.92 2.88 2.83 2.75 2.72 2.64] #>0.25top 60 #>0.05top 64 min 0.723
2 W_q 1 top [3.19 2.88 2.82 2.74 2.7  2.65 2.62 2.57 2.54 2.52 2.47 2.45] #>0.25top 64 #>0.05top 64 min 0.923
2 W_q 2 top [3.12 2.75 2.45 2.3
```

## Call 3

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=180)
def rs(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False)
    return Vt[:k].T,S[:k],U[:k].T
def overlap(A,B):
    s=np.linalg.svd(A.T@B,compute_uv=False)
    return s
D1=W['ln1_w.0']; D2=W['ln2_w.0']
print('corr ln1_w.0 ln2_w.0',np.corrcoef(D1,D2)[0,1])
print('hist ln1_w.0',np.histogram(D1,bins=[-.1,.01,.05,.1,.2,.3,.5,.8,1.2])[0])
print('hist ln2_w.0',np.histogram(D2,bins=[-.1,.01,.05,.1,.3,.5,.8,1.2,1.6])[0])
print('sorted ln1_w.0 head',np.sort(D1)[:10],'tail',np.sort(D1)[-20:])
s=lambda j:slice(64*j,64*j+64)
Q=W['W_q.0'];K=W['W_k.0'];Vv=W['W_v.0'];Fc=W['W_fc.0'];O=W['W_o.0']
sub={}
for j in range(6):
    sub[f'q{j}']=rs(Q[s(j)]*D1,10 if j!=2 else 28)[0]
    sub[f'k{j}']=rs(K[s(j)]*D1,10 if j!=2 else 28)[0]
    sub[f'v{j}']=rs(Vv[s(j)]*D1,10 if j!=2 else 28)[0]
sub['fc']=rs(Fc*D2,28)[0]
sub['fc_raw']=rs(Fc,28)[0]
names=list(sub.keys())
print(names)
M=np.zeros((len(names),len(names)))
for a,na in enumerate(names):
    for b,nb in enumerate(names):
        o=overlap(sub[na],sub[nb])
        M[a,b]=(o**2).sum()/min(sub[na].shape[1],sub[nb].shape[1])
import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
print(pd.DataFrame(M,index=names,columns=names).round(2).to_string())
print('q2 vs k2 overlaps',overlap(sub['q2'],sub['k2']).round(2))
print('q2 vs v2 overlaps',overlap(sub['q2'],sub['v2']).round(2))
print('k2 vs v2 overlaps',overlap(sub['k2'],sub['v2']).round(2))
print('v2 vs fc overlaps',overlap(sub['v2'],sub['fc']).round(2))
print('q2 vs fc overlaps',overlap(sub['q2'],sub['fc']).round(2))

```

Output:

```
corr ln1_w.0 ln2_w.0 -0.40426269730054426
hist ln1_w.0 [ 19  33  73 118  40  47  31  23]
hist ln2_w.0 [  1   0   1  26  51 118 131  56]
sorted ln1_w.0 head [-0. -0. -0. -0. -0.  0.  0.  0.  0.  0.] tail [0.84 0.84 0.84 0.84 0.85 0.86 0.86 0.87 0.88 0.89 0.89 0.9  0.93 0.94 0.94 0.94 0.94 0.98 0.98 1.03]
['q0', 'k0', 'v0', 'q1', 'k1', 'v1', 'q2', 'k2', 'v2', 'q3', 'k3', 'v3', 'q4', 'k4', 'v4', 'q5', 'k5', 'v5', 'fc', 'fc_raw']
          q0    k0    v0    q1    k1    v1    q2    k2    v2    q3    k3    v3    q4    k4    v4    q5    k5    v5    fc  fc_raw
q0      1.00  0.17  0.27  0.37  0.20  0.26  0.26  0.31  0.17  0.41  0.23  0.14  0.44  0.22  0.25  0.34  0.17  0.26  0.03    0.06
k0      0.17  1.00  0.14  0.17  0.20  0.14  0.20  0.20  0.21  0.21  0.13  0.16  0.17  0.23  0.12  0.16  0.21  0.14  0.06    0.09
v0      0.27  0.14  1.00  0.19  0.15  0.46  0.39  0.39  0.29  0.34  0.30  0.28  0.27  0.18  0.43  0.20  0.15  0.45  0.12    0.15
q1      0.37  0.17  0.19  1.00  0.19  0.19  0.22  0.25  0.16  0.33  0.19  0.11  0.41  0.21  0.17  0.30  0.14  0.19  0.02    0.05
k1      0.20  0.20  0.15  0.19  1.00  0.15  0.26  0.27  0.20  0.20  0.12  0.14  0.22  0.21  0.12  0.17  0.18  0.13  0.03    0.06
v1      0.26  0.14  0.46  0.19  0.15  1.00  0.39  0.40  0.28  0.31  0.28  0.26  0.26  0.19  0.43  0.21  0.13  0.49  0.11    0.13
q2      0.26  0.20  0.39  0.22  0.26  0.39  1.00  0.79  0.26  0.48  0.51  0.28  0.24  0.26  0.35  0.24  0.28  0.36  0.02    0.05
k2      0.31  0.20  0.39  0.25  0.27  0.40  0.79  1.00  0.29  0.48  0.57  0.22  0.26  0.27  0.36  0.28  0.31  0.38  0.02    0.05
v2      0.17  0.21  0.29  0.16  0.20  0.28  0.26  0.29  1.00  0.21  0.28  0.35  0.17  0.19  0.27  0.15  0.19  0.33  0.19    0.16
q3      0.41  0.21  0.34  0.33  0.20  0.31  0.48  0.48  0.21  1.00  0.44  0.19  0.41  0.24  0.29  0.32  0.15  0.33  0.04    0.08
k3      0.23  0.13  0.30  0.19  0.12  0.28  0.51  0.57  0.28  0.44  1.00  0.17  0.22  0.16  0.27  0.18  0.15  0.31  0.06    0.10
v3      0.14  0.16  0.28  0.11  0.14  0.26  0.28  0.22  0.35  0.19  0.17  1.00  0.14  0.15  0.28  0.12  0.13  0.25  0.15    0.18
q4      0.44  0.17  0.27  0.41  0.22  0.26  0.24  0.26  0.17  0.41  0.22  0.14  1.00  0.23  0.23  0.30  0.17  0.26  0.03    0.06
k4      0.22  0.23  0.18  0.21  0.21  0.19  0.26  0.27  0.19  0.24  0.16  0.15  0.23  1.00  0.16  0.17  0.21  0.16  0.05    0.08
v4      0.25  0.12  0.43  0.17  0.12  0.43  0.35  0.36  0.27  0.29  0.27  0.28  0.23  0.16  1.00  0.18  0.10  0.46  0.09    0.12
q5      0.34  0.16  0.20  0.30  0.17  0.21  0.24  0.28  0.15  0.32  0.18  0.12  0.30  0.17  0.18  1.00  0.16  0.20  0.02    0.05
k5      0.17  0.21  0.15  0.14  0.18  0.13  0.28  0.31  0.19  0.15  0.15  0.13  0.17  0.21  0.10  0.16  1.00  0.10  0.04    0.07
v5      0.26  0.14  0.45  0.19  0.13  0.49  0.36  0.38  0.33  0.33  0.31  0.25  0.26  0.16  0.46  0.20  0.10  1.00  0.10    0.12
fc      0.03  0.06  0.12  0.02  0.03  0.11  0.02  0.02  0.19  0.04  0.06  0.15  0.03  0.05  0.09  0.02  0.04  0.10  1.00    0.78
fc_raw  0.06  0.09  0.15  0.05  0.06  0.13  0.05  0.05  0.16  0.08  0.10  0.18  0.06  0.08  0.12  0.05  0.07  0.12  0.78    1.00
q2 vs k2 overlaps [0.99 0.99 0.99 0.99 0.99 0.98 0.98 0.98 0.98 0.97 0.97 0.97 0.96 0.95 0.95 0.94 0.93 0.92 0.92 0.9  0.88 0.86 0.79 0.75 0.67 0.6  0.11 0.01]
q2 vs v2 overlaps [0.91 0.9  0.88 0.83 0.79 0.74 0.71 0.69 0.62 0.59 0.5  0.43 0.41 0.37 0.33 0.32 0.28 0.24 0.21 0.21 0.19 0.12 0.12 0.09 0.07 0.06 0.04 0.  ]
k2 vs v2 overlaps [0.94 0.92 0.88 0.88 0.86 0.82 0.76 0.71 0.64 0.58 0.51 0.47 0.44 0.41 0.35 0.33 0.32 0.3  0.24 0.2  0.18 0.15 0.13 0.11 0.1  0.06 0.06 0.01]
v2 vs fc overlaps [0.77 0.74 0.72 0.67 0.64 0.58 0.57 0.53 0.5  0.47 0.46 0.45 0.43 0.4  0.39 0.36 0.34 0.31 0.3  0.22 0.2  0.16 0.13 0.1  0.08 0.04 0.03 0.01]
q2 vs fc overlaps [0.39 0.3  0.26 0.22 0.18 0.17 0.17 0.15 0.14 0.13 0.12 0.1  0.1  0.09 0.08 0.08 0.07 0.06 0.06 0.05 0.04 0.04 0.03 0.02 0.02 0.01 0.   0.  ]

```

## Call 4

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=180)
s=lambda j:slice(64*j,64*j+64)
for l in range(4):
    D=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'];K=W[f'W_k.{l}']
    for j in range(6):
        q=Q[s(j)]*D;k=K[s(j)]*D
        # form q^T k => eig of k q^T (64x64)
        M=k@q.T
        ev=np.linalg.eigvals(M)
        ev=ev[np.argsort(-np.abs(ev))]
        tr=np.trace(M); 
        cos=tr/(np.linalg.norm(q)*np.linalg.norm(k))
        print(f'L{l}H{j} sym-cos {cos:.3f} |q|{np.linalg.norm(q):.1f} |k|{np.linalg.norm(k):.1f} top|ev|',np.abs(ev[:6]).round(1),'angles(deg)',np.degrees(np.angle(ev[:10])).round(0),
              'frac energy real part',(ev.real**2).sum()/(np.abs(ev)**2).sum())
q=W['W_q.0'][s(2)]*W['ln1_w.0'];print(np.linalg.svd(q,compute_uv=False)[:40].round(2))
k=W['W_k.0'][s(2)]*W['ln1_w.0'];print(np.linalg.svd(k,compute_uv=False)[:40].round(2))
print(np.linalg.svd(W['W_q.0'][s(2)],compute_uv=False)[:40].round(2))
print(np.linalg.svd(W['W_k.0'][s(2)],compute_uv=False)[:40].round(2))

```

Output:

```
L0H0 sym-cos -0.024 |q|1.4 |k|1.1 top|ev| [0.1 0.  0.  0.  0.  0. ] angles(deg) [   0.  180.  180.  108. -108.   80.  -80.  160. -160.  142.] frac energy real part 0.8447742
L0H1 sym-cos -0.097 |q|1.1 |k|1.0 top|ev| [0. 0. 0. 0. 0. 0.] angles(deg) [ 166. -166.    0.  155. -155.  180.  138. -138.   98.  -98.] frac energy real part 0.79950887
L0H2 sym-cos 0.063 |q|11.5 |k|13.2 top|ev| [8.2 8.2 7.7 7.7 7.5 7.5] angles(deg) [  52.  -52.  104. -104.   69.  -69.  157. -157.  180.   32.] frac energy real part 0.46441332
L0H3 sym-cos 0.192 |q|2.8 |k|3.0 top|ev| [2.1 0.6 0.3 0.3 0.1 0.1] angles(deg) [   0.    0.   72.  -72.  163. -163.  180.  170. -170.  180.] frac energy real part 0.9740525
L0H4 sym-cos -0.239 |q|1.7 |k|1.4 top|ev| [0.3 0.3 0.1 0.  0.  0. ] angles(deg) [ 135. -135.  180.  180.  128. -128.  180.   63.  -63.  119.] frac energy real part 0.5098397
L0H5 sym-cos -0.179 |q|1.4 |k|1.1 top|ev| [0.2 0.  0.  0.  0.  0. ] angles(deg) [ 180.   68.  -68.  166. -166.  115. -115.   55.  -55.  156.] frac energy real part 0.984254
L1H0 sym-cos -0.182 |q|4.9 |k|5.7 top|ev| [0.9 0.9 0.6 0.6 0.6 0.4] angles(deg) [  75.  -75.  180.  157. -157.  173. -173.  180.  180.  158.] frac energy real part 0.55426276
L1H1 sym-cos -0.205 |q|6.2 |k|7.3 top|ev| [1.6 1.6 1.2 0.9 0.9 0.9] angles(deg) [  70.  -70.  180.  167. -167.  177. -177.  180.  171. -171.] frac energy real part 0.5776393
L1H2 sym-cos -0.236 |q|6.7 |k|8.2 top|ev| [2.1 2.1 1.3 1.2 1.2 1.1] angles(deg) [  69.  -69.  180.  155. -155.  176. -176.  164. -164.  121.] frac energy real part 0.5603531
L1H3 sym-cos -0.202 |q|6.2 |k|7.4 top|ev| [1.8 1.8 1.4 0.9 0.9 0.9] angles(deg) [  65.  -65.  180.  176. -176.  180.  114. -114.  173. -173.] frac energy real part 0.5670367
L1H4 sym-cos -0.209 |q|6.3 |k|7.6 top|ev| [1.8 1.8 1.1 1.1 1.  0.8] angles(deg) [  74.  -74.  176. -176.  180.  180.  147. -147.  180.  161.] frac energy real part 0.5499402
L1H5 sym-cos -0.171 |q|6.2 |k|7.6 top|ev| [1.9 1.9 1.  1.  0.8 0.8] angles(deg) [  68.  -68.  174. -174.  178. -178.  143. -143.  157. -157.] frac energy real part 0.48711398
L2H0 sym-cos -0.001 |q|11.9 |k|11.9 top|ev| [2.5 2.5 1.5 1.5 1.1 1.1] angles(deg) [  80.  -80.   79.  -79.   34.  -34.  128. -128.  179. -179.] frac energy real part 0.3436858
L2H1 sym-cos 0.034 |q|11.3 |k|11.0 top|ev| [1.2 1.2 1.  1.  0.9 0.9] angles(deg) [  79.  -79.    5.   -5.   19.  -19.  140. -140.   64.  -64.] frac energy real part 0.55666584
L2H2 sym-cos -0.093 |q|9.9 |k|9.7 top|ev| [1.4 1.4 0.8 0.7 0.7 0.7] angles(deg) [  78.  -78.  180.   39.  -39.  170. -170.  170. -170.  119.] frac energy real part 0.47653875
L2H3 sym-cos -0.045 |q|11.4 |k|11.5 top|ev| [2.5 2.5 0.9 0.9 0.9 0.9] angles(deg) [  77.  -77.  132. -132.  158. -158. -177.  177. -164.  164.] frac energy real part 0.3861011
L2H4 sym-cos -0.027 |q|12.0 |k|12.0 top|ev| [2.5 2.5 1.  1.  1.  1. ] angles(deg) [  79.  -79.   53.  -53.  132. -132.  180. -166.  166.  113.] frac energy real part 0.3819992
L2H5 sym-cos -0.041 |q|11.7 |k|11.6 top|ev| [2.4 2.4 1.1 1.  1.  1. ] angles(deg) [  81.  -81.  180. -123.  123.  163. -163.   65.  -65.  125.] frac energy real part 0.3769513
L3H0 sym-cos 0.199 |q|11.9 |k|11.5 top|ev| [1.5 1.3 1.3 1.3 1.3 1.2] angles(deg) [  0.  29. -29.   7.  -7.  21. -21.  14. -14.   2.] frac energy real part 0.7080061
L3H1 sym-cos 0.440 |q|10.7 |k|10.7 top|ev| [1.5 1.5 1.5 1.5 1.4 1.4] angles(deg) [-21.  21. -19.  19. -29.  29.  11. -11.  19. -19.] frac energy real part 0.8568881
L3H2 sym-cos 0.019 |q|9.6 |k|9.5 top|ev| [1.1 1.1 0.7 0.7 0.7 0.7] angles(deg) [  77.  -77.   26.  -26.  113. -113.    0.  180.   28.  -28.] frac energy real part 0.47917897
L3H3 sym-cos 0.009 |q|10.0 |k|9.9 top|ev| [1.3 1.3 0.8 0.8 0.8 0.7] angles(deg) [  82.  -82.    0.  104. -104.   25.  -25.    0.   41.  -41.] frac energy real part 0.41074657
L3H4 sym-cos 0.063 |q|10.9 |k|10.5 top|ev| [0.9 0.9 0.9 0.9 0.8 0.8] angles(deg) [ 54. -54.  89. -89.  22. -22.   0.  43. -43. -50.] frac energy real part 0.49717027
L3H5 sym-cos 0.016 |q|7.2 |k|6.9 top|ev| [0.7 0.7 0.6 0.4 0.4 0.3] angles(deg) [  25.  -25.    0.   66.  -66.   58.  -58.   16.  -16. -142.] frac energy real part 0.6288216
[3.26 3.12 3.05 2.93 2.86 2.75 2.66 2.63 2.59 2.42 2.4  2.29 2.23 2.2  1.97 1.9  1.84 1.72 1.51 1.51 1.3  1.13 1.09 1.   0.95 0.85 0.78 0.7  0.64 0.61 0.58 0.53 0.5  0.46 0.44
 0.43 0.42 0.38 0.37 0.36]
[4.47 3.89 3.69 3.43 3.24 3.23 3.16 3.06 2.97 2.8  2.76 2.72 2.47 2.37 2.08 2.03 1.91 1.8  1.6  1.52 1.24 1.16 1.08 0.96 0.92 0.85 0.78 0.69 0.63 0.59 0.58 0.52 0.49 0.47 0.43
 0.39 0.38 0.34 0.33 0.29]
[8.23 8.2  7.94 7.8  7.66 7.62 7.49 7.18 6.92 6.79 6.37 6.1  5.74 5.5  5.09 4.82 4.8  4.42 3.96 3.7  3.4  3.19 3.14 3.04 2.85 2.64 2.31 2.16 2.02 2.01 1.86 1.74 1.66 1.56 1.5
 1.47 1.33 1.29 1.28 1.2 ]
[7.6  7.4  7.28 6.87 6.78 6.61 6.43 6.4  6.33 6.08 5.96 5.88 5.43 5.25 4.91 4.72 4.6  4.18 3.97 3.88 3.54 3.23 2.91 2.85 2.64 2.42 2.12 2.11 1.96 1.69 1.64 1.59 1.53 1.46 1.34
 1.2  1.15 1.13 1.09 1.03]

```

## Call 5

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=200)
s=lambda j:slice(64*j,64*j+64)
D=W['ln1_w.0']
Q=W['W_q.0'][s(2)];K=W['W_k.0'][s(2)]
# shared subspace of q,k head 2 in raw coordinates
U,S,Vt=np.linalg.svd(np.vstack([Q,K]),full_matrices=False)
print('joint sv',S[:40].round(2))
r=26
P=Vt[:r].T # raw-dim basis
print('D on P: rms ln1 weight along P',np.sqrt(((P**2)*(D**2)[:,None]).sum(0)).round(2))
# dims heavy in P?
lev=(P**2).sum(1)
print('corr leverage with D',np.corrcoef(lev,D)[0,1],' top leverage dims D',D[np.argsort(-lev)[:15]])
qP=Q@P; kP=K@P
A=np.linalg.lstsq(kP,qP,rcond=None)[0]  # k A? want K p_{i-1} = a Q p_i -> p_{i-1}=K^+ Q p_i
ev,vec=np.linalg.eig(A)
o=np.argsort(np.abs(np.angle(ev)))
print('A eig abs',np.abs(ev[o]).round(2))
print('A eig deg',np.degrees(np.angle(ev[o])).round(0))
# and form B = qP^T kP eigen
B=qP.T@kP
ev2=np.linalg.eigvals(B);o=np.argsort(np.abs(np.angle(ev2)))
print('B eig abs',np.abs(ev2[o]).round(1));print('B deg',np.degrees(np.angle(ev2[o])).round(0))
# sym vs antisym
print('sym frac',np.linalg.norm((B+B.T)/2)**2/np.linalg.norm(B)**2)
# all heads L0 : project on P
for j in range(6):
    for nm in ['W_q','W_k','W_v']:
        M=W[nm+'.0'][s(j)]
        print(j,nm,'frac energy in P %.3f'%(np.linalg.norm(M@P)**2/np.linalg.norm(M)**2),'abs energy %.1f'%np.linalg.norm(M@P)**2, end=' | ')
    print()
print('fc frac in P',np.linalg.norm(W['W_fc.0']@P)**2/np.linalg.norm(W['W_fc.0'])**2)

```

Output:

```
joint sv [10.93 10.79 10.38 10.18 10.11  9.84  9.62  9.44  9.05  8.91  8.48  8.33  7.62  7.17  6.8   6.48  6.34  6.09  5.95  5.56  5.43  4.98  4.7   4.2   4.02  3.7   3.58  3.16  2.84  2.79  2.65  2.49  2.42
  2.29  2.18  2.13  2.05  2.    1.97  1.91]
D on P: rms ln1 weight along P [0.37 0.37 0.39 0.39 0.38 0.38 0.39 0.39 0.42 0.38 0.44 0.42 0.42 0.41 0.46 0.41 0.42 0.53 0.77 0.54 0.61 0.43 0.37 0.42 0.39 0.41]
corr leverage with D 0.567278169783327  top leverage dims D [1.03 0.98 0.98 0.26 0.63 0.6  0.54 0.49 0.16 0.49 0.37 0.45 0.33 0.41 0.82]
A eig abs [0.91 1.01 1.01 1.01 1.06 1.06 1.08 1.08 0.63 0.63 0.97 0.97 1.06 1.06 1.07 1.07 1.12 1.12 1.15 1.15 1.23 1.23 1.08 1.08 1.2  1.2 ]
A eig deg [   0.    0.  -10.   10.  -31.   31.  -47.   47.  -51.   51.  -72.   72.  -75.   75.   98.  -98. -116.  116. -133.  133. -156.  156. -162.  162. -179.  179.]
B eig abs [ 7.1 31.1 26.  26.  42.4 42.4 14.5 14.5 37.9 37.9 43.6 43.6  8.6  8.6 47.5 47.5 13.4 13.4  9.   9.  23.4 23.4 38.7 38.7 15.4 40.2]
B deg [   0.    0.  -11.   11.   31.  -31.  -51.   51.   60.  -60.   69.  -69.   70.  -70.   99.  -99.  138. -138.  139. -139.  147. -147. -150.  150.  180.  180.]
sym frac 0.4843826
0 W_q frac energy in P 0.186 abs energy 2.0 | 0 W_k frac energy in P 0.121 abs energy 2.1 | 0 W_v frac energy in P 0.290 abs energy 10.6 | 
1 W_q frac energy in P 0.115 abs energy 1.1 | 1 W_k frac energy in P 0.138 abs energy 1.9 | 1 W_v frac energy in P 0.263 abs energy 7.6 | 
2 W_q frac energy in P 0.913 abs energy 867.5 | 2 W_k frac energy in P 0.909 abs energy 728.1 | 2 W_v frac energy in P 0.084 abs energy 5.1 | 
3 W_q frac energy in P 0.251 abs energy 11.7 | 3 W_k frac energy in P 0.285 abs energy 14.9 | 3 W_v frac energy in P 0.108 abs energy 3.2 | 
4 W_q frac energy in P 0.097 abs energy 1.5 | 4 W_k frac energy in P 0.113 abs energy 2.8 | 4 W_v frac energy in P 0.181 abs energy 4.9 | 
5 W_q frac energy in P 0.154 abs energy 1.6 | 5 W_k frac energy in P 0.142 abs energy 2.0 | 5 W_v frac energy in P 0.256 abs energy 9.6 | 
fc frac in P 0.068921596

```

## Call 6

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
Q=W['W_q.0'][s(2)];K=W['W_k.0'][s(2)]
U,S,Vt=np.linalg.svd(np.vstack([Q,K]),full_matrices=False)
r=26
P=Vt[:r].T
A=np.linalg.lstsq(K@P,Q@P,rcond=None)[0]
ev,V=np.linalg.eig(A)
# normalise to unit modulus, keep eigenvectors, stationary covariance C = V V^H with unit vec
phase=ev/np.abs(ev)
Vn=V/np.linalg.norm(V,axis=0,keepdims=True)
Vi=np.linalg.inv(Vn)
C=(Vn@Vn.conj().T).real
def Apow(d): return ((Vn*phase**d)@Vi).real
def kernel(q,k,maxd=40):
    Bf=(P.T@q.T)@(k@P)
    return np.array([np.trace(Bf@Apow(d)@C) for d in range(maxd)])
for l in range(4):
    Dl=W[f'ln1_w.{l}']
    for j in range(6):
        q=W[f'W_q.{l}'][s(j)]*Dl;k=W[f'W_k.{l}'][s(j)]*Dl
        f=kernel(q,k,24)
        B_full=q.T@k
        B_pp=(P.T@q.T)@(k@P)
        print(f'L{l}H{j} PP-form frac {np.linalg.norm(B_pp)/np.linalg.norm(B_full):.2f} kernel',(f/np.abs(f).max()).round(2),' max',round(float(np.abs(f).max()),1),'argmax',f.argmax())

```

Output:

```
L0H0 PP-form frac 0.21 kernel [-1.   -0.59  0.28  0.78  0.87  0.3  -0.51 -0.82 -0.56  0.09  0.81  0.93  0.42 -0.49 -0.96 -0.66  0.19  0.84  0.96  0.24 -0.44 -0.84 -0.63  0.18]  max 0.1 argmax 18
L0H1 PP-form frac 0.22 kernel [-1.   -0.38  0.31  0.34  0.14 -0.14 -0.76 -0.45 -0.11  0.24  0.41  0.33 -0.02 -0.46 -0.47 -0.19  0.32  0.86  0.64 -0.08 -0.37 -0.33 -0.32  0.51]  max 0.0 argmax 17
L0H2 PP-form frac 0.67 kernel [ 0.1   1.    0.12 -0.   -0.24 -0.01  0.    0.1   0.14  0.1   0.1   0.03 -0.01 -0.17 -0.1   0.46  0.19  0.12 -0.65 -0.16 -0.06  0.04  0.06  0.29]  max 94.6 argmax 1
L0H3 PP-form frac 0.66 kernel [ 0.81  0.78  0.02 -0.79 -0.97 -0.46  0.42  1.    0.8  -0.   -0.77 -0.97 -0.4   0.42  0.96  0.79  0.01 -0.65 -0.96 -0.51  0.43  0.98  0.75  0.06]  max 2.4 argmax 7
L0H4 PP-form frac 0.14 kernel [-0.69  0.15  0.81  0.59  0.13 -0.56 -0.93 -0.53  0.12  0.68  0.74  0.09 -0.61 -1.   -0.71  0.24  0.77  0.78  0.28 -0.64 -0.83 -0.51  0.09  0.76]  max 0.1 argmax 2
L0H5 PP-form frac 0.27 kernel [-0.99 -0.92 -0.18  0.54  0.95  0.51 -0.36 -0.95 -0.89 -0.25  0.6   0.94  0.5  -0.28 -0.94 -0.93 -0.2   0.66  1.    0.57 -0.2  -0.92 -0.89 -0.14]  max 0.1 argmax 18
L1H0 PP-form frac 0.08 kernel [-0.3   0.79  0.67  0.38 -0.25 -0.58 -0.68  0.18  0.43  0.72  0.32 -0.31 -0.26 -0.29 -0.26  0.52  0.51  0.44 -0.18 -0.45 -0.46  0.17  0.33  1.  ]  max 0.3 argmax 23
L1H1 PP-form frac 0.08 kernel [-0.23  0.42  0.47  0.25 -0.15 -0.28 -0.75 -0.61  0.02  0.99  0.58 -0.03 -0.52 -0.6  -0.06  0.62  1.    0.35 -0.   -0.1  -0.09 -0.13  0.17  0.88]  max 0.4 argmax 16
L1H2 PP-form frac 0.07 kernel [-1.    0.17  0.12 -0.24 -0.43 -0.72 -0.63  0.2   0.5   0.41 -0.64 -0.58 -0.21 -0.14 -0.02  0.31  0.34  0.58 -0.04 -0.31  0.17  0.06  0.47  0.87]  max 0.4 argmax 23
L1H3 PP-form frac 0.07 kernel [-0.77  0.43  0.16  0.13 -0.6  -1.   -0.58  0.03  0.32  0.68 -0.32 -0.37 -0.18 -0.19 -0.32  0.36  0.59  0.79 -0.12 -0.33 -0.05  0.15  0.78  0.98]  max 0.5 argmax 23
L1H4 PP-form frac 0.08 kernel [-0.51  0.46  0.65  0.35 -0.19 -0.85 -0.72  0.15  0.52  0.85  0.08 -0.31 -0.42 -0.2  -0.18  0.29  0.66  0.66 -0.06 -0.62 -0.29 -0.11  0.6   1.  ]  max 0.6 argmax 23
L1H5 PP-form frac 0.08 kernel [-0.84  0.51  0.41  0.1  -0.38 -1.   -0.38 -0.06  0.22  0.89 -0.02 -0.72 -0.63 -0.14 -0.16  0.6   0.7   0.79 -0.02 -0.23  0.46  0.2   0.61  0.89]  max 0.3 argmax 23
L2H0 PP-form frac 0.06 kernel [-0.56 -0.05  0.88  0.58  0.18 -0.39 -0.45  0.29  0.84  0.38 -0.13 -0.39  0.44  0.32  0.41 -0.54  0.26  0.64  0.13 -0.64 -0.27 -0.13  0.57  1.  ]  max 0.6 argmax 23
L2H1 PP-form frac 0.06 kernel [ 0.39 -0.01  0.54 -0.1   0.31 -0.38  0.33 -0.38  1.   -0.16  0.3  -0.83  0.8   0.26 -0.05 -0.04  0.58 -0.47  0.26 -0.52 -0.13 -0.17  0.99  0.19]  max 0.4 argmax 8
L2H2 PP-form frac 0.06 kernel [-1.    0.26 -0.22  0.48  0.04 -0.23 -0.53 -0.67  0.06  0.11  0.41  0.1   0.06 -0.26  0.01  0.01  0.18  0.81  0.07  0.12 -0.06 -0.38 -0.07  0.44]  max 0.7 argmax 17
L2H3 PP-form frac 0.06 kernel [-0.9   0.19 -0.29  1.    0.18  0.31 -0.02 -0.46 -0.06  0.36  0.22 -0.08 -0.18  0.05 -0.11  0.52 -0.06  0.94  0.03  0.56 -0.02 -0.03 -0.19 -0.03]  max 0.4 argmax 3
L2H4 PP-form frac 0.06 kernel [-0.62  0.56  0.03 -0.36  0.54 -0.02 -0.65  0.32  0.3   0.66  0.64  0.91 -0.65 -1.    0.35  0.56  0.58  0.46  0.14 -0.52  0.33 -0.62 -0.25  0.55]  max 0.3 argmax 11
L2H5 PP-form frac 0.06 kernel [-1.    0.16  0.39  0.04  0.19 -0.21 -0.38 -0.08  0.73  0.73  0.28 -0.1  -0.01  0.22 -0.53  0.5   0.67  0.84  0.46 -0.31 -0.16 -0.47  0.03  0.83]  max 0.6 argmax 17
L3H0 PP-form frac 0.06 kernel [ 1.   -0.02 -0.04  0.06  0.23 -0.15  0.03  0.21  0.31  0.18  0.09 -0.21 -0.08  0.01  0.22 -0.1  -0.07 -0.68 -0.    0.06 -0.2  -0.22  0.53  0.03]  max 1.5 argmax 0
L3H1 PP-form frac 0.07 kernel [ 1.    0.15  0.13 -0.11 -0.   -0.02  0.11  0.13 -0.01  0.23  0.2   0.2  -0.14 -0.07  0.27  0.18  0.18 -0.52 -0.09 -0.1   0.    0.04  0.33 -0.03]  max 2.9 argmax 0
L3H2 PP-form frac 0.07 kernel [-0.22  0.41  0.52  0.07 -1.   -0.47 -0.18 -0.16  0.66 -0.24  0.25 -0.   -0.46 -0.57 -0.16  0.37  0.3   0.46 -0.33 -0.56 -0.46  0.57  0.35  0.03]  max 0.6 argmax 8
L3H3 PP-form frac 0.07 kernel [ 1.    0.23  0.2  -0.01 -0.11 -0.24  0.36  0.15  0.19  0.28 -0.08 -0.19 -0.57 -0.2  -0.01  0.1   0.47 -0.57 -0.39 -0.18 -0.02 -0.06  0.67 -0.21]  max 0.7 argmax 0
L3H4 PP-form frac 0.07 kernel [ 1.    0.44  0.81  0.45  0.22 -0.24 -0.4   0.18 -0.01 -0.4   0.13  0.2  -0.94 -0.13 -0.04  0.3   0.54 -0.36 -0.46 -0.84 -0.74 -0.5   0.67  0.  ]  max 0.4 argmax 0
L3H5 PP-form frac 0.07 kernel [ 0.66  0.22 -0.65 -0.27 -0.47  0.04 -0.68  0.29 -0.88  0.07 -0.11 -0.84  0.25 -0.44  0.23 -0.4  -0.48 -0.85 -1.    0.41 -0.36  0.46 -0.52  0.41]  max 0.2 argmax 0

```

## Call 7

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
D1=W['ln1_w.0'];D2=W['ln2_w.0']
V=W['W_v.0'];O=W['W_o.0'];Fc=W['W_fc.0']
for j in range(6):
    M=O[:,s(j)]@V[s(j)]*D1
    u,sv,vt=np.linalg.svd(M)
    print('L0 head',j,'o@v*D sv',sv[:45].round(2))
M=O[:,s(2)]@V[s(2)]*D1
u,sv,vt=np.linalg.svd(M)
print(sv[:60].round(2))
Fs=Fc*D2
uF,sF,vF=np.linalg.svd(Fs,full_matrices=False)
print('fc*D2 sv',sF[:60].round(1))
for r in [20,24,26,28,30,34]:
    Tv=vt[:r].T; Tp=u[:,:r]
    print(r,'fc energy frac Tv %.3f Tp %.3f chance %.3f'%(np.linalg.norm(Fs@Tv)**2/np.linalg.norm(Fs)**2,np.linalg.norm(Fs@Tp)**2/np.linalg.norm(Fs)**2,r/384))
    print('   overlap Tv/Tp',np.linalg.svd(Tv.T@Tp,compute_uv=False)[:6].round(2))
# fc top right vectors: how are they explained by Tv (28) / Tp (28)?
Tv=vt[:28].T;Tp=u[:,:28]
for i in range(14):
    print(i,sF[i].round(1),'in Tv %.2f in Tp %.2f'%(np.linalg.norm(Tv.T@vF[i])**2,np.linalg.norm(Tp.T@vF[i])**2))

```

Output:

```
L0 head 0 o@v*D sv [8.07 3.33 2.1  0.67 0.56 0.32 0.26 0.19 0.15 0.14 0.12 0.11 0.1  0.09 0.08 0.07 0.07 0.07 0.06 0.06 0.05 0.05 0.05 0.05 0.04 0.04 0.04 0.04 0.04 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.02 0.02 0.02 0.02 0.02 0.02 0.02
 0.02 0.02]
L0 head 1 o@v*D sv [5.66 3.83 1.06 0.5  0.43 0.27 0.22 0.17 0.15 0.11 0.09 0.09 0.08 0.08 0.07 0.07 0.06 0.06 0.06 0.05 0.05 0.04 0.04 0.04 0.04 0.04 0.04 0.03 0.03 0.03 0.03 0.03 0.03 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02
 0.02 0.02]
L0 head 2 o@v*D sv [0.79 0.72 0.68 0.67 0.64 0.6  0.57 0.56 0.55 0.52 0.52 0.49 0.48 0.47 0.44 0.42 0.4  0.39 0.35 0.3  0.28 0.28 0.26 0.23 0.22 0.2  0.18 0.17 0.16 0.16 0.13 0.12 0.11 0.1  0.09 0.08 0.07 0.06 0.05 0.05 0.04 0.04 0.04
 0.03 0.03]
L0 head 3 o@v*D sv [1.03 0.97 0.62 0.5  0.45 0.23 0.2  0.18 0.17 0.15 0.14 0.12 0.11 0.09 0.09 0.08 0.08 0.07 0.06 0.06 0.05 0.05 0.05 0.05 0.04 0.04 0.04 0.03 0.03 0.03 0.03 0.03 0.03 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02
 0.02 0.01]
L0 head 4 o@v*D sv [3.4  1.64 1.08 0.61 0.38 0.33 0.23 0.2  0.17 0.14 0.13 0.12 0.1  0.1  0.08 0.08 0.07 0.06 0.06 0.06 0.05 0.05 0.05 0.04 0.04 0.04 0.04 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02
 0.02 0.02]
L0 head 5 o@v*D sv [8.33 3.61 1.68 0.77 0.51 0.39 0.34 0.31 0.25 0.19 0.17 0.13 0.11 0.11 0.1  0.09 0.09 0.07 0.07 0.06 0.06 0.06 0.06 0.05 0.05 0.05 0.04 0.04 0.04 0.04 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.02 0.02 0.02 0.02 0.02
 0.02 0.02]
[0.79 0.72 0.68 0.67 0.64 0.6  0.57 0.56 0.55 0.52 0.52 0.49 0.48 0.47 0.44 0.42 0.4  0.39 0.35 0.3  0.28 0.28 0.26 0.23 0.22 0.2  0.18 0.17 0.16 0.16 0.13 0.12 0.11 0.1  0.09 0.08 0.07 0.06 0.05 0.05 0.04 0.04 0.04
 0.03 0.03 0.02 0.02 0.02 0.02 0.02 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01]
fc*D2 sv [18.  10.4  7.6  7.5  7.1  6.8  6.7  6.4  6.4  6.2  6.1  6.1  6.   5.9  5.8  5.6  5.6  5.5  5.4  5.3  5.3  5.1  5.1  5.   4.9  4.8  4.8  4.7  4.6  4.6  4.5  4.3  4.1  4.1  4.1  3.9  3.8  3.8  3.7  3.5  3.5  3.4  3.3
  3.3  3.2  3.2  3.1  3.   2.9  2.8  2.7  2.7  2.6  2.5  2.4  2.3  2.2  2.2  2.2  2.1]
20 fc energy frac Tv 0.139 Tp 0.113 chance 0.052
   overlap Tv/Tp [0.41 0.34 0.29 0.28 0.26 0.23]
24 fc energy frac Tv 0.149 Tp 0.119 chance 0.062
   overlap Tv/Tp [0.51 0.44 0.41 0.39 0.34 0.31]
26 fc energy frac Tv 0.153 Tp 0.121 chance 0.068
   overlap Tv/Tp [0.6  0.56 0.47 0.44 0.4  0.37]
28 fc energy frac Tv 0.157 Tp 0.124 chance 0.073
   overlap Tv/Tp [0.68 0.61 0.57 0.55 0.5  0.45]
30 fc energy frac Tv 0.162 Tp 0.126 chance 0.078
   overlap Tv/Tp [0.71 0.66 0.64 0.57 0.57 0.54]
34 fc energy frac Tv 0.169 Tp 0.136 chance 0.089
   overlap Tv/Tp [0.72 0.69 0.68 0.66 0.64 0.58]
0 18.0 in Tv 0.04 in Tp 0.05
1 10.4 in Tv 0.09 in Tp 0.04
2 7.6 in Tv 0.23 in Tp 0.09
3 7.5 in Tv 0.29 in Tp 0.07
4 7.1 in Tv 0.30 in Tp 0.30
5 6.8 in Tv 0.17 in Tp 0.11
6 6.7 in Tv 0.27 in Tp 0.09
7 6.4 in Tv 0.28 in Tp 0.09
8 6.4 in Tv 0.14 in Tp 0.10
9 6.2 in Tv 0.22 in Tp 0.13
10 6.1 in Tv 0.20 in Tp 0.08
11 6.1 in Tv 0.16 in Tp 0.09
12 6.0 in Tv 0.25 in Tp 0.14
13 5.9 in Tv 0.18 in Tp 0.06

```

## Call 8

```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=220)
Wp=W['W_proj.3']  # 384 x 1536
U,S,Vt=np.linalg.svd(Wp,full_matrices=False)
print(S[:36].round(2))
for r in [24,28,30]:
    C=(Vt[:r]*1).T  # 1536 x r whitened-ish
    ica=FastICA(n_components=r,whiten='unit-variance',max_iter=2000,tol=1e-5,random_state=0)
    Sx=ica.fit_transform(C)
    k=kurtosis(Sx,axis=0)
    sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
    print(r,'kurt',np.sort(k)[::-1].round(1))
    print(r,'skew',sk[np.argsort(-k)].round(1))
print('kurtosis of singular comps',kurtosis(Vt[:36],axis=1).round(1))
print('skew of singular comps',(((Vt[:36].T-Vt[:36].mean(1))/Vt[:36].std(1))**3).mean(0).round(1))
# column norms of proj
cn=np.linalg.norm(Wp,axis=0);print('col norm quantiles',np.quantile(cn,[0,.1,.5,.9,.99,1]).round(2))
rown=np.linalg.norm(W['W_fc.3'],axis=1);print('fc row norm quantiles',np.quantile(rown,[0,.1,.5,.9,.99,1]).round(2))
print('corr col norm/row norm',np.corrcoef(cn,rown)[0,1])

```

Output:

```
[26.31 20.68 19.7  19.26 18.59 18.2  17.86 17.67 17.04 16.75 16.04 15.15 14.76 14.33 14.05 13.5  13.29 12.53 12.4  11.96 11.74 11.59 10.37  8.35  8.06  7.91  7.46  7.17  5.83  4.07  3.22  3.14  2.84  2.8   2.69  2.57]
24 kurt [ 3.8  1.3  1.2  1.1  1.1  1.1  1.1  1.   0.9  0.9  0.8  0.8  0.8  0.5 -0.4 -0.4 -0.5 -0.5 -0.6 -0.6 -0.6 -0.6 -0.6 -0.6]
24 skew [-0.4 -0.3  0.1 -0.4  0.3 -0.2 -0.2  0.3 -0.1  0.  -0.  -0.2  0.1  0.1 -0.2 -0.   0.1 -0.   0.1  0.1  0.1  0.   0.  -0. ]
28 kurt [ 1.3  0.5  0.3  0.3  0.3  0.2  0.2  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.   0.   0.  -0.  -0.  -0.  -0.  -0.1 -0.1 -0.1 -0.1 -0.1 -0.2 -0.2]
28 skew [-0.1 -0.  -0.   0.   0.1 -0.2  0.1 -0.1 -0.  -0.   0.1 -0.1  0.  -0.2 -0.1  0.1 -0.2  0.  -0.1  0.  -0.1 -0.  -0.1  0.1  0.2 -0.1 -0.1 -0.1]
30 kurt [ 0.1  0.1  0.  -0.  -0.  -0.  -0.  -0.  -0.  -0.  -0.  -0.  -0.  -0.  -0.  -0.  -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1]
30 skew [ 0.  -0.1 -0.1  0.   0.  -0.1 -0.1  0.1 -0.1  0.1 -0.1 -0.1 -0.1  0.1 -0.1  0.  -0.1 -0.1  0.1  0.1  0.1 -0.1 -0.1  0.1 -0.1  0.1 -0.1  0.  -0.2 -0.1]
kurtosis of singular comps [ 0.5 -0.1 -0.4  0.   0.6  0.4  0.   0.1  0.1  0.2 -0.1  0.6  0.4  0.   0.4  0.1  0.1  0.4  0.1  0.3  1.   0.2  0.1  0.7 23.6  0.1  0.4  0.1  3.3  3.6  4.2  5.4  0.5  0.2  0.6  0.5]
skew of singular comps [-0.1 -0.1  0.1 -0.1 -0.3 -0.   0.1  0.1  0.1 -0.2 -0.1 -0.  -0.3 -0.1  0.1  0.  -0.1  0.2  0.1 -0.1  0.1 -0.1 -0.2 -0.1  0.6 -0.2  0.2 -0.  -1.1  0.1 -0.8 -0.8 -0.2  0.1 -0.   0. ]
col norm quantiles [1.45 1.81 2.06 2.4  2.88 4.82]
fc row norm quantiles [1.78 1.91 1.98 2.05 2.27 3.13]
corr col norm/row norm 0.671683005444817

[stderr]
/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/sklearn/decomposition/_fastica.py:132: ConvergenceWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
  warnings.warn(
/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/sklearn/decomposition/_fastica.py:132: ConvergenceWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
  warnings.warn(
/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/sklearn/decomposition/_fastica.py:132: ConvergenceWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
  warnings.warn(
```

## Call 9

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
U3,S3,Vt3=np.linalg.svd(W['W_proj.3'],full_matrices=False)
Y=U3[:,:30]
for l in [1,2,3]:
    Dl=W[f'ln1_w.{l}']
    for j in range(6):
        M=W[f'W_o.{l}'][:,s(j)]@W[f'W_v.{l}'][s(j)]*Dl
        u,sv,vt=np.linalg.svd(M)
        O=W[f'W_o.{l}'][:,s(j)]
        print(f'L{l}H{j} sv',sv[:36:5].round(2),'#>.5top',(sv>.5*sv[0]).sum(),'W_o energy in Y %.3f (chance .078)'%(np.linalg.norm(Y.T@O)**2/np.linalg.norm(O)**2),
              'left-top12 in Y',np.round((Y.T@u[:,:12]**1).__pow__(2).sum(0),2))
# proj of layers 0..2 into Y
for l in range(3):
    Pj=W[f'W_proj.{l}'];print('W_proj',l,'energy in Y %.3f'%(np.linalg.norm(Y.T@Pj)**2/np.linalg.norm(Pj)**2))
# how Y read by q,k,v of each layer
for l in range(4):
    for nm in ['W_q','W_k','W_v','W_fc']:
        Mx=W[f'{nm}.{l}']
        print(l,nm,'read energy in Y %.3f'%(np.linalg.norm(Mx@Y)**2/np.linalg.norm(Mx)**2),end=' | ')
    print()

```

Output:

```
L1H0 sv [4.01 2.69 1.82 1.1  0.73 0.46 0.26 0.15] #>.5top 10 W_o energy in Y 0.096 (chance .078) left-top12 in Y [0.25 0.06 0.08 0.09 0.04 0.12 0.1  0.07 0.09 0.06 0.06 0.07]
L1H1 sv [3.02 2.43 2.05 1.69 1.33 1.01 0.66 0.42] #>.5top 17 W_o energy in Y 0.078 (chance .078) left-top12 in Y [0.06 0.08 0.08 0.1  0.14 0.08 0.07 0.08 0.09 0.09 0.09 0.08]
L1H2 sv [2.87 2.22 1.9  1.45 1.03 0.67 0.39 0.23] #>.5top 16 W_o energy in Y 0.071 (chance .078) left-top12 in Y [0.07 0.06 0.06 0.09 0.09 0.06 0.05 0.06 0.05 0.04 0.08 0.09]
L1H3 sv [3.7  2.56 1.95 1.43 0.64 0.37 0.18 0.13] #>.5top 12 W_o energy in Y 0.066 (chance .078) left-top12 in Y [0.06 0.05 0.08 0.06 0.06 0.05 0.04 0.06 0.05 0.07 0.07 0.08]
L1H4 sv [3.08 2.52 2.18 1.81 1.21 0.59 0.35 0.18] #>.5top 18 W_o energy in Y 0.072 (chance .078) left-top12 in Y [0.1  0.06 0.06 0.05 0.07 0.06 0.08 0.06 0.06 0.06 0.07 0.06]
L1H5 sv [3.03 2.53 2.24 1.84 1.35 0.95 0.43 0.22] #>.5top 19 W_o energy in Y 0.074 (chance .078) left-top12 in Y [0.09 0.1  0.06 0.08 0.07 0.05 0.08 0.1  0.06 0.1  0.07 0.09]
L2H0 sv [3.52 3.02 2.84 2.6  2.34 2.04 1.88 1.68] #>.5top 34 W_o energy in Y 0.068 (chance .078) left-top12 in Y [0.09 0.08 0.07 0.05 0.07 0.05 0.05 0.05 0.03 0.07 0.06 0.08]
L2H1 sv [3.91 3.29 3.01 2.69 2.49 2.27 2.04 1.86] #>.5top 33 W_o energy in Y 0.066 (chance .078) left-top12 in Y [0.16 0.06 0.07 0.06 0.07 0.04 0.05 0.05 0.03 0.05 0.07 0.06]
L2H2 sv [5.7  4.04 3.38 3.07 2.72 2.41 2.19 1.9 ] #>.5top 18 W_o energy in Y 0.060 (chance .078) left-top12 in Y [0.05 0.07 0.05 0.06 0.03 0.06 0.06 0.05 0.03 0.07 0.05 0.05]
L2H3 sv [4.47 3.26 2.88 2.65 2.38 2.16 1.98 1.74] #>.5top 24 W_o energy in Y 0.063 (chance .078) left-top12 in Y [0.07 0.07 0.07 0.04 0.04 0.09 0.07 0.06 0.04 0.03 0.07 0.07]
L2H4 sv [3.62 3.07 2.81 2.58 2.41 2.13 1.94 1.74] #>.5top 34 W_o energy in Y 0.065 (chance .078) left-top12 in Y [0.04 0.06 0.06 0.09 0.06 0.09 0.08 0.08 0.04 0.07 0.06 0.07]
L2H5 sv [3.46 3.07 2.81 2.61 2.4  2.14 1.93 1.76] #>.5top 37 W_o energy in Y 0.069 (chance .078) left-top12 in Y [0.07 0.05 0.05 0.04 0.06 0.07 0.06 0.05 0.08 0.08 0.06 0.06]
L3H0 sv [6.41 4.29 3.75 3.28 2.97 2.7  2.41 2.24] #>.5top 16 W_o energy in Y 0.149 (chance .078) left-top12 in Y [0.41 0.36 0.14 0.1  0.13 0.14 0.14 0.08 0.11 0.1  0.14 0.11]
L3H1 sv [10.63  9.31  8.2   7.12  5.85  4.15  2.57  1.75] #>.5top 23 W_o energy in Y 0.434 (chance .078) left-top12 in Y [0.49 0.45 0.42 0.48 0.52 0.49 0.4  0.45 0.49 0.42 0.52 0.56]
L3H2 sv [9.14 5.9  4.66 3.7  3.2  2.71 2.38 2.14] #>.5top 11 W_o energy in Y 0.143 (chance .078) left-top12 in Y [0.46 0.27 0.25 0.08 0.17 0.29 0.1  0.14 0.1  0.08 0.06 0.07]
L3H3 sv [9.52 5.78 4.56 3.74 3.22 2.92 2.53 2.27] #>.5top 8 W_o energy in Y 0.123 (chance .078) left-top12 in Y [0.42 0.2  0.21 0.14 0.04 0.06 0.09 0.12 0.09 0.06 0.09 0.04]
L3H4 sv [7.27 4.65 4.02 3.43 3.16 2.77 2.43 2.18] #>.5top 14 W_o energy in Y 0.124 (chance .078) left-top12 in Y [0.39 0.43 0.05 0.08 0.13 0.05 0.04 0.07 0.06 0.07 0.09 0.04]
L3H5 sv [16.89 12.19  8.33  5.62  3.93  2.99  2.27  1.79] #>.5top 10 W_o energy in Y 0.167 (chance .078) left-top12 in Y [0.41 0.15 0.13 0.15 0.06 0.14 0.23 0.1  0.14 0.07 0.23 0.21]
W_proj 0 energy in Y 0.155
W_proj 1 energy in Y 0.083
W_proj 2 energy in Y 0.116
0 W_q read energy in Y 0.070 | 0 W_k read energy in Y 0.071 | 0 W_v read energy in Y 0.089 | 0 W_fc read energy in Y 0.087 | 
1 W_q read energy in Y 0.107 | 1 W_k read energy in Y 0.193 | 1 W_v read energy in Y 0.101 | 1 W_fc read energy in Y 0.097 | 
2 W_q read energy in Y 0.082 | 2 W_k read energy in Y 0.136 | 2 W_v read energy in Y 0.084 | 2 W_fc read energy in Y 0.093 | 
3 W_q read energy in Y 0.077 | 3 W_k read energy in Y 0.092 | 3 W_v read energy in Y 0.095 | 3 W_fc read energy in Y 0.095 | 

```

## Call 10

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
def top(M,r): 
    u,sv,vt=np.linalg.svd(M,full_matrices=False); return vt[:r].T,sv,u[:,:r]
Dl=W['ln1_w.3']
for j in [0,1,5]:
    M=W['W_o.3'][:,s(j)]@W['W_v.3'][s(j)]*Dl
    print('L3H',j,'o@v*D sv',np.linalg.svd(M,compute_uv=False)[:44].round(1))
    q=W['W_q.3'][s(j)]*Dl;k=W['W_k.3'][s(j)]*Dl
    print('   joint qk sv',np.linalg.svd(np.vstack([q,k]),compute_uv=False)[:44].round(2))
# build subspaces
Tv=top(W['W_v.0'][s(2)]*W['ln1_w.0'],30)[0]
M2=W['W_o.0'][:,s(2)]@W['W_v.0'][s(2)]*W['ln1_w.0']
u2,s2,v2=np.linalg.svd(M2); TvO=v2[:30].T; Tp=u2[:,:30]
Pq=top(np.vstack([W['W_q.0'][s(2)],W['W_k.0'][s(2)]]),26)[0]
Y=np.linalg.svd(W['W_proj.3'],full_matrices=False)[0][:,:30]
Y0=np.linalg.svd(W['W_proj.0'],full_matrices=False)[0][:,:30]
sub={'Tv0':TvO,'Tp0':Tp,'P':Pq,'Y':Y,'Y0':Y0}
for l in [1,2,3]:
    for j in range(6):
        Dl=W[f'ln1_w.{l}']
        sub[f'qk{l}{j}']=top(np.vstack([W[f'W_q.{l}'][s(j)]*Dl,W[f'W_k.{l}'][s(j)]*Dl]),30)[0]
        M=W[f'W_o.{l}'][:,s(j)]@W[f'W_v.{l}'][s(j)]*Dl
        u,sv,vt=np.linalg.svd(M); sub[f'rd{l}{j}']=vt[:30].T
names=list(sub.keys())
import pandas as pd
ref=['Tv0','Tp0','P','Y','Y0']
T=pd.DataFrame({r:[float((np.linalg.svd(sub[n].T@sub[r],compute_uv=False)**2).sum()/30) for n in names] for r in ref},index=names)
print(T.round(2).to_string())
print('chance 30/384=',30/384)

```

Output:

```
L3H 0 o@v*D sv [6.4 5.8 4.9 4.7 4.5 4.3 4.1 4.  3.9 3.8 3.8 3.6 3.5 3.5 3.4 3.3 3.2 3.2 3.  3.  3.  2.9 2.8 2.8 2.7 2.7 2.7 2.6 2.5 2.5 2.4 2.4 2.4 2.3 2.2 2.2 2.2 2.1 2.1 2.  2.  1.9 1.9 1.9]
   joint qk sv [2.24 2.23 2.21 2.2  2.18 2.16 2.12 2.11 2.09 2.06 2.04 2.02 2.02 2.   1.97 1.96 1.95 1.94 1.92 1.91 1.9  1.89 1.87 1.86 1.85 1.83 1.82 1.8  1.79 1.77 1.76 1.75 1.74 1.73 1.72 1.7  1.7  1.68 1.67 1.66 1.65 1.65 1.63
 1.62]
L3H 1 o@v*D sv [10.6 10.3 10.2  9.5  9.4  9.3  9.   9.   8.6  8.4  8.2  7.7  7.6  7.3  7.3  7.1  6.8  6.5  6.5  6.3  5.8  5.7  5.6  5.2  4.3  4.1  4.   3.6  3.3  3.   2.6  2.3  2.2  2.1  1.8  1.8  1.7  1.6  1.6  1.5  1.4  1.3  1.2
  1.2]
   joint qk sv [2.17 2.16 2.13 2.1  2.09 2.06 2.05 2.04 2.   1.99 1.98 1.95 1.94 1.94 1.92 1.9  1.89 1.87 1.85 1.84 1.83 1.81 1.79 1.79 1.77 1.76 1.76 1.74 1.73 1.73 1.71 1.69 1.68 1.67 1.66 1.65 1.63 1.62 1.6  1.59 1.57 1.56 1.55
 1.55]
L3H 5 o@v*D sv [16.9 14.7 13.4 12.9 12.3 12.2 10.8 10.7  9.3  8.9  8.3  7.2  7.1  6.3  5.8  5.6  5.1  4.9  4.6  4.   3.9  3.7  3.4  3.3  3.1  3.   2.8  2.6  2.5  2.4  2.3  2.1  2.   1.9  1.9  1.8  1.7  1.6  1.6  1.6  1.5  1.4  1.3
  1.3]
   joint qk sv [2.09 1.78 1.72 1.61 1.6  1.45 1.4  1.35 1.32 1.3  1.28 1.26 1.24 1.22 1.19 1.18 1.17 1.15 1.15 1.12 1.12 1.11 1.1  1.09 1.07 1.06 1.05 1.03 1.03 1.02 1.02 1.   1.   0.98 0.98 0.97 0.95 0.95 0.94 0.93 0.93 0.92 0.91
 0.9 ]
       Tv0   Tp0     P     Y    Y0
Tv0   1.00  0.12  0.16  0.09  0.12
Tp0   0.12  1.00  0.33  0.08  0.10
P     0.16  0.33  0.87  0.06  0.05
Y     0.09  0.08  0.06  1.00  0.18
Y0    0.12  0.10  0.05  0.18  1.00
qk10  0.17  0.21  0.06  0.12  0.20
rd10  0.13  0.30  0.07  0.10  0.20
qk11  0.19  0.19  0.05  0.12  0.21
rd11  0.18  0.25  0.05  0.11  0.24
qk12  0.22  0.20  0.05  0.13  0.22
rd12  0.18  0.29  0.06  0.10  0.26
qk13  0.20  0.25  0.06  0.12  0.20
rd13  0.17  0.39  0.07  0.10  0.22
qk14  0.20  0.22  0.05  0.12  0.20
rd14  0.19  0.27  0.05  0.10  0.25
qk15  0.18  0.25  0.06  0.12  0.20
rd15  0.23  0.22  0.04  0.11  0.26
qk20  0.10  0.09  0.06  0.10  0.11
rd20  0.10  0.11  0.05  0.09  0.12
qk21  0.09  0.09  0.05  0.10  0.11
rd21  0.09  0.09  0.05  0.08  0.11
qk22  0.09  0.08  0.06  0.10  0.10
rd22  0.09  0.08  0.05  0.09  0.10
qk23  0.09  0.08  0.06  0.10  0.10
rd23  0.09  0.08  0.05  0.08  0.12
qk24  0.09  0.08  0.06  0.10  0.11
rd24  0.10  0.10  0.05  0.08  0.13
qk25  0.10  0.08  0.06  0.10  0.10
rd25  0.09  0.12  0.06  0.08  0.11
qk30  0.09  0.08  0.06  0.08  0.09
rd30  0.08  0.10  0.07  0.13  0.08
qk31  0.08  0.11  0.05  0.05  0.11
rd31  0.15  0.06  0.04  0.11  0.11
qk32  0.08  0.06  0.07  0.09  0.08
rd32  0.08  0.08  0.06  0.08  0.08
qk33  0.08  0.07  0.07  0.08  0.07
rd33  0.08  0.07  0.06  0.09  0.07
qk34  0.08  0.07  0.07  0.08  0.07
rd34  0.08  0.08  0.07  0.08  0.07
qk35  0.07  0.08  0.07  0.08  0.07
rd35  0.07  0.07  0.06  0.08  0.06
chance 30/384= 0.078125

```

## Call 11

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
for l in range(4):
    Dl=W[f'ln1_w.{l}'];b=W[f'ln1_b.{l}'];bh=b/np.linalg.norm(b)
    b2=W[f'ln2_b.{l}'];b2h=b2/np.linalg.norm(b2)
    # fc top vector vs ln2_b
    uf,sf,vf=np.linalg.svd(W[f'W_fc.{l}']*W[f'ln2_w.{l}'],full_matrices=False)
    uf2,sf2,vf2=np.linalg.svd(W[f'W_fc.{l}'],full_matrices=False)
    print(f'L{l} fc top rsv cos with ln2_b: {vf2[0]@b2h:.2f}, {vf2[1]@b2h:.2f}; sign mean of U col0: {uf2[:,0].mean()/uf2[:,0].std():.2f}; |b2|={np.linalg.norm(b2):.2f} |ln1b|={np.linalg.norm(b):.2f} cos(ln1b,ln2b)={bh@b2h:.2f}')
    print('   fc @ ln2_b mean %.3f std %.3f'%((W[f'W_fc.{l}']@b2).mean(),(W[f'W_fc.{l}']@b2).std()))
    for j in range(6):
        q=W[f'W_q.{l}'][s(j)]*Dl;k=W[f'W_k.{l}'][s(j)]*Dl
        Mkq=k@q.T
        ev,vec=np.linalg.eig(Mkq);o=np.argsort(-np.abs(ev));ev=ev[o]
        B=q.T@k   # form
        ub,sb,vb=np.linalg.svd(B)
        qb=W[f'W_q.{l}'][s(j)]@b; kb=W[f'W_k.{l}'][s(j)]@b
        print(f'  H{j} form sv',sb[:5].round(1),'u0.b %.2f v0.b %.2f u1.b %.2f v1.b %.2f'%(ub[:,0]@bh,vb[0]@bh,ub[:,1]@bh,vb[1]@bh),'|Wq b| %.2f |Wk b| %.2f cos %.2f'%(np.linalg.norm(qb),np.linalg.norm(kb),qb@kb/np.linalg.norm(qb)/np.linalg.norm(kb)))

```

Output:

```
L0 fc top rsv cos with ln2_b: -0.21, 0.05; sign mean of U col0: 0.56; |b2|=1.34 |ln1b|=0.56 cos(ln1b,ln2b)=0.05
   fc @ ln2_b mean -0.047 std 0.153
  H0 form sv [0.4 0.1 0.1 0.  0. ] u0.b -0.25 v0.b 0.06 u1.b 0.08 v1.b 0.08 |Wq b| 0.31 |Wk b| 0.18 cos -0.78
  H1 form sv [0.2 0.1 0.  0.  0. ] u0.b 0.13 v0.b 0.01 u1.b -0.24 v1.b 0.00 |Wq b| 0.14 |Wk b| 0.08 cos 0.11
  H2 form sv [11.3 10.2  9.7  9.6  9.3] u0.b -0.05 v0.b -0.02 u1.b 0.07 v1.b -0.08 |Wq b| 0.67 |Wk b| 0.71 cos -0.00
  H3 form sv [2.9 1.  0.6 0.3 0.3] u0.b -0.18 v0.b -0.11 u1.b 0.21 v1.b 0.15 |Wq b| 0.49 |Wk b| 0.33 cos -0.27
  H4 form sv [0.7 0.4 0.1 0.1 0. ] u0.b -0.07 v0.b 0.10 u1.b 0.17 v1.b -0.12 |Wq b| 0.26 |Wk b| 0.13 cos -0.37
  H5 form sv [0.6 0.1 0.  0.  0. ] u0.b -0.21 v0.b 0.19 u1.b 0.03 v1.b 0.03 |Wq b| 0.26 |Wk b| 0.21 cos -0.79
L1 fc top rsv cos with ln2_b: 0.83, -0.02; sign mean of U col0: -3.75; |b2|=2.67 |ln1b|=3.28 cos(ln1b,ln2b)=-0.18
   fc @ ln2_b mean -1.185 std 0.312
  H0 form sv [4.8 3.9 2.  1.6 1.1] u0.b -0.30 v0.b 0.27 u1.b 0.29 v1.b 0.61 |Wq b| 3.86 |Wk b| 9.19 cos 0.51
  H1 form sv [7.1 5.3 3.3 2.1 1.6] u0.b -0.40 v0.b 0.08 u1.b -0.16 v1.b -0.68 |Wq b| 4.97 |Wk b| 11.72 cos 0.39
  H2 form sv [8.  6.9 3.1 2.6 2.1] u0.b 0.42 v0.b 0.16 u1.b 0.02 v1.b -0.65 |Wq b| 4.68 |Wk b| 13.73 cos 0.42
  H3 form sv [7.4 5.9 3.3 2.  1.8] u0.b 0.39 v0.b 0.01 u1.b 0.13 v1.b 0.64 |Wq b| 4.70 |Wk b| 11.70 cos 0.49
  H4 form sv [7.6 6.9 2.9 2.1 1.5] u0.b 0.39 v0.b 0.02 u1.b -0.06 v1.b -0.68 |Wq b| 4.46 |Wk b| 12.67 cos 0.37
  H5 form sv [7.1 5.9 3.  1.8 1.6] u0.b -0.33 v0.b -0.02 u1.b 0.10 v1.b 0.66 |Wq b| 4.20 |Wk b| 11.86 cos 0.39
L2 fc top rsv cos with ln2_b: 0.92, -0.08; sign mean of U col0: -5.46; |b2|=3.01 |ln1b|=2.99 cos(ln1b,ln2b)=-0.38
   fc @ ln2_b mean -1.065 std 0.221
  H0 form sv [5.6 5.  4.8 4.5 4.2] u0.b 0.45 v0.b 0.03 u1.b 0.01 v1.b -0.12 |Wq b| 3.69 |Wk b| 8.96 cos 0.31
  H1 form sv [4.2 3.6 3.5 3.3 3.3] u0.b 0.12 v0.b -0.03 u1.b 0.16 v1.b 0.03 |Wq b| 2.35 |Wk b| 6.71 cos 0.23
  H2 form sv [4.9 4.  3.2 2.5 2.4] u0.b -0.24 v0.b -0.47 u1.b 0.17 v1.b -0.46 |Wq b| 2.29 |Wk b| 7.63 cos 0.35
  H3 form sv [4.5 4.2 3.9 3.5 3.3] u0.b -0.47 v0.b 0.10 u1.b -0.08 v1.b -0.28 |Wq b| 3.00 |Wk b| 9.37 cos 0.22
  H4 form sv [5.  4.8 4.3 4.1 3.7] u0.b -0.46 v0.b -0.27 u1.b 0.10 v1.b -0.59 |Wq b| 3.21 |Wk b| 9.43 cos 0.24
  H5 form sv [5.1 4.3 3.9 3.8 3.7] u0.b 0.43 v0.b -0.03 u1.b 0.12 v1.b 0.63 |Wq b| 3.25 |Wk b| 9.18 cos 0.23
L3 fc top rsv cos with ln2_b: 0.88, -0.03; sign mean of U col0: -4.64; |b2|=2.50 |ln1b|=1.68 cos(ln1b,ln2b)=-0.33
   fc @ ln2_b mean -0.782 std 0.187
  H0 form sv [3.6 3.4 3.3 3.2 3.2] u0.b 0.04 v0.b -0.01 u1.b 0.02 v1.b 0.01 |Wq b| 1.26 |Wk b| 1.87 cos 0.34
  H1 form sv [2.6 2.5 2.5 2.4 2.4] u0.b -0.02 v0.b 0.02 u1.b 0.02 v1.b 0.01 |Wq b| 1.42 |Wk b| 0.88 cos 0.62
  H2 form sv [3.8 3.3 2.8 2.7 2.5] u0.b 0.20 v0.b 0.07 u1.b 0.03 v1.b 0.06 |Wq b| 1.19 |Wk b| 2.74 cos 0.27
  H3 form sv [4.  3.2 2.7 2.4 2.4] u0.b 0.21 v0.b 0.04 u1.b 0.07 v1.b 0.01 |Wq b| 1.21 |Wk b| 3.13 cos 0.21
  H4 form sv [4.2 3.3 3.2 3.  2.9] u0.b 0.12 v0.b 0.02 u1.b -0.07 v1.b 0.00 |Wq b| 1.32 |Wk b| 2.43 cos 0.21
  H5 form sv [2.1 1.9 1.7 1.6 1.5] u0.b 0.11 v0.b 0.02 u1.b 0.03 v1.b 0.04 |Wq b| 1.10 |Wk b| 0.86 cos -0.06

```

## Call 12

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
G={}
for l in range(4):
    Dl=W[f'ln1_w.{l}'];b=W[f'ln1_b.{l}'];bh=b/np.linalg.norm(b)
    Pi=np.eye(384)-np.outer(bh,bh)
    for j in range(6):
        Wq=W[f'W_q.{l}'][s(j)];Wk=W[f'W_k.{l}'][s(j)]
        q=(Wq*Dl)@Pi;k=(Wk*Dl)@Pi
        Mk=k@q.T
        ev=np.linalg.eigvals(Mk);o=np.argsort(-np.abs(ev));ev=ev[o]
        cos=np.trace(Mk)/(np.linalg.norm(q)*np.linalg.norm(k))
        g=(Wk*Dl).T@(Wq@b)   # static key preference in h-delta coordinates
        G[(l,j)]=g
        print(f'L{l}H{j} sym-cos {cos:.3f} top|ev|',np.abs(ev[:8]).round(2),'deg',np.degrees(np.angle(ev[:8])).round(0),'realfrac %.2f'%((ev.real**2).sum()/(np.abs(ev)**2).sum()),'|g| %.1f'%np.linalg.norm(g))
    # compare g
    Gm=np.array([G[(l,j)]/np.linalg.norm(G[(l,j)]) for j in range(6)])
    print(' cos between static key prefs\n',(Gm@Gm.T).round(2))

```

Output:

```
L0H0 sym-cos -0.021 top|ev| [0.07 0.03 0.02 0.01 0.01 0.01 0.01 0.01] deg [   0.  180.  180.  111. -111.  159. -159.  148.] realfrac 0.88 |g| 0.1
L0H1 sym-cos -0.098 top|ev| [0.04 0.04 0.02 0.01 0.01 0.01 0.01 0.01] deg [ 165. -165.    0.  155. -155.  180.  139. -139.] realfrac 0.79 |g| 0.0
L0H2 sym-cos 0.061 top|ev| [8.   8.   7.7  7.7  7.5  7.5  6.76 6.76] deg [  52.  -52.  104. -104.   69.  -69.  157. -157.] realfrac 0.47 |g| 1.1
L0H3 sym-cos 0.184 top|ev| [2.05 0.58 0.26 0.26 0.15 0.15 0.14 0.12] deg [   0.    0.   73.  -73.  163. -163.  180. -170.] realfrac 0.97 |g| 0.4
L0H4 sym-cos -0.234 top|ev| [0.31 0.31 0.05 0.02 0.01 0.01 0.01 0.01] deg [ 135. -135.  180.  180.  126. -126.  180.   63.] realfrac 0.50 |g| 0.1
L0H5 sym-cos -0.166 top|ev| [0.22 0.01 0.01 0.01 0.01 0.01 0.01 0.01] deg [ 180.   70.  -70.  161. -161.  115. -115.   55.] realfrac 0.98 |g| 0.1
 cos between static key prefs
 [[ 1.    0.26  0.05  0.18 -0.12 -0.17]
 [ 0.26  1.   -0.17  0.09 -0.06 -0.43]
 [ 0.05 -0.17  1.    0.45 -0.23  0.02]
 [ 0.18  0.09  0.45  1.   -0.45 -0.3 ]
 [-0.12 -0.06 -0.23 -0.45  1.    0.37]
 [-0.17 -0.43  0.02 -0.3   0.37  1.  ]]
L1H0 sym-cos -0.205 top|ev| [0.74 0.67 0.67 0.43 0.43 0.37 0.37 0.37] deg [ 180.  167. -167.  171. -171.  180.   43.  -43.] realfrac 0.89 |g| 10.9
L1H1 sym-cos -0.227 top|ev| [1.25 1.07 0.87 0.87 0.86 0.7  0.7  0.69] deg [ 180.  180.  170. -170.  180.   34.  -34.  180.] realfrac 0.90 |g| 18.3
L1H2 sym-cos -0.261 top|ev| [1.44 1.25 1.25 1.08 1.08 0.87 0.87 0.73] deg [ 180.  155. -155.  177. -177.  164. -164.  123.] realfrac 0.85 |g| 20.7
L1H3 sym-cos -0.226 top|ev| [1.39 1.21 0.98 0.84 0.84 0.73 0.73 0.67] deg [ 180.  180.    0.  178. -178.  169. -169.  122.] realfrac 0.87 |g| 18.8
L1H4 sym-cos -0.230 top|ev| [1.12 1.09 1.09 0.8  0.8  0.77 0.72 0.72] deg [ 180.  173. -173.  166. -166.    0.  180.  154.] realfrac 0.92 |g| 18.3
L1H5 sym-cos -0.190 top|ev| [1.01 1.01 0.81 0.8  0.8  0.73 0.73 0.73] deg [ 173. -173.  180.   55.  -55.  143. -143.  180.] realfrac 0.80 |g| 14.0
 cos between static key prefs
 [[1.   0.97 0.96 0.96 0.95 0.96]
 [0.97 1.   0.98 0.96 0.97 0.96]
 [0.96 0.98 1.   0.97 0.97 0.97]
 [0.96 0.96 0.97 1.   0.96 0.97]
 [0.95 0.97 0.97 0.96 1.   0.97]
 [0.96 0.96 0.97 0.97 0.97 1.  ]]
L2H0 sym-cos -0.003 top|ev| [1.68 1.68 1.24 1.24 1.13 1.13 1.02 1.02] deg [  78.  -78.   78.  -78.   34.  -34.  125. -125.] realfrac 0.42 |g| 13.6
L2H1 sym-cos 0.034 top|ev| [0.99 0.99 0.93 0.93 0.85 0.85 0.84 0.84] deg [   5.   -5.   20.  -20.  140. -140.   65.  -65.] realfrac 0.60 |g| 6.0
L2H2 sym-cos -0.097 top|ev| [0.9  0.9  0.84 0.68 0.68 0.66 0.66 0.61] deg [  80.  -80.  180.   32.  -32. -171.  171.  149.] realfrac 0.56 |g| 6.0
L2H3 sym-cos -0.047 top|ev| [1.59 1.59 0.91 0.91 0.89 0.89 0.84 0.84] deg [  74.  -74.  132. -132.  159. -159.  177. -177.] realfrac 0.49 |g| 11.7
L2H4 sym-cos -0.028 top|ev| [1.48 1.48 1.03 1.03 0.98 0.98 0.93 0.93] deg [  73.  -73.   54.  -54.  132. -132. -165.  165.] realfrac 0.48 |g| 12.6
L2H5 sym-cos -0.043 top|ev| [1.46 1.46 1.13 1.07 1.07 1.   1.   0.99] deg [  77.  -77.  180. -125.  125. -163.  163.  -68.] realfrac 0.47 |g| 12.1
 cos between static key prefs
 [[1.   0.84 0.88 0.91 0.93 0.91]
 [0.84 1.   0.86 0.84 0.85 0.83]
 [0.88 0.86 1.   0.86 0.88 0.85]
 [0.91 0.84 0.86 1.   0.93 0.91]
 [0.93 0.85 0.88 0.93 1.   0.9 ]
 [0.91 0.83 0.85 0.91 0.9  1.  ]]
L3H0 sym-cos 0.199 top|ev| [1.46 1.29 1.29 1.27 1.27 1.23 1.23 1.17] deg [  0.  29. -29.  -7.   7.  21. -21.  14.] realfrac 0.71 |g| 2.1
L3H1 sym-cos 0.440 top|ev| [1.47 1.47 1.47 1.47 1.36 1.36 1.33 1.33] deg [-22.  22.  19. -19. -29.  29. -11.  11.] realfrac 0.86 |g| 1.9
L3H2 sym-cos 0.018 top|ev| [0.86 0.86 0.7  0.7  0.69 0.69 0.67 0.61] deg [  76.  -76.  114. -114.   25.  -25.    0.  180.] realfrac 0.52 |g| 2.3
L3H3 sym-cos 0.008 top|ev| [0.93 0.93 0.8  0.8  0.75 0.7  0.7  0.67] deg [  83.  -83.  105. -105.    0.   26.  -26.    0.] realfrac 0.45 |g| 2.9
L3H4 sym-cos 0.063 top|ev| [0.93 0.93 0.84 0.84 0.83 0.77 0.77 0.71] deg [ 54. -54.  21. -21.   0. -44.  44. 130.] realfrac 0.52 |g| 2.5
L3H5 sym-cos 0.016 top|ev| [0.67 0.67 0.63 0.39 0.39 0.34 0.34 0.31] deg [ 25. -25.   0.  72. -72.  57. -57.  16.] realfrac 0.63 |g| 1.3
 cos between static key prefs
 [[ 1.    0.08  0.4   0.34  0.36  0.19]
 [ 0.08  1.   -0.05 -0.17 -0.01 -0.03]
 [ 0.4  -0.05  1.    0.7   0.65  0.2 ]
 [ 0.34 -0.17  0.7   1.    0.63  0.26]
 [ 0.36 -0.01  0.65  0.63  1.    0.23]
 [ 0.19 -0.03  0.2   0.26  0.23  1.  ]]

```

## Call 13

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
def gvec(l):
    Dl=W[f'ln1_w.{l}'];b=W[f'ln1_b.{l}']
    gs=[]
    for j in range(6):
        Wq=W[f'W_q.{l}'][s(j)];Wk=W[f'W_k.{l}'][s(j)]
        gs.append((Wk*Dl).T@(Wq@b))
    return np.array(gs)
g1=gvec(1);g2=gvec(2);g3=gvec(3)
gm1=g1.sum(0);gm2=g2.sum(0);gm3=g3.sum(0)
gh1=gm1/np.linalg.norm(gm1);gh2=gm2/np.linalg.norm(gm2);gh3=gm3/np.linalg.norm(gm3)
print('cos g1,g2',gh1@gh2,'g1,g3',gh1@gh3,'g2,g3',gh2@gh3)
# inverse participation
print('IPR-effective dims g1',1/(gh1**4).sum(),'g2',1/(gh2**4).sum())
print('corr g1 with ln1_w.1',np.corrcoef(gh1,W['ln1_w.1'])[0,1],'ln1_b.1',np.corrcoef(gh1,W['ln1_b.1'])[0,1])
print('corr g2 with ln1_b.2',np.corrcoef(gh2,W['ln1_b.2'])[0,1],' ln1_b.1',np.corrcoef(gh2,W['ln1_b.1'])[0,1],' ln2_b.0',np.corrcoef(gh1,W['ln2_b.0'])[0,1],'ln2_b.1',np.corrcoef(gh2,W['ln2_b.1'])[0,1],'ln1_b.0',np.corrcoef(gh1,W['ln1_b.0'])[0,1])
# which L0 path writes along g1?
D0=W['ln1_w.0']
print('--- L0 attention path into g1: sv of g^T O_j V_j D (norm) & cos of output dir')
for j in range(6):
    O=W['W_o.0'][:,s(j)];V=W['W_v.0'][s(j)]
    m=gh1@O@V*D0
    u,sv,vt=np.linalg.svd(O@V*D0)
    print(j,'|g^T M| %.2f'%np.linalg.norm(m),'cos with top-3 left sv',(gh1@u[:,:5]).round(2),'sv',sv[:5].round(2))
print('L0 proj: |g^T W_proj.0| %.2f vs typical row %.2f'%(np.linalg.norm(gh1@W['W_proj.0']),np.linalg.norm(W['W_proj.0'])/np.sqrt(384)))
uP,sP,vP=np.linalg.svd(W['W_proj.0'],full_matrices=False)
print('proj0 sv',sP[:8].round(1),'g1 on left sv',(gh1@uP[:,:8]).round(2))
print('g1 frac within top-30 of proj0',((gh1@uP[:,:30])**2).sum().round(3),'top-60',((gh1@uP[:,:60])**2).sum().round(3))
# L1 path to g2
D1=W['ln1_w.1']
print('--- L1 into g2')
for j in range(6):
    O=W['W_o.1'][:,s(j)];V=W['W_v.1'][s(j)]
    u,sv,vt=np.linalg.svd(O@V*D1)
    print(j,'|g2^T M| %.2f'%np.linalg.norm(gh2@O@V*D1),'cos top-5',(gh2@u[:,:5]).round(2),'sv',sv[:5].round(2))
print('L1 proj: |g2^T W_proj.1| %.2f typical %.2f'%(np.linalg.norm(gh2@W['W_proj.1']),np.linalg.norm(W['W_proj.1'])/np.sqrt(384)))
print('g1 read by fc.1 etc: |W_fc.1 g1|/typical',np.linalg.norm(W['W_fc.1']@gh1)/ (np.linalg.norm(W['W_fc.1'])/np.sqrt(384)))

```

Output:

```
cos g1,g2 0.8454149 g1,g3 0.6354648 g2,g3 0.80107343
IPR-effective dims g1 3.1396706 g2 2.1705868
corr g1 with ln1_w.1 0.1473200148141181 ln1_b.1 0.13492622359611064
corr g2 with ln1_b.2 0.047676173137831096  ln1_b.1 0.019980530331614482  ln2_b.0 0.06533447322747346 ln2_b.1 -0.012051283711291445 ln1_b.0 0.06779126751361834
--- L0 attention path into g1: sv of g^T O_j V_j D (norm) & cos of output dir
0 |g^T M| 4.32 cos with top-3 left sv [-0.45  0.62  0.55  0.06 -0.04] sv [8.07 3.33 2.1  0.67 0.56]
1 |g^T M| 4.07 cos with top-3 left sv [-0.47  0.8  -0.19 -0.03 -0.  ] sv [5.66 3.83 1.06 0.5  0.43]
2 |g^T M| 0.04 cos with top-3 left sv [-0.03 -0.02 -0.02 -0.02  0.01] sv [0.79 0.72 0.68 0.67 0.64]
3 |g^T M| 0.14 cos with top-3 left sv [ 0.09  0.02 -0.09  0.1   0.04] sv [1.03 0.97 0.62 0.5  0.45]
4 |g^T M| 2.06 cos with top-3 left sv [ 0.47 -0.79 -0.2   0.1   0.03] sv [3.4  1.64 1.08 0.61 0.38]
5 |g^T M| 7.60 cos with top-3 left sv [ 0.91 -0.11 -0.28 -0.01  0.02] sv [8.33 3.61 1.68 0.77 0.51]
L0 proj: |g^T W_proj.0| 4.16 vs typical row 2.01
proj0 sv [9.8 7.4 6.3 5.9 5.8 5.7 5.5 5.5] g1 on left sv [-0.39  0.08 -0.1   0.07  0.03 -0.01  0.04  0.02]
g1 frac within top-30 of proj0 0.189 top-60 0.236
--- L1 into g2
0 |g2^T M| 0.77 cos top-5 [-0.07 -0.03 -0.1  -0.   -0.01] sv [4.01 3.55 2.95 2.89 2.79]
1 |g2^T M| 0.23 cos top-5 [ 0.01 -0.04  0.01  0.01 -0.03] sv [3.02 2.8  2.65 2.6  2.55]
2 |g2^T M| 0.16 cos top-5 [-0.01  0.01  0.01  0.03  0.02] sv [2.87 2.74 2.49 2.38 2.32]
3 |g2^T M| 0.19 cos top-5 [-0.01 -0.01  0.02 -0.   -0.03] sv [3.7  3.25 3.1  2.79 2.71]
4 |g2^T M| 0.18 cos top-5 [-0.01  0.   -0.   -0.02  0.02] sv [3.08 2.98 2.82 2.68 2.58]
5 |g2^T M| 0.19 cos top-5 [ 0.01 -0.    0.01  0.02 -0.03] sv [3.03 2.83 2.76 2.67 2.56]
L1 proj: |g2^T W_proj.1| 1.19 typical 3.86
g1 read by fc.1 etc: |W_fc.1 g1|/typical 0.6183101244386086

```

## Call 14

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
def gvec(l):
    Dl=W[f'ln1_w.{l}'];b=W[f'ln1_b.{l}']
    return sum((W[f'W_k.{l}'][s(j)]*Dl).T@(W[f'W_q.{l}'][s(j)]@b) for j in range(6))
gs={l:gvec(l) for l in [1,2,3]}
for l in gs:
    g=gs[l];o=np.argsort(-np.abs(g))[:10]
    print('g',l,'dims',o,'vals',(g[o]/np.linalg.norm(g)).round(2))
dims=sorted(set(np.concatenate([np.argsort(-np.abs(gs[l]))[:6] for l in gs]).tolist()))
print('dims',dims)
for nm in ['ln1_w','ln1_b','ln2_w','ln2_b']:
    for l in range(4):
        v=W[f'{nm}.{l}']
        print(nm,l,v[dims].round(2),' median %.2f'%np.median(v))
# outlier dims for LN weights
for nm in ['ln1_w','ln2_w']:
    for l in range(4):
        v=W[f'{nm}.{l}'];m=np.median(v);mad=np.median(np.abs(v-m))
        z=(v-m)/(1.48*mad)
        o=np.argsort(-np.abs(z))[:8]
        print(nm,l,'outlier dims',o,'vals',v[o].round(2),'z',z[o].round(1))
# row-norms of the raw dims in W_o,W_proj (write strength per raw dim)
for l in range(4):
    wo=np.linalg.norm(W[f'W_o.{l}'],axis=1);wp=np.linalg.norm(W[f'W_proj.{l}'],axis=1)
    print(l,'W_o row norm top',np.argsort(-wo)[:6],np.sort(wo)[::-1][:6].round(1),'median',np.median(wo).round(2),' W_proj top',np.argsort(-wp)[:6],np.sort(wp)[::-1][:6].round(1),'median',np.median(wp).round(2))
    rq=np.linalg.norm(W[f'W_k.{l}'],axis=0);rv=np.linalg.norm(W[f'W_v.{l}'],axis=0);rf=np.linalg.norm(W[f'W_fc.{l}'],axis=0)
    print('  W_k col top',np.argsort(-rq)[:6],np.sort(rq)[::-1][:6].round(1),'med',np.median(rq).round(2),' W_v col top',np.argsort(-rv)[:6],np.sort(rv)[::-1][:6].round(1),'med',np.median(rv).round(2),' W_fc col top',np.argsort(-rf)[:6],np.sort(rf)[::-1][:6].round(1),'med',np.median(rf).round(2))

```

Output:

```
g 1 dims [ 72  78 222 273 254  25 360 197 355 135] vals [-0.63  0.54  0.53 -0.13  0.06 -0.06  0.03  0.03  0.02 -0.02]
g 2 dims [ 78 222  72  25 360 254 114 273 103  19] vals [ 0.74  0.63 -0.16 -0.04  0.04  0.04  0.03 -0.02 -0.02 -0.02]
g 3 dims [ 78 222 103 137 134 187 309  15  98 147] vals [ 0.64  0.53 -0.15 -0.08 -0.08 -0.08  0.08 -0.08 -0.08  0.08]
dims [25, 72, 78, 103, 134, 137, 187, 222, 254, 273, 360]
ln1_w 0 [0.36 0.71 0.72 0.43 0.2  0.12 0.26 0.84 0.33 0.78 0.82]  median 0.14
ln1_w 1 [0.35 0.8  0.78 0.24 0.58 0.54 0.27 1.12 0.34 0.46 0.14]  median 0.45
ln1_w 2 [0.44 0.39 0.91 0.3  0.71 0.57 0.69 1.01 0.47 0.51 0.21]  median 0.72
ln1_w 3 [0.69 0.5  0.76 0.59 0.74 0.76 0.82 0.83 0.7  0.65 0.29]  median 0.76
ln1_b 0 [-0.05 -0.02  0.08 -0.06 -0.02  0.    0.01 -0.04  0.03  0.06 -0.05]  median 0.00
ln1_b 1 [-0.9  -0.77  0.87 -0.51 -0.06  0.17  0.01 -1.47  1.07 -0.66  1.15]  median -0.01
ln1_b 2 [-0.75 -0.51  0.97 -0.61  0.01  0.06  0.02 -1.38  0.82 -0.27  1.05]  median -0.00
ln1_b 3 [-0.14 -0.22  0.55 -0.25  0.02 -0.03 -0.03 -0.61  0.18  0.05  0.65]  median 0.01
ln2_w 0 [0.59 0.25 0.11 0.76 0.33 1.09 0.25 0.05 0.5  0.3  0.46]  median 0.79
ln2_w 1 [ 0.32 -0.    0.    1.02  1.11  1.32  0.64 -0.    0.2   0.06  0.44]  median 1.08
ln2_w 2 [1.14 0.35 0.06 0.89 1.27 1.17 1.23 0.   1.2  1.13 0.73]  median 1.32
ln2_w 3 [1.15 0.85 0.7  0.74 1.31 1.26 1.29 0.44 1.12 1.29 1.08]  median 1.29
ln2_b 0 [-0.09 -0.21  0.27 -0.3   0.04 -0.    0.02 -0.5   0.27 -0.19  0.31]  median 0.01
ln2_b 1 [-0.35  0.   -0.    0.12  0.04 -0.36 -0.06  0.01  0.37 -0.07  0.3 ]  median -0.00
ln2_b 2 [ 0.1  -0.38  0.16  0.23  0.    0.    0.01 -0.   -0.   -0.25  0.16]  median 0.01
ln2_b 3 [-0.1   0.03 -0.04 -0.11 -0.2   0.11  0.16 -0.67  0.07 -0.23 -0.05]  median 0.00
ln1_w 0 outlier dims [162  60 223 311 239 300  35 180] vals [1.03 0.98 0.98 0.94 0.94 0.94 0.94 0.93] z [7.2 6.8 6.8 6.5 6.4 6.4 6.4 6.4]
ln1_w 1 outlier dims [222  72  78 360 302 118 151 351] vals [1.12 0.8  0.78 0.14 0.77 0.71 0.21 0.7 ] z [ 7.3  3.8  3.6 -3.5  3.4  2.8 -2.7  2.7]
ln1_w 2 outlier dims [360 103  72 222  25 307 254  92] vals [0.21 0.3  0.39 1.01 0.44 0.46 0.47 0.49] z [-9.3 -7.5 -6.   5.4 -5.1 -4.6 -4.5 -4.2]
ln1_w 3 outlier dims [360  72 302 226 249 292 343 103] vals [0.29 0.5  0.53 0.55 0.58 0.58 0.59 0.59] z [-11.5  -6.4  -5.6  -5.3  -4.5  -4.5  -4.2  -4.1]
ln2_w 0 outlier dims [151 222 161 164  78 155 366 348] vals [0.   0.05 1.51 1.51 0.11 1.45 0.15 1.42] z [-2.1 -2.   1.9  1.9 -1.8  1.8 -1.7  1.7]
ln2_w 1 outlier dims [ 72 222  78 114 273 254 136  25] vals [-0.   -0.    0.    0.    0.06  0.2   0.21  0.32] z [-8.3 -8.3 -8.3 -8.3 -7.8 -6.8 -6.7 -5.9]
ln2_w 2 outlier dims [222  78  72 360 103 136 273 302] vals [0.   0.06 0.35 0.73 0.89 1.11 1.13 1.14] z [-23.7 -22.6 -17.4 -10.7  -7.8  -3.8  -3.5  -3.4]
ln2_w 3 outlier dims [222  78 103  72 249 360 254 219] vals [0.44 0.7  0.74 0.85 1.05 1.08 1.12 1.14] z [-19.  -13.1 -12.3  -9.8  -5.2  -4.6  -3.8  -3.3]
0 W_o row norm top [222  78 360 103  72 254] [6.5 5.2 4.3 2.8 2.8 1.6] median 0.64  W_proj top [103 222 360 292 249 226] [7.3 4.7 3.  2.6 2.5 2.4] median 1.96
  W_k col top [162  35  60 223 204 299] [3.2 3.  2.9 2.7 2.5 2.5] med 1.35  W_v col top [223 162 300 311  59 351] [1.3 1.3 1.2 1.2 1.1 1.1] med 0.75  W_fc col top [306 164 368 291 316 297] [3.7 3.7 3.5 3.3 3.3 3.2] med 2.26
1 W_o row norm top [360 103 273  81 307 292] [3.6 3.4 1.6 1.6 1.5 1.5] median 1.3  W_proj top [103 360 302 249  94 287] [7.  4.9 4.2 4.1 4.1 4.1] median 3.87
  W_k col top [222  72  78 254 273  25] [9.3 8.4 8.  4.  4.  3.2] med 1.22  W_v col top [101  81 134  70  15 256] [1.8 1.8 1.8 1.8 1.7 1.7] med 1.41  W_fc col top [ 70 198 160 314 241 142] [4.4 4.3 4.3 4.3 4.3 4.3] med 4.1
2 W_o row norm top [103  72 360 254 136 273] [3.3 3.2 2.6 2.2 2.  2. ] median 1.65  W_proj top [103   4 252  96 332 128] [6.5 4.2 4.2 4.2 4.2 4.1] median 3.92
  W_k col top [222  78  72 360 254  25] [8.9 8.7 4.4 3.  2.7 2.5] med 1.8  W_v col top [192  18  29 234 186 118] [1.9 1.9 1.9 1.9 1.9 1.9] med 1.72  W_fc col top [282 198 330 185 265 180] [4.2 4.2 4.2 4.2 4.2 4.2] med 4.08
3 W_o row norm top [222  78 103  72 360 249] [8.9 6.6 6.5 3.2 3.1 2.6] median 1.95  W_proj top [103 249 226 124 327 184] [17.  10.9 10.5  8.8  7.9  7.4] median 3.98
  W_k col top [222  78 136 360 382 123] [5.2 5.  2.  1.9 1.8 1.8] med 1.6  W_v col top [ 97 186 328 237 300  40] [2.4 2.3 2.3 2.3 2.3 2.3] med 2.07  W_fc col top [110  96 157  65 281 233] [4.1 4.1 4.1 4.1 4.1 4.1] med 4.01

```

## Call 15

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
D1=W['ln1_w.0'];D2=W['ln2_w.0']
Q=W['W_q.0'][s(2)];K=W['W_k.0'][s(2)]
_,_,Vt=np.linalg.svd(np.vstack([Q,K]),full_matrices=False);P=Vt[:26].T
hi=D1>0.4; print('n dims ln1_w.0>.4',hi.sum(),'P energy in those',(P[hi]**2).sum()/26)
print('mean ln2_w.0 hi/lo',D2[hi].mean(),D2[~hi].mean())
reads={}
for j in [0,1,4,5,3]:
    O=W['W_o.0'][:,s(j)];V=W['W_v.0'][s(j)]
    M=O@V*D1
    u,sv,vt=np.linalg.svd(M)
    for k in range(3):
        r=vt[k]; w=u[:,k]
        reads[(j,k)]=r
        o=np.argsort(-np.abs(r))[:6]; o2=np.argsort(-np.abs(w))[:6]
        print(f'H{j} comp{k} sv {sv[k]:.2f} read: energy in P {((P.T@r)**2).sum():.3f}, in hi-dims {(r[hi]**2).sum():.3f}, IPR {1/(r**4).sum():.0f}; write top dims {o2} vals {w[o2].round(2)} IPR {1/(w**4).sum():.1f}; read@ln1b {r@W["ln1_b.0"]:.3f}')
keys=list(reads.keys())
R=np.array([reads[k] for k in keys])
print(keys);print((R@R.T).round(2))
# Do the broad heads read the same subspace?

```

Output:

```
n dims ln1_w.0>.4 72 P energy in those 0.3185807
mean ln2_w.0 hi/lo 0.5379447 0.86117566
H0 comp0 sv 8.07 read: energy in P 0.880, in hi-dims 0.987, IPR 17; write top dims [222 360 254  25 197 103] vals [-0.92  0.26  0.14 -0.1   0.08 -0.07] IPR 1.4; read@ln1b -0.023
H0 comp1 sv 3.33 read: energy in P 0.338, in hi-dims 0.966, IPR 19; write top dims [ 78 360 103 222  72 254] vals [ 0.67  0.49 -0.32  0.19 -0.18  0.15] IPR 3.6; read@ln1b -0.206
H0 comp2 sv 2.10 read: energy in P 0.465, in hi-dims 0.951, IPR 20; write top dims [ 72  78 103 360 273  92] vals [-0.54  0.44  0.41 -0.32 -0.11 -0.09] IPR 6.2; read@ln1b 0.006
H1 comp0 sv 5.66 read: energy in P 0.783, in hi-dims 0.983, IPR 18; write top dims [222 360 103 254  25 197] vals [-0.87  0.36 -0.18  0.17 -0.12  0.08] IPR 1.7; read@ln1b -0.049
H1 comp1 sv 3.83 read: energy in P 0.731, in hi-dims 0.986, IPR 18; write top dims [ 78  72 360 222 103 254] vals [ 0.79 -0.36  0.32  0.2  -0.15  0.13] IPR 2.4; read@ln1b -0.158
H1 comp2 sv 1.06 read: energy in P 0.119, in hi-dims 0.917, IPR 26; write top dims [ 72 103 360 222  78 273] vals [ 0.43 -0.42  0.41  0.3  -0.14  0.13] IPR 9.5; read@ln1b -0.057
H4 comp0 sv 3.40 read: energy in P 0.807, in hi-dims 0.982, IPR 17; write top dims [222 360 103 254  25 197] vals [ 0.89 -0.35  0.17 -0.14  0.09 -0.07] IPR 1.6; read@ln1b 0.044
H4 comp1 sv 1.64 read: energy in P 0.606, in hi-dims 0.946, IPR 18; write top dims [ 78  72 360 103 222  25] vals [-0.81  0.35 -0.31  0.18 -0.18  0.11] IPR 2.2; read@ln1b 0.194
H4 comp2 sv 1.08 read: energy in P 0.210, in hi-dims 0.901, IPR 31; write top dims [360 103  72 222  78 273] vals [ 0.48 -0.46  0.45  0.29 -0.16  0.12] IPR 6.9; read@ln1b -0.050
H5 comp0 sv 8.33 read: energy in P 0.789, in hi-dims 0.993, IPR 20; write top dims [222  78  72 360 103 273] vals [ 0.65  0.64 -0.3   0.17 -0.09 -0.08] IPR 2.8; read@ln1b -0.092
H5 comp1 sv 3.61 read: energy in P 0.714, in hi-dims 0.968, IPR 24; write top dims [222  78  72 360 254  25] vals [ 0.7  -0.43  0.32 -0.29 -0.17  0.15] IPR 3.4; read@ln1b 0.112
H5 comp2 sv 1.68 read: energy in P 0.052, in hi-dims 0.935, IPR 24; write top dims [360 103  72 307 254 197] vals [ 0.59 -0.47  0.39 -0.12  0.12  0.12] IPR 5.3; read@ln1b -0.119
H3 comp0 sv 1.03 read: energy in P 0.079, in hi-dims 0.832, IPR 21; write top dims [360  78 103 222 254 197] vals [ 0.66  0.42 -0.37 -0.3   0.14  0.11] IPR 4.1; read@ln1b -0.191
H3 comp1 sv 0.97 read: energy in P 0.095, in hi-dims 0.556, IPR 55; write top dims [187 180 147 339 246 230] vals [-0.18  0.16  0.15 -0.14 -0.13  0.12] IPR 118.9; read@ln1b -0.016
H3 comp2 sv 0.62 read: energy in P 0.061, in hi-dims 0.402, IPR 84; write top dims [ 78 103 187 109  68 356] vals [-0.19  0.15  0.14  0.13  0.13 -0.13] IPR 119.0; read@ln1b -0.184
[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2), (3, 0), (3, 1), (3, 2)]
[[ 1.    0.    0.    0.95 -0.05 -0.2  -0.96 -0.01 -0.21 -0.69 -0.67  0.01 -0.04 -0.04 -0.06]
 [ 0.    1.    0.    0.15  0.81  0.46 -0.08 -0.8   0.36  0.53 -0.48  0.6   0.75 -0.08 -0.15]
 [ 0.    0.    1.   -0.15  0.54 -0.74  0.18 -0.47 -0.78  0.44 -0.43 -0.69 -0.35 -0.05  0.22]
 [ 0.95  0.15 -0.15  1.    0.   -0.   -0.96 -0.02 -0.07 -0.64 -0.67  0.22  0.18 -0.06 -0.07]
 [-0.05  0.81  0.54  0.    1.   -0.    0.09 -0.91 -0.13  0.73 -0.62  0.11  0.41 -0.13 -0.02]
 [-0.2   0.46 -0.74 -0.   -0.    1.    0.   -0.02  0.85  0.05  0.25  0.87  0.59 -0.07 -0.25]
 [-0.96 -0.08  0.18 -0.96  0.09  0.    1.   -0.   -0.    0.71  0.63 -0.21 -0.13 -0.01  0.11]
 [-0.01 -0.8  -0.47 -0.02 -0.91 -0.02 -0.    1.   -0.   -0.63  0.61 -0.18 -0.43  0.04  0.02]
 [-0.21  0.36 -0.78 -0.07 -0.13  0.85 -0.   -0.    1.   -0.03  0.31  0.83  0.56  0.06 -0.29]
 [-0.69  0.53  0.44 -0.64  0.73  0.05  0.71 -0.63 -0.03  1.   -0.   -0.    0.26 -0.01  0.04]
 [-0.67 -0.48 -0.43 -0.67 -0.62  0.25  0.63  0.61  0.31 -0.    1.   -0.   -0.22 -0.07  0.02]
 [ 0.01  0.6  -0.69  0.22  0.11  0.87 -0.21 -0.18  0.83 -0.   -0.    1.    0.75  0.04 -0.26]
 [-0.04  0.75 -0.35  0.18  0.41  0.59 -0.13 -0.43  0.56  0.26 -0.22  0.75  1.   -0.   -0.  ]
 [-0.04 -0.08 -0.05 -0.06 -0.13 -0.07 -0.01  0.04  0.06 -0.01 -0.07  0.04 -0.    1.    0.  ]
 [-0.06 -0.15  0.22 -0.07 -0.02 -0.25  0.11  0.02 -0.29  0.04  0.02 -0.26 -0.    0.    1.  ]]

```

## Call 16

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
D0=W['ln1_w.0']
M2=W['W_o.0'][:,s(2)]@W['W_v.0'][s(2)]*D0
u2,s2,v2=np.linalg.svd(M2)
print(s2[:34].round(2))
for r in [24,28,32]:
    U=u2[:,:r];V=v2[:r].T;S=np.diag(s2[:r])
    print('rank',r,' overlap U,V top',np.linalg.svd(U.T@V,compute_uv=False)[:5].round(2))
    for l in [1,2,3]:
        Dl=W[f'ln1_w.{l}']
        row=[]
        for j in range(6):
            q=W[f'W_q.{l}'][s(j)]*Dl;k=W[f'W_k.{l}'][s(j)]*Dl
            F=q.T@k
            C=V.T@F@U@S      # q reads T, k reads prevT
            C0=V.T@F@V       # q reads T, k reads T
            R=U.T@F@V@S      # q reads prevT , k reads T
            CC=U.T@F@U
            sc=lambda A:np.trace(A)/ (np.linalg.norm(A)*np.sqrt(r)+1e-9)
            row.append((sc(C),sc(C0),sc(R),sc(CC),np.linalg.norm(C),np.linalg.norm(C0),np.linalg.norm(CC)))
        row=np.array(row)
        print(f' L{l} copy(T,prevT)',row[:,0].round(2),'same(T,T)',row[:,1].round(2),'rev',row[:,2].round(2),'prev-prev',row[:,3].round(2))
        print(f'     norms C',row[:,4].round(1),'C0',row[:,5].round(1),'CC',row[:,6].round(1))

```

Output:

```
[0.79 0.72 0.68 0.67 0.64 0.6  0.57 0.56 0.55 0.52 0.52 0.49 0.48 0.47 0.44 0.42 0.4  0.39 0.35 0.3  0.28 0.28 0.26 0.23 0.22 0.2  0.18 0.17 0.16 0.16 0.13 0.12 0.11 0.1 ]
rank 24  overlap U,V top [0.51 0.44 0.41 0.39 0.34]
 L1 copy(T,prevT) [-0.08 -0.    0.13  0.15  0.14  0.05] same(T,T) [-0.29 -0.37 -0.51 -0.38 -0.41 -0.32] rev [-0.33 -0.16  0.12  0.13 -0.06 -0.14] prev-prev [-0.22 -0.09 -0.17 -0.26 -0.23 -0.28]
     norms C [0.3 0.6 0.7 0.5 0.5 0.7] C0 [1.  1.7 1.8 1.5 1.7 1.5] CC [0.7 1.3 1.7 1.7 1.2 1.8]
 L2 copy(T,prevT) [-0.07 -0.09  0.02  0.06 -0.12  0.01] same(T,T) [-0.1  -0.1  -0.1  -0.21 -0.16 -0.17] rev [ 0.01  0.27 -0.1   0.1  -0.03  0.06] prev-prev [-0.09 -0.02 -0.31 -0.21 -0.15 -0.17]
     norms C [0.8 0.6 0.4 0.6 0.7 0.7] C0 [1.6 1.3 1.1 1.3 1.5 1.5] CC [1.6 1.3 0.8 1.1 1.3 1.4]
 L3 copy(T,prevT) [-0.03  0.62  0.05  0.02 -0.07  0.04] same(T,T) [ 0.2   0.18  0.03 -0.05  0.06  0.05] rev [ 0.25 -0.3   0.03  0.08  0.03  0.04] prev-prev [0.26 0.4  0.   0.01 0.06 0.  ]
     norms C [0.6 1.1 0.3 0.4 0.4 0.2] C0 [1.3 1.1 0.9 0.9 0.9 0.4] CC [1.2 1.5 0.6 0.7 0.8 0.4]
rank 28  overlap U,V top [0.68 0.61 0.57 0.55 0.5 ]
 L1 copy(T,prevT) [-0.07 -0.    0.12  0.13  0.13  0.04] same(T,T) [-0.27 -0.34 -0.46 -0.34 -0.37 -0.29] rev [-0.3  -0.15  0.11  0.11 -0.06 -0.13] prev-prev [-0.2  -0.09 -0.16 -0.23 -0.21 -0.26]
     norms C [0.3 0.7 0.7 0.6 0.6 0.7] C0 [1.  1.8 1.9 1.6 1.8 1.6] CC [0.7 1.3 1.7 1.7 1.3 1.9]
 L2 copy(T,prevT) [-0.06 -0.07  0.01  0.05 -0.1   0.01] same(T,T) [-0.1  -0.08 -0.1  -0.2  -0.14 -0.15] rev [ 0.01  0.24 -0.08  0.09 -0.02  0.06] prev-prev [-0.07  0.01 -0.28 -0.19 -0.12 -0.16]
     norms C [0.9 0.6 0.4 0.6 0.7 0.7] C0 [1.8 1.4 1.2 1.4 1.7 1.7] CC [1.7 1.4 1.  1.3 1.5 1.5]
 L3 copy(T,prevT) [-0.02  0.56  0.04  0.02 -0.06  0.04] same(T,T) [ 0.22  0.23  0.02 -0.01  0.08  0.05] rev [ 0.22 -0.26  0.03  0.06  0.03  0.03] prev-prev [ 0.22  0.4  -0.01  0.02  0.06 -0.01]
     norms C [0.7 1.2 0.4 0.4 0.5 0.2] C0 [1.4 1.3 1.  1.  1.  0.4] CC [1.4 1.6 0.7 0.9 1.  0.4]
rank 32  overlap U,V top [0.71 0.68 0.66 0.63 0.58]
 L1 copy(T,prevT) [-0.06 -0.    0.11  0.11  0.11  0.03] same(T,T) [-0.2  -0.28 -0.4  -0.26 -0.32 -0.23] rev [-0.28 -0.14  0.1   0.1  -0.05 -0.12] prev-prev [-0.19 -0.09 -0.15 -0.22 -0.21 -0.24]
     norms C [0.3 0.7 0.8 0.6 0.6 0.8] C0 [1.2 2.  2.2 1.9 2.  1.9] CC [0.8 1.3 1.7 1.7 1.3 1.9]
 L2 copy(T,prevT) [-0.05 -0.06  0.01  0.03 -0.09  0.01] same(T,T) [-0.08 -0.07 -0.08 -0.16 -0.12 -0.13] rev [ 0.01  0.21 -0.07  0.08 -0.02  0.05] prev-prev [-0.05  0.01 -0.25 -0.15 -0.1  -0.13]
     norms C [1.  0.7 0.5 0.6 0.8 0.8] C0 [2.  1.6 1.4 1.7 2.  1.9] CC [1.9 1.6 1.1 1.4 1.6 1.7]
 L3 copy(T,prevT) [-0.02  0.52  0.04  0.02 -0.06  0.03] same(T,T) [ 0.23  0.26  0.02 -0.01  0.09  0.05] rev [ 0.19 -0.23  0.03  0.05  0.02  0.02] prev-prev [ 0.22  0.39 -0.01  0.03  0.08  0.02]
     norms C [0.8 1.2 0.4 0.4 0.5 0.2] C0 [1.6 1.4 1.1 1.2 1.2 0.5] CC [1.6 1.7 0.9 1.  1.1 0.5]

```

## Call 17

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
D0=W['ln1_w.0']
M2=W['W_o.0'][:,s(2)]@W['W_v.0'][s(2)]*D0
u2,s2,v2=np.linalg.svd(M2)
r=28
V=v2[:r].T
T0=V; T1=M2@V
def Mh(l,j): return W[f'W_o.{l}'][:,s(j)]@W[f'W_v.{l}'][s(j)]*W[f'ln1_w.{l}']
def score(A):
    return np.trace(A)/(np.linalg.norm(A)*np.sqrt(A.shape[0])+1e-12)
# eigen structure of L3H1 copy form
Dl=W['ln1_w.3']
q=W['W_q.3'][s(1)]*Dl;k=W['W_k.3'][s(1)]*Dl;F=q.T@k
C=T0.T@F@T1
ev=np.linalg.eigvals(C);o=np.argsort(-ev.real)
print('L3H1 C eig real',ev[o].real.round(2));print('imag',ev[o].imag.round(2))
sym=(C+C.T)/2
print('sym eig',np.linalg.eigvalsh(sym)[::-1].round(2))
print('sym frac',np.linalg.norm(sym)**2/np.linalg.norm(C)**2)
# table: for L1 heads, map of T0 and T1
T={'T0':T0,'T1':T1}
for j in range(6):
    T[f'M{j}T0']=Mh(1,j)@T0
    T[f'M{j}T1']=Mh(1,j)@T1
names=list(T)
for l in [1,2,3]:
    print('layer',l)
    Dl=W[f'ln1_w.{l}']
    for h in range(6):
        q=W[f'W_q.{l}'][s(h)]*Dl;k=W[f'W_k.{l}'][s(h)]*Dl;F=q.T@k
        if l==1:
            rows=['T0','T1']
            cols=['T0','T1']
        else:
            rows=['T0','T1']+[f'M{j}T0' for j in range(6)]
            cols=names
        tab=np.array([[score(T[a].T@F@T[b]) for b in cols] for a in rows])
        i,jj=np.unravel_index(np.argmax(tab),tab.shape)
        print(f' L{l}H{h} best q:{rows[i]} k:{cols[jj]} score {tab[i,jj]:.2f}; q:T0 row',tab[0].round(2))
        if l==3 and h in (0,1): 
            print('    q:T1 row',tab[1].round(2))
print(names)

```

Output:

```
L3H1 C eig real [ 0.41  0.41  0.33  0.28  0.28  0.25  0.25  0.25  0.21  0.21  0.12  0.12  0.11  0.08  0.08  0.04  0.04  0.03  0.02  0.02  0.01  0.01  0.    0.   -0.01 -0.02 -0.02 -0.02]
imag [ 0.1  -0.1   0.    0.09 -0.09  0.04 -0.04  0.    0.01 -0.01  0.05 -0.05  0.   -0.    0.   -0.08  0.08  0.    0.03 -0.03  0.    0.    0.01 -0.01  0.    0.02 -0.02  0.  ]
sym eig [ 0.5   0.39  0.37  0.34  0.32  0.31  0.25  0.23  0.21  0.2   0.18  0.13  0.1   0.09  0.07  0.05  0.03  0.01  0.01  0.01 -0.   -0.02 -0.02 -0.03 -0.03 -0.04 -0.05 -0.07]
sym frac 0.8183517
layer 1
 L1H0 best q:T0 k:T1 score -0.07; q:T0 row [-0.27 -0.07]
 L1H1 best q:T0 k:T1 score -0.00; q:T0 row [-0.34 -0.  ]
 L1H2 best q:T0 k:T1 score 0.12; q:T0 row [-0.46  0.12]
 L1H3 best q:T0 k:T1 score 0.13; q:T0 row [-0.34  0.13]
 L1H4 best q:T0 k:T1 score 0.13; q:T0 row [-0.37  0.13]
 L1H5 best q:T0 k:T1 score 0.04; q:T0 row [-0.29  0.04]
layer 2
 L2H0 best q:M5T0 k:M5T0 score 0.39; q:T0 row [-0.1  -0.06 -0.17  0.04 -0.01 -0.05  0.03 -0.11 -0.14 -0.   -0.03 -0.11 -0.01 -0.02]
 L2H1 best q:M5T0 k:M5T0 score 0.40; q:T0 row [-0.08 -0.07 -0.09 -0.07 -0.13 -0.02  0.05 -0.05  0.08 -0.07 -0.08 -0.02  0.   -0.01]
 L2H2 best q:M4T0 k:M4T1 score 0.14; q:T0 row [-0.1   0.01 -0.02 -0.01  0.04 -0.01  0.03  0.02  0.03  0.01  0.03  0.01  0.05 -0.03]
 L2H3 best q:M5T0 k:M5T0 score 0.25; q:T0 row [-0.2   0.05 -0.02  0.03  0.01  0.   -0.01 -0.02  0.02 -0.02 -0.01 -0.02 -0.07  0.03]
 L2H4 best q:M4T0 k:M4T0 score 0.23; q:T0 row [-0.14 -0.1  -0.07  0.02 -0.03 -0.   -0.01 -0.02 -0.07 -0.   -0.04 -0.06 -0.04 -0.03]
 L2H5 best q:M5T0 k:M5T0 score 0.28; q:T0 row [-0.15  0.01 -0.04 -0.01  0.01  0.02  0.07 -0.08 -0.04 -0.    0.02 -0.04 -0.11  0.03]
layer 3
 L3H0 best q:M5T0 k:M5T0 score 0.43; q:T0 row [ 0.22 -0.02  0.01 -0.06 -0.1   0.06 -0.    0.02 -0.06  0.01 -0.07  0.02 -0.03 -0.03]
    q:T1 row [ 0.22  0.2  -0.04 -0.07  0.04 -0.14  0.03 -0.17 -0.06 -0.15 -0.03 -0.07 -0.15 -0.04]
 L3H1 best q:M5T0 k:M5T0 score 0.58; q:T0 row [ 0.23  0.56  0.22 -0.12  0.26 -0.07  0.37 -0.13  0.23 -0.03  0.25 -0.08  0.14 -0.06]
    q:T1 row [-0.24  0.32  0.26  0.13  0.27  0.35  0.26  0.4   0.1   0.46  0.23  0.3   0.26  0.07]
 L3H2 best q:M0T0 k:M0T0 score 0.12; q:T0 row [ 0.02  0.04  0.01 -0.05 -0.03  0.    0.    0.   -0.01 -0.05 -0.04  0.04 -0.03 -0.01]
 L3H3 best q:M5T0 k:M0T0 score 0.10; q:T0 row [-0.01  0.02 -0.02 -0.04 -0.08  0.01  0.01 -0.08  0.05  0.02  0.03  0.03 -0.01 -0.  ]
 L3H4 best q:M4T0 k:M4T0 score 0.18; q:T0 row [ 0.08 -0.06 -0.03  0.03 -0.04 -0.01  0.01 -0.04 -0.03  0.01 -0.05  0.06  0.03 -0.03]
 L3H5 best q:M1T0 k:M1T0 score 0.10; q:T0 row [ 0.05  0.04  0.07  0.01 -0.02 -0.    0.04  0.04  0.02  0.04 -0.01  0.01  0.04  0.  ]
['T0', 'T1', 'M0T0', 'M0T1', 'M1T0', 'M1T1', 'M2T0', 'M2T1', 'M3T0', 'M3T1', 'M4T0', 'M4T1', 'M5T0', 'M5T1']

```

## Call 18

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
U,S,Vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
from scipy.stats import kurtosis
ipr=1/(U**4).sum(0)
print('sv  ',S[:40].round(1))
print('IPR ',ipr[:40].round(0))
print('kurt',kurtosis(Vt[:40],axis=1).round(1))
print('top raw dim',np.abs(U[:,:40]).argmax(0))
print('top raw val',np.abs(U[:,:40]).max(0).round(2))
# remove the special raw dims and redo
rows=np.linalg.norm(W['W_proj.3'],axis=1);o=np.argsort(-rows)
print('row norms top',o[:20],rows[o[:20]].round(1),'median',np.median(rows).round(2))
for thr in [6,5.2,4.8]:
    keep=rows<thr
    print('thr',thr,'dropped',(~keep).sum(),'sv',np.linalg.svd(W['W_proj.3'][keep],compute_uv=False)[:36].round(1))

```

Output:

```
sv   [26.3 20.7 19.7 19.3 18.6 18.2 17.9 17.7 17.  16.7 16.  15.2 14.8 14.3 14.1 13.5 13.3 12.5 12.4 12.  11.7 11.6 10.4  8.4  8.1  7.9  7.5  7.2  5.8  4.1  3.2  3.1  2.8  2.8  2.7  2.6  2.4  2.3  2.2  2.2]
IPR  [  5. 148. 145. 141. 145. 148. 147. 125. 148. 144. 130. 139. 118. 104. 132. 119. 102. 123. 118. 127. 121. 123. 136. 113. 142. 109. 108. 117.  10.   7.  11.  25.  61.  31.  54.  96. 122.  47. 103.  67.]
kurt [ 0.5 -0.1 -0.4  0.   0.6  0.4  0.   0.1  0.1  0.2 -0.1  0.6  0.4  0.   0.4  0.1  0.1  0.4  0.1  0.3  1.   0.2  0.1  0.7 23.6  0.1  0.4  0.1  3.3  3.6  4.2  5.4  0.5  0.2  0.6  0.5  2.2  0.2  0.4  0.2]
top raw dim [103 327 327 321 164 343 267 370 251  14  78  18 190 260 352 349 196  94  92  26 160  17  76 381 226 152 212 341 327  78 222 103 124 249 360 302  72 360 292 360]
top raw val [0.63 0.15 0.15 0.14 0.14 0.13 0.15 0.16 0.13 0.13 0.15 0.16 0.16 0.22 0.16 0.19 0.2  0.18 0.16 0.15 0.17 0.2  0.14 0.18 0.14 0.16 0.18 0.19 0.51 0.55 0.47 0.36 0.27 0.34 0.33 0.22 0.18 0.31 0.2  0.28]
row norms top [103 249 226 124 327 184 292 302 222 360  78 157 242  92  90 266 374  22 329 137] [17.  10.9 10.5  8.8  7.9  7.4  6.5  6.5  6.2  5.4  5.1  4.7  4.6  4.5  4.5  4.5  4.4  4.4  4.4  4.3] median 3.98
thr 6 dropped 9 sv [20.2 19.3 19.1 18.4 17.9 17.8 17.6 16.9 16.6 15.9 15.  14.7 14.2 13.9 13.3 13.3 12.4 12.2 11.9 11.6 11.5 10.2  8.3  7.9  7.8  7.4  7.1  6.2  3.6  3.2  2.7  2.7  2.5  2.4  2.4  2.3]
thr 5.2 dropped 10 sv [20.2 19.3 19.1 18.4 17.8 17.7 17.6 16.9 16.6 15.9 15.  14.7 14.2 13.9 13.3 13.3 12.4 12.2 11.9 11.6 11.5 10.2  8.3  7.9  7.8  7.4  7.1  5.4  3.6  3.1  2.7  2.6  2.5  2.4  2.4  2.2]
thr 4.8 dropped 11 sv [20.2 19.2 19.1 18.4 17.8 17.6 17.6 16.9 16.6 15.7 15.  14.6 14.2 13.9 13.3 13.2 12.4 12.2 11.9 11.6 11.5 10.2  8.3  7.9  7.8  7.4  7.1  5.4  3.3  2.7  2.6  2.5  2.4  2.4  2.2  2.2]

```

## Call 19

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
U,S,Vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
# hidden loadings: mean vs std
Vm=Vt[:36]
print('t-mean of V comps',(Vm.mean(1)/Vm.std(1)*np.sqrt(1536)).round(1))
print('frac positive',(Vm>0).mean(1).round(2))
sp=np.array([W['W_proj.3'].sum(1)]);print('|sum of columns|',np.linalg.norm(sp).round(1))
u_mean=W['W_proj.3'].mean(1);u_mean/=np.linalg.norm(u_mean);print('mean col dir on U top30',(u_mean@U[:,:32]).round(2))
# Y dense subspace
idx=list(range(1,28))
Y=U[:,idx]
Dl=W['ln1_w.3']
for j in range(6):
    M=W['W_o.3'][:,s(j)]@W['W_v.3'][s(j)]*Dl
    sv=np.linalg.svd(Y.T@M,compute_uv=False)
    print(f'L3H{j} Y^T M sv',sv[:27].round(1))
# also layers' W_proj.2, etc into Y per component
for l in range(3):
    print('L',l,'proj energy per Y comp',(np.linalg.norm(Y.T@W[f'W_proj.{l}'],axis=1)).round(1))
for j in range(6):
    print('L3 W_o head',j,'per comp',(np.linalg.norm(Y.T@W['W_o.3'][:,s(j)],axis=1)).round(1))
print('L3 proj per comp',S[idx].round(1))

```

Output:

```
t-mean of V comps [ -5.9  -0.2   0.6  -0.6  -0.   -0.3   0.3  -0.3  -0.3   1.    1.1  -0.1   0.6  -1.8   0.7  -0.1   1.4  -0.5  -0.5   0.1  -1.4  -0.3   1.   -1.1  -3.1   1.7   0.2  -1.5  -5.2  11.9 -12.    1.6  -2.7   0.3  -0.1   0.5]
frac positive [0.46 0.49 0.5  0.5  0.51 0.5  0.5  0.5  0.49 0.51 0.51 0.51 0.52 0.49 0.5  0.5  0.53 0.49 0.5  0.5  0.48 0.49 0.53 0.5  0.46 0.53 0.49 0.49 0.48 0.61 0.38 0.52 0.5  0.5  0.5  0.51]
|sum of columns| 184.8
mean col dir on U top30 [-0.83 -0.03  0.06 -0.06 -0.   -0.03  0.02 -0.03 -0.02  0.09  0.1  -0.01  0.05 -0.14  0.06 -0.01  0.1  -0.03 -0.03  0.01 -0.09 -0.02  0.05 -0.05 -0.14  0.07  0.01 -0.06 -0.16  0.25 -0.2   0.03]
L3H0 Y^T M sv [2.4 2.3 2.  1.8 1.7 1.6 1.5 1.5 1.4 1.4 1.3 1.3 1.2 1.1 1.1 1.1 1.  0.9 0.8 0.8 0.7 0.7 0.7 0.6 0.6 0.5 0.5]
L3H1 Y^T M sv [7.5 7.1 7.  6.8 6.6 6.5 6.3 6.2 5.9 5.7 5.7 5.5 5.2 5.  4.9 4.8 4.8 4.5 4.4 4.1 3.9 3.7 3.3 3.3 3.1 3.  2.6]
L3H2 Y^T M sv [4.9 2.9 2.3 2.  1.9 1.6 1.6 1.5 1.4 1.3 1.3 1.2 1.1 1.  1.  1.  0.9 0.8 0.8 0.7 0.6 0.6 0.5 0.5 0.4 0.4 0.3]
L3H3 Y^T M sv [3.  2.6 2.1 2.  1.8 1.6 1.5 1.4 1.3 1.3 1.2 1.1 1.1 1.  1.  1.  0.9 0.8 0.8 0.8 0.7 0.6 0.6 0.6 0.5 0.5 0.4]
L3H4 Y^T M sv [2.3 2.  1.8 1.6 1.5 1.5 1.4 1.3 1.3 1.2 1.1 1.1 1.  1.  1.  0.9 0.9 0.8 0.7 0.7 0.7 0.6 0.6 0.5 0.5 0.4 0.4]
L3H5 Y^T M sv [10.2  4.9  4.7  3.7  3.5  3.2  2.9  2.7  2.6  2.2  2.   1.8  1.6  1.5  1.4  1.3  1.2  1.1  1.   1.   0.9  0.8  0.7  0.6  0.6  0.5  0.4]
L 0 proj energy per Y comp [2.4 2.8 2.2 2.5 2.4 2.7 2.7 3.1 2.7 2.6 3.2 2.6 2.8 2.8 2.7 2.3 3.2 2.7 2.2 2.4 2.6 2.7 2.2 2.8 2.1 2.3 2.3]
L 1 proj energy per Y comp [3.9 4.  3.8 3.9 3.9 3.9 3.8 3.9 3.8 3.8 3.8 3.9 3.8 3.9 4.  4.  4.  4.2 4.  4.1 3.9 4.1 4.  3.5 4.1 3.9 3.9]
L 2 proj energy per Y comp [5.2 5.6 4.9 4.9 4.9 5.2 4.9 4.4 5.  4.8 4.7 4.8 4.7 4.7 5.2 5.1 4.7 4.9 4.9 4.8 4.9 4.6 4.2 3.7 4.5 4.  4.1]
L3 W_o head 0 per comp [1.1 0.9 0.7 0.9 0.9 0.9 0.8 0.7 0.9 0.9 0.8 0.9 0.9 0.9 0.9 0.8 0.7 0.9 0.7 1.  0.8 0.8 0.8 0.7 0.8 0.9 0.8]
L3 W_o head 1 per comp [2.1 2.2 2.5 2.4 2.4 2.1 2.3 2.1 2.4 2.3 2.3 2.3 2.3 2.2 2.1 2.2 2.3 2.3 2.2 2.2 2.2 2.1 2.  2.2 1.8 2.  1.8]
L3 W_o head 2 per comp [1.1 0.9 0.8 1.  1.  0.9 0.5 0.8 0.6 0.8 0.8 0.7 0.7 0.7 0.8 0.6 0.7 0.7 0.8 0.8 0.7 0.8 1.  2.1 0.9 0.8 0.7]
L3 W_o head 3 per comp [0.9 0.7 0.8 0.9 0.8 0.9 0.6 0.8 0.7 0.8 0.8 0.8 0.7 0.8 0.8 0.8 0.9 0.9 0.8 0.8 0.8 0.9 0.7 1.  0.8 0.8 0.8]
L3 W_o head 4 per comp [0.9 0.8 0.7 0.8 0.8 0.9 0.6 0.7 0.8 0.7 0.7 0.8 0.7 0.7 0.8 0.6 0.7 0.9 0.8 0.7 0.8 0.7 0.8 0.9 0.8 0.7 0.7]
L3 W_o head 5 per comp [1.  1.  0.8 0.9 1.2 1.1 0.8 0.9 0.9 1.1 1.  0.9 0.9 0.9 1.2 1.  1.  1.2 1.2 1.  1.  1.3 1.5 3.  1.2 1.2 1.1]
L3 proj per comp [20.7 19.7 19.3 18.6 18.2 17.9 17.7 17.  16.7 16.  15.2 14.8 14.3 14.1 13.5 13.3 12.5 12.4 12.  11.7 11.6 10.4  8.4  8.1  7.9  7.5  7.2]

```

## Call 20

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
D3=W['ln1_w.3']
M=W['W_o.3'][:,s(1)]@W['W_v.3'][s(1)]*D3
u,sv,vt=np.linalg.svd(M)
for r in [24,28]:
    R=vt[:r].T
    print('rank',r)
    for l in range(3):
        for nm,A in [('proj',W[f'W_proj.{l}'])]+[(f'O{j}',W[f'W_o.{l}'][:,s(j)]@W[f'W_v.{l}'][s(j)]*W[f'ln1_w.{l}']) for j in range(6)]:
            U,S,Vt=np.linalg.svd(A,full_matrices=False)
            for k in [30,60]:
                B=U[:,:k]
                print(f'  R vs L{l}{nm} top{k}: {np.linalg.norm(B.T@R)**2/r:.3f} (chance {k/384:.3f})',end=';')
        print()
    # fc rows
    for l in range(4):
        U,S,Vt=np.linalg.svd(W[f'W_fc.{l}'],full_matrices=False)
        print(f'  R vs right-sv fc.{l} top30 {np.linalg.norm(Vt[:30]@R)**2/r:.3f}, top60 {np.linalg.norm(Vt[:60]@R)**2/r:.3f}')
    # read by earlier V?
    for l in range(3):
        Vv=W[f'W_v.{l}']*W[f'ln1_w.{l}']
        U,S,Vt=np.linalg.svd(Vv,full_matrices=False)
        print(f'  R vs right-sv W_v.{l} top30 {np.linalg.norm(Vt[:30]@R)**2/r:.3f} top60 {np.linalg.norm(Vt[:60]@R)**2/r:.3f}; sv {S[:6].round(1)} ..{S[28:32].round(1)}..{S[56:62].round(1)}')

```

Output:

```
rank 24
  R vs L0proj top30: 0.105 (chance 0.078);  R vs L0proj top60: 0.458 (chance 0.156);  R vs L0O0 top30: 0.045 (chance 0.078);  R vs L0O0 top60: 0.091 (chance 0.156);  R vs L0O1 top30: 0.039 (chance 0.078);  R vs L0O1 top60: 0.085 (chance 0.156);  R vs L0O2 top30: 0.055 (chance 0.078);  R vs L0O2 top60: 0.107 (chance 0.156);  R vs L0O3 top30: 0.044 (chance 0.078);  R vs L0O3 top60: 0.086 (chance 0.156);  R vs L0O4 top30: 0.043 (chance 0.078);  R vs L0O4 top60: 0.087 (chance 0.156);  R vs L0O5 top30: 0.041 (chance 0.078);  R vs L0O5 top60: 0.094 (chance 0.156);
  R vs L1proj top30: 0.036 (chance 0.078);  R vs L1proj top60: 0.078 (chance 0.156);  R vs L1O0 top30: 0.054 (chance 0.078);  R vs L1O0 top60: 0.123 (chance 0.156);  R vs L1O1 top30: 0.053 (chance 0.078);  R vs L1O1 top60: 0.127 (chance 0.156);  R vs L1O2 top30: 0.050 (chance 0.078);  R vs L1O2 top60: 0.114 (chance 0.156);  R vs L1O3 top30: 0.047 (chance 0.078);  R vs L1O3 top60: 0.112 (chance 0.156);  R vs L1O4 top30: 0.044 (chance 0.078);  R vs L1O4 top60: 0.113 (chance 0.156);  R vs L1O5 top30: 0.046 (chance 0.078);  R vs L1O5 top60: 0.118 (chance 0.156);
  R vs L2proj top30: 0.030 (chance 0.078);  R vs L2proj top60: 0.067 (chance 0.156);  R vs L2O0 top30: 0.037 (chance 0.078);  R vs L2O0 top60: 0.094 (chance 0.156);  R vs L2O1 top30: 0.032 (chance 0.078);  R vs L2O1 top60: 0.087 (chance 0.156);  R vs L2O2 top30: 0.034 (chance 0.078);  R vs L2O2 top60: 0.092 (chance 0.156);  R vs L2O3 top30: 0.030 (chance 0.078);  R vs L2O3 top60: 0.077 (chance 0.156);  R vs L2O4 top30: 0.029 (chance 0.078);  R vs L2O4 top60: 0.082 (chance 0.156);  R vs L2O5 top30: 0.036 (chance 0.078);  R vs L2O5 top60: 0.098 (chance 0.156);
  R vs right-sv fc.0 top30 0.142, top60 0.257
  R vs right-sv fc.1 top30 0.205, top60 0.458
  R vs right-sv fc.2 top30 0.164, top60 0.315
  R vs right-sv fc.3 top30 0.192, top60 0.327
  R vs right-sv W_v.0 top30 0.130 top60 0.217; sv [3.5 2.4 1.5 0.9 0.7 0.7] ..[0.4 0.4 0.3 0.3]..[0.2 0.2 0.2 0.2 0.2 0.2]
  R vs right-sv W_v.1 top30 0.263 top60 0.404; sv [2.5 2.4 2.3 2.2 2.2 2.2] ..[1.5 1.5 1.4 1.4]..[0.8 0.8 0.7 0.7 0.7 0.7]
  R vs right-sv W_v.2 top30 0.130 top60 0.237; sv [2.7 2.6 2.6 2.6 2.5 2.5] ..[2.2 2.1 2.1 2.1]..[1.9 1.9 1.8 1.8 1.8 1.8]
rank 28
  R vs L0proj top30: 0.115 (chance 0.078);  R vs L0proj top60: 0.451 (chance 0.156);  R vs L0O0 top30: 0.046 (chance 0.078);  R vs L0O0 top60: 0.094 (chance 0.156);  R vs L0O1 top30: 0.040 (chance 0.078);  R vs L0O1 top60: 0.088 (chance 0.156);  R vs L0O2 top30: 0.056 (chance 0.078);  R vs L0O2 top60: 0.110 (chance 0.156);  R vs L0O3 top30: 0.044 (chance 0.078);  R vs L0O3 top60: 0.087 (chance 0.156);  R vs L0O4 top30: 0.043 (chance 0.078);  R vs L0O4 top60: 0.088 (chance 0.156);  R vs L0O5 top30: 0.042 (chance 0.078);  R vs L0O5 top60: 0.097 (chance 0.156);
  R vs L1proj top30: 0.042 (chance 0.078);  R vs L1proj top60: 0.087 (chance 0.156);  R vs L1O0 top30: 0.055 (chance 0.078);  R vs L1O0 top60: 0.124 (chance 0.156);  R vs L1O1 top30: 0.053 (chance 0.078);  R vs L1O1 top60: 0.127 (chance 0.156);  R vs L1O2 top30: 0.051 (chance 0.078);  R vs L1O2 top60: 0.118 (chance 0.156);  R vs L1O3 top30: 0.049 (chance 0.078);  R vs L1O3 top60: 0.116 (chance 0.156);  R vs L1O4 top30: 0.047 (chance 0.078);  R vs L1O4 top60: 0.120 (chance 0.156);  R vs L1O5 top30: 0.047 (chance 0.078);  R vs L1O5 top60: 0.121 (chance 0.156);
  R vs L2proj top30: 0.034 (chance 0.078);  R vs L2proj top60: 0.075 (chance 0.156);  R vs L2O0 top30: 0.039 (chance 0.078);  R vs L2O0 top60: 0.100 (chance 0.156);  R vs L2O1 top30: 0.034 (chance 0.078);  R vs L2O1 top60: 0.091 (chance 0.156);  R vs L2O2 top30: 0.039 (chance 0.078);  R vs L2O2 top60: 0.097 (chance 0.156);  R vs L2O3 top30: 0.034 (chance 0.078);  R vs L2O3 top60: 0.083 (chance 0.156);  R vs L2O4 top30: 0.032 (chance 0.078);  R vs L2O4 top60: 0.088 (chance 0.156);  R vs L2O5 top30: 0.040 (chance 0.078);  R vs L2O5 top60: 0.103 (chance 0.156);
  R vs right-sv fc.0 top30 0.143, top60 0.253
  R vs right-sv fc.1 top30 0.182, top60 0.410
  R vs right-sv fc.2 top30 0.150, top60 0.286
  R vs right-sv fc.3 top30 0.176, top60 0.303
  R vs right-sv W_v.0 top30 0.128 top60 0.214; sv [3.5 2.4 1.5 0.9 0.7 0.7] ..[0.4 0.4 0.3 0.3]..[0.2 0.2 0.2 0.2 0.2 0.2]
  R vs right-sv W_v.1 top30 0.241 top60 0.395; sv [2.5 2.4 2.3 2.2 2.2 2.2] ..[1.5 1.5 1.4 1.4]..[0.8 0.8 0.7 0.7 0.7 0.7]
  R vs right-sv W_v.2 top30 0.122 top60 0.226; sv [2.7 2.6 2.6 2.6 2.5 2.5] ..[2.2 2.1 2.1 2.1]..[1.9 1.9 1.8 1.8 1.8 1.8]

```

## Call 21

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
for l in range(4):
    D2=W[f'ln2_w.{l}']
    ML=W[f'W_proj.{l}']@(W[f'W_fc.{l}']*D2)
    u,sv,vt=np.linalg.svd(ML)
    print('L',l,'proj@fc*D sv',sv[:70].round(1))
    # remove fc top (bias direction) influence?
D2=W['ln2_w.0']
ML=W['W_proj.0']@(W['W_fc.0']*D2)
u,sv,vt=np.linalg.svd(ML)
# compare left subspace vs R
D3=W['ln1_w.3']
M31=W['W_o.3'][:,s(1)]@W['W_v.3'][s(1)]*D3
u3,s3,v3=np.linalg.svd(M31);R=v3[:24].T
print('R energy per left sv of proj@fc (L0) first 70',(np.linalg.norm(u[:,:70].T@R,axis=1)**2).round(2))
Up,Sp,Vp=np.linalg.svd(W['W_proj.0'],full_matrices=False)
print('R energy per left sv of proj.0 first 70',(np.linalg.norm(Up[:,:70].T@R,axis=1)**2).round(2))
print('proj0 sv',Sp[:70].round(1))

```

Output:

```
L 0 proj@fc*D sv [126.4  39.9  32.5  30.8  28.   27.1  26.1  25.6  24.   23.9  23.3  22.9  22.4  22.   21.7  20.8  20.6  20.   19.7  19.   18.5  17.8  17.3  17.1  16.5  16.   15.5  15.2  14.8  14.1  13.7  13.   12.4  12.2  11.2  10.9
  10.6  10.5   9.9   9.7   9.4   8.8   8.6   8.5   7.9   7.8   7.5   7.1   6.9   6.8   6.8   6.7   6.4   6.4   6.2   5.8   5.4   5.3   5.1   4.8   4.7   4.5   4.3   4.1   3.9   3.8   3.7   3.6   3.5   3.2]
L 1 proj@fc*D sv [76.2 33.  28.  26.9 25.3 24.5 24.1 23.2 23.1 22.4 22.2 22.  21.4 21.2 21.1 20.4 20.4 19.9 19.5 19.2 18.9 18.7 18.6 18.2 18.1 18.  17.6 17.4 17.3 16.9 16.8 16.3 16.1 16.  15.7 15.6 15.4 15.1 14.9 14.8 14.6 14.5 14.2
 14.1 13.8 13.8 13.7 13.3 13.2 13.1 12.7 12.4 12.3 12.2 12.1 12.  11.6 11.5 11.3 11.1 10.8 10.8 10.6 10.5 10.4 10.4 10.1  9.9  9.7  9.7]
L 2 proj@fc*D sv [68.  29.9 27.2 26.7 26.1 25.8 25.3 25.1 24.5 24.2 23.9 23.7 22.8 22.5 22.1 21.6 21.3 21.  20.6 20.5 20.3 20.2 20.1 19.8 19.6 19.3 18.9 18.7 18.6 18.5 18.4 18.3 18.  17.9 17.8 17.7 17.6 17.5 17.4 17.1 17.1 16.9 16.7
 16.6 16.5 16.3 16.2 16.  15.8 15.8 15.7 15.6 15.4 15.4 15.3 15.2 15.  14.9 14.9 14.8 14.8 14.5 14.5 14.3 14.2 14.1 14.  13.9 13.8 13.8]
L 3 proj@fc*D sv [133.3  55.4  48.8  45.1  38.7  38.2  37.4  36.2  35.7  35.1  34.   33.2  31.9  30.8  30.   29.8  29.4  28.1  26.7  25.4  25.3  24.5  24.1  23.3  22.5  21.6  20.3  19.3  18.7  18.3  17.2  16.8  16.4  15.9  15.2  13.7
  13.1  12.5  12.4  12.1  11.6  11.1  10.8  10.5  10.4  10.3  10.2  10.    9.8   9.3   9.2   9.1   8.9   8.6   8.5   8.4   8.4   8.3   8.1   8.    7.8   7.8   7.8   7.6   7.5   7.4   7.3   7.2   7.1   7. ]
R energy per left sv of proj@fc (L0) first 70 [0.01 0.02 0.02 0.02 0.02 0.02 0.05 0.03 0.06 0.03 0.03 0.03 0.04 0.02 0.05 0.02 0.05 0.03 0.04 0.05 0.06 0.03 0.05 0.06 0.03 0.08 0.12 0.07 0.05 0.1  0.13 0.19 0.27 0.25 0.26 0.25 0.43 0.31 0.39 0.46 0.39 0.45 0.39
 0.35 0.37 0.41 0.28 0.24 0.36 0.35 0.4  0.3  0.33 0.29 0.24 0.31 0.24 0.24 0.3  0.28 0.2  0.25 0.2  0.17 0.18 0.09 0.1  0.19 0.08 0.14]
R energy per left sv of proj.0 first 70 [0.01 0.1  0.04 0.04 0.05 0.04 0.06 0.06 0.04 0.05 0.03 0.06 0.05 0.04 0.09 0.07 0.04 0.09 0.03 0.08 0.04 0.05 0.12 0.08 0.11 0.12 0.24 0.2  0.15 0.31 0.26 0.24 0.25 0.27 0.12 0.21 0.23 0.26 0.34 0.26 0.25 0.24 0.3
 0.29 0.25 0.26 0.28 0.29 0.4  0.39 0.34 0.33 0.35 0.34 0.31 0.28 0.28 0.32 0.3  0.26 0.38 0.3  0.23 0.25 0.15 0.27 0.2  0.17 0.15 0.12]
proj0 sv [9.8 7.4 6.3 5.9 5.8 5.7 5.5 5.5 5.3 5.2 5.2 5.1 5.  4.9 4.9 4.8 4.8 4.7 4.6 4.6 4.6 4.5 4.4 4.4 4.3 4.2 4.1 4.1 4.  4.  3.9 3.8 3.8 3.8 3.8 3.7 3.7 3.6 3.6 3.5 3.5 3.5 3.5 3.4 3.4 3.4 3.3 3.3 3.3 3.3 3.2 3.2 3.1 3.1
 3.1 3.1 3.1 3.  3.  3.  3.  2.9 2.9 2.8 2.8 2.8 2.7 2.7 2.6 2.6]

```

## Call 22

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
D3=W['ln1_w.3']
M31=W['W_o.3'][:,s(1)]@W['W_v.3'][s(1)]*D3
u3,s3,v3=np.linalg.svd(M31)
for r in [24,28]:
    R=v3[:r].T
    for nm,A in [('fc1',W['W_fc.1']),('fc2',W['W_fc.2']),('fc3',W['W_fc.3']),('fc0',W['W_fc.0'])]:
        X=A@R
        print(nm,r,'kurtosis raw comps',kurtosis(X,axis=0).round(1)[:12])
        ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-4,random_state=0,fun='cube')
        Sx=ica.fit_transform(X)
        k=kurtosis(Sx,axis=0);sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
        o=np.argsort(-k)
        print('   ICA kurt',k[o].round(1));print('   skew',sk[o].round(1))
    # L0 proj columns in R
    X=(R.T@W['W_proj.0']).T
    print('proj0 cols in R kurt',kurtosis(X,axis=0).round(1)[:12])
    ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-4,random_state=0,fun='cube')
    Sx=ica.fit_transform(X);k=kurtosis(Sx,axis=0);print('   ICA kurt',np.sort(k)[::-1].round(1))

```

Output:

```
fc1 24 kurtosis raw comps [-0.  -0.2 -0.1 -0.2  0.2 -0.1 -0.   0.1 -0.2 -0.1 -0.1  0. ]

[stderr]
Traceback (most recent call last):
  File "<string>", line 19, in <module>
    Sx=ica.fit_transform(X)
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/sklearn/utils/_set_output.py", line 327, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/sklearn/decomposition/_fastica.py", line 724, in fit_transform
    return self._fit_transform(X, compute_sources=True)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/sklearn/decomposition/_fastica.py", line 671, in _fit_transform
    W, n_iter = _ica_par(X1, **kwargs)
                ~~~~~~~~^^^^^^^^^^^^^^
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/sklearn/decomposition/_fastica.py", line 122, in _ica_par
    W1 = _sym_decorrelation(np.dot(gwtx, X.T) / p_ - g_wtx[:, np.newaxis] * W)
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/sklearn/decomposition/_fastica.py", line 65, in _sym_decorrelation
    s, u = linalg.eigh(np.dot(W, W.T))
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/scipy/_lib/_util.py", line 1138, in wrapper
    return f(*arrays, *other_args, **kwargs)
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/scipy/linalg/_decomp.py", line 487, in eigh
    a1 = _asarray_validated(a, check_finite=check_finite)
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/scipy/_lib/_util.py", line 409, in _asarray_validated
    a = toarray(a)
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/numpy/lib/_function_base_impl.py", line 685, in asarray_chkfinite
    raise ValueError(
        "array must not contain infs or NaNs")
ValueError: array must not contain infs or NaNs
```

## Call 23

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
D3=W['ln1_w.3']
M31=(W['W_o.3'][:,s(1)]@W['W_v.3'][s(1)]*D3).astype(np.float64)
u3,s3,v3=np.linalg.svd(M31)
r=26
R=v3[:r].T
for nm,A in [('fc1',W['W_fc.1']),('fc2',W['W_fc.2']),('fc3',W['W_fc.3']),('fc0',W['W_fc.0'])]:
    X=A.astype(np.float64)@R
    ica=FastICA(n_components=r,whiten='unit-variance',max_iter=1000,tol=1e-3,random_state=0,fun='logcosh')
    Sx=ica.fit_transform(X)
    k=kurtosis(Sx,axis=0);sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
    o=np.argsort(-k)
    print(nm,'ICA kurt',k[o].round(1));print('   skew',sk[o].round(1))
# same for T0 subspace on fc0, and L0 proj@... rows in fc0 top subspace
D2=W['ln2_w.0']
uF,sF,vF=np.linalg.svd((W['W_fc.0']*D2).astype(np.float64),full_matrices=False)
for r in [28,56]:
    X=uF[:,2:r+2]
    ica=FastICA(n_components=r,whiten='unit-variance',max_iter=1000,tol=1e-3,random_state=0,fun='logcosh')
    Sx=ica.fit_transform(X)
    k=kurtosis(Sx,axis=0);sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
    o=np.argsort(-k)
    print('fc0 top',r,'ICA kurt',k[o].round(1));print('   skew',sk[o].round(1))
print('fc0 raw left comps kurt',kurtosis(uF[:,:20],axis=0).round(1))
print('row norm quantiles fc0',np.quantile(np.linalg.norm(W['W_fc.0'],axis=1),[0,.1,.5,.9,1]).round(2))

```

Output:

```
fc1 ICA kurt [ 1.9  1.4  1.3  1.   1.   0.9  0.9  0.9  0.7  0.7  0.7  0.7  0.7  0.2 -0.1 -0.4 -0.4 -0.4 -0.5 -0.5 -0.5 -0.5 -0.5 -0.6 -0.6 -0.7]
   skew [-0.   0.2  0.   0.2 -0.1 -0.  -0.   0.1  0.1  0.   0.1 -0.3  0.3  0.  -0.  -0.   0.   0.  -0.  -0.   0.   0.   0.   0.1  0.  -0. ]
fc2 ICA kurt [ 1.4  1.1  0.9  0.8  0.8  0.8  0.7  0.7  0.5  0.   0.  -0.  -0.  -0.1 -0.3 -0.3 -0.4 -0.5 -0.5 -0.5 -0.5 -0.5 -0.6 -0.6 -0.6 -0.7]
   skew [ 0.1 -0.1  0.1 -0.1 -0.1 -0.1 -0.  -0.  -0.   0.1  0.  -0.   0.   0.   0.  -0.  -0.   0.   0.1 -0.  -0.  -0.1 -0.  -0.  -0.1  0. ]
fc3 ICA kurt [ 4.6  2.8  2.6  1.5  1.3  1.2  1.1  1.   1.   0.9  0.9  0.8  0.8  0.7  0.6  0.6  0.4  0.3  0.2  0.1  0.  -0.1 -0.3 -0.5 -0.6 -0.6]
   skew [ 1.1  0.6  0.3  0.2  0.  -0.1 -0.1 -0.  -0.  -0.  -0.2 -0.1  0.1 -0.  -0.1  0.   0.  -0.  -0.   0.1 -0.  -0.   0.1 -0.   0.1  0. ]
fc0 ICA kurt [161.8 128.4 120.2 104.9 101.1  86.9  81.4  79.7  69.1  66.3  66.   61.6  61.4  57.8  54.9  43.9  43.8  40.9  38.5  37.6  37.5  33.6  27.8  23.   21.9  14.2]
   skew [ 8.8 -8.1  7.1  8.2 -7.8  7.7 -6.7  6.5 -6.6  5.5 -6.2 -6.7  4.8  5.1 -4.9 -3.1 -4.6 -4.2 -2.6 -3.1 -4.2  1.3  3.5  1.1  2.6  0.4]
fc0 top 28 ICA kurt [246.6 183.1 167.8 157.9 139.  137.6 134.8 133.8 122.7 114.9 114.  110.3 110.2 105.9 104.8  92.6  81.6  64.7  61.1  55.6  52.3  44.7  42.5  30.   29.4  24.6  23.   15.7]
   skew [-14.   -9.9  -9.6  10.4  10.3 -10.2   9.9 -10.1   8.7  -8.7  -8.5  -9.    9.2  -9.   -7.9  -8.    7.1   6.6  -5.3   4.5  -5.6  -5.1  -3.5  -2.3   1.3  -1.2   0.2  -3.2]
fc0 top 56 ICA kurt [361.2 300.  238.1 232.4 229.8 226.3 214.5 199.5 197.4 178.4 174.2 173.  167.6 159.8 158.4 157.6 156.1 149.3 146.1 145.5 137.2 132.9 129.6 123.9 119.2 117.3 116.1 115.4 113.  108.8 107.7 104.8 104.   91.6  91.5  91.3
  87.1  86.   79.7  77.1  75.3  74.9  71.6  67.6  61.3  59.7  55.6  54.5  52.5  48.1  43.4  41.8  41.6  31.2  25.7  23.6]
   skew [-14.8 -15.4  12.7  14.2 -12.8 -13.1  12.4  10.8 -12.1  10.4 -11.3  11.1 -11.1 -10.6   9.2  -3.1  10.   10.   10.1  -8.8  -9.9  -9.5  -9.9  10.    7.7  -7.8   3.8   7.9  -8.3   7.8  -7.4   9.   -7.2  -6.6  -7.6  -7.8
   6.5   7.3  -4.7   6.8   6.1   6.4  -3.7  -5.4   6.1  -4.7   5.3  -4.8  -5.5  -4.   -3.4   2.    2.    1.4   2.7   0.4]
fc0 raw left comps kurt [ 9.  15.3 20.9 23.7 42.4 25.1 23.  17.6 22.5 16.3 17.4 13.5 17.8 20.9 14.  25.3  9.9 13.5  9.  16.8]
row norm quantiles fc0 [0.33 0.37 0.42 2.26 4.42]

```

## Call 24

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=2,suppress=True,linewidth=220)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);cn=np.linalg.norm(Pj,axis=0)
print('hist row norm',np.histogram(rn,bins=[0,.3,.5,.7,1,1.5,2,2.5,3,3.5,4,5]))
print('hist col norm',np.histogram(cn,bins=[0,.1,.2,.3,.5,1,1.5,2,3,4,5,8]))
act=np.where(rn>1)[0];print('n active',len(act))
print('corr row/col norm',np.corrcoef(rn,cn)[0,1])
Fa=F[act];Fn=Fa/np.linalg.norm(Fa,axis=1,keepdims=True)
C=Fn@Fn.T
print('cos between active rows: quantiles',np.quantile(C[np.triu_indices(len(act),1)],[.001,.01,.1,.5,.9,.99,.999]).round(2))
# sv of active rows
print('sv active',np.linalg.svd(Fa,compute_uv=False)[:64].round(1))
# fc@ln2_b for active
b2=W['ln2_b.0']
print('active: fc@b mean',(Fa@b2).mean().round(3),'quantiles',np.quantile(Fa@b2,[0,.1,.5,.9,1]).round(2))
for l in range(1,4):
    rn2=np.linalg.norm(W[f'W_fc.{l}'],axis=1);print('L',l,'row-norm quantiles',np.quantile(rn2,[0,.1,.5,.9,1]).round(2))

```

Output:

```
hist row norm (array([   0, 1025,   97,   33,   79,   94,  104,   65,   21,   13,    5]), array([0. , 0.3, 0.5, 0.7, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 5. ]))
hist col norm (array([  0, 448, 534, 116,  40,  51, 201, 144,   1,   1,   0]), array([0. , 0.1, 0.2, 0.3, 0.5, 1. , 1.5, 2. , 3. , 4. , 5. , 8. ]))
n active 381
corr row/col norm 0.9611175811023585
cos between active rows: quantiles [-0.29 -0.16 -0.06  0.04  0.3   0.54  0.93]
sv active [17.9  9.8  8.4  7.2  7.   7.   6.8  6.6  6.4  6.4  6.3  6.1  6.1  6.   5.9  5.8  5.7  5.6  5.5  5.4  5.4  5.2  5.1  5.   5.   4.9  4.8  4.8  4.7  4.5  4.5  4.4  4.4  4.3  4.2  4.1  3.9  3.8  3.7  3.7  3.6  3.6  3.5
  3.4  3.4  3.4  3.3  3.2  3.1  3.1  3.   3.   2.8  2.8  2.6  2.5  2.5  2.4  2.3  2.3  2.2  2.2  2.1  2.1]
active: fc@b mean -0.155 quantiles [-0.87 -0.52 -0.12  0.16  1.04]
L 1 row-norm quantiles [0.39 1.83 2.   2.21 3.37]
L 2 row-norm quantiles [1.72 1.93 2.01 2.09 2.48]
L 3 row-norm quantiles [1.78 1.91 1.98 2.05 3.13]

```

## Call 25

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.cluster.hierarchy import linkage,fcluster
np.set_printoptions(precision=2,suppress=True,linewidth=220)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1)
act=np.where(rn>1)[0]
Fa=F[act]; Pa=Pj[:,act]
Fn=Fa/np.linalg.norm(Fa,axis=1,keepdims=True)
Pn=(Pa/np.linalg.norm(Pa,axis=0,keepdims=True)).T
Z=linkage(Fn,'average',metric='cosine')
for t in [.3,.5,.7,.85]:
    lab=fcluster(Z,t,'distance')
    cnt=np.bincount(lab)[1:]
    print('dist',t,'nclusters',cnt.size,'sizes sorted',np.sort(cnt)[::-1][:50])
lab=fcluster(Z,.5,'distance')
cnt=np.bincount(lab)
o=np.argsort(-cnt)[:45]
cents=[]
for c in o:
    if cnt[c]<2:break
    idx=np.where(lab==c)[0]
    m=Fa[idx].mean(0)
    oc=Pa[:,idx].mean(1)
    # internal cos of outputs
    cc=(Pn[idx]@Pn[idx].T)
    iu=np.triu_indices(len(idx),1)
    cc2=(Fn[idx]@Fn[idx].T)
    print('cl',c,'n',cnt[c],'mean rownorm %.2f'%np.linalg.norm(Fa[idx],axis=1).mean(),'in cos %.2f'%cc2[iu].mean(),'out cos %.2f'%cc[iu].mean())
# cos row vs col for active
print('corr of row vs col cos (structure)',np.corrcoef((Fn@Fn.T)[np.triu_indices(len(act),1)],(Pn@Pn.T)[np.triu_indices(len(act),1)])[0,1])

```

Output:

```
dist 0.3 nclusters 244 sizes sorted [7 5 5 5 5 5 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2]
dist 0.5 nclusters 140 sizes sorted [11  8  8  7  6  6  6  6  6  6  6  6  6  6  6  6  5  5  5  5  5  5  5  5  5  5  5  5  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  3  3  3  3  3  3]
dist 0.7 nclusters 57 sizes sorted [78 20 12 12 11 10 10 10  9  8  8  8  8  7  7  6  6  6  6  6  6  6  6  6  5  5  5  5  5  5  5  4  4  4  4  4  4  4  4  4  4  3  3  3  3  3  2  2  2  2]
dist 0.85 nclusters 35 sizes sorted [126  39  17  12  12  12  10  10  10   9   8   7   7   6   6   6   6   6   6   6   6   5   5   5   5   4   4   4   4   4   4   3   3   2   2]
cl 67 n 11 mean rownorm 2.04 in cos 0.66 out cos 0.79
cl 61 n 8 mean rownorm 2.04 in cos 0.68 out cos 0.61
cl 79 n 8 mean rownorm 2.35 in cos 0.56 out cos 0.12
cl 32 n 7 mean rownorm 2.95 in cos 0.61 out cos 0.35
cl 63 n 6 mean rownorm 2.19 in cos 0.73 out cos 0.73
cl 62 n 6 mean rownorm 2.62 in cos 0.68 out cos 0.65
cl 71 n 6 mean rownorm 2.10 in cos 0.73 out cos 0.75
cl 58 n 6 mean rownorm 2.76 in cos 0.66 out cos 0.50
cl 56 n 6 mean rownorm 2.48 in cos 0.80 out cos 0.85
cl 136 n 6 mean rownorm 1.52 in cos 0.72 out cos 0.63
cl 64 n 6 mean rownorm 2.07 in cos 0.69 out cos 0.68
cl 49 n 6 mean rownorm 2.96 in cos 0.56 out cos 0.53
cl 76 n 6 mean rownorm 2.32 in cos 0.79 out cos 0.85
cl 50 n 6 mean rownorm 2.57 in cos 0.71 out cos 0.67
cl 74 n 6 mean rownorm 2.37 in cos 0.67 out cos 0.56
cl 36 n 6 mean rownorm 2.22 in cos 0.75 out cos 0.69
cl 55 n 5 mean rownorm 2.75 in cos 0.79 out cos 0.78
cl 37 n 5 mean rownorm 2.67 in cos 0.76 out cos 0.70
cl 39 n 5 mean rownorm 3.06 in cos 0.72 out cos 0.64
cl 41 n 5 mean rownorm 1.34 in cos 0.69 out cos 0.90
cl 43 n 5 mean rownorm 2.71 in cos 0.71 out cos 0.71
cl 46 n 5 mean rownorm 2.86 in cos 0.84 out cos 0.87
cl 47 n 5 mean rownorm 2.04 in cos 0.76 out cos 0.85
cl 65 n 5 mean rownorm 2.63 in cos 0.66 out cos 0.64
cl 19 n 5 mean rownorm 1.73 in cos 0.65 out cos 0.72
cl 69 n 5 mean rownorm 2.20 in cos 0.79 out cos 0.76
cl 7 n 5 mean rownorm 1.55 in cos 0.66 out cos 0.64
cl 66 n 5 mean rownorm 2.18 in cos 0.72 out cos 0.78
cl 48 n 4 mean rownorm 1.74 in cos 0.75 out cos 0.92
cl 87 n 4 mean rownorm 1.38 in cos 0.64 out cos 0.53
cl 44 n 4 mean rownorm 1.59 in cos 0.76 out cos 0.82
cl 4 n 4 mean rownorm 1.31 in cos 0.87 out cos 0.98
cl 124 n 4 mean rownorm 1.56 in cos 0.59 out cos 0.67
cl 6 n 4 mean rownorm 1.38 in cos 0.66 out cos 0.77
cl 52 n 4 mean rownorm 2.78 in cos 0.62 out cos 0.55
cl 70 n 4 mean rownorm 2.36 in cos 0.78 out cos 0.75
cl 14 n 4 mean rownorm 2.12 in cos 0.71 out cos 0.41
cl 135 n 4 mean rownorm 1.51 in cos 0.72 out cos 0.87
cl 121 n 4 mean rownorm 1.33 in cos 0.72 out cos 0.77
cl 29 n 4 mean rownorm 1.95 in cos 0.66 out cos 0.62
cl 18 n 4 mean rownorm 2.10 in cos 0.65 out cos 0.76
cl 68 n 4 mean rownorm 1.77 in cos 0.61 out cos 0.74
cl 132 n 4 mean rownorm 1.51 in cos 0.74 out cos 0.66
cl 12 n 4 mean rownorm 1.51 in cos 0.75 out cos 0.82
cl 75 n 3 mean rownorm 2.54 in cos 0.76 out cos 0.73
corr of row vs col cos (structure) 0.40327494544457687

```

## Call 26

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.cluster.hierarchy import linkage,fcluster
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D0=W['ln1_w.0'].astype(np.float64);D2=W['ln2_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=(F[act]*D2)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
u2,s2,v2=np.linalg.svd(M2)
for r in [24,28,32,40]:
    U=u2[:,:r]
    fr=(np.linalg.norm(Fa@U,axis=1)**2/np.linalg.norm(Fa,axis=1)**2)
    print(r,'frac of active row energy in T1 subspace: quantiles',np.quantile(fr,[.1,.5,.9]).round(2),'chance',r/384)
# find subspace of active rows, remove b-direction top comps
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
print(sa[:70].round(1))
print('top2 right vecs: IPR',1/(va[:4]**4).sum(1),'top dims',np.abs(va[:4]).argmax(1),np.abs(va[:4]).max(1).round(2))
print('left loading of comp0 quantiles',np.quantile(ua[:,0],[0,.1,.5,.9,1]).round(3),'comp1',np.quantile(ua[:,1],[0,.1,.5,.9,1]).round(3))
# overlap of active row subspace (k) with T0 (v2), T1 (u2), P
Q=W['W_q.0'][s(2)];K=W['W_k.0'][s(2)]
_,_,Vt=np.linalg.svd(np.vstack([Q,K]),full_matrices=False);P=Vt[:26].T
for k in [10,20,30,40,50,60,80]:
    B=va[:k].T
    print(k,'T0(28) energy in top-k %.2f, T1(28) %.2f, P %.2f; chance %.2f'%((np.linalg.norm(B.T@v2[:28].T)**2)/28,(np.linalg.norm(B.T@u2[:,:28])**2)/28,(np.linalg.norm(B.T@P)**2)/26,k/384))
print('per comp energy in T0',(np.linalg.norm(v2[:28]@va[:70].T,axis=0)**2).round(2))
print('per comp energy in T1',(np.linalg.norm(u2[:,:28].T@va[:70].T,axis=0)**2).round(2))

```

Output:

```
24 frac of active row energy in T1 subspace: quantiles [0.02 0.12 0.23] chance 0.0625
28 frac of active row energy in T1 subspace: quantiles [0.02 0.12 0.24] chance 0.07291666666666667
32 frac of active row energy in T1 subspace: quantiles [0.03 0.14 0.24] chance 0.08333333333333333
40 frac of active row energy in T1 subspace: quantiles [0.04 0.17 0.27] chance 0.10416666666666667
[17.8  9.1  7.5  7.3  6.9  6.7  6.7  6.3  6.2  6.2  6.1  5.9  5.9  5.8  5.7  5.5  5.5  5.4  5.3  5.2  5.2  5.   5.   4.9  4.8  4.7  4.7  4.6  4.5  4.4  4.3  4.3  4.   4.   3.9  3.8  3.7  3.6  3.6  3.4  3.4  3.3  3.2
  3.2  3.1  3.   3.   2.8  2.8  2.7  2.6  2.6  2.5  2.3  2.3  2.2  2.1  2.1  2.   2.   1.9  1.8  1.8  1.8  1.7  1.7  1.6  1.6  1.5  1.5]
top2 right vecs: IPR [70.19 77.15 45.   54.31] top dims [306 376 379 161] [0.22 0.21 0.28 0.27]
left loading of comp0 quantiles [-0.02 -0.    0.02  0.09  0.15] comp1 [-0.22 -0.09 -0.01  0.02  0.11]
10 T0(28) energy in top-k 0.08, T1(28) 0.04, P 0.01; chance 0.03
20 T0(28) energy in top-k 0.15, T1(28) 0.08, P 0.02; chance 0.05
30 T0(28) energy in top-k 0.23, T1(28) 0.13, P 0.03; chance 0.08
40 T0(28) energy in top-k 0.30, T1(28) 0.22, P 0.05; chance 0.10
50 T0(28) energy in top-k 0.35, T1(28) 0.34, P 0.07; chance 0.13
60 T0(28) energy in top-k 0.37, T1(28) 0.42, P 0.09; chance 0.16
80 T0(28) energy in top-k 0.40, T1(28) 0.51, P 0.13; chance 0.21
per comp energy in T0 [0.04 0.14 0.25 0.24 0.3  0.17 0.28 0.28 0.19 0.22 0.2  0.22 0.18 0.18 0.24 0.24 0.23 0.2  0.15 0.17 0.18 0.21 0.22 0.25 0.15 0.24 0.26 0.23 0.23 0.32 0.22 0.2  0.18 0.22 0.11 0.12 0.18 0.23 0.32 0.18 0.22 0.13 0.14
 0.15 0.16 0.2  0.09 0.12 0.13 0.1  0.07 0.13 0.06 0.04 0.04 0.03 0.05 0.05 0.04 0.03 0.03 0.05 0.04 0.02 0.08 0.04 0.04 0.03 0.06 0.03]
per comp energy in T1 [0.05 0.05 0.08 0.19 0.19 0.13 0.08 0.1  0.11 0.13 0.08 0.12 0.08 0.07 0.11 0.18 0.09 0.1  0.18 0.03 0.23 0.1  0.09 0.17 0.09 0.16 0.07 0.21 0.18 0.13 0.13 0.12 0.22 0.34 0.17 0.42 0.26 0.2  0.27 0.39 0.39 0.32 0.29
 0.32 0.23 0.35 0.48 0.32 0.39 0.3  0.33 0.33 0.33 0.22 0.1  0.16 0.2  0.19 0.17 0.16 0.12 0.11 0.13 0.11 0.09 0.13 0.1  0.14 0.17 0.16]

```

## Call 27

```python
import numpy as np, warnings, pickle
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);D2=W['ln2_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
for r in [30,34,40,56]:
    X=ua[:,1:r+1]*sa[1:r+1]
    best=None
    ica=FastICA(n_components=r,whiten='unit-variance',max_iter=2000,tol=1e-4,random_state=1,fun='logcosh')
    Sx=ica.fit_transform(X)
    k=kurtosis(Sx,axis=0);sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
    o=np.argsort(-k)
    print('r',r,'kurt',k[o].round(1));print('   skew',sk[o].round(1))
    # sign fix
    Sf=Sx*np.sign(sk)
    th=2.5
    print('   n units with |z|>2.5 in each comp',(np.abs(Sf)>th).sum(0)[o])
    print('   units per row with |z|>2: mean',(np.abs(Sf)>2).sum(1).mean(),' dist',np.bincount((np.abs(Sf)>2).sum(1))[:8])

```

Output:

```
r 30 kurt [59.7 49.9 48.  37.9 37.4 34.3 34.3 34.1 31.1 29.9 27.9 27.5 27.1 25.5 23.2 22.9 22.3 19.5 16.9 15.5 15.  14.4 11.6 11.  10.4  9.9  7.5  5.8  4.4  3.9]
   skew [-7.1  5.4 -5.3 -5.3 -5.7  5.3  5.3  5.2 -4.9 -4.6 -4.3 -4.8  4.6 -4.1  4.3 -4.1  4.  -3.2 -3.  -3.  -3.3 -2.8 -2.9 -2.  -2.6  2.  -1.8  0.  -0.3 -0.1]
   n units with |z|>2.5 in each comp [ 7  8 11 11  8 10  8 11 11  9 12 10 12 11 14 11 12  9  9 13 15  9 15 14 14 11 11 13 15 14]
   units per row with |z|>2: mean 1.178477690288714  dist [ 56 227  80  13   3   1   1]
r 34 kurt [62.2 55.1 50.2 40.9 39.7 36.8 35.  33.9 32.5 32.1 31.7 30.1 30.  28.2 28.  27.5 26.8 26.7 25.2 20.6 17.7 16.7 16.  15.1 14.1 13.4 11.4 10.1  7.8  6.7  5.9  5.2  3.9  3.3]
   skew [ 7.   5.8 -5.4  5.9  5.5  0.9 -5.4 -5.4  5.  -5.1  4.9 -4.6 -4.6  4.8  4.4 -4.8  4.4  4.3 -4.4 -3.8  3.7 -2.8  3.2 -2.9  3.3 -2.6 -2.8 -2.3  1.7  1.  -0.6  0.  -0.9 -0.3]
   n units with |z|>2.5 in each comp [ 7 11  9  7 10  7 10  9 11 12 10 12 12 12  9 10 12 11 11 10 14 12 14  9 15 12 15 11 15 17 15 14 12 18]
   units per row with |z|>2: mean 1.3333333333333333  dist [ 46 203 103  23   3   1   1   1]
r 40 kurt [64.9 54.9 53.1 44.6 42.8 41.3 37.7 37.2 32.9 32.8 31.3 30.8 30.1 29.3 29.1 29.  27.3 26.3 26.2 24.1 23.9 21.2 20.6 19.5 17.8 17.2 16.7 15.9 15.3 14.8 14.6 12.3 11.7  6.7  5.8  5.1  4.9  4.8  3.5  3.1]
   skew [-7.3 -5.8  5.6  6.2 -1.2  5.5 -5.5 -5.4 -4.9  5.1  4.9 -4.7  5.   4.7 -4.2 -4.5 -4.5  4.1  3.5 -4.1  4.3  3.6 -3.2 -3.2  2.9 -2.9  3.3  3.  -2.9 -3.  -3.4  1.7 -2.6  0.5 -0.3  0.8 -0.9  0.3 -0.8 -0.6]
   n units with |z|>2.5 in each comp [ 7 11  9  7  7 10  9 10 12 12 12 13 12 11  8  9 11  8  6  9 12 12  5 13 11  7 14 11 13 12 16  9 10 12 16 14 14 13 15 13]
   units per row with |z|>2: mean 1.5511811023622046  dist [ 39 174 114  34  15   3   2]
r 56 kurt [113.9  73.9  60.7  59.5  57.8  57.3  56.5  53.3  52.9  47.8  47.5  45.2  44.6  42.9  42.   41.8  41.1  40.2  38.2  38.   36.6  36.4  36.1  32.   31.5  30.9  30.5  29.9  29.7  28.9  27.4  25.3  25.1  24.8  23.6  23.5
  22.1  22.   21.8  21.3  20.5  20.   17.7  17.6  17.5  15.4  15.4  13.3  12.6  12.4  11.9  10.7   8.1   7.8   7.3   6.2]
   skew [ 8.5  7.8 -6.5 -2.5  6.8 -6.1 -7.1 -5.8  6.2 -5.6 -6.1 -5.8 -5.1  5.8 -5.6  4.9  5.5  5.3 -5.3  4.8  5.  -4.7  4.7 -4.7 -4.5  4.2 -5.1  3.9 -4.9 -4.4 -3.9 -2.6 -4.  -4.6  4.1  3.5 -3.9  3.2  4.  -3.4 -3.1 -3.3  2.7
  2.9 -2.9 -2.5 -3.2 -2.6 -0.9 -2.  -2.1  2.1  0.9  0.5  0.3  1.1]
   n units with |z|>2.5 in each comp [ 5  7  3  7  4  7  7  7  6  9  6  5  7 10  7 10  6  9  6 10  8  8  7  8  9  7 11  8 13  8  8 13  8 12 10 12 10  9 10  9 11  8 10 11 11 11 13 14 13 12 12 18 13 14 16 14]
   units per row with |z|>2: mean 1.8766404199475066  dist [ 17 188  90  45  17  12   6   3]

```

## Call 28

```python
import numpy as np, warnings, pickle
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=1,suppress=True,linewidth=250)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=56
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk); 
A=ica.mixing_*np.sign(sk)        # X = S A^T
# axis vectors in raw 384 space (input direction)
Ax=(A.T@va[1:r+1]).T if False else va[1:r+1].T@A    # 384 x r
Axn=Ax/np.linalg.norm(Ax,axis=0)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
u2,s2,v2=np.linalg.svd(M2)
eT1=(np.linalg.norm(u2[:,:30].T@Axn,axis=0)**2)
eT0=(np.linalg.norm(v2[:30]@Axn,axis=0)**2)
kk=kurtosis(Sx,axis=0)
o=np.argsort(-(eT1-eT0))
print('axis energy in T1 subspace, T0 subspace, kurt (sorted by T1-T0):')
print(np.vstack([eT1[o],eT0[o],kk[o]/10]).round(2))
isT1=eT1>eT0
print('n T1 type',isT1.sum(),'n T0 type',(~isT1).sum())
print('gram Ax cos quantiles',np.quantile((Axn.T@Axn)[np.triu_indices(r,1)],[.01,.5,.99]).round(2))
np.savez('/tmp/ica56.npz',Sx=Sx,Ax=Axn,act=act,isT1=isT1,sa=sa)
# bipartite adjacency: count of units with z>1.5 on both
pos=Sx>1.8
print('units per axis',pos.sum(0)[o])
print('unit types: (#T1 axes, #T0 axes):')
from collections import Counter
c=Counter((int(pos[i][isT1].sum()),int(pos[i][~isT1].sum())) for i in range(len(act)));print(sorted(c.items(),key=lambda x:-x[1])[:12])

```

Output:

```
axis energy in T1 subspace, T0 subspace, kurt (sorted by T1-T0):
[[ 0.8  0.5  0.5  0.4  0.4  0.3  0.3  0.3  0.2  0.3  0.3  0.2  0.2  0.2  0.3  0.2  0.2  0.2  0.1  0.2  0.2  0.2  0.2  0.   0.1  0.2  0.   0.   0.2  0.1  0.   0.   0.   0.   0.   0.1  0.   0.2  0.1  0.2  0.   0.1  0.1  0.   0.1  0.1  0.1  0.1  0.1
   0.   0.   0.1  0.1  0.1  0.   0. ]
 [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.   0.1  0.1  0.1  0.2  0.   0.1  0.2  0.1  0.1  0.2  0.1  0.1  0.1  0.2  0.2  0.2  0.3  0.2  0.4  0.3  0.4  0.3  0.3  0.3  0.3  0.3  0.4  0.4  0.4  0.4
   0.4  0.4  0.5  0.5  0.5  0.5  0.6]
 [ 0.8  0.6  0.8  0.7  1.9  3.1  2.   2.   2.3  3.7  1.2  2.9  2.2  1.8  6.1  4.8  2.5  3.6  1.1  3.8  2.2  4.5  2.7  1.2  5.8  4.5  6.   5.7  3.   4.3  5.3  4.   2.5  7.4  3.   1.2  5.9 11.4  4.1  1.5  4.8  4.2  2.1  3.   3.1  1.3  3.6  2.4  5.3
   1.5  3.2  2.4  2.2  1.8  3.4  4.2]]
n T1 type 24 n T0 type 32
gram Ax cos quantiles [-0.2  0.   0.3]

[stderr]
Traceback (most recent call last):
  File "<string>", line 35, in <module>
    np.savez('/tmp/ica56.npz',Sx=Sx,Ax=Axn,act=act,isT1=isT1,sa=sa)
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/numpy/lib/_npyio_impl.py", line 673, in savez
    _savez(file, args, kwds, False, allow_pickle=allow_pickle)
    ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/numpy/lib/_npyio_impl.py", line 779, in _savez
    zipf = zipfile_factory(file, mode="w", compression=compression)
  File "/Users/agnesbeaudry/.claude-account2/jobs/98de758c/tmp/exp3-sandbox-venv/lib/python3.14/site-packages/numpy/lib/_npyio_impl.py", line 112, in zipfile_factory
    return zipfile.ZipFile(file, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/zipfile/__init__.py", line 1454, in __init__
    self.fp = io.open(file, filemode)
              ~~~~~~~^^^^^^^^^^^^^^^^
PermissionError: [Errno 1] Operation not permitted: '/tmp/ica56.npz'
```

## Call 29

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from collections import Counter
np.set_printoptions(precision=1,suppress=True,linewidth=250)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=56
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
Ax=va[1:r+1].T@A;Axn=Ax/np.linalg.norm(Ax,axis=0)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
u2,s2,v2=np.linalg.svd(M2)
eT1=(np.linalg.norm(u2[:,:30].T@Axn,axis=0)**2);eT0=(np.linalg.norm(v2[:30]@Axn,axis=0)**2)
isT1=eT1>eT0
pos=Sx>1.8
iT1=np.where(isT1)[0];iT0=np.where(~isT1)[0]
iT1=iT1[np.argsort(-pos[:,iT1].sum(0))];iT0=iT0[np.argsort(-pos[:,iT0].sum(0))]
print('units per axis T1',pos[:,iT1].sum(0));print('units per axis T0',pos[:,iT0].sum(0))
Adj=np.zeros((len(iT1),len(iT0)),int)
for u in range(len(act)):
    for a,ia in enumerate(iT1):
        for b,ib in enumerate(iT0):
            if pos[u,ia] and pos[u,ib]: Adj[a,b]+=1
print('rows=T1 axes, cols=T0 axes')
for a in range(len(iT1)):
    print(''.join(f'{v:2d}' if v else ' .' for v in Adj[a]))
print('T0-T0 cooccur');C=(pos[:,iT0].T.astype(int)@pos[:,iT0].astype(int));np.fill_diagonal(C,0)
for a in range(len(iT0)):
    print(''.join(f'{v:2d}' if v else ' .' for v in C[a]))
print('T1-T1 cooccur');C=(pos[:,iT1].T.astype(int)@pos[:,iT1].astype(int));np.fill_diagonal(C,0)
for a in range(len(iT1)):
    print(''.join(f'{v:2d}' if v else ' .' for v in C[a]))
# negatives count
neg=Sx<-1.8
print('neg per axis',neg.sum(0)[np.r_[iT1,iT0]])

```

Output:

```
units per axis T1 [17 16 16 15 15 15 14 13 13 13 12 12 11 11 11 11 10 10  9  9  8  6  6  5]
units per axis T0 [20 15 15 14 13 13 13 12 12 12 11 11 11 10 10  9  9  9  9  9  9  9  8  8  8  7  7  7  7  6  5  5]
rows=T1 axes, cols=T0 axes
 1 1 . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . 1
 3 1 . . . . 1 1 1 . . 1 1 2 . 1 . 1 . . 1 1 1 1 . 2 . . 1 . . .
 . 1 3 1 . . 1 1 1 . 1 . 1 . 1 . 1 . . 2 . . . . . . . . . . 1 .
 2 . . 1 . . 1 1 1 1 3 . . . . 1 . 1 . . . . . 1 . . . . . . . .
 . . . . . 1 1 1 1 . . . . . . . . . . . . . . . . 1 . . . . . .
 . . . . . 3 . . . 2 . . 1 . . . . . . . . . 3 . 1 . . . 1 . . .
 . 1 . 1 . . 2 . . . . . . . . . . . . . . . . . . . . 1 . . . .
 . . . 1 . . . . . . . . . . . . . . . . 1 . . . . . . . . . . .
 3 3 . . . . 1 . . 1 . . 1 . . . . . . . . . 1 . . . . . . . . .
 . . . 1 1 . 1 2 . . . . . . . . 1 1 . . . . . . . . . 1 . . . .
 1 . 1 . 1 1 . . . . 1 1 1 . . . . . . . . . . . . . . . . . . 1
 . . . 1 . . . . . . . 1 . . . . 2 . . . 1 . . . . . . . . . . .
 1 . . . . 1 . 1 . . 1 . 1 1 . . . . . . 4 . 2 . 1 . . . 1 . . .
 . . 2 . . . . . . . 1 . . 1 1 . . . . . . . . . . . 1 . . . . .
 . . 1 . 1 . . . . . . . . . . . . . . . . 1 . . . . . . . . . .
 1 . . 2 1 1 . 2 . . . . 3 . . . . . . . . . . . . . . . . . . 1
 . . . 1 . . . . 1 . . 1 . . . . . . 1 2 . . . . . 1 . . . . . .
 1 1 . . 1 1 . . 1 . . . . . . . . . 3 . . . . . . . . . . 1 . .
 . . 1 . 1 1 1 . . . . . . . . . . . . . . . . . 1 . . . . . . .
 . . . 1 . . . . . . . . . . . 1 . . 1 . . . . . . . . . . . . .
 . . . . 1 . . . . . . . . . . 1 . . . . . . . . . . . . . . . 1
 . . . . . . . . 1 . . 1 . . . . . . . . . . . . . . 1 . . . . .
 . . . . . . . . . . . . . . . . . . . . . . 1 . 1 . . . . . . .
 . . . . . 1 . . . . . . . . . . . . 1 . . . . . . . . . . . . .
T0-T0 cooccur
 . 1 . 1 . . . . 1 . 1 . . 1 . . . . . . . . . . . 1 . . . . . 1
 1 . . . 1 . . . . 1 . . . . . . . . . . . . . . . . . . . . . 2
 . . . . . 1 1 . 1 1 1 . . 1 1 . . . . . . . . . 1 . 1 . . . . .
 1 . . . . . . 1 1 . . 2 2 . 1 . 1 2 1 1 2 . . 1 . . . . . . . 1
 . 1 . . . . . . . . . . . . 1 . 2 . . . . 1 . . . . . . . . . 1
 . . 1 . . . . 1 . 1 . . 1 . . 1 . . 1 . . . . . 2 . . . . . . .
 . . 1 . . . . 1 . . . . . . . . . 2 . . . . . . . . . 1 . . . .
 . . . 1 . 1 1 . 1 . 1 . 1 1 . . . 1 . . 1 1 . 1 . . . . . . . .
 1 . 1 1 . . . 1 . . . 1 . . . 1 . 1 1 1 . . 1 1 . . . . . . . .
 . 1 1 . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . 1
 1 . 1 . . . . 1 . . . . . 1 2 1 1 . . . . . . . 1 . . . . . . .
 . . . 2 . . . . 1 . . . . . . . 1 . 2 1 1 . . . . 1 1 . . . . .
 . . . 2 . 1 . 1 . . . . . . . . . . . . . . 1 1 . . . . . . . .
 1 . 1 . . . . 1 . . 1 . . . . . . 1 . 1 . . . . . 1 1 . . . . .
 . . 1 1 1 . . . . . 2 . . . . 1 2 . . . . . . 1 . . . . . . . .
 . . . . . 1 . . 1 . 1 . . . 1 . 1 . . . . . . . . . . . 1 . . .
 . . . 1 2 . . . . . 1 1 . . 2 1 . . . . 1 . . . . . . . . . . .
 . . . 2 . . 2 1 1 . . . . 1 . . . . . 1 1 1 . 2 . . . . . . . .
 . . . 1 . 1 . . 1 . . 2 . . . . . . . 1 . . 1 . . . . . . . . .
 . . . 1 . . . . 1 . . 1 . 1 . . . 1 1 . . . . . . . . . . 1 . .
 . . . 2 . . . 1 . . . 1 . . . . 1 1 . . . 1 . 1 . . . . . . . .
 . . . . 1 . . 1 . . . . . . . . . 1 . . 1 . . 1 1 . . . . . . .
 . . . . . . . . 1 . . . 1 . . . . . 1 . . . . . 1 . . . . . 1 .
 . . . 1 . . . 1 1 . . . 1 . 1 . . 2 . . 1 1 . . . . . . . . . .
 . . 1 . . 2 . . . . 1 . . . . . . . . . . 1 1 . . . . . . . . .
 1 . . . . . . . . . . 1 . 1 . . . . . . . . . . . . . . . . . .
 . . 1 . . . . . . . . 1 . 1 . . . . . . . . . . . . . . . . . .
 . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . .
 . . . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . .
 1 2 . 1 1 . . . . 1 . . . . . . . . . . . . . . . . . . . . . .
T1-T1 cooccur
 . . . 1 . . . . 1 1 . . . . . . . . 1 . . . . .
 . . 2 1 . . 1 . 3 . . . 2 . . . . 1 . . . . . .
 . 2 . 1 . . . 1 1 1 . 1 1 . 1 . 1 1 . 1 1 . . .
 1 1 1 . . 1 1 . . . 1 . . 1 . . 1 . 1 . . 1 . .
 . . . . . . . . . . 1 1 . . . 2 1 . . . 1 . . .
 . . . 1 . . . . 1 . . . 2 . . . . . 1 . . . 2 .
 . 1 . 1 . . . 1 . 1 1 . . . 2 1 . . . . . . . .
 . . 1 . . . 1 . 1 . . 1 . 1 . 1 1 . 1 1 . . . .
 1 3 1 . . 1 . 1 . . 1 . 1 . . . . . . . . . . .
 1 . 1 . . . 1 . . . . . . . 1 . . . . . . . . .
 . . . 1 1 . 1 . 1 . . 2 . . 1 1 . . . . 1 . . .
 . . 1 . 1 . . 1 . . 2 . . 1 . . . . . . . . . .
 . 2 1 . . 2 . . 1 . . . . . 1 . . . . . . . . .
 . . . 1 . . . 1 . . . 1 . . 1 . . . . . 1 . . .
 . . 1 . . . 2 . . 1 1 . 1 1 . . . . . . . . . .
 . . . . 2 . 1 1 . . 1 . . . . . . . 1 1 . . . .
 . . 1 1 1 . . 1 . . . . . . . . . . . . . . . .
 . 1 1 . . . . . . . . . . . . . . . . 1 1 1 . 2
 1 . . 1 . 1 . 1 . . . . . . . 1 . . . . . . . .
 . . 1 . . . . 1 . . . . . . . 1 . 1 . . 1 . . .
 . . 1 . 1 . . . . . 1 . . 1 . . . 1 . 1 . . . 1
 . . . 1 . . . . . . . . . . . . . 1 . . . . . .
 . . . . . 2 . . . . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . . . . 2 . . 1 . . .
neg per axis [ 7 13  9  8  5  6  0  6 15  4  2  7 13  2  3  6  3  6  2  5  2  1  2  3  3  1  0  8  0  0  5  9  5  1  6  6  2  2  4  3  4  8  1  4  2  2  0  5  2  0  1  2  0  1  1  8]

```

## Call 30

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
np.set_printoptions(precision=2,suppress=True,linewidth=250)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=56
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
Ax=va[1:r+1].T@A;Axn=Ax/np.linalg.norm(Ax,axis=0)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
# image under M2 of each axis
Im=M2@Axn
print('norm of M2 a_i',np.sort(np.linalg.norm(Im,axis=0))[::-1].round(2))
Imn=Im/np.linalg.norm(Im,axis=0)
Sim=Imn.T@Axn   # (i,j): cos(M2 a_i , a_j)
np.fill_diagonal(Sim,0)
m=Sim.max(1);j=Sim.argmax(1)
print('best cos(M2 a_i,a_j) per i sorted',np.sort(m)[::-1].round(2))
print('min (neg) per i',np.sort(Sim.min(1)).round(2)[:12])
# scaled by norm
Sim2=(Im.T@Axn);np.fill_diagonal(Sim2,0)
rowi,colj=linear_sum_assignment(-np.maximum(Sim,0))
vals=Sim[rowi,colj]
print('assignment top',np.sort(vals)[::-1][:40].round(2))
strong=[(i,j,round(Sim[i,j],2),round(float(np.linalg.norm(Im[:,i])),2)) for i,j in zip(rowi,colj) if Sim[i,j]>.3]
print(len(strong),strong)
# null: cos of random vector images
rng=np.random.default_rng(0)
g=rng.standard_normal((384,56));g/=np.linalg.norm(g,axis=0)
Imr=M2@Axn[:,rng.permutation(56)]
print('sim sign: mean positive tail',np.quantile(Sim[np.triu_indices(56,1)],[.5,.9,.99,.999]).round(2))

```

Output:

```
norm of M2 a_i [0.38 0.38 0.37 0.36 0.36 0.35 0.35 0.35 0.35 0.34 0.34 0.32 0.31 0.31 0.3  0.29 0.28 0.28 0.27 0.27 0.23 0.23 0.22 0.21 0.21 0.21 0.2  0.2  0.2  0.2  0.19 0.19 0.19 0.18 0.18 0.17 0.17 0.17 0.17 0.16 0.16 0.16 0.16 0.16 0.16 0.16 0.16 0.15 0.14
 0.14 0.13 0.13 0.13 0.12 0.09 0.08]
best cos(M2 a_i,a_j) per i sorted [0.69 0.57 0.5  0.49 0.48 0.48 0.47 0.47 0.47 0.46 0.46 0.45 0.44 0.43 0.42 0.4  0.39 0.38 0.38 0.37 0.37 0.36 0.36 0.36 0.36 0.35 0.35 0.34 0.33 0.32 0.32 0.32 0.32 0.31 0.3  0.29 0.29 0.24 0.23 0.23 0.23 0.22 0.22 0.22 0.21 0.2  0.2  0.18 0.16
 0.16 0.15 0.15 0.14 0.12 0.12 0.11]
min (neg) per i [-0.56 -0.42 -0.38 -0.36 -0.32 -0.31 -0.31 -0.3  -0.3  -0.3  -0.3  -0.29]
assignment top [0.69 0.57 0.5  0.48 0.46 0.46 0.44 0.42 0.4  0.39 0.38 0.37 0.37 0.36 0.36 0.35 0.35 0.35 0.33 0.32 0.31 0.3  0.3  0.29 0.19 0.19 0.16 0.15 0.14 0.14 0.14 0.13 0.12 0.12 0.11 0.1  0.1  0.1  0.09 0.09]
21 [(np.int64(0), np.int64(39), np.float64(0.35), 0.31), (np.int64(4), np.int64(50), np.float64(0.44), 0.36), (np.int64(12), np.int64(27), np.float64(0.48), 0.35), (np.int64(13), np.int64(5), np.float64(0.42), 0.28), (np.int64(16), np.int64(31), np.float64(0.35), 0.34), (np.int64(17), np.int64(21), np.float64(0.38), 0.36), (np.int64(18), np.int64(6), np.float64(0.37), 0.38), (np.int64(23), np.int64(8), np.float64(0.32), 0.17), (np.int64(24), np.int64(44), np.float64(0.33), 0.35), (np.int64(25), np.int64(20), np.float64(0.4), 0.37), (np.int64(26), np.int64(3), np.float64(0.69), 0.35), (np.int64(28), np.int64(37), np.float64(0.57), 0.38), (np.int64(34), np.int64(22), np.float64(0.37), 0.19), (np.int64(35), np.int64(51), np.float64(0.35), 0.35), (np.int64(36), np.int64(42), np.float64(0.36), 0.32), (np.int64(41), np.int64(54), np.float64(0.31), 0.16), (np.int64(43), np.int64(34), np.float64(0.46), 0.21), (np.int64(44), np.int64(10), np.float64(0.36), 0.17), (np.int64(45), np.int64(32), np.float64(0.46), 0.3), (np.int64(46), np.int64(2), np.float64(0.5), 0.34), (np.int64(53), np.int64(30), np.float64(0.39), 0.31)]
sim sign: mean positive tail [0.   0.08 0.3  0.47]

```

## Call 31

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=250)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64);D3=W['ln1_w.3'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
best=None
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0)
o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o];k=k[o]
Ax=va[1:r+1].T@A;Axn=Ax/np.linalg.norm(Ax,axis=0)
print('kurt',k.round(0))
print('units z>2',(Sx>2).sum(0));print('units z<-2',(Sx<-2).sum(0))
# Token-level output code: regression of Pa on Sx
Bt=np.linalg.lstsq(Sx,Pa.T,rcond=None)[0].T  # 384 x r ; code written per unit of axis
print('norm code',np.linalg.norm(Bt,axis=0).round(2))
print('cos code gram quantiles',np.quantile((lambda B:(B.T@B))(Bt/np.linalg.norm(Bt,axis=0))[np.triu_indices(r,1)],[.01,.5,.99]).round(2))
# weighted mean instead: mean output of top units per axis
Cm=np.zeros((384,r))
for i in range(r):
    idx=np.argsort(-Sx[:,i])[:6]
    Cm[:,i]=(Pa[:,idx]).mean(1)
print('top-unit mean code norm',np.linalg.norm(Cm,axis=0).round(2))
Cn=Cm/np.linalg.norm(Cm,axis=0)
G=Cn.T@Cn;print('gram of codes offdiag quantiles',np.quantile(G[np.triu_indices(r,1)],[.01,.5,.9,.99]).round(2))
M31=(W['W_o.3'][:,s(1)]@W['W_v.3'][s(1)]*D3).astype(np.float64)
U3,S3,Vt3=np.linalg.svd(W['W_proj.3'].astype(np.float64),full_matrices=False)
Y=U3[:,1:28]
L=Y.T@M31@Cm     # copy logit vector for each token code
print('copy logit norm',np.linalg.norm(L,axis=0).round(1))
Ln=L/np.linalg.norm(L,axis=0)
GL=Ln.T@Ln
print('gram of copy logit offdiag q',np.quantile(GL[np.triu_indices(r,1)],[.01,.5,.9,.99]).round(2))
print(GL[:12,:12].round(1))

```

Output:

```
kurt [60. 50. 48. 38. 37. 34. 34. 34. 31. 30. 28. 28. 27. 25. 23. 23. 22. 20. 17. 15. 15. 14. 12. 11. 10. 10.  8.  6.  5.  4.]
units z>2 [ 8 15 13 11  8 11 10 13 13 12 12 10 13 13 14 12 12  9 13 16 16 15 18 14 17 12 14 13 14 14]
units z<-2 [ 0  0  0  0  0  0  0  2  0  0  1  0  0  0  0  0  1  6  3  2  1  3  1  8  1  3  5 10  9 10]
norm code [0.18 0.16 0.16 0.19 0.2  0.22 0.19 0.16 0.18 0.2  0.18 0.23 0.21 0.2  0.23 0.17 0.2  0.2  0.19 0.13 0.21 0.22 0.21 0.2  0.17 0.19 0.19 0.26 0.17 0.15]
cos code gram quantiles [-0.18 -0.01  0.22]
top-unit mean code norm [1.39 1.13 1.33 1.32 1.56 1.68 1.49 1.11 1.43 1.61 1.43 1.64 1.49 1.35 1.44 1.21 1.51 1.52 1.44 0.86 1.5  1.47 1.61 1.38 1.09 1.2  1.44 1.19 1.14 0.93]
gram of codes offdiag quantiles [-0.07  0.04  0.12  0.21]
copy logit norm [2.  1.9 1.1 1.7 1.6 0.9 1.5 2.1 1.2 1.2 1.2 0.9 1.4 0.8 2.1 1.1 1.  1.  0.9 1.2 1.1 1.1 1.8 1.  2.  1.1 1.9 1.8 0.9 1.3]
gram of copy logit offdiag q [-0.43  0.02  0.27  0.48]
[[ 1.  -0.1 -0.1 -0.   0.  -0.   0.   0.2 -0.2 -0.2  0.1  0.1]
 [-0.1  1.  -0.   0.  -0.1 -0.1 -0.1 -0.2  0.3 -0.  -0.3 -0.5]
 [-0.1 -0.   1.  -0.1  0.2 -0.1  0.1  0.   0.2  0.2  0.4  0.3]
 [-0.   0.  -0.1  1.  -0.1 -0.2 -0.1  0.1  0.2 -0.2 -0.1  0.1]
 [ 0.  -0.1  0.2 -0.1  1.  -0.3 -0.   0.1 -0.1  0.1 -0.   0.1]
 [-0.  -0.1 -0.1 -0.2 -0.3  1.   0.4 -0.4  0.   0.2  0.4  0.2]
 [ 0.  -0.1  0.1 -0.1 -0.   0.4  1.   0.1 -0.1  0.   0.2  0.2]
 [ 0.2 -0.2  0.   0.1  0.1 -0.4  0.1  1.  -0.3 -0.3  0.1 -0. ]
 [-0.2  0.3  0.2  0.2 -0.1  0.  -0.1 -0.3  1.   0.2 -0.   0.1]
 [-0.2 -0.   0.2 -0.2  0.1  0.2  0.  -0.3  0.2  1.   0.1  0.2]
 [ 0.1 -0.3  0.4 -0.1 -0.   0.4  0.2  0.1 -0.   0.1  1.   0.4]
 [ 0.1 -0.5  0.3  0.1  0.1  0.2  0.2 -0.   0.1  0.2  0.4  1. ]]

```

## Call 32

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=1,suppress=True,linewidth=250)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
Ax=va[1:r+1].T@A;Axn=Ax/np.linalg.norm(Ax,axis=0)
K=26
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
f=M2@Axn[:,:K];f=f/np.linalg.norm(f,axis=0)
Ftot=Fa
S0=Ftot@Axn[:,:K];S1=Ftot@f
print('corr of S0 vs Sx',np.corrcoef(S0[:,0],Sx[:,0])[0,1])
S1z=(S1-np.median(S1,axis=0))/ (1.48*np.median(np.abs(S1-np.median(S1,axis=0)),axis=0))
S0z=(S0-np.median(S0,axis=0))/ (1.48*np.median(np.abs(S0-np.median(S0,axis=0)),axis=0))
print('kurt S1',kurtosis(S1,axis=0).round(0))
print('skew S1',(((S1-S1.mean(0))/S1.std(0))**3).mean(0).round(1))
print('S1 z>4 count',(S1z>4).sum(0))
print('S0 z>4 count',(S0z>4).sum(0))
# adjacency: for each unit, top cur axis, top prev axis
cur=S0z.argmax(1);prev=S1z.argmax(1)
Adj=np.zeros((K,K),int)
for u in range(len(act)):
    if S0z[u,cur[u]]>4 and S1z[u,prev[u]]>4: Adj[prev[u],cur[u]]+=1
print('adj rows=prev, cols=cur, total',Adj.sum())
for a in range(K): print(''.join(f'{v:2d}' if v else ' .' for v in Adj[a]))
Cont=np.maximum(S0z,0).T@np.maximum(S1z,0)
print('sv Cont',np.linalg.svd(Cont,compute_uv=False)[:10].round(0))
print('unit counts by has cur/prev',((S0z.max(1)>4)&(S1z.max(1)>4)).sum(),((S0z.max(1)>4)&(S1z.max(1)<=4)).sum(),((S0z.max(1)<=4)&(S1z.max(1)>4)).sum(),((S0z.max(1)<=4)&(S1z.max(1)<=4)).sum())

```

Output:

```
corr of S0 vs Sx 0.9709728381043492
kurt S1 [ 7.  9. 16.  0.  3.  4.  9.  8.  4.  8. 12.  2.  3.  8.  2.  3.  1.  6.  3.  4.  2.  3.  3.  3.  2.  4.]
skew S1 [ 0.9  1.3  1.9 -0.1  0.1  0.3  1.4 -0.7  0.3  1.5  2.2  0.4 -0.2  1.8  0.4  0.6 -0.   1.3  0.7 -0.7 -0.5  0.8  0.5  0.4  0.9  0.2]
S1 z>4 count [ 7  8  7  0  2  7  4  5  6 11 11  4  3  8  2  6  0  6  4  2  1  8 11  8  9  7]
S0 z>4 count [11 22 15 11  9 12 10 17 15 12 14 10 14 13 18 13 12 10  9 16 15  7 18 12 16  8]
adj rows=prev, cols=cur, total 85
 3 . . . . . . . . . . . 1 . . . . . . . . . . . 1 .
 . 3 . . . . . . . . . . . . 1 . . . . . . . . . . .
 . . 4 . . . . . . . . . . . . . . . . . 1 . . . . .
 . . . . . . . . . . . . . . . . . . . . . . . . . .
 . . . . 1 . . . . . . . . . . . . . . . . . . . . .
 . . . . . 4 . . . . . . . . . . 1 . . . . . . . . .
 . . . . . . 3 . . . . . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . . . . . . 2 1 . . . . .
 . . . . . . . . 2 . . . . . . . . . . . . . . . . .
 . . . . . 1 . . . 5 . . . . . 1 . . . . . . . . . .
 . . . . . . 1 . . . 5 . . . . . . . . . . . . 1 . .
 . . . . . . . . . . . 3 . . . . . . . . . . . . . .
 . . . . . . . . . . . . 1 . . . . . . . . . . . . .
 . . . . . . . . . . . . . 5 . . . . . . . . . . 1 .
 . . . . . . . . . . . . . . 2 . . . . . . . . . . .
 . . 1 . . . . . . . . . . . . 1 . . . . . . . . . .
 . . . . . . . . . . . . . . . . . . . . . . . . . .
 1 . . . 1 . . . . . . . . . . . . 3 . . . . . . . .
 . . . . 1 . . . . . . . 1 . . . . . 2 . . . . . . .
 . . . . . . . . . . . . . . . . . . . . . . . . . .
 . . . . . . . . . . . . . . . . . . . . . . . . . .
 . 2 . . . . . 1 . . . . . . . . . . . 1 . 1 . . . .
 . 1 . . . . . . . . . . . . . 1 . . . . . . 2 . . .
 . . . . 1 . . . . . . . . . . . . . . . . . . 1 . .
 . . . 1 . . . . . . 1 . . . . . . . . . . . . . 4 .
 . 1 . . . . . 1 . . . . . . . . . . 1 . . . . . . .
sv Cont [3954. 1086.  847.  730.  667.  544.  521.  478.  459.  412.]
unit counts by has cur/prev 85 199 19 78

```

## Call 33

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=250)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
ev=np.linalg.eigvals(M2);o=np.argsort(-np.abs(ev))
print('M2 eig abs',np.abs(ev[o[:34]]).round(2));print('M2 eig deg',np.degrees(np.angle(ev[o[:34]])).round(0))
print('trace M2',np.trace(M2).round(3),'|M2|F',np.linalg.norm(M2).round(2))
print('L0 other heads: trace, top eig')
for j in range(6):
    Mx=W['W_o.0'][:,s(j)].astype(np.float64)@W['W_v.0'][s(j)].astype(np.float64)*D0
    e=np.linalg.eigvals(Mx);print(j,'trace %.2f'%np.trace(Mx),'top eig',e[np.argsort(-np.abs(e))[:4]].round(2))
for r in [30,56]:
    X=ua[:,1:r+1]*sa[1:r+1]
    ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
    Sx=ica.fit_transform(X)
    sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
    Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
    k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
    Ax=va[1:r+1].T@A;Axn=Ax/np.linalg.norm(Ax,axis=0)
    Im=M2@Axn;Imn=Im/np.linalg.norm(Im,axis=0)
    print('r',r,'diag cos(a,M2 a)',np.diag(Axn.T@Imn).round(2))
    Sim=Imn.T@Axn; 
    print(' argmax j for each i',Sim.argmax(0)[:30],'max',Sim.max(0)[:30].round(2))

```

Output:

```
M2 eig abs [0.14 0.14 0.13 0.13 0.13 0.13 0.12 0.12 0.12 0.11 0.11 0.09 0.09 0.09 0.09 0.09 0.08 0.08 0.08 0.07 0.07 0.07 0.07 0.05 0.05 0.05 0.05 0.04 0.04 0.04 0.04 0.04 0.04 0.03]
M2 eig deg [ 122. -122.  156. -156.   91.  -91.   30.  -30.    0.  139. -139. -126.  126.   68.  -68.  180.  180.  144. -144. -157.  157.   10.  -10.  -72.   72.  122. -122.    0.  180.   29.  -29.  111. -111.  129.]
trace M2 -0.626 |M2|F 2.54
L0 other heads: trace, top eig
0 trace 0.02 top eig [ 0.99+0.j   -0.39+0.47j -0.39-0.47j -0.04+0.12j]
1 trace -0.19 top eig [ 0.73+0.j   -0.68+0.j   -0.02+0.16j -0.02-0.16j]
2 trace -0.63 top eig [-0.08+0.12j -0.08-0.12j -0.12+0.05j -0.12-0.05j]
3 trace -0.25 top eig [ 0.23+0.j   -0.1 +0.j   -0.08+0.04j -0.08-0.04j]
4 trace -0.35 top eig [ 0.28+0.j   -0.25+0.05j -0.25-0.05j  0.08+0.06j]
5 trace 0.05 top eig [ 0.73+0.j   -0.38+0.41j -0.38-0.41j  0.25+0.j  ]
r 30 diag cos(a,M2 a) [ 0.05  0.33  0.34  0.03  0.08  0.12  0.21 -0.12  0.24  0.22  0.24  0.14  0.14  0.29  0.09  0.16  0.15  0.12  0.21  0.1   0.11  0.24  0.22  0.17  0.17  0.24  0.22  0.04  0.06 -0.26]
 argmax j for each i [ 5  1  2 24  0  5  6 14  8  9 10 20 12 13 14 15 28  7 25  7 29 21 22 23 24 25 26  0 27 28] max [0.05 0.33 0.34 0.12 0.12 0.12 0.21 0.07 0.24 0.22 0.24 0.17 0.14 0.29 0.09 0.16 0.2  0.13 0.23 0.12 0.12 0.24 0.22 0.17 0.17 0.24 0.22 0.07 0.13 0.45]
r 56 diag cos(a,M2 a) [ 0.36 -0.01  0.36 -0.02 -0.09  0.23  0.02 -0.16 -0.04  0.28 -0.09  0.31  0.21  0.17 -0.19 -0.09 -0.17 -0.03  0.17  0.47  0.35 -0.04  0.07 -0.09  0.43 -0.1   0.11  0.31 -0.1   0.14  0.33  0.39  0.09 -0.07 -0.24  0.11  0.35  0.29 -0.09 -0.13  0.37
  0.32  0.42 -0.02  0.33 -0.04 -0.29 -0.09 -0.16  0.11  0.31 -0.06  0.3  -0.63 -0.02  0.05]
 argmax j for each i [53  5 10 22 31 16 13 41 35  7  7 11  0 13 51 29 49 14  8 19 15 49 13 12 24 49 26 46 51 14] max [0.37 0.06 0.39 0.05 0.09 0.29 0.08 0.1  0.1  0.36 0.09 0.31 0.3  0.17 0.14 0.09 0.1  0.07 0.32 0.47 0.37 0.09 0.09 0.07 0.43 0.14 0.11 0.35 0.06 0.42]

```

## Call 34

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from scipy.cluster.hierarchy import linkage,leaves_list,fcluster
np.set_printoptions(precision=2,suppress=True,linewidth=250)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o]
K=27
Sp=np.maximum(Sx[:,:K],0)
U3,S3,Vt3=np.linalg.svd(W['W_proj.3'].astype(np.float64),full_matrices=False)
Y=U3[:,1:28]
YP=Y.T@Pa      # 27 x units
beta=np.linalg.lstsq(Sp,YP.T,rcond=None)[0]   # K x 27
print('beta row norms',np.linalg.norm(beta,axis=1).round(2))
print('sv beta',np.linalg.svd(beta,compute_uv=False).round(2))
res=YP.T-Sp@beta;print('R2',1-(res**2).sum()/(YP**2).sum())
# also full output code (not just Y)
code=np.linalg.lstsq(Sp,Pa.T,rcond=None)[0]
print('code R2',1-((Pa.T-Sp@code)**2).sum()/(Pa**2).sum(),'sv',np.linalg.svd(code,compute_uv=False).round(1))
bn=beta/np.linalg.norm(beta,axis=1,keepdims=True)
G=bn@bn.T
Z=linkage(bn,'average',metric='cosine');lv=leaves_list(Z)
print('order',lv)
print((G[lv][:,lv]*10).round(0).astype(int))
print('clusters at .9',fcluster(Z,.9,'distance')[lv])
cn=code/np.linalg.norm(code,axis=1,keepdims=True)
Gc=cn@cn.T
Z2=linkage(cn,'average',metric='cosine');lv2=leaves_list(Z2);print('code order',lv2);print((Gc[lv2][:,lv2]*10).round(0).astype(int))

```

Output:

```
beta row norms [0.06 0.1  0.1  0.07 0.09 0.1  0.07 0.04 0.07 0.08 0.12 0.09 0.1  0.13 0.09 0.07 0.11 0.1  0.14 0.05 0.12 0.13 0.1  0.14 0.11 0.09 0.12]
sv beta [0.2  0.18 0.16 0.15 0.14 0.14 0.13 0.12 0.11 0.1  0.09 0.09 0.08 0.08 0.07 0.07 0.06 0.06 0.04 0.04 0.04 0.03 0.02 0.02 0.01 0.01 0.01]
R2 0.4412932704782303
code R2 0.30650840444169036 sv [0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.1]
order [21 22  1  8 12  3 17  9 24  0 18 15 26  7 25 20 23 10  5 19 13  2  6 14 16  4 11]
[[10  4  1  2  1 -2 -4 -2  1  0  0  0  2 -1 -2 -1 -3 -1 -2 -3  0  1  1 -1 -2 -5 -2]
 [ 4 10  1 -1  2 -1 -3  1 -2 -1 -1  0 -1  2 -2 -4 -3 -4 -2  0  3  1 -1 -1  0 -4 -2]
 [ 1  1 10  2  3  0  2  1  2 -1 -1  1  1  0 -1  0 -2 -1 -1 -1 -2 -2  1  1 -1 -2 -3]
 [ 2 -1  2 10  4  2  2 -4  0 -1  1  4  1 -2 -4  2 -3  2 -2  0 -2 -3 -2  2 -3  0 -1]
 [ 1  2  3  4 10  0 -1  0 -1  0  0  1  0  1  1 -2 -1 -1 -3  1 -2 -3 -3 -1 -2 -3 -3]
 [-2 -1  0  2  0 10  3  2  2 -1  1  0  1 -2 -3  1  1  2 -2  1 -2  3 -2  0  1  2  0]
 [-4 -3  2  2 -1  3 10  2  3  0 -1  0  2 -3 -3  1 -1  2  1 -1 -1 -2  1 -1  1  3  2]
 [-2  1  1 -4  0  2  2 10  3  2  2  0  0  0  0 -1  0 -3 -2 -4 -1  1  1 -2  2  0 -1]
 [ 1 -2  2  0 -1  2  3  3 10  1  1  1  2  1 -2  1 -3 -2 -2 -3 -3  0  1  0 -3  0 -1]
 [ 0 -1 -1 -1  0 -1  0  2  1 10  3  0  1 -3  1  0 -1  0 -1 -4  0  0  2 -1 -1  1  0]
 [ 0 -1 -1  1  0  1 -1  2  1  3 10  2  2  1 -1 -2 -2 -2 -1 -2 -3  2 -2  1 -2 -3 -3]
 [ 0  0  1  4  1  0  0  0  1  0  2 10  3  1 -2  0 -2  1  1 -2 -2  0  0  2 -2 -1  1]
 [ 2 -1  1  1  0  1  2  0  2  1  2  3 10 -1 -1  1  1  0  0 -3 -2 -1 -2  0 -3 -4 -1]
 [-1  2  0 -2  1 -2 -3  0  1 -3  1  1 -1 10  3 -2 -2 -3  0  0 -3  1  0  0 -1  0  0]
 [-2 -2 -1 -4  1 -3 -3  0 -2  1 -1 -2 -1  3 10 -2  0  0  2  2 -3 -1 -2 -1 -1  0 -2]
 [-1 -4  0  2 -2  1  1 -1  1  0 -2  0  1 -2 -2 10  4  2 -2 -2 -2 -2 -1  3  0  2  4]
 [-3 -3 -2 -3 -1  1 -1  0 -3 -1 -2 -2  1 -2  0  4 10 -1  0  0  0 -1  0 -2  1  0  2]
 [-1 -4 -1  2 -1  2  2 -3 -2  0 -2  1  0 -3  0  2 -1 10  3  2  0  0  2  3  0  1  3]
 [-2 -2 -1 -2 -3 -2  1 -2 -2 -1 -1  1  0  0  2 -2  0  3 10  5  3 -1  1  1  0  1  2]
 [-3  0 -1  0  1  1 -1 -4 -3 -4 -2 -2 -3  0  2 -2  0  2  5 10  1 -1 -4  0  1  0  1]
 [ 0  3 -2 -2 -2 -2 -1 -1 -3  0 -3 -2 -2 -3 -3 -2  0  0  3  1 10  2  4  0  3  1  1]
 [ 1  1 -2 -3 -3  3 -2  1  0  0  2  0 -1  1 -1 -2 -1  0 -1 -1  2 10  5  3  2  1  2]
 [ 1 -1  1 -2 -3 -2  1  1  1  2 -2  0 -2  0 -2 -1  0  2  1 -4  4  5 10  1  1  1  2]
 [-1 -1  1  2 -1  0 -1 -2  0 -1  1  2  0  0 -1  3 -2  3  1  0  0  3  1 10  1  1  2]
 [-2  0 -1 -3 -2  1  1  2 -3 -1 -2 -2 -3 -1 -1  0  1  0  0  1  3  2  1  1 10  3  2]
 [-5 -4 -2  0 -3  2  3  0  0  1 -3 -1 -4  0  0  2  0  1  1  0  1  1  1  1  3 10  4]
 [-2 -2 -3 -1 -3  0  2 -1 -1  0 -3  1 -1  0 -2  4  2  3  2  1  1  2  2  2  2  4 10]]
clusters at .9 [1 1 1 1 1 2 2 2 2 3 3 3 3 4 4 5 5 6 6 6 7 7 7 7 7 7 7]
code order [ 9 18  8 12 22  5 10 15 23 19 25 21 26  1 17  7 20 14  4 11 16  2 13 24  3  0  6]
[[10  1  0  1  0 -1  0  0  0 -1  1  0  0  0  0  0 -1 -1  0  1  0  0 -1  1  0  0  0]
 [ 1 10  0  1  0  0  0  0  0  0  0  1  0  0 -1  0 -1  0  0 -1  0  0 -2  0  0  0  0]
 [ 0  0 10  1 -1  0  0  0  0 -1 -1  0  0  0  1  0  0  0  0  0 -1 -1  0 -1  0  0 -1]
 [ 1  1  1 10 -1  0  0  1  0  0  0  0  0  1  0  0 -1  0 -2 -1  0 -1 -1 -1  0 -1  0]
 [ 0  0 -1 -1 10 -1  0  1  0  0 -1  1  1  0  0  0 -1 -1 -1  0  0  0  1  0  0  0  0]
 [-1  0  0  0 -1 10  1  1  0  0  0  0  0 -1  0  0  0 -1  0  0  0  0  0 -1 -1  1  0]
 [ 0  0  0  0  0  1 10  1  0  0  1  1  0  0  1 -1 -1  0  0  1  0  1  1  0  0  0  0]
 [ 0  0  0  1  1  1  1 10  0  0  0  0  1  0  0  0  1 -1 -1  0  0  0 -1  0  0 -1  0]
 [ 0  0  0  0  0  0  0  0 10  2  1  0  1  0  0 -1  1 -1  0  0  0  1  1 -1  0 -1  0]
 [-1  0 -1  0  0  0  0  0  2 10  2  1  0  0  0 -1  0 -1 -1  0  0  0  0  0  0 -1 -1]
 [ 1  0 -1  0 -1  0  1  0  1  2 10  0  1  0  1  1 -1 -1  0 -1  0  1  0  1  1 -1  0]
 [ 0  1  0  0  1  0  1  0  0  1  0 10  2  1  0  0 -1 -2 -2  0 -1  1  0  1 -1  0 -1]
 [ 0  0  0  0  1  0  0  1  1  0  1  2 10  1  1  0  0 -1 -2  0 -1  0  0  0  1 -1 -1]
 [ 0  0  0  1  0 -1  0  0  0  0  0  1  1 10  2 -1  1  0 -1 -1  0  0 -1  1  0 -1  0]
 [ 0 -1  1  0  0  0  1  0  0  0  1  0  1  2 10  0  1 -2  0  0  0 -1 -1  1  1  0  0]
 [ 0  0  0  0  0  0 -1  0 -1 -1  1  0  0 -1  0 10 -1  0  0  0  0  0  0  0  0  0  0]
 [-1 -1  0 -1 -1  0 -1  1  1  0 -1 -1  0  1  1 -1 10  1  1  0  0  0 -1  0 -1  0  0]
 [-1  0  0  0 -1 -1  0 -1 -1 -1 -1 -2 -1  0 -2  0  1 10  2  1  0  1  0  1  0  0  0]
 [ 0  0  0 -2 -1  0  0 -1  0 -1  0 -2 -2 -1  0  0  1  2 10  2  0  0  1  0 -1  0  0]
 [ 1 -1  0 -1  0  0  1  0  0  0 -1  0  0 -1  0  0  0  1  2 10 -1  0  1  0  0  1  1]
 [ 0  0 -1  0  0  0  0  0  0  0  0 -1 -1  0  0  0  0  0  0 -1 10  1  1 -1  1  0  1]
 [ 0  0 -1 -1  0  0  1  0  1  0  1  1  0  0 -1  0  0  1  0  0  1 10  1  1  0  0  1]
 [-1 -2  0 -1  1  0  1 -1  1  0  0  0  0 -1 -1  0 -1  0  1  1  1  1 10  0  0  0  1]
 [ 1  0 -1 -1  0 -1  0  0 -1  0  1  1  0  1  1  0  0  1  0  0 -1  1  0 10  0  0  1]
 [ 0  0  0  0  0 -1  0  0  0  0  1 -1  1  0  1  0 -1  0 -1  0  1  0  0  0 10  1  0]
 [ 0  0  0 -1  0  1  0 -1 -1 -1 -1  0 -1 -1  0  0  0  0  0  1  0  0  0  0  1 10  1]
 [ 0  0 -1  0  0  0  0  0  0 -1  0 -1 -1  0  0  0  0  0  0  1  1  1  1  1  0  1 10]]

```

## Call 35

```python
import numpy as np, warnings, re
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=200)
EN = """The city was founded in the early nineteenth century by settlers from the eastern part of the country and became an important trading center during the following decades. According to the census of 2010 the population of the municipality was about forty thousand people, most of whom were employed in agriculture, mining and light industry. The river which flows through the town is a tributary of the larger system that drains the northern plains.
He was born in London and educated at the University of Cambridge, where he studied mathematics and natural philosophy. After graduating he joined the army and served in several campaigns in Europe before returning to England in 1815. In the following year he was elected to the Royal Society and published his first major work on the theory of light. His later research dealt with the properties of gases and the structure of crystals, and he received many honours during his lifetime.
The species is found in tropical forests of South America, where it feeds mainly on fruit and small insects. Adults reach a length of about twenty centimetres and have a dark brown back with a pale yellow belly. The female lays two or three eggs in a nest built high in the trees, and both parents take care of the young. It is considered vulnerable by the International Union for Conservation of Nature because of habitat loss and hunting.
The album was released in 1994 by the record company and received generally positive reviews from music critics, who praised the production and the songwriting of the band. It reached number five on the national chart and was certified gold in the United States and Canada. The first single from the album became a major hit and the accompanying video was directed by a well known film maker. The group then toured extensively through Europe, Japan and Australia in support of the record.
During the Second World War the airfield was used by the air force as a training base for pilots and later as a station for bombers. After the end of the war the site was returned to the local authority and part of it was developed as a civil airport. The government announced in 1972 that a new terminal would be constructed, and the building was opened three years later. Today the airport handles more than two million passengers each year and serves as a hub for several regional airlines.
The school was established in 1923 and is one of the oldest educational institutions in the district. It offers courses in science, commerce and the arts, and has about one thousand students. The election was held on the same day as the presidential vote, and the party won a majority of the seats in the national assembly. The team finished third in the league that season and reached the final of the national cup, where it lost to the defending champions."""
PY = """import os
import sys
from typing import list, optional
class Config(object):
    def __init__(self, name, value=None, parent=None):
        self.name = name
        self.value = value
        self.parent = parent
        self._children = []
    def add_child(self, child):
        child.parent = self
        self._children.append(child)
        return child
    def get(self, key, default=None):
        if key in self.__dict__:
            return self.__dict__[key]
        if self.parent is not None:
            return self.parent.get(key, default)
        return default
    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self.name)
def load_file(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        data = f.read()
    lines = data.split('\\n')
    result = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        result.append(line)
    return result
class TestRequest(unittest.TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create(username='test', email='test@example.com')
    def test_get_returns_response(self):
        response = self.client.get('/api/items/', follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('items', response.data)
    def test_post_invalid_data(self):
        data = {'name': '', 'value': None}
        response = self.client.post('/api/items/', data=data)
        self.assertEqual(response.status_code, 400)
def compute_loss(x, y, weights, axis=0):
    n = x.shape[0]
    pred = np.dot(x, weights)
    diff = pred - y
    loss = np.sum(diff ** 2, axis=axis) / n
    if loss > max_value:
        raise ValueError('loss too large: {}'.format(loss))
    return loss, pred
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='out.txt')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    return args
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_size, hidden_size))
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(torch.relu(layer(x)))
        return x
def main():
    args = parse_args()
    logger = logging.getLogger(__name__)
    try:
        result = process(args.input, args.output)
    except (IOError, KeyError) as e:
        logger.error('failed: %s', str(e))
        sys.exit(1)
    if result is None:
        return
    for key, value in sorted(result.items()):
        print(key, value)
if __name__ == '__main__':
    main()
"""
RO = """sono hi wa asa kara ame ga futte ita. watashi wa heya no naka de hon wo yonde ita ga, yagate tachiagatte mado no soto wo mita. niwa no ki wa nurete, tooku no yama wa kumo ni kakurete ita. haha wa daidokoro de nani ka wo shite iru rashikatta. kare wa sono toki hajimete kanojo no kao wo mita no de aru. sore wa utsukushii onna de atta ga, doko ka kanashige na tokoro ga atta. anata wa dou omoimasu ka to kiku to, otoko wa shibaraku damatte ita ga, yagate shizuka ni kuchi wo hiraita.
watakushi wa sono sensei wo tsune ni sensei to yonde ita. dakara koko demo tada sensei to kaku dake de honmyou wa uchiakenai. kore wa seken wo habakaru enryo to iu yori mo, sono hou ga watakushi ni totte shizen dakara de aru. watakushi wa sono hito no kioku wo yobiokosu goto ni, sugu sensei to iitaku naru. fude wo totte mo kokoromochi wa onaji koto de aru.
kuni no hito wa minna sou iu fuu ni kangaete ita. mura no kodomotachi wa kawa no hotori de asonde ita. toshi wo totta shoumin no kurashi wa mazushiku, fuyu ni naru to yuki ga fukaku tsumotte michi mo wakaranaku natta. shikashi haru ga kuru to sakura no hana ga saite, hitobito wa yorokonde yama e dekaketa."""
def norm(t):
    out=[]
    for ln in t.lower().split('\n'):
        m=re.match(r'^ *',ln);lead=m.group(0)
        body=re.sub(r'[^a-z]+',' ',ln[len(lead):]).rstrip()
        out.append(lead+body)
    return '\n'.join(out)
V="\n abcdefghijklmnopqrstuvwxyz"
def bigram(t):
    t=norm(t);C=np.zeros((28,28))
    for a,b in zip(t,t[1:]): C[V.index(a),V.index(b)]+=1
    return C,np.bincount([V.index(c) for c in t],minlength=28)
EN2=EN.replace('\n','\n')
for nm,t in [('EN',EN),('PY',PY),('RO',RO)]:
    C,f=bigram(t);print(nm,'chars',int(f.sum()),'top freq',''.join(repr(V[i])[1:-1]+' ' for i in np.argsort(-f)[:12]),'rare count',{V[i]:int(f[i]) for i in np.argsort(f)[:8]})
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
Ax=va[1:r+1].T@A;Axn=Ax/np.linalg.norm(Ax,axis=0)
K=26
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
f=M2@Axn[:,:K];f=f/np.linalg.norm(f,axis=0)
S0=Fa@Axn[:,:K];S1=Fa@f
z=lambda S:(S-np.median(S,axis=0))/(1.48*np.median(np.abs(S-np.median(S,axis=0)),axis=0))
S0z=np.maximum(z(S0),0);S1z=np.maximum(z(S1),0)
# remove leak of a_t into f_t by regressing S1 on S0 same index
Cont=S1z.T@S0z   # rows prev, cols cur
Cc=Cont.copy().astype(float)
np.fill_diagonal(Cc,np.nan)
def dc(M):
    M=M.copy();
    for _ in range(5):
        M=M-np.nanmean(M,axis=1,keepdims=True);M=M-np.nanmean(M,axis=0,keepdims=True)
    return M
Ac=dc(np.log1p(Cc))
Ac=np.nan_to_num(Ac)
def templ(C):
    T=np.log(C+.5)
    return T
rng=np.random.default_rng(0)
def qap(A,B,restarts=300,sweeps=25):
    n=A.shape[0];N=B.shape[0]
    best=-1e9;bp=None
    for rs in range(restarts):
        p=rng.permutation(N)[:n].copy()
        rest=[x for x in range(N) if x not in set(p.tolist())]
        def score(p):
            sub=B[np.ix_(p,p)].copy();np.fill_diagonal(sub,np.nan)
            sub=np.nan_to_num(dc(sub))
            return (A*sub).sum()/np.sqrt((A**2).sum()*(sub**2).sum())
        cur=score(p)
        for it in range(sweeps*n):
            i,j=rng.integers(n),rng.integers(n+len(rest))
            q=p.copy()
            if j<n:
                q[i],q[j]=q[j],q[i]
            else:
                t=rest[j-n];rest2=rest.copy();rest2[j-n]=q[i];q[i]=t
            sc=score(q)
            if sc>cur:
                cur=sc;p=q
                if j>=n: rest=rest2
        if cur>best:best=cur;bp=p.copy()
    return best,bp
import time
t0=time.time()
for nm,t in [('EN',EN),('PY',PY),('RO',RO)]:
    C,fq=bigram(t);T=templ(C)
    b,p=qap(Ac,T,restarts=12,sweeps=12)
    print(nm,'best corr',round(b,3),'labels',[V[i] if V[i]>' ' else repr(V[i]) for i in p],'time',round(time.time()-t0))

```

Output:

```
EN chars 2757 top freq   e t a n i r o s h d l  rare count {'z': 0, 'q': 0, 'x': 1, 'j': 5, '\n': 5, 'k': 6, 'v': 20, 'b': 35}
PY chars 2591 top freq   e s r t a n l i o \n d  rare count {'j': 2, 'b': 3, 'z': 4, 'q': 4, 'w': 6, 'k': 7, 'v': 12, 'x': 15}
RO chars 1123 top freq   a o t i k n u e s h w  rare count {'l': 0, 'x': 0, 'v': 0, 'q': 0, 'p': 0, '\n': 2, 'z': 3, 'j': 3}
EN best corr 0.295 labels ['z', 'n', 'k', 'y', 'b', "'\\n'", 'e', 'i', 'j', "' '", 'p', 'u', 'f', 'm', 'l', 'a', 'd', 'x', 'o', 'r', 'v', 'g', 'h', 't', 'q', 'w'] time 0
PY best corr 0.322 labels ['b', 'u', 'q', 'c', 'd', 'x', 'f', "' '", 'j', 'h', 'n', 'o', 'w', 'z', 'v', "'\\n'", 'k', 'e', 'y', 'r', 'm', 's', 'l', 'a', 'g', 't'] time 1
RO best corr 0.314 labels ["'\\n'", 't', 'q', 'l', 'r', 'j', 's', 'b', 'v', 'h', 'm', 'u', 'e', 'd', 'p', 'z', 'x', 'n', 'k', "' '", 'f', 'i', 'a', 'w', 'y', 'o'] time 1

```

## Call 36

```python
import numpy as np, warnings, re, time
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=200)
EN = """The city was founded in the early nineteenth century by settlers from the eastern part of the country and became an important trading center during the following decades. According to the census of 2010 the population of the municipality was about forty thousand people, most of whom were employed in agriculture, mining and light industry. The river which flows through the town is a tributary of the larger system that drains the northern plains.
He was born in London and educated at the University of Cambridge, where he studied mathematics and natural philosophy. After graduating he joined the army and served in several campaigns in Europe before returning to England in 1815. In the following year he was elected to the Royal Society and published his first major work on the theory of light. His later research dealt with the properties of gases and the structure of crystals, and he received many honours during his lifetime.
The species is found in tropical forests of South America, where it feeds mainly on fruit and small insects. Adults reach a length of about twenty centimetres and have a dark brown back with a pale yellow belly. The female lays two or three eggs in a nest built high in the trees, and both parents take care of the young. It is considered vulnerable by the International Union for Conservation of Nature because of habitat loss and hunting.
The album was released in 1994 by the record company and received generally positive reviews from music critics, who praised the production and the songwriting of the band. It reached number five on the national chart and was certified gold in the United States and Canada. The first single from the album became a major hit and the accompanying video was directed by a well known film maker. The group then toured extensively through Europe, Japan and Australia in support of the record.
During the Second World War the airfield was used by the air force as a training base for pilots and later as a station for bombers. After the end of the war the site was returned to the local authority and part of it was developed as a civil airport. The government announced in 1972 that a new terminal would be constructed, and the building was opened three years later. Today the airport handles more than two million passengers each year and serves as a hub for several regional airlines.
The school was established in 1923 and is one of the oldest educational institutions in the district. It offers courses in science, commerce and the arts, and has about one thousand students. The election was held on the same day as the presidential vote, and the party won a majority of the seats in the national assembly. The team finished third in the league that season and reached the final of the national cup, where it lost to the defending champions."""
PY = """import os
import sys
from typing import list, optional
class Config(object):
    def __init__(self, name, value=None, parent=None):
        self.name = name
        self.value = value
        self.parent = parent
        self._children = []
    def add_child(self, child):
        child.parent = self
        self._children.append(child)
        return child
    def get(self, key, default=None):
        if key in self.__dict__:
            return self.__dict__[key]
        if self.parent is not None:
            return self.parent.get(key, default)
        return default
    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self.name)
def load_file(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        data = f.read()
    lines = data.split('\\n')
    result = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        result.append(line)
    return result
class TestRequest(unittest.TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create(username='test', email='test@example.com')
    def test_get_returns_response(self):
        response = self.client.get('/api/items/', follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('items', response.data)
    def test_post_invalid_data(self):
        data = {'name': '', 'value': None}
        response = self.client.post('/api/items/', data=data)
        self.assertEqual(response.status_code, 400)
def compute_loss(x, y, weights, axis=0):
    n = x.shape[0]
    pred = np.dot(x, weights)
    diff = pred - y
    loss = np.sum(diff ** 2, axis=axis) / n
    if loss > max_value:
        raise ValueError('loss too large: {}'.format(loss))
    return loss, pred
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='out.txt')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    return args
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_size, hidden_size))
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(torch.relu(layer(x)))
        return x
def main():
    args = parse_args()
    logger = logging.getLogger(__name__)
    try:
        result = process(args.input, args.output)
    except (IOError, KeyError) as e:
        logger.error('failed: %s', str(e))
        sys.exit(1)
    if result is None:
        return
    for key, value in sorted(result.items()):
        print(key, value)
if __name__ == '__main__':
    main()
"""
RO = """sono hi wa asa kara ame ga futte ita. watashi wa heya no naka de hon wo yonde ita ga, yagate tachiagatte mado no soto wo mita. niwa no ki wa nurete, tooku no yama wa kumo ni kakurete ita. haha wa daidokoro de nani ka wo shite iru rashikatta. kare wa sono toki hajimete kanojo no kao wo mita no de aru. sore wa utsukushii onna de atta ga, doko ka kanashige na tokoro ga atta. anata wa dou omoimasu ka to kiku to, otoko wa shibaraku damatte ita ga, yagate shizuka ni kuchi wo hiraita.
watakushi wa sono sensei wo tsune ni sensei to yonde ita. dakara koko demo tada sensei to kaku dake de honmyou wa uchiakenai. kore wa seken wo habakaru enryo to iu yori mo, sono hou ga watakushi ni totte shizen dakara de aru. watakushi wa sono hito no kioku wo yobiokosu goto ni, sugu sensei to iitaku naru. fude wo totte mo kokoromochi wa onaji koto de aru.
kuni no hito wa minna sou iu fuu ni kangaete ita. mura no kodomotachi wa kawa no hotori de asonde ita. toshi wo totta shoumin no kurashi wa mazushiku, fuyu ni naru to yuki ga fukaku tsumotte michi mo wakaranaku natta. shikashi haru ga kuru to sakura no hana ga saite, hitobito wa yorokonde yama e dekaketa."""
def norm(t):
    out=[]
    for ln in t.lower().split('\n'):
        m=re.match(r'^ *',ln);lead=m.group(0)
        body=re.sub(r'[^a-z]+',' ',ln[len(lead):]).rstrip()
        out.append(lead+body)
    return '\n'.join(out)
V="\n abcdefghijklmnopqrstuvwxyz"
def bigram(t):
    t=norm(t);C=np.zeros((28,28))
    for a,b in zip(t,t[1:]): C[V.index(a),V.index(b)]+=1
    return C
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64);D3=W['ln1_w.3'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o]
K=27
Sp=np.maximum(Sx[:,:K],0)
U3,S3,Vt3=np.linalg.svd(W['W_proj.3'].astype(np.float64),full_matrices=False)
Y=U3[:,1:28]
code=np.linalg.lstsq(Sp,Pa.T,rcond=None)[0]   # K x 384
M31=(W['W_o.3'][:,s(1)]@W['W_v.3'][s(1)]*D3).astype(np.float64)
Lc=(Y.T@M31@code.T).T   # K x 27 copy-logit vectors
beta=code@Y              # K x 27 direct bigram logit vectors
B=beta@np.linalg.pinv(Lc).T*0+beta@Lc.T   # K x K  (cur -> next)
print('B sv',np.linalg.svd(B,compute_uv=False)[:10].round(3))
def dc(M):
    M=M.copy()
    for _ in range(6):
        M=M-np.nanmean(M,axis=1,keepdims=True);M=M-np.nanmean(M,axis=0,keepdims=True)
    return M
Bc=dc(B)
print('symmetry of B',np.corrcoef(Bc.ravel(),Bc.T.ravel())[0,1],'diag mean z',(np.diag(Bc)/Bc.std()).mean())
rng=np.random.default_rng(0)
def qap(A,T,restarts,sweeps):
    n=A.shape[0];N=T.shape[0];best=-9;bp=None
    A=A/np.linalg.norm(A)
    for rs in range(restarts):
        p=rng.permutation(N)[:n].copy();rest=[x for x in range(N) if x not in set(p.tolist())]
        def sc(p):
            sub=dc(T[np.ix_(p,p)]);return (A*sub).sum()/np.linalg.norm(sub)
        cur=sc(p)
        for it in range(sweeps*n*n):
            i=rng.integers(n);j=rng.integers(n+len(rest));q=p.copy()
            if j<n: q[i],q[j]=q[j],q[i]
            else: q[i]=rest[j-n]
            v=sc(q)
            if v>cur:
                if j>=n: rest[j-n]=p[i]
                p=q;cur=v
        if cur>best: best=cur;bp=p.copy()
    return best,bp
t0=time.time()
for nm,t in [('EN',EN),('PY',PY),('RO',RO)]:
    C=bigram(t);T=np.log(C+.5)
    b,p=qap(Bc,T,restarts=6,sweeps=2)
    print(nm,'best corr',round(b,3),[V[i] if V[i]>' ' else repr(V[i]) for i in p],round(time.time()-t0))

```

Output:

```
B sv [0.05 0.04 0.04 0.04 0.03 0.03 0.03 0.02 0.02 0.02]
symmetry of B 0.1449954530538057 diag mean z 0.6292194989546028
EN best corr 0.359 ['z', 'w', 'm', 'i', 'r', 'n', 'b', 'p', 'g', 'f', 'a', 'k', 'q', 'o', 'h', 'x', 'e', "'\\n'", 'v', 'j', 'c', 't', 'd', 'u', 'y', 'l', "' '"] 1
PY best corr 0.337 ['l', "' '", 'x', 'j', 'y', 'p', 'a', "'\\n'", 'z', 'w', 'd', 'h', 'k', 't', 'n', 'v', 's', 'q', 'b', 'u', 'g', 'e', 'o', 'f', 'm', 'c', 'r'] 1
RO best corr 0.327 ['p', 'u', 'h', 'c', 'b', "'\\n'", 'k', 'i', 't', 'v', 'g', 'z', 's', 'm', "' '", 'f', 'd', 'l', 'x', 'y', 'j', 'n', 'w', 'o', 'r', 'q', 'a'] 2

```

## Call 37

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=1,suppress=True,linewidth=230)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);D2=W['ln2_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);A=A[:,o]
Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0);K=27;a=a[:,:K]
gain=[222,78,72,360,103,254,273,25,249,226]
mask=np.ones(384,bool);mask[gain]=False
tab=[]
for l in range(4):
    Dl=W[f'ln1_w.{l}'].astype(np.float64);b=W[f'ln1_b.{l}'].astype(np.float64)
    for j in range(6):
        Wq=W[f'W_q.{l}'][s(j)].astype(np.float64);Wk=W[f'W_k.{l}'][s(j)].astype(np.float64)
        g=(Wk*Dl).T@(Wq@b)
        g2=g*mask
        w=g2@a
        tab.append(w/w.std())
        print(f'L{l}H{j} |g| {np.linalg.norm(g):5.1f} |g no gain| {np.linalg.norm(g2):5.2f} token pref z',w.__div__(w.std()).round(1) if False else (w/w.std()).round(0).astype(int))
tab=np.array(tab)
print('mean by axis L1-3',tab[6:].mean(0).round(1))

```

Output:

```
L0H0 |g|   0.1 |g no gain|  0.10 token pref z [ 0 -1  0  0 -1  0  1  3  0  0  0  0  0  0  0 -2  0  0  0  3 -1  0  0  0 -1  0  0]
L0H1 |g|   0.0 |g no gain|  0.02 token pref z [-1  0 -2  2 -1  0 -1  3  0  0 -1  0  2  1  0 -1  0  0  0  1  1  0  1  0 -1  1  0]
L0H2 |g|   1.1 |g no gain|  1.10 token pref z [ 0 -1  2  0  0  0  1  1  1  2 -1  3  1 -1  0 -1  0  2  1  1  0  1 -1  1  1 -1  2]
L0H3 |g|   0.4 |g no gain|  0.41 token pref z [ 1  0  2  2  0  1  1 -1  0  1  0  2 -1  0  2  0  0  2  0  0  2  1  0  2  0  3  1]
L0H4 |g|   0.1 |g no gain|  0.09 token pref z [ 0 -2 -1  1 -2 -1  1 -1 -1 -1 -1 -1  0 -2  0 -1  0 -1  0  2 -1  0  0  0 -2 -3 -1]
L0H5 |g|   0.1 |g no gain|  0.10 token pref z [ 0  0 -1  0 -2  0  0  2 -1 -1 -1 -1  0 -1 -1 -1 -1 -1 -1  3 -2 -1 -1 -1 -1 -2 -2]
L1H0 |g|  10.9 |g no gain|  2.50 token pref z [ 0  0  1  0  1  1  0 -2  1  0  1  1  0  1  1  0  0  1  2 -3  1  0  0  0  2 -1  1]
L1H1 |g|  18.3 |g no gain|  3.41 token pref z [ 0  2  0  0  1 -1 -1 -1  0  0 -1  0  0  1  1 -1  0  0 -1 -1  0  1  1  3  0  2 -1]
L1H2 |g|  20.7 |g no gain|  3.39 token pref z [-1 -1  0  1 -1  0  1 -1  0  0 -1 -1 -3  0  1  0  0  0  0  0  1  2  0  0 -1  1  1]
L1H3 |g|  18.8 |g no gain|  3.67 token pref z [ 1 -1  1  0  1  0  0  0  0  0  0  1  0  1  0 -2  1  0  0  1 -1 -1 -2 -2  0 -1 -1]
L1H4 |g|  18.3 |g no gain|  3.00 token pref z [-1  1  0  0  1  0  0  2 -1 -2  0  1  1 -1  0  0  0 -1  1  3  0 -1  1  1 -1  1 -1]
L1H5 |g|  14.0 |g no gain|  2.75 token pref z [ 3  0  2  1  2  1 -1  0  0 -1  0  1  0  2  1  1 -1  2  2  2  0  0  0  0  1  1  0]
L2H0 |g|  13.6 |g no gain|  4.28 token pref z [ 0  1  1  2  0  0 -1  0  1 -1  0  1  0 -1  1  1  0 -2  0  0 -1 -2 -1 -1 -1  0  1]
L2H1 |g|   6.0 |g no gain|  2.63 token pref z [ 0  3  1 -1 -1 -1  0  0  0  1  0 -1  0 -2 -1 -2  0  0  1 -1  1  2 -1  1  0  0 -1]
L2H2 |g|   6.0 |g no gain|  2.25 token pref z [-1  2 -1 -1  0 -1  0  0  0  0  0  0  0  1  0 -1  0  0  2 -2  1  0  1  2  0  1 -1]
L2H3 |g|  11.7 |g no gain|  3.27 token pref z [ 0 -1 -1 -1 -1 -1  0  0  1  1  0  0 -1 -2  1  0 -1 -1 -1  2  1 -1 -1 -2  0 -2 -1]
L2H4 |g|  12.6 |g no gain|  3.74 token pref z [ 0  1 -1 -1  1 -2  0  1 -1  0 -2 -1 -1  1 -1 -1  0  1  0  3  0  0  0 -1 -1 -1  0]
L2H5 |g|  12.1 |g no gain|  3.93 token pref z [ 0 -2  0  0 -1  0  0 -1  0  0 -1 -1 -1 -3  0  2 -1 -2 -2  1 -1 -1  0  1  1 -1 -1]
L3H0 |g|   2.1 |g no gain|  1.88 token pref z [ 0  0  0 -1  0  1  1  2  2  1  0  0  0 -2 -1  2  0 -1 -1 -1  1 -1  0  0  0 -2 -2]
L3H1 |g|   1.9 |g no gain|  1.85 token pref z [ 0  0  0 -1  0 -2  0  0  0 -1  0  0  0  1 -1 -1  2  0  1 -1  0  0  2 -1  0  1  2]
L3H2 |g|   2.3 |g no gain|  1.31 token pref z [ 0 -1  1  0 -1  1  0  1  0  1  1  1  0  0  0  0 -1 -2  0  0 -2  1 -2 -2 -2  0 -1]
L3H3 |g|   2.9 |g no gain|  1.46 token pref z [ 0  2  0 -1 -1  1 -2 -1  1  1  0  0  0  0 -1  1 -1  1 -1  1  1 -2 -1  0 -1 -2 -1]
L3H4 |g|   2.5 |g no gain|  1.63 token pref z [ 0  1  0  0 -1  0 -2  0  0  1  1 -1  1  0  0  0  1 -2 -1 -1  1 -2  1 -1 -1 -1 -2]
L3H5 |g|   1.3 |g no gain|  1.28 token pref z [ 1  1  0 -2  1 -1 -1  0  0  0  0  1 -1 -1 -1 -2  2  1 -1 -1 -2  0  1  0  0  0 -1]
mean by axis L1-3 [ 0.1  0.4  0.2 -0.2  0.  -0.2 -0.4  0.   0.2  0.  -0.2  0.  -0.2 -0.1  0.  -0.2  0.1 -0.2  0.1  0.   0.  -0.2 -0.1 -0.2 -0.2 -0.1 -0.3]

```

## Call 38

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=1,suppress=True,linewidth=230)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);A=A[:,o]
Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0);K=26;a=a[:,:K]
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
f=M2@a;fn=f/np.linalg.norm(f,axis=0)
def dz(Mx):
    d=np.diag(Mx);o=Mx[~np.eye(K,dtype=bool)]
    return (d.mean()-o.mean())/o.std()
for l in range(1,4):
    Dl=W[f'ln1_w.{l}'].astype(np.float64)
    for j in range(6):
        q=W[f'W_q.{l}'][s(j)].astype(np.float64)*Dl;kk=W[f'W_k.{l}'][s(j)].astype(np.float64)*Dl
        Fm=q.T@kk
        vals=[dz(a.T@Fm@fn),dz(a.T@Fm@a),dz(fn.T@Fm@a),dz(fn.T@Fm@fn)]
        print(f'L{l}H{j} diag z: q:cur k:prev {vals[0]:5.1f} | q:cur k:cur {vals[1]:5.1f} | q:prev k:cur {vals[2]:5.1f} | q:prev k:prev {vals[3]:5.1f}')
Dl=W['ln1_w.3'].astype(np.float64)
q=W['W_q.3'][s(1)].astype(np.float64)*Dl;kk=W['W_k.3'][s(1)].astype(np.float64)*Dl
Mx=a.T@(q.T@kk)@fn
print('L3H1 token copy diag',np.diag(Mx).round(1));print('offdiag std',Mx[~np.eye(K,dtype=bool)].std().round(2))

```

Output:

```
L1H0 diag z: q:cur k:prev  -0.4 | q:cur k:cur  -2.8 | q:prev k:cur  -1.7 | q:prev k:prev  -1.0
L1H1 diag z: q:cur k:prev  -0.2 | q:cur k:cur  -3.1 | q:prev k:cur  -0.4 | q:prev k:prev  -0.3
L1H2 diag z: q:cur k:prev   0.0 | q:cur k:cur  -2.8 | q:prev k:cur   0.4 | q:prev k:prev  -0.9
L1H3 diag z: q:cur k:prev  -0.3 | q:cur k:cur  -2.1 | q:prev k:cur   0.2 | q:prev k:prev  -1.4
L1H4 diag z: q:cur k:prev  -0.1 | q:cur k:cur  -3.1 | q:prev k:cur  -0.2 | q:prev k:prev  -1.2
L1H5 diag z: q:cur k:prev  -0.8 | q:cur k:cur  -2.5 | q:prev k:cur  -1.0 | q:prev k:prev  -1.3
L2H0 diag z: q:cur k:prev  -0.7 | q:cur k:cur  -0.1 | q:prev k:cur  -0.2 | q:prev k:prev  -0.5
L2H1 diag z: q:cur k:prev   0.2 | q:cur k:cur   0.4 | q:prev k:cur   1.1 | q:prev k:prev   0.0
L2H2 diag z: q:cur k:prev   0.0 | q:cur k:cur  -0.8 | q:prev k:cur  -0.7 | q:prev k:prev  -1.5
L2H3 diag z: q:cur k:prev  -0.0 | q:cur k:cur  -0.7 | q:prev k:cur   0.3 | q:prev k:prev  -1.3
L2H4 diag z: q:cur k:prev  -0.8 | q:cur k:cur  -0.9 | q:prev k:cur  -0.1 | q:prev k:prev  -1.2
L2H5 diag z: q:cur k:prev  -0.1 | q:cur k:cur  -0.7 | q:prev k:cur  -0.1 | q:prev k:prev  -1.0
L3H0 diag z: q:cur k:prev   0.4 | q:cur k:cur   1.6 | q:prev k:cur   1.0 | q:prev k:prev   1.1
L3H1 diag z: q:cur k:prev   3.0 | q:cur k:cur   2.9 | q:prev k:cur   0.1 | q:prev k:prev   1.2
L3H2 diag z: q:cur k:prev   0.2 | q:cur k:cur   0.4 | q:prev k:cur   0.3 | q:prev k:prev   0.3
L3H3 diag z: q:cur k:prev   0.0 | q:cur k:cur  -0.4 | q:prev k:cur   0.1 | q:prev k:prev  -0.0
L3H4 diag z: q:cur k:prev  -0.1 | q:cur k:cur   0.5 | q:prev k:cur  -0.1 | q:prev k:prev   0.4
L3H5 diag z: q:cur k:prev   0.5 | q:cur k:cur   0.4 | q:prev k:cur   0.1 | q:prev k:prev   0.1
L3H1 token copy diag [ 0.1  0.5  0.3  0.1  0.1  0.2  0.2  0.2  0.2  0.3  0.4 -0.   0.1  0.3  0.   0.2  0.2  0.2  0.2  0.2  0.1  0.2  0.2  0.2  0.2  0.4]
offdiag std 0.07

```

## Call 39

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=200)
Wp=W['W_proj.3'].astype(np.float64);Fc=W['W_fc.3'].astype(np.float64)
U,S,Vt=np.linalg.svd(Wp,full_matrices=False)
for idx in [21,23,24,25,26,27,28,29,30,31]:
    v=Vt[idx];o=np.argsort(-np.abs(v))[:6]
    print('comp',idx,'sv',S[idx].round(1),'top units',o,'load',v[o].round(2),'colnorm',np.linalg.norm(Wp[:,o],axis=0).round(1),'rownorm',np.linalg.norm(Fc[o],axis=1).round(1),'energy in top6 %.2f'%((v[o]**2).sum()))
cn=np.linalg.norm(Wp,axis=0);o=np.argsort(-cn)[:12]
print('top col units',o,cn[o].round(2))
Y=U[:,1:28]
print('energy of those cols in Y',(np.linalg.norm(Y.T@Wp[:,o],axis=0)/cn[o]).round(2))
print('their top raw dims',np.abs(Wp[:,o]).argmax(0),np.abs(Wp[:,o]).max(0).round(2))
# units that write to raw dim 103 etc.
for d in [103,249,226,124,222,78]:
    w=Wp[d];oo=np.argsort(-np.abs(w))[:5]
    print('raw dim',d,'norm',np.linalg.norm(w).round(1),'mean',w.mean().round(3),'std',w.std().round(3),'top units',oo,w[oo].round(2))
# relation among hidden loadings of raw dims and b direction
b=W['ln2_b.3'];fb=Fc@b
print('corr of raw dim 103 weights with fc@b',np.corrcoef(Wp[103],fb)[0,1])

```

Output:

```
comp 21 sv 11.6 top units [1533   28 1373  695 1051 1238] load [ 0.09  0.09 -0.09 -0.08 -0.08 -0.08] colnorm [2.1 1.9 2.9 2.  2.  2. ] rownorm [2.  2.  2.1 2.  2.  2. ] energy in top6 0.04
comp 23 sv 8.4 top units [ 356  589 1287  814 1290  866] load [-0.14  0.1  -0.09  0.08 -0.08 -0.08] colnorm [3.2 2.5 2.2 2.3 2.5 2.4] rownorm [2.2 2.  2.1 2.  2.1 2.5] energy in top6 0.06
comp 24 sv 8.1 top units [ 356 1316 1233  284  866  442] load [ 0.31 -0.23  0.17 -0.17  0.15 -0.11] colnorm [3.2 2.6 2.1 2.2 2.4 2. ] rownorm [2.2 2.3 2.  2.  2.5 2. ] energy in top6 0.24
comp 25 sv 7.9 top units [342 781 151 196 395 719] load [-0.1  -0.09 -0.08 -0.08  0.08 -0.07] colnorm [2.  2.2 2.  2.5 2.3 2. ] rownorm [2.  2.  1.9 2.1 2.1 2. ] energy in top6 0.04
comp 26 sv 7.5 top units [1315 1467  873 1534  231  775] load [ 0.1   0.09 -0.09  0.09  0.08  0.08] colnorm [2.5 2.4 2.3 2.3 2.4 2. ] rownorm [2.1 2.  2.1 2.1 2.1 2. ] energy in top6 0.05
comp 27 sv 7.2 top units [ 948  479  590  394 1351   74] load [ 0.08 -0.08 -0.08 -0.08  0.07  0.07] colnorm [2.2 2.4 1.9 2.  2.  2.1] rownorm [2.  2.  2.  2.  1.9 2. ] energy in top6 0.04
comp 28 sv 5.8 top units [ 656  977  771 1495   94  729] load [-0.13 -0.13 -0.12 -0.12 -0.12 -0.11] colnorm [2.4 2.9 2.4 2.3 2.4 2.4] rownorm [2.1 2.5 2.2 2.2 2.5 2.2] energy in top6 0.09
comp 29 sv 4.1 top units [1123  452  114  796  167 1107] load [ 0.18 -0.17  0.11  0.09  0.09  0.09] colnorm [3.1 2.9 1.7 1.7 1.6 1.5] rownorm [2.1 2.2 1.9 1.9 1.9 1.9] energy in top6 0.10
comp 30 sv 3.2 top units [ 452  564  182  976  937 1481] load [-0.21 -0.13 -0.12 -0.12 -0.11 -0.1 ] colnorm [2.9 2.8 2.  1.9 1.9 2.2] rownorm [2.2 2.2 2.  1.9 1.9 2. ] energy in top6 0.11
comp 31 sv 3.1 top units [1123  254  452 1497  771   94] load [-0.24 -0.11 -0.09  0.09 -0.09 -0.09] colnorm [3.1 4.1 2.9 2.  2.4 2.4] rownorm [2.1 2.3 2.2 2.  2.2 2.5] energy in top6 0.10
top col units [1473  254  655 1084  800  356  480 1123 1301  265  165  977] [4.82 4.05 3.45 3.28 3.26 3.19 3.12 3.11 2.97 2.94 2.94 2.93]
energy of those cols in Y [0.95 0.97 0.75 0.94 0.39 0.98 0.98 0.76 0.91 0.9  0.75 0.66]
their top raw dims [103 123 103 249 103 224 160 249 103 103 103 103] [0.73 0.54 1.42 0.51 2.01 0.44 0.41 1.09 0.72 0.79 1.03 1.19]
raw dim 103 norm 17.0 mean 0.058 std 0.429 top units [ 800  564 1379  655  659] [-2.01 -1.52  1.51 -1.42  1.41]
raw dim 249 norm 10.9 mean -0.04 std 0.276 top units [ 800 1123 1255  655  951] [ 1.13 -1.09 -1.02  0.95 -0.92]
raw dim 226 norm 10.5 mean -0.044 std 0.264 top units [1528  977 1507  362  830] [-1.17 -1.06 -1.06 -1.04 -1.  ]
raw dim 124 norm 8.8 mean 0.032 std 0.221 top units [ 165  107  800  659 1379] [-0.77  0.7  -0.69  0.69  0.67]
raw dim 222 norm 6.2 mean 0.001 std 0.158 top units [1123  452  564  182  137] [ 0.78 -0.71 -0.66 -0.52 -0.51]
raw dim 78 norm 5.1 mean 0.03 std 0.126 top units [1123    8 1502 1111 1101] [ 0.84 -0.49  0.46  0.42  0.41]
corr of raw dim 103 weights with fc@b 0.08544076691383069

```

## Call 40

```python
import numpy as np, warnings, re, time
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=200)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64);D3=W['ln1_w.3'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0);K=27;a=a[:,:K]
Sp=np.maximum(Sx[:,:K],0)
code=np.linalg.lstsq(Sp,Pa.T,rcond=None)[0]   # K x 384: token code written by L0 MLP
U3,S3,Vt3=np.linalg.svd(W['W_proj.3'].astype(np.float64),full_matrices=False)
Y=U3[:,1:28]
LR=[]
for l in range(1,4):
    Lin=(Y.T@W[f'W_proj.{l}'].astype(np.float64))@(W[f'W_fc.{l}'].astype(np.float64)*W[f'ln2_w.{l}'].astype(np.float64))
    LR.append(Lin)
    print('layer',l,'linear readout sv',np.linalg.svd(Lin,compute_uv=False)[:12].round(1))
L3lin=LR[2]
print('readout energy on a:',(np.linalg.norm(L3lin@a,axis=0)).round(1))
print('readout energy on code dirs:',(np.linalg.norm(L3lin@(code/np.linalg.norm(code,axis=1,keepdims=True)).T,axis=0)).round(1))
rand=np.random.default_rng(0).standard_normal((384,27));rand/=np.linalg.norm(rand,axis=0)
print('rand dir',np.linalg.norm(L3lin@rand,axis=0).mean().round(1))
cn=(code/np.linalg.norm(code,axis=1,keepdims=True)).T
Lc=Y.T@(W['W_o.3'][:,s(1)].astype(np.float64)@W['W_v.3'][s(1)].astype(np.float64)*D3)@cn
print('copy logit norm',np.linalg.norm(Lc,axis=0).round(1))
B=(L3lin@cn).T@Lc   # cur -> next
print('B sv',np.linalg.svd(B,compute_uv=False)[:14].round(0))
print('corr B3-diag:',(np.diag(B)-B.mean())/B.std())
# M: same with a

```

Output:

```
layer 1 linear readout sv [21.1 13.  11.8 11.3 10.8 10.7 10.3  9.9  9.6  9.3  8.8  8.3]
layer 2 linear readout sv [18.3 17.8 16.8 16.5 15.9 15.4 15.2 14.9 14.3 13.9 13.7 13.3]
layer 3 linear readout sv [51.6 45.  41.8 37.9 37.6 36.6 36.3 35.  34.1 33.3 32.6 30.7]
readout energy on a: [ 9.1 13.1 10.8  7.2  9.2  9.7  8.9  8.6  8.   8.3  6.   6.   8.5  8.2  7.7  8.2  8.4 11.5  5.6 10.4  6.8  8.9 12.4  9.5  9.2 13.3  9. ]
readout energy on code dirs: [ 9.1  8.9 11.5  9.7  8.8 10.9  9.9  9.8 10.   7.   8.6  6.3 11.  11.5  8.7 10.9 10.3 10.9  9.4 16.9  9.9  9.4  9.6  9.7 11.   7.4  8.3]
rand dir 7.8
copy logit norm [1.9 0.8 0.6 1.2 1.1 0.6 1.  1.9 1.  0.6 0.6 0.7 0.7 0.6 1.2 0.9 0.9 0.7 0.8 1.4 1.  0.5 0.7 0.5 1.1 0.5 1. ]
B sv [36. 20. 16. 15. 13. 11.  9.  8.  7.  6.  6.  5.  4.  4.]
corr B3-diag: [ 2.   -0.15 -1.43  0.69 -0.21  0.77  0.82  5.   -0.83  0.55 -0.49 -0.   -0.33 -0.36 -0.5  -1.7   1.73 -0.09  0.92 -3.68  2.36  0.84 -0.01  0.46  1.16  0.26 -0.91]

```

## Call 41

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=1,suppress=True,linewidth=230)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0);K=26;a=a[:,:K]
Sp=np.maximum(Sx[:,:K],0)
code=np.linalg.lstsq(Sp,Pa.T,rcond=None)[0].T;c=code/np.linalg.norm(code,axis=0)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
f=M2@a;f=f/np.linalg.norm(f,axis=0)
sets={'a':a,'f':f,'c':c}
print('overlaps diag a-c',np.diag(a.T@c).round(2)[:10],'f-c',np.diag(f.T@c).round(2)[:10],'offdiag a-c std',(a.T@c).std().round(3))
def dz(Mx):
    d=np.diag(Mx);o=Mx[~np.eye(K,dtype=bool)]
    return (d.mean()-o.mean())/o.std()
pairs=[('a','f'),('a','a'),('f','a'),('f','f'),('c','c'),('c','f'),('f','c'),('a','c'),('c','a')]
print('      '+'  '.join(f'q:{x},k:{y}' for x,y in pairs))
for l in range(1,4):
    Dl=W[f'ln1_w.{l}'].astype(np.float64)
    for j in range(6):
        q=W[f'W_q.{l}'][s(j)].astype(np.float64)*Dl;kk=W[f'W_k.{l}'][s(j)].astype(np.float64)*Dl
        Fm=q.T@kk
        print(f'L{l}H{j} '+'  '.join(f'{dz(sets[x].T@Fm@sets[y]):7.1f}' for x,y in pairs))
print('V reads: energy of W_v rows on sets (per unit, chance=1)')
for l in range(1,4):
    Dl=W[f'ln1_w.{l}'].astype(np.float64)
    for j in range(6):
        V=W[f'W_v.{l}'][s(j)].astype(np.float64)*Dl
        tot=np.linalg.norm(V)**2/384
        print(f'L{l}H{j} v on a {(np.linalg.norm(V@a)**2/K/tot):.1f} f {(np.linalg.norm(V@f)**2/K/tot):.1f} c {(np.linalg.norm(V@c)**2/K/tot):.1f} | q on a {(np.linalg.norm((W[f"W_q.{l}"][s(j)]*Dl)@a)**2/K/(np.linalg.norm(W[f"W_q.{l}"][s(j)]*Dl)**2/384)):.1f} f {(np.linalg.norm((W[f"W_q.{l}"][s(j)]*Dl)@f)**2/K/(np.linalg.norm(W[f"W_q.{l}"][s(j)]*Dl)**2/384)):.1f} c {(np.linalg.norm((W[f"W_q.{l}"][s(j)]*Dl)@c)**2/K/(np.linalg.norm(W[f"W_q.{l}"][s(j)]*Dl)**2/384)):.1f} | k on a {(np.linalg.norm((W[f"W_k.{l}"][s(j)]*Dl)@a)**2/K/(np.linalg.norm(W[f"W_k.{l}"][s(j)]*Dl)**2/384)):.1f} f {(np.linalg.norm((W[f"W_k.{l}"][s(j)]*Dl)@f)**2/K/(np.linalg.norm(W[f"W_k.{l}"][s(j)]*Dl)**2/384)):.1f} c {(np.linalg.norm((W[f"W_k.{l}"][s(j)]*Dl)@c)**2/K/(np.linalg.norm(W[f"W_k.{l}"][s(j)]*Dl)**2/384)):.1f}')

```

Output:

```
overlaps diag a-c [0.4 0.1 0.  0.3 0.5 0.2 0.2 0.2 0.3 0.2] f-c [ 0.  -0.  -0.1 -0.  -0.1 -0.1 -0.1  0.  -0.1 -0. ] offdiag a-c std 0.062
      q:a,k:f  q:a,k:a  q:f,k:a  q:f,k:f  q:c,k:c  q:c,k:f  q:f,k:c  q:a,k:c  q:c,k:a
L1H0    -0.4     -2.8     -1.7     -1.0     -1.1      0.4     -0.5     -0.8     -1.3
L1H1    -0.2     -3.1     -0.4     -0.3     -1.4      0.1      0.1     -1.6     -1.8
L1H2     0.0     -2.8      0.4     -0.9     -1.8      0.4      0.9     -1.5     -2.4
L1H3    -0.3     -2.1      0.2     -1.4     -1.1      0.2      1.0     -0.6     -1.1
L1H4    -0.1     -3.1     -0.2     -1.2     -1.1      0.1     -0.2     -1.6     -1.5
L1H5    -0.8     -2.5     -1.0     -1.3     -0.9     -0.0     -0.3     -1.1     -1.2
L2H0    -0.7     -0.1     -0.2     -0.5     -0.3     -0.6      0.4      0.2     -0.6
L2H1     0.2      0.4      1.1      0.0     -0.3     -0.8      1.1      0.1     -2.0
L2H2     0.0     -0.8     -0.7     -1.5     -0.8      0.2     -0.1     -0.3     -0.3
L2H3    -0.0     -0.7      0.3     -1.3     -1.3     -0.3      0.3     -0.6     -1.2
L2H4    -0.8     -0.9     -0.1     -1.2     -0.6     -0.0      0.2     -0.2     -0.5
L2H5    -0.1     -0.7     -0.1     -1.0     -0.6      0.1      0.5     -0.2     -0.7
L3H0     0.4      1.6      1.0      1.1      1.2     -0.8      1.3      0.4     -1.4
L3H1     3.0      2.9      0.1      1.2      1.8      3.0     -2.8     -3.7      0.6
L3H2     0.2      0.4      0.3      0.3     -0.5     -0.2      0.2     -0.2     -0.1
L3H3     0.0     -0.4      0.1     -0.0     -0.2     -0.2      0.4     -0.0     -0.4
L3H4    -0.1      0.5     -0.1      0.4      0.2     -0.4     -0.1     -0.4     -0.4
L3H5     0.5      0.4      0.1      0.1     -0.4      0.1     -0.1     -0.4      0.1
V reads: energy of W_v rows on sets (per unit, chance=1)
L1H0 v on a 2.1 f 4.6 c 2.3 | q on a 2.7 f 3.8 c 1.9 | k on a 1.2 f 1.3 c 2.4
L1H1 v on a 2.6 f 4.5 c 3.4 | q on a 3.2 f 3.7 c 2.6 | k on a 1.4 f 1.9 c 2.6
L1H2 v on a 2.5 f 5.8 c 3.2 | q on a 2.9 f 4.3 c 2.5 | k on a 1.1 f 1.6 c 2.6
L1H3 v on a 2.1 f 9.2 c 2.2 | q on a 2.3 f 4.6 c 1.9 | k on a 1.1 f 2.1 c 2.3
L1H4 v on a 2.9 f 5.7 c 3.3 | q on a 2.8 f 4.4 c 2.1 | k on a 1.1 f 1.6 c 2.6
L1H5 v on a 3.3 f 4.7 c 3.6 | q on a 2.7 f 4.9 c 2.2 | k on a 1.1 f 2.9 c 2.6
L2H0 v on a 1.2 f 1.6 c 1.5 | q on a 1.2 f 1.3 c 1.3 | k on a 1.0 f 1.3 c 1.5
L2H1 v on a 1.1 f 1.2 c 1.4 | q on a 1.2 f 1.4 c 1.2 | k on a 1.2 f 1.1 c 1.3
L2H2 v on a 1.2 f 1.2 c 1.3 | q on a 1.1 f 1.1 c 1.0 | k on a 1.0 f 1.0 c 1.4
L2H3 v on a 1.1 f 1.2 c 1.5 | q on a 1.1 f 1.2 c 1.2 | k on a 0.9 f 0.9 c 1.3
L2H4 v on a 1.2 f 1.4 c 1.5 | q on a 1.2 f 1.3 c 1.2 | k on a 0.9 f 1.0 c 1.4
L2H5 v on a 1.2 f 1.6 c 1.4 | q on a 1.3 f 1.2 c 1.2 | k on a 1.0 f 1.1 c 1.3
L3H0 v on a 1.1 f 1.2 c 0.9 | q on a 1.1 f 1.1 c 1.1 | k on a 1.2 f 1.1 c 1.0
L3H1 v on a 1.9 f 0.7 c 0.6 | q on a 1.4 f 1.4 c 1.1 | k on a 1.0 f 2.6 c 1.1
L3H2 v on a 1.0 f 1.0 c 0.9 | q on a 1.0 f 0.8 c 0.9 | k on a 1.0 f 0.9 c 1.0
L3H3 v on a 1.0 f 0.9 c 1.0 | q on a 1.0 f 0.8 c 0.9 | k on a 0.9 f 0.8 c 1.0
L3H4 v on a 1.0 f 1.0 c 0.9 | q on a 1.0 f 0.8 c 0.9 | k on a 1.0 f 1.0 c 0.9
L3H5 v on a 1.0 f 1.0 c 0.6 | q on a 1.1 f 0.8 c 0.9 | k on a 1.0 f 1.0 c 0.9

```

## Call 42

```python
import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=200)
s=lambda j:slice(64*j,64*j+64)
gain=[222,78,72,360,103,254,273,25,249,226,124]
for l in [2,3]:
    Dl=W[f'ln1_w.{l}'].astype(np.float64)
    for j in range(6):
        O=W[f'W_o.{l}'][:,s(j)].astype(np.float64);V=W[f'W_v.{l}'][s(j)].astype(np.float64)
        M=O@V*Dl
        u,sv,vt=np.linalg.svd(M)
        ge=np.array([(u[gain,i]**2).sum() for i in range(6)])
        print(f'L{l}H{j} M sv',sv[:6].round(1),'write energy on gain dims',ge.round(2),'top write dims',np.abs(u[:,:3]).argmax(0),' read IPR',(1/(vt[:3]**4).sum(1)).round(0),'read top dims',np.abs(vt[:3]).argmax(1))
# compare q/k forms between heads L3
forms={}
for j in range(6):
    Dl=W['ln1_w.3'].astype(np.float64)
    q=W['W_q.3'][s(j)].astype(np.float64)*Dl;k=W['W_k.3'][s(j)].astype(np.float64)*Dl
    F=(q.T@k);forms[j]=F/np.linalg.norm(F)
print('cos of forms L3',np.array([[np.sum(forms[a]*forms[b]) for b in range(6)] for a in range(6)]).round(2))
forms2={}
for j in range(6):
    Dl=W['ln1_w.2'].astype(np.float64)
    q=W['W_q.2'][s(j)].astype(np.float64)*Dl;k=W['W_k.2'][s(j)].astype(np.float64)*Dl
    F=(q.T@k);forms2[j]=F/np.linalg.norm(F)
print('cos of forms L2',np.array([[np.sum(forms2[a]*forms2[b]) for b in range(6)] for a in range(6)]).round(2))
forms1={}
for j in range(6):
    Dl=W['ln1_w.1'].astype(np.float64)
    q=W['W_q.1'][s(j)].astype(np.float64)*Dl;k=W['W_k.1'][s(j)].astype(np.float64)*Dl
    F=(q.T@k);forms1[j]=F/np.linalg.norm(F)
print('cos of forms L1',np.array([[np.sum(forms1[a]*forms1[b]) for b in range(6)] for a in range(6)]).round(2))
# entropy-like measure: effective rank of q,k after removing gain dims
for l in [1,2,3]:
    Dl=W[f'ln1_w.{l}'].astype(np.float64).copy()
    mask=np.ones(384);mask[gain]=0
    for j in range(6):
        q=W[f'W_q.{l}'][s(j)].astype(np.float64)*Dl*mask;k=W[f'W_k.{l}'][s(j)].astype(np.float64)*Dl*mask
        F=q.T@k
        sv=np.linalg.svd(F,compute_uv=False)
        tr=np.trace(F)
        print(f'L{l}H{j} form(no gain) |F| {np.linalg.norm(F):.1f} sym part frac {np.linalg.norm((F+F.T)/2)**2/np.linalg.norm(F)**2:.2f} trace/|F| {tr/np.linalg.norm(F):.2f} top sv',sv[:5].round(1),'eff rank %.0f'%((sv.sum())**2/(sv**2).sum()))

```

Output:

```
L2H0 M sv [3.5 3.3 3.2 3.2 3.1 3. ] write energy on gain dims [0.11 0.05 0.06 0.01 0.01 0.05] top write dims [103 360 235]  read IPR [126. 115. 118.] read top dims [262 300  31]
L2H1 M sv [3.9 3.7 3.6 3.4 3.4 3.3] write energy on gain dims [0.35 0.04 0.07 0.04 0.06 0.07] top write dims [103  51 273]  read IPR [127. 129. 118.] read top dims [155 377 259]
L2H2 M sv [5.7 5.2 4.9 4.3 4.3 4. ] write energy on gain dims [0.07 0.06 0.02 0.04 0.02 0.1 ] top write dims [ 72  72 383]  read IPR [114. 129. 109.] read top dims [287  94 266]
L2H3 M sv [4.5 4.2 4.  3.6 3.3 3.3] write energy on gain dims [0.1  0.03 0.03 0.03 0.03 0.14] top write dims [360 292 230]  read IPR [122. 130. 115.] read top dims [330 262 274]
L2H4 M sv [3.6 3.4 3.3 3.2 3.1 3.1] write energy on gain dims [0.09 0.1  0.12 0.02 0.05 0.09] top write dims [ 72 360  72]  read IPR [113. 116. 127.] read top dims [113   9 293]
L2H5 M sv [3.5 3.4 3.2 3.2 3.1 3.1] write energy on gain dims [0.1  0.02 0.06 0.03 0.04 0.05] top write dims [254 324 360]  read IPR [137. 125. 112.] read top dims [247   6  22]
L3H0 M sv [6.4 5.8 4.9 4.7 4.5 4.3] write energy on gain dims [0.71 0.35 0.22 0.07 0.1  0.13] top write dims [222 103 222]  read IPR [124. 110. 121.] read top dims [152  74  58]
L3H1 M sv [10.6 10.3 10.2  9.5  9.4  9.3] write energy on gain dims [0.11 0.38 0.27 0.04 0.08 0.04] top write dims [ 78 103 103]  read IPR [105.  97. 103.] read top dims [296  92 218]
L3H2 M sv [9.1 7.5 6.4 6.4 6.1 5.9] write energy on gain dims [0.72 0.36 0.07 0.06 0.04 0.04] top write dims [103  78 222]  read IPR [111. 118. 108.] read top dims [214 172 331]
L3H3 M sv [9.5 7.9 6.9 6.1 5.9 5.8] write energy on gain dims [0.64 0.51 0.16 0.16 0.04 0.13] top write dims [103 222 222]  read IPR [134. 126. 120.] read top dims [368  35 344]
L3H4 M sv [7.3 6.8 5.4 5.3 5.  4.6] write energy on gain dims [0.81 0.68 0.07 0.04 0.17 0.06] top write dims [222  78 103]  read IPR [145. 137. 113.] read top dims [186 181  97]
L3H5 M sv [16.9 14.7 13.4 12.9 12.3 12.2] write energy on gain dims [0.46 0.14 0.07 0.09 0.03 0.08] top write dims [103 222 222]  read IPR [121. 116. 115.] read top dims [344 140  94]
cos of forms L3 [[1.   0.07 0.04 0.04 0.08 0.03]
 [0.07 1.   0.02 0.02 0.03 0.05]
 [0.04 0.02 1.   0.09 0.1  0.05]
 [0.04 0.02 0.09 1.   0.08 0.07]
 [0.08 0.03 0.1  0.08 1.   0.05]
 [0.03 0.05 0.05 0.07 0.05 1.  ]]
cos of forms L2 [[1.   0.08 0.13 0.1  0.11 0.11]
 [0.08 1.   0.09 0.09 0.08 0.07]
 [0.13 0.09 1.   0.12 0.13 0.12]
 [0.1  0.09 0.12 1.   0.12 0.11]
 [0.11 0.08 0.13 0.12 1.   0.11]
 [0.11 0.07 0.12 0.11 0.11 1.  ]]
cos of forms L1 [[1.   0.58 0.42 0.45 0.46 0.47]
 [0.58 1.   0.55 0.49 0.56 0.53]
 [0.42 0.55 1.   0.61 0.67 0.56]
 [0.45 0.49 0.61 1.   0.64 0.56]
 [0.46 0.56 0.67 0.64 1.   0.58]
 [0.47 0.53 0.56 0.56 0.58 1.  ]]
L1H0 form(no gain) |F| 3.9 sym part frac 0.57 trace/|F| -1.55 top sv [2.2 1.6 1.2 1.1 1. ] eff rank 20
L1H1 form(no gain) |F| 6.2 sym part frac 0.59 trace/|F| -1.70 top sv [3.8 2.2 1.6 1.4 1.4] eff rank 19
L1H2 form(no gain) |F| 6.9 sym part frac 0.60 trace/|F| -2.03 top sv [3.5 2.6 2.1 1.8 1.8] eff rank 20
L1H3 form(no gain) |F| 6.1 sym part frac 0.58 trace/|F| -1.69 top sv [3.7 2.  1.8 1.5 1.3] eff rank 19
L1H4 form(no gain) |F| 5.8 sym part frac 0.60 trace/|F| -1.90 top sv [3.2 2.1 1.5 1.5 1.4] eff rank 21
L1H5 form(no gain) |F| 5.9 sym part frac 0.57 trace/|F| -1.54 top sv [3.6 1.8 1.6 1.5 1.4] eff rank 19
L2H0 form(no gain) |F| 18.2 sym part frac 0.50 trace/|F| -0.07 top sv [4.9 4.6 4.3 4.2 4.1] eff rank 46
L2H1 form(no gain) |F| 15.9 sym part frac 0.51 trace/|F| 0.22 top sv [4.  3.4 3.4 3.3 3.1] eff rank 51
L2H2 form(no gain) |F| 11.2 sym part frac 0.51 trace/|F| -0.87 top sv [3.  2.7 2.4 2.3 2.2] eff rank 53
L2H3 form(no gain) |F| 15.6 sym part frac 0.51 trace/|F| -0.45 top sv [4.1 3.7 3.4 3.1 3.1] eff rank 51
L2H4 form(no gain) |F| 17.4 sym part frac 0.50 trace/|F| -0.27 top sv [4.2 4.1 3.9 3.7 3.6] eff rank 49
L2H5 form(no gain) |F| 16.8 sym part frac 0.50 trace/|F| -0.39 top sv [4.3 3.8 3.8 3.7 3.5] eff rank 49
L3H0 form(no gain) |F| 17.5 sym part frac 0.53 trace/|F| 1.52 top sv [3.5 3.3 3.3 3.2 3.1] eff rank 57
L3H1 form(no gain) |F| 14.4 sym part frac 0.61 trace/|F| 3.42 top sv [2.6 2.5 2.4 2.4 2.4] eff rank 60
L3H2 form(no gain) |F| 11.4 sym part frac 0.51 trace/|F| 0.12 top sv [3.2 2.8 2.7 2.5 2.4] eff rank 52
L3H3 form(no gain) |F| 12.1 sym part frac 0.50 trace/|F| 0.04 top sv [3.3 3.  2.7 2.4 2.3] eff rank 55
L3H4 form(no gain) |F| 14.3 sym part frac 0.50 trace/|F| 0.47 top sv [3.8 3.2 3.  2.8 2.7] eff rank 55
L3H5 form(no gain) |F| 6.5 sym part frac 0.51 trace/|F| 0.11 top sv [2.  1.8 1.6 1.5 1.4] eff rank 50

```

## Call 43

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=2,suppress=True,linewidth=200)
s=lambda j:slice(64*j,64*j+64)
gain=[222,78,72,360,103,254,273,25,249,226,124]
mask=np.ones(384);mask[gain]=0
Dl=W['ln1_w.1'].astype(np.float64)*mask
Fb=0
for j in range(6):
    q=W['W_q.1'][s(j)].astype(np.float64)*Dl;k=W['W_k.1'][s(j)].astype(np.float64)*Dl
    F=q.T@k;Fb=Fb+F/np.linalg.norm(F)
u,sv,vt=np.linalg.svd(Fb)
print('mean form sv',sv[:30].round(2))
Up,Sp,Vp=np.linalg.svd(W['W_proj.0'].astype(np.float64),full_matrices=False)
D0=W['ln1_w.0'].astype(np.float64)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
u2,s2,v2=np.linalg.svd(M2)
Q=W['W_q.0'][s(2)];K=W['W_k.0'][s(2)]
_,_,Pt=np.linalg.svd(np.vstack([Q,K]),full_matrices=False);P=Pt[:26].T
F0=W['W_fc.0'].astype(np.float64)*W['ln2_w.0'].astype(np.float64)
_,_,Fv=np.linalg.svd(F0,full_matrices=False)
subs={'P':P,'T0':v2[:30].T,'T1':u2[:,:30],'pj1-30':Up[:,:30],'pj31-60':Up[:,30:60],'fc0top55':Fv[:55].T}
for k1 in range(12):
    print(k1,'sv %.2f'%sv[k1],'qside',{n:round(float(np.linalg.norm(B.T@u[:,k1])**2),2) for n,B in subs.items()},'kside',{n:round(float(np.linalg.norm(B.T@vt[k1])**2),2) for n,B in subs.items()},'u.v %.2f'%(u[:,k1]@vt[k1]))
Fs=(Fb+Fb.T)/2;e=np.linalg.eigvalsh(Fs);print('sym eig low',e[:12].round(2),'high',e[-8:].round(2))
Fa=(Fb-Fb.T)/2;print('antisym top sv',np.linalg.svd(Fa,compute_uv=False)[:10].round(2))
print('total energy in subspaces, q-side',{n:round(float(np.linalg.norm(B.T@u[:,:20])**2/20),2) for n,B in subs.items()},' k-side',{n:round(float(np.linalg.norm(B.T@vt[:20].T)**2/20),2) for n,B in subs.items()})
print('chance',{n:round(B.shape[1]/384,2) for n,B in subs.items()})

```

Output:

```
mean form sv [1.33 1.   0.81 0.73 0.69 0.63 0.59 0.58 0.54 0.51 0.5  0.48 0.48 0.47 0.43 0.42 0.41 0.39 0.37 0.37 0.36 0.34 0.32 0.31 0.3  0.29 0.28 0.26 0.24 0.23]
0 sv 1.33 qside {'P': 0.08, 'T0': 0.07, 'T1': 0.12, 'pj1-30': 0.19, 'pj31-60': 0.16, 'fc0top55': 0.33} kside {'P': 0.06, 'T0': 0.15, 'T1': 0.09, 'pj1-30': 0.52, 'pj31-60': 0.11, 'fc0top55': 0.38} u.v -0.13
1 sv 1.00 qside {'P': 0.07, 'T0': 0.15, 'T1': 0.08, 'pj1-30': 0.1, 'pj31-60': 0.35, 'fc0top55': 0.39} kside {'P': 0.07, 'T0': 0.21, 'T1': 0.04, 'pj1-30': 0.08, 'pj31-60': 0.41, 'fc0top55': 0.31} u.v -0.53
2 sv 0.81 qside {'P': 0.06, 'T0': 0.18, 'T1': 0.12, 'pj1-30': 0.29, 'pj31-60': 0.2, 'fc0top55': 0.4} kside {'P': 0.05, 'T0': 0.19, 'T1': 0.15, 'pj1-30': 0.3, 'pj31-60': 0.2, 'fc0top55': 0.44} u.v -0.63
3 sv 0.73 qside {'P': 0.04, 'T0': 0.16, 'T1': 0.39, 'pj1-30': 0.15, 'pj31-60': 0.36, 'fc0top55': 0.58} kside {'P': 0.04, 'T0': 0.25, 'T1': 0.08, 'pj1-30': 0.24, 'pj31-60': 0.43, 'fc0top55': 0.39} u.v -0.51
4 sv 0.69 qside {'P': 0.05, 'T0': 0.23, 'T1': 0.27, 'pj1-30': 0.15, 'pj31-60': 0.4, 'fc0top55': 0.53} kside {'P': 0.04, 'T0': 0.27, 'T1': 0.26, 'pj1-30': 0.18, 'pj31-60': 0.38, 'fc0top55': 0.5} u.v -0.71
5 sv 0.63 qside {'P': 0.03, 'T0': 0.2, 'T1': 0.19, 'pj1-30': 0.17, 'pj31-60': 0.42, 'fc0top55': 0.51} kside {'P': 0.03, 'T0': 0.32, 'T1': 0.06, 'pj1-30': 0.21, 'pj31-60': 0.44, 'fc0top55': 0.51} u.v -0.76
6 sv 0.59 qside {'P': 0.02, 'T0': 0.39, 'T1': 0.18, 'pj1-30': 0.13, 'pj31-60': 0.39, 'fc0top55': 0.47} kside {'P': 0.04, 'T0': 0.43, 'T1': 0.13, 'pj1-30': 0.17, 'pj31-60': 0.35, 'fc0top55': 0.46} u.v -0.73
7 sv 0.58 qside {'P': 0.07, 'T0': 0.23, 'T1': 0.28, 'pj1-30': 0.15, 'pj31-60': 0.28, 'fc0top55': 0.48} kside {'P': 0.03, 'T0': 0.32, 'T1': 0.13, 'pj1-30': 0.17, 'pj31-60': 0.37, 'fc0top55': 0.44} u.v -0.74
8 sv 0.54 qside {'P': 0.03, 'T0': 0.22, 'T1': 0.2, 'pj1-30': 0.12, 'pj31-60': 0.41, 'fc0top55': 0.44} kside {'P': 0.02, 'T0': 0.29, 'T1': 0.09, 'pj1-30': 0.17, 'pj31-60': 0.37, 'fc0top55': 0.46} u.v -0.82
9 sv 0.51 qside {'P': 0.03, 'T0': 0.29, 'T1': 0.15, 'pj1-30': 0.2, 'pj31-60': 0.39, 'fc0top55': 0.5} kside {'P': 0.03, 'T0': 0.32, 'T1': 0.11, 'pj1-30': 0.22, 'pj31-60': 0.4, 'fc0top55': 0.45} u.v -0.77
10 sv 0.50 qside {'P': 0.04, 'T0': 0.24, 'T1': 0.13, 'pj1-30': 0.17, 'pj31-60': 0.33, 'fc0top55': 0.5} kside {'P': 0.03, 'T0': 0.23, 'T1': 0.15, 'pj1-30': 0.16, 'pj31-60': 0.34, 'fc0top55': 0.5} u.v -0.63
11 sv 0.48 qside {'P': 0.03, 'T0': 0.23, 'T1': 0.16, 'pj1-30': 0.2, 'pj31-60': 0.28, 'fc0top55': 0.42} kside {'P': 0.06, 'T0': 0.26, 'T1': 0.16, 'pj1-30': 0.22, 'pj31-60': 0.29, 'fc0top55': 0.44} u.v -0.57
sym eig low [-0.88 -0.77 -0.66 -0.6  -0.54 -0.52 -0.5  -0.48 -0.46 -0.45 -0.42 -0.41] high [0.1  0.11 0.13 0.14 0.17 0.27 0.35 0.64]
antisym top sv [0.72 0.72 0.42 0.42 0.35 0.35 0.31 0.31 0.26 0.26]
total energy in subspaces, q-side {'P': 0.04, 'T0': 0.23, 'T1': 0.18, 'pj1-30': 0.17, 'pj31-60': 0.35, 'fc0top55': 0.46}  k-side {'P': 0.04, 'T0': 0.27, 'T1': 0.13, 'pj1-30': 0.22, 'pj31-60': 0.35, 'fc0top55': 0.45}
chance {'P': 0.07, 'T0': 0.08, 'T1': 0.08, 'pj1-30': 0.08, 'pj31-60': 0.08, 'fc0top55': 0.14}

```

## Call 44

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0)
for j in [0,1,3,4,5]:
    V=W['W_v.0'][s(j)].astype(np.float64);O=W['W_o.0'][:,s(j)].astype(np.float64)
    M=O@V*D0
    u,sv,vt=np.linalg.svd(M)
    for c in range(3):
        rd=vt[c]
        w=rd@a
        oo=np.argsort(-np.abs(w))[:4]
        print(f'H{j} comp{c} sv {sv[c]:.2f} |proj on a| {np.linalg.norm(w):.2f} top axes {oo} vals {w[oo].round(2)}')
# all V of L0 onto a (with D0)
Vv=(W['W_v.0'].astype(np.float64)*D0)
Vn=np.linalg.norm(Vv@a,axis=0)
print('V read of each token axis',Vn.round(2))
Q=W['W_q.0'].astype(np.float64)*D0;K=W['W_k.0'].astype(np.float64)*D0
for j in range(6):
    print('head',j,'q on axes',np.linalg.norm(Q[s(j)]@a,axis=0).round(2)[:30])
for j in range(6):
    print('head',j,'k on axes',np.linalg.norm(K[s(j)]@a,axis=0).round(2)[:30])
print('D0-weighted leverage of axes: mean ln1_w along axis',np.sqrt(((a**2)*(D0**2)[:,None]).sum(0)).round(2))
print('D2-weighted: ',np.sqrt(((a**2)*(D2**2)[:,None]).sum(0)).round(2))
print('kurt',k[o].round(0))

```

Output:

```
H0 comp0 sv 8.07 |proj on a| 0.12 top axes [27 18 29  9] vals [-0.06 -0.04 -0.04 -0.03]
H0 comp1 sv 3.33 |proj on a| 0.26 top axes [27 25 17 11] vals [ 0.15 -0.1  -0.07 -0.06]
H0 comp2 sv 2.10 |proj on a| 0.17 top axes [19 27  9 17] vals [-0.09 -0.07 -0.05 -0.05]
H1 comp0 sv 5.66 |proj on a| 0.18 top axes [25 18 13 10] vals [-0.07 -0.06 -0.05 -0.05]
H1 comp1 sv 3.83 |proj on a| 0.23 top axes [27 25 17 14] vals [ 0.1  -0.09 -0.08 -0.06]
H1 comp2 sv 1.06 |proj on a| 0.25 top axes [27  1 28  9] vals [0.14 0.09 0.08 0.07]
H3 comp0 sv 1.03 |proj on a| 0.36 top axes [19 27 25 26] vals [ 0.2   0.18 -0.11 -0.09]
H3 comp1 sv 0.97 |proj on a| 0.61 top axes [ 7 20 19  6] vals [-0.53  0.15 -0.12  0.07]
H3 comp2 sv 0.62 |proj on a| 0.65 top axes [19 24 27 28] vals [ 0.54 -0.14 -0.11 -0.11]
H4 comp0 sv 3.40 |proj on a| 0.11 top axes [29 23 13 28] vals [ 0.04 -0.04  0.04 -0.03]
H4 comp1 sv 1.64 |proj on a| 0.16 top axes [27  1 29 21] vals [-0.08 -0.05  0.05 -0.05]
H4 comp2 sv 1.08 |proj on a| 0.28 top axes [27 28  9  2] vals [0.13 0.11 0.1  0.09]
H5 comp0 sv 8.33 |proj on a| 0.16 top axes [27 17 25 19] vals [ 0.09 -0.05 -0.05 -0.04]
H5 comp1 sv 3.61 |proj on a| 0.20 top axes [ 7 19 29 17] vals [0.11 0.09 0.05 0.05]
H5 comp2 sv 1.68 |proj on a| 0.25 top axes [27 19 28 13] vals [ 0.17  0.08  0.07 -0.07]
V read of each token axis [0.24 0.25 0.25 0.26 0.18 0.2  0.25 0.45 0.2  0.23 0.27 0.24 0.21 0.24 0.26 0.26 0.2  0.27 0.23 0.48 0.22 0.19 0.21 0.24 0.24 0.34 0.24 0.42 0.25 0.24]
head 0 q on axes [0.02 0.05 0.03 0.02 0.03 0.04 0.05 0.04 0.02 0.03 0.03 0.05 0.02 0.03 0.03 0.02 0.02 0.07 0.04 0.03 0.03 0.02 0.02 0.04 0.03 0.07 0.05 0.15 0.03 0.04]
head 1 q on axes [0.03 0.04 0.03 0.03 0.03 0.04 0.03 0.03 0.02 0.03 0.03 0.04 0.02 0.03 0.04 0.02 0.03 0.05 0.03 0.02 0.04 0.03 0.02 0.04 0.03 0.07 0.06 0.1  0.04 0.04]
head 2 q on axes [0.21 0.29 0.26 0.27 0.21 0.27 0.24 0.35 0.21 0.23 0.27 0.33 0.24 0.38 0.21 0.21 0.3  0.27 0.34 0.32 0.28 0.29 0.24 0.22 0.19 0.44 0.32 0.4  0.24 0.38]
head 3 q on axes [0.06 0.11 0.08 0.06 0.1  0.07 0.11 0.21 0.08 0.07 0.08 0.1  0.07 0.1  0.11 0.06 0.07 0.12 0.08 0.19 0.11 0.06 0.06 0.1  0.07 0.15 0.13 0.3  0.08 0.07]
head 4 q on axes [0.04 0.04 0.05 0.04 0.03 0.05 0.04 0.07 0.04 0.06 0.04 0.06 0.03 0.06 0.05 0.03 0.04 0.07 0.04 0.03 0.04 0.03 0.02 0.04 0.03 0.1  0.07 0.13 0.05 0.05]
head 5 q on axes [0.03 0.02 0.04 0.04 0.03 0.02 0.03 0.03 0.04 0.04 0.06 0.06 0.04 0.06 0.06 0.02 0.05 0.06 0.06 0.02 0.04 0.02 0.03 0.02 0.04 0.1  0.05 0.05 0.03 0.06]
head 0 k on axes [0.03 0.04 0.03 0.03 0.06 0.04 0.04 0.13 0.03 0.03 0.04 0.04 0.03 0.04 0.03 0.07 0.03 0.04 0.03 0.15 0.05 0.03 0.02 0.03 0.03 0.05 0.03 0.06 0.04 0.05]
head 1 k on axes [0.03 0.03 0.03 0.03 0.04 0.03 0.02 0.05 0.02 0.03 0.03 0.03 0.03 0.04 0.03 0.04 0.03 0.03 0.04 0.03 0.03 0.02 0.03 0.03 0.03 0.04 0.03 0.06 0.03 0.05]
head 2 k on axes [0.27 0.3  0.28 0.26 0.21 0.31 0.25 0.16 0.22 0.29 0.3  0.38 0.28 0.39 0.27 0.25 0.34 0.31 0.38 0.27 0.28 0.29 0.27 0.23 0.26 0.46 0.41 0.46 0.3  0.44]
head 3 k on axes [0.09 0.1  0.17 0.1  0.05 0.08 0.11 0.19 0.07 0.09 0.12 0.13 0.09 0.08 0.15 0.09 0.07 0.11 0.07 0.14 0.12 0.1  0.07 0.13 0.09 0.15 0.11 0.19 0.1  0.12]
head 4 k on axes [0.04 0.05 0.06 0.05 0.07 0.05 0.04 0.05 0.04 0.05 0.05 0.05 0.04 0.08 0.05 0.06 0.04 0.08 0.05 0.08 0.07 0.06 0.04 0.05 0.06 0.13 0.14 0.07 0.12 0.05]
head 5 k on axes [0.03 0.03 0.05 0.04 0.06 0.03 0.03 0.06 0.03 0.05 0.05 0.05 0.03 0.06 0.05 0.05 0.04 0.06 0.05 0.1  0.07 0.04 0.03 0.06 0.05 0.11 0.08 0.06 0.07 0.05]
D0-weighted leverage of axes: mean ln1_w along axis [0.2  0.21 0.2  0.2  0.21 0.2  0.2  0.19 0.2  0.22 0.22 0.24 0.19 0.22 0.21 0.2  0.19 0.2  0.23 0.2  0.2  0.2  0.19 0.19 0.2  0.24 0.24 0.24 0.19 0.32]
D2-weighted:  [1.12 1.19 1.16 1.11 1.09 1.14 1.14 1.21 1.15 1.11 1.14 1.11 1.11 1.12 1.11 1.16 1.13 1.15 1.07 1.24 1.13 1.13 1.15 1.17 1.18 1.14 1.11 1.11 1.16 1.05]
kurt [60. 50. 48. 38. 37. 34. 34. 34. 31. 30. 28. 28. 27. 25. 23. 23. 22. 20. 17. 15. 15. 14. 12. 11. 10. 10.  8.  6.  5.  4.]

```

## Call 45

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0)
for j in range(6):
    M=W['W_o.0'][:,s(j)].astype(np.float64)@W['W_v.0'][s(j)].astype(np.float64)*D0
    print('M head',j,'norm on axes',np.linalg.norm(M@a,axis=0).round(2))
# head 3 forms in token axes
for j in [3,0,4,5]:
    q=W['W_q.0'][s(j)].astype(np.float64)*D0;kk=W['W_k.0'][s(j)].astype(np.float64)*D0
    Fm=q.T@kk
    T=a.T@Fm@a
    idx=[7,19,27,25]
    print('head',j,'form on axes 7,19,27,25:\n',T[np.ix_(idx,idx)].round(3),' typical |offdiag|',np.abs(T).mean().round(4))
    print(' row norm of T',np.linalg.norm(T,axis=1).round(2))
# units on axes 7,19: details
for t in [7,19,27,25]:
    u=np.argsort(-Sx[:,t])[:8]
    print('axis',t,'top z',Sx[u,t].round(1),'row norm',np.linalg.norm(F[act[u]],axis=1).round(2),'col norm',np.linalg.norm(Pj[:,act[u]],axis=0).round(2))
    # second axes on those units
    print('   other axes per unit',[int(np.argsort(-Sx[x])[1]) for x in u],[round(float(np.sort(Sx[x])[-2]),1) for x in u])
# relation of a7,a19 to P, and cos between
Q=W['W_q.0'][s(2)];K=W['W_k.0'][s(2)]
_,_,Pt=np.linalg.svd(np.vstack([Q,K]),full_matrices=False);P=Pt[:26].T
print('energy of axes in P',(np.linalg.norm(P.T@a,axis=0)**2).round(2))
print('cos a7,a19',a[:,7]@a[:,19],'a7,a27',a[:,7]@a[:,27],'a19,a27',a[:,19]@a[:,27],'a25,a27',a[:,25]@a[:,27])

```

Output:

```
M head 0 norm on axes [0.17 0.29 0.17 0.18 0.1  0.23 0.25 0.37 0.26 0.29 0.22 0.22 0.15 0.31 0.26 0.16 0.24 0.27 0.31 0.29 0.16 0.1  0.15 0.22 0.12 0.41 0.23 0.73 0.13 0.31]
M head 1 norm on axes [0.18 0.21 0.23 0.19 0.16 0.14 0.16 0.1  0.24 0.22 0.3  0.28 0.14 0.32 0.31 0.08 0.27 0.38 0.37 0.32 0.21 0.1  0.19 0.12 0.21 0.55 0.28 0.41 0.22 0.3 ]
M head 2 norm on axes [0.2  0.3  0.27 0.18 0.14 0.2  0.27 0.29 0.16 0.21 0.33 0.15 0.19 0.22 0.17 0.3  0.19 0.21 0.22 0.39 0.21 0.26 0.28 0.27 0.31 0.26 0.22 0.17 0.25 0.19]
M head 3 norm on axes [0.08 0.09 0.06 0.11 0.1  0.06 0.09 0.52 0.12 0.06 0.08 0.11 0.06 0.1  0.1  0.2  0.07 0.1  0.08 0.41 0.15 0.03 0.05 0.09 0.15 0.14 0.13 0.21 0.09 0.07]
M head 4 norm on axes [0.09 0.12 0.14 0.16 0.08 0.08 0.12 0.12 0.11 0.14 0.14 0.11 0.12 0.13 0.14 0.09 0.09 0.1  0.09 0.12 0.08 0.11 0.06 0.14 0.07 0.12 0.06 0.23 0.17 0.18]
M head 5 norm on axes [0.22 0.27 0.24 0.23 0.13 0.25 0.25 0.51 0.13 0.2  0.17 0.34 0.1  0.14 0.28 0.16 0.1  0.45 0.15 0.53 0.17 0.1  0.08 0.34 0.1  0.46 0.35 0.78 0.28 0.21]
head 3 form on axes 7,19,27,25:
 [[ 0.01  0.01  0.03 -0.01]
 [ 0.03  0.01  0.02 -0.01]
 [-0.04 -0.01  0.03 -0.01]
 [ 0.01  0.   -0.01  0.01]]  typical |offdiag| 0.0031
 row norm of T [0.01 0.02 0.02 0.01 0.03 0.01 0.02 0.06 0.02 0.01 0.01 0.02 0.02 0.02 0.02 0.01 0.01 0.03 0.01 0.05 0.03 0.01 0.01 0.03 0.01 0.03 0.02 0.07 0.01 0.01]
head 0 form on axes 7,19,27,25:
 [[ 0.    0.   -0.    0.  ]
 [-0.   -0.01  0.    0.  ]
 [-0.02 -0.02  0.    0.  ]
 [ 0.01  0.01 -0.   -0.  ]]  typical |offdiag| 0.0007
 row norm of T [0.   0.01 0.01 0.   0.   0.01 0.01 0.   0.   0.   0.   0.01 0.   0.01 0.   0.   0.   0.01 0.   0.01 0.01 0.   0.   0.01 0.   0.01 0.01 0.03 0.   0.  ]
head 4 form on axes 7,19,27,25:
 [[-0.    0.    0.   -0.01]
 [ 0.    0.    0.    0.  ]
 [ 0.01 -0.01 -0.    0.01]
 [-0.    0.01  0.   -0.01]]  typical |offdiag| 0.0011
 row norm of T [0.01 0.01 0.01 0.   0.01 0.   0.01 0.01 0.01 0.01 0.   0.01 0.01 0.02 0.01 0.   0.   0.01 0.   0.   0.   0.   0.   0.   0.   0.01 0.01 0.03 0.   0.01]
head 5 form on axes 7,19,27,25:
 [[ 0.    0.    0.   -0.  ]
 [-0.   -0.   -0.    0.  ]
 [-0.   -0.01 -0.    0.01]
 [ 0.01  0.01  0.   -0.01]]  typical |offdiag| 0.0015
 row norm of T [0.01 0.   0.01 0.01 0.01 0.   0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.02 0.01 0.   0.01 0.01 0.01 0.   0.01 0.   0.01 0.   0.01 0.03 0.01 0.01 0.   0.01]
axis 7 top z [9.2 6.8 6.6 5.1 5.1 4.6 4.4 3.9] row norm [3.51 3.96 3.28 2.46 2.49 2.65 2.12 1.5 ] col norm [2.46 4.8  2.54 2.12 2.28 2.48 1.85 1.65]
   other axes per unit [16, 8, 26, 14, 27, 27, 23, 23] [0.8, 1.7, 2.0, 2.7, 1.3, 2.2, 1.0, 0.6]
axis 19 top z [7.5 6.2 5.1 4.1 3.8 3.8 3.7 3.4] row norm [3.06 2.88 1.61 1.83 1.76 1.26 1.22 1.46] col norm [2.49 2.22 1.44 1.81 1.74 1.3  1.33 1.62]
   other axes per unit [27, 27, 26, 29, 7, 29, 29, 24] [2.1, 2.5, 0.4, 0.7, 1.4, 0.3, 0.3, 0.4]
axis 27 top z [4.7 4.2 3.9 3.4 3.1 2.7 2.5 2.4] row norm [4.26 2.34 3.65 4.42 3.42 2.45 2.88 3.52] col norm [2.86 2.26 2.65 2.9  2.55 2.32 2.22 3.35]
   other axes per unit [27, 14, 27, 27, 27, 21, 27, 0] [4.7, 2.5, 3.9, 3.4, 3.1, 2.4, 2.5, 4.5]
axis 25 top z [6.4 5.6 4.8 4.5 4.  3.8 3.  3. ] row norm [2.97 2.68 2.71 1.46 2.26 1.24 1.68 1.61] col norm [2.05 2.   2.55 1.65 1.94 1.58 1.83 1.76]
   other axes per unit [19, 20, 1, 17, 24, 29, 21, 23] [0.4, 0.5, 2.9, 0.4, 0.8, 0.5, 1.4, 1.4]
energy of axes in P [0.01 0.02 0.03 0.01 0.02 0.03 0.03 0.06 0.03 0.03 0.02 0.02 0.02 0.04 0.02 0.02 0.03 0.03 0.03 0.05 0.03 0.03 0.01 0.02 0.02 0.05 0.03 0.04 0.06 0.03]
cos a7,a19 0.08143041782414365 a7,a27 0.16932310176814008 a19,a27 0.02283860996215821 a25,a27 -0.21749686006484562

```

## Call 46

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
def ica_axes(r,seed=1):
    X=ua[:,1:r+1]*sa[1:r+1]
    ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=seed,fun='logcosh')
    Sx=ica.fit_transform(X)
    sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
    Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
    k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o];k=k[o]
    Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0)
    return Sx,a,k
S30,a30,k30=ica_axes(30)
S56,a56,k56=ica_axes(56)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
# map r=30 axes to r=56 axes
C=a30.T@a56
print('axis7 -> best 56 axes',np.argsort(-C[7])[:3],np.sort(C[7])[::-1][:3].round(2))
print('axis19-> best 56 axes',np.argsort(-C[19])[:3],np.sort(C[19])[::-1][:3].round(2))
M7=M2@a30[:,7];M19=M2@a30[:,19]
M7/=np.linalg.norm(M7);M19/=np.linalg.norm(M19)
print('M2 a7 -> best 56 axes',np.argsort(-(M7@a56))[:4],np.sort(M7@a56)[::-1][:4].round(2),' worst',np.argsort((M7@a56))[:3],np.sort(M7@a56)[:3].round(2))
print('M2 a19-> best 56 axes',np.argsort(-(M19@a56))[:4],np.sort(M19@a56)[::-1][:4].round(2),' worst',np.argsort((M19@a56))[:3],np.sort(M19@a56)[:3].round(2))
# all token images: for each of first 27 r30 axes best match in a56, excluding its own
own=np.argmax(C,axis=1)
print('own56',own[:27])
best=[];
for t in range(27):
    m=M2@a30[:,t];m/=np.linalg.norm(m);sim=m@a56;sim[own[t]]=-9
    best.append((t,int(np.argmax(sim)),round(float(sim.max()),2)))
print(best)

```

Output:

```
axis7 -> best 56 axes [28  3 55] [0.99 0.28 0.14]
axis19-> best 56 axes [45 54  3] [0.94 0.24 0.16]
M2 a7 -> best 56 axes [54 29 12  9] [0.48 0.18 0.1  0.1 ]  worst [53 14 47] [-0.18 -0.14 -0.13]
M2 a19-> best 56 axes [54 41 31 44] [0.55 0.13 0.13 0.1 ]  worst [53 47 16] [-0.17 -0.13 -0.12]
own56 [ 1  0 42  4  6  5  2 28 13 11 31 18 26 19 17 21 32  9 24 45 29 20 36 27 38 30 44]
[(0, 53, 0.43), (1, 52, 0.39), (2, 48, 0.15), (3, 39, 0.14), (4, 53, 0.2), (5, 53, 0.25), (6, 50, 0.1), (7, 54, 0.48), (8, 53, 0.29), (9, 37, 0.1), (10, 55, 0.18), (11, 20, 0.3), (12, 29, 0.24), (13, 55, 0.1), (14, 50, 0.35), (15, 37, 0.34), (16, 5, 0.18), (17, 53, 0.13), (18, 11, 0.13), (19, 54, 0.55), (20, 37, 0.16), (21, 52, 0.35), (22, 52, 0.47), (23, 52, 0.55), (24, 41, 0.4), (25, 52, 0.66), (26, 55, 0.33)]

```

## Call 47

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=220)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>0.7)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
print('n units',len(act))
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
def ica_axes(r,seed=1):
    X=ua[:,1:r+1]*sa[1:r+1]
    ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=seed,fun='logcosh')
    Sx=ica.fit_transform(X)
    sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
    Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
    k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o];k=k[o]
    Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0)
    return Sx,a,k
Sx,a,k=ica_axes(30)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
rows=[]
Rn=np.linalg.norm(Fa,axis=1)
print('kurt',k.round(0))
for t in range(30):
    z=Sx[:,t];sel=np.where(z>2.5)[0]
    sel=sel[np.argsort(-z[sel])]
    # dominance: unit assigned to axis if it is its max
    dom=sel[np.argmax(Sx[sel],axis=1)==t] if len(sel) else sel
    rows.append((t,len(sel),len(dom),Rn[sel].mean() if len(sel) else 0,Rn[sel].max() if len(sel) else 0,z[sel].max(),np.linalg.norm(a[:,t]@Fa.T)))
rows=sorted(rows,key=lambda x:-x[3])
print('axis n dom meanrow maxrow maxz  |a.Fa|')
for r in rows: print(r[0],r[1],r[2],round(r[3],2),round(r[4],2),round(r[5],1),round(r[6],1))
print('|M2 a|',np.linalg.norm(M2@a,axis=0).round(2))

```

Output:

```
n units 414
kurt [64. 54. 51. 40. 40. 38. 37. 37. 34. 32. 31. 30. 30. 28. 25. 24. 24. 21. 18. 17. 16. 15. 13. 12. 11. 11.  8.  6.  5.  5.]
axis n dom meanrow maxrow maxz  |a.Fa|
0 7 5 2.91 3.7 11.4 6.4
4 8 7 2.7 3.02 8.4 6.3
7 8 8 2.62 3.71 8.2 6.2
5 12 12 2.46 3.75 9.6 7.5
3 11 8 2.35 3.3 9.3 5.5
14 14 13 2.35 2.89 8.5 6.6
11 10 10 2.35 3.07 7.8 6.3
28 6 6 2.33 2.77 4.8 6.4
12 12 12 2.3 3.74 8.8 6.1
6 10 10 2.28 3.35 8.6 6.1
1 11 8 2.28 4.34 12.0 6.5
10 11 11 2.26 3.53 9.7 5.8
13 11 9 2.25 3.8 8.7 5.8
21 8 7 2.24 3.2 7.4 6.1
16 12 10 2.24 3.32 8.3 5.9
17 9 8 2.23 3.29 7.6 5.7
8 12 10 2.23 3.35 8.3 5.7
27 9 8 2.21 3.67 4.7 7.7
29 9 2 2.19 2.76 4.1 6.0
19 15 10 2.19 2.79 6.8 5.9
2 11 10 2.12 4.18 11.9 6.0
18 9 9 2.1 2.84 7.8 5.7
9 10 10 2.09 3.36 8.6 5.9
15 12 11 2.06 3.47 8.6 5.7
24 14 13 2.06 2.56 5.9 5.8
22 13 12 1.92 3.38 7.8 6.4
26 13 10 1.91 2.55 5.9 5.5
25 11 11 1.88 2.89 6.6 6.5
23 17 14 1.84 2.89 6.8 5.8
20 14 14 1.8 3.1 7.8 6.3
|M2 a| [0.2  0.3  0.27 0.18 0.14 0.29 0.2  0.27 0.16 0.21 0.33 0.15 0.2  0.22 0.17 0.18 0.31 0.21 0.22 0.21 0.4  0.26 0.27 0.28 0.31 0.27 0.25 0.17 0.25 0.19]

```

## Call 48

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from scipy.cluster.hierarchy import linkage,leaves_list,fcluster
np.set_printoptions(precision=2,suppress=True,linewidth=240)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0)
K=27
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
Im=M2@a[:,:K];n=np.linalg.norm(Im,axis=0);Imn=Im/n
G=Imn.T@Imn
Z=linkage(Imn.T,'average',metric='cosine');lv=leaves_list(Z)
print('order',lv)
print((G[lv][:,lv]*10).round(0).astype(int))
print('cos M2a7,M2a19',G[7,19])
print('top partners of 7',np.argsort(-G[7])[:5],np.sort(G[7])[::-1][:5].round(2))
print('top partners of 19',np.argsort(-G[19])[:5],np.sort(G[19])[::-1][:5].round(2))
print('V-level: cos(V a_s,V a_t) head2')
Vh=W['W_v.0'][s(2)].astype(np.float64)*D0@a[:,:K];Vh/=np.linalg.norm(Vh,axis=0)
print('V cos 7,19',(Vh.T@Vh)[7,19])
# code similarity
print('fc0 units: cos a7,a19 %.2f'%(a[:,7]@a[:,19]))

```

Output:

```
order [23  1 22 25  2  7 19 11 21 10  6 12 18 26 13 17  3  9  5 16  0 14 24  8 20  4 15]
[[10  2  3  2  0 -1  0  0  2 -1 -1 -2 -2  1 -2  1 -1 -1  0 -2  1  0  0  2  0 -2  1]
 [ 2 10  3  3  1  0 -1  0  0 -1  1 -1  0 -2  0 -1  0  1 -2  0 -4  0  0 -2 -1 -2  0]
 [ 3  3 10  3 -2 -1 -1 -1  4 -2  2 -1  1 -1 -1  1  0 -1  1 -1 -1 -1  0  2  0 -4  0]
 [ 2  3  3 10 -1  0 -1 -2  2 -1  1 -1  2 -1 -3  1 -1  0  0 -1 -2 -2 -2 -2  0 -1 -2]
 [ 0  1 -2 -1 10  0 -1  2  0 -1 -1 -1  0  1 -2 -1 -1 -1 -4  1  0  0 -1  0 -1  0 -1]
 [-1  0 -1  0  0 10  8  4 -1 -1 -1  0 -2 -1 -3 -1 -1 -3  0 -1 -3 -4 -2 -3  0 -3 -1]
 [ 0 -1 -1 -1 -1  8 10  4 -1 -1 -1 -2 -3 -1 -2  0 -1 -3  0 -2 -1 -2 -2 -4 -2 -3 -2]
 [ 0  0 -1 -2  2  4  4 10  5  0 -4  0  2  0 -1  0 -2 -1 -2  0 -2 -4 -3 -1  0  0 -1]
 [ 2  0  4  2  0 -1 -1  5 10 -2  0 -2  1 -1  0  1  1  1  2 -1  1 -3 -3  0 -2  0  0]
 [-1 -1 -2 -1 -1 -1 -1  0 -2 10  0  0 -1  3  0  0 -2 -1  0 -1 -1  1  2 -2  0  1 -2]
 [-1  1  2  1 -1 -1 -1 -4  0  0 10  2  1  0  0 -3  2  1 -1 -1 -2  3 -2 -5 -2 -4  2]
 [-2 -1 -1 -1 -1  0 -2  0 -2  0  2 10  2  1  1 -1  0  0  0  3 -5 -3  0  0  4  0  1]
 [-2  0  1  2  0 -2 -3  2  1 -1  1  2 10  3 -2 -2 -1  2 -3  1 -2 -1 -1  0  1  1 -1]
 [ 1 -2 -1 -1  1 -1 -1  0 -1  3  0  1  3 10 -1 -1  1  0 -2  0 -1  0  1  0  0  0 -2]
 [-2  0 -1 -3 -2 -3 -2 -1  0  0  0  1 -2 -1 10  1  1  2  0 -1  0 -1 -1  2  1 -1  2]
 [ 1 -1  1  1 -1 -1  0  0  1  0 -3 -1 -2 -1  1 10  3  1  3 -1  2  1 -1  3  0  3  1]
 [-1  0  0 -1 -1 -1 -1 -2  1 -2  2  0 -1  1  1  3 10  3  2  1  1  2 -3  1  0  1  1]
 [-1  1 -1  0 -1 -3 -3 -1  1 -1  1  0  2  0  2  1  3 10  0 -1  2 -2 -3  0 -3  3  0]
 [ 0 -2  1  0 -4  0  0 -2  2  0 -1  0 -3 -2  0  3  2  0 10  5  2  0  0  2  2  0  0]
 [-2  0 -1 -1  1 -1 -2  0 -1 -1 -1  3  1  0 -1 -1  1 -1  5 10  0  1  0  2  2 -1 -3]
 [ 1 -4 -1 -2  0 -3 -1 -2  1 -1 -2 -5 -2 -1  0  2  1  2  2  0 10  3  1  4 -2  4  1]
 [ 0  0 -1 -2  0 -4 -2 -4 -3  1  3 -3 -1  0 -1  1  2 -2  0  1  3 10  2  1  0  1  2]
 [ 0  0  0 -2 -1 -2 -2 -3 -3  2 -2  0 -1  1 -1 -1 -3 -3  0  0  1  2 10  2  0  4  1]
 [ 2 -2  2 -2  0 -3 -4 -1  0 -2 -5  0  0  0  2  3  1  0  2  2  4  1  2 10  5  3  2]
 [ 0 -1  0  0 -1  0 -2  0 -2  0 -2  4  1  0  1  0  0 -3  2  2 -2  0  0  5 10  2  4]
 [-2 -2 -4 -1  0 -3 -3  0  0  1 -4  0  1  0 -1  3  1  3  0 -1  4  1  4  3  2 10  4]
 [ 1  0  0 -2 -1 -1 -2 -1  0 -2  2  1 -1 -2  2  1  1  0  0 -3  1  2  1  2  4  4 10]]
cos M2a7,M2a19 0.796850195258434
top partners of 7 [ 7 19 11  1  2] [1.   0.8  0.36 0.04 0.  ]
top partners of 19 [19  7 11  5 17] [ 1.    0.8   0.39  0.01 -0.  ]
V-level: cos(V a_s,V a_t) head2
V cos 7,19 0.6587009979857605
fc0 units: cos a7,a19 0.08

```

## Call 49

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=240)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
def ica_axes(r,seed=1):
    X=ua[:,1:r+1]*sa[1:r+1]
    ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=seed,fun='logcosh')
    Sx=ica.fit_transform(X)
    sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
    Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
    k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o];k=k[o]
    Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0)
    return Sx,a,k
S30,a30,k30=ica_axes(30)
S56,a56,k56=ica_axes(56)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
ws=(M2@a30[:,7])/np.linalg.norm(M2@a30[:,7])+(M2@a30[:,19])/np.linalg.norm(M2@a30[:,19]);ws/=np.linalg.norm(ws)
# unit loading on whitespace-as-previous direction, from the raw row
z=Fa@ws; z=(z-np.median(z))/(1.48*np.median(np.abs(z-np.median(z))))
print('prev-whitespace loading: quantiles',np.quantile(z,[.01,.1,.5,.9,.99]).round(1),'max',z.max().round(1),'min',z.min().round(1))
print('n units z>3',(z>3).sum(),'z<-3',(z<-3).sum())
# for units with strong prev-whitespace, list cur axis
S0=Fa@a30[:,:27]
Sz=(S0-np.median(S0,0))/(1.48*np.median(np.abs(S0-np.median(S0,0)),0))
print('units with prev-WS strongly positive -> their best cur axis and z')
top=np.argsort(-z)[:30]
print([ (int(u),round(float(z[u]),1),int(Sz[u].argmax()),round(float(Sz[u].max()),1)) for u in top])
print('units with prev-WS strongly negative')
top=np.argsort(z)[:20]
print([ (int(u),round(float(z[u]),1),int(Sz[u].argmax()),round(float(Sz[u].max()),1)) for u in top])
# units with cur 7 and cur 19 : their prev-WS loading
for t in [7,19]:
    idx=np.argsort(-Sz[:,t])[:12]
    print('cur axis',t,'units',[(round(float(Sz[u,t]),1),round(float(z[u]),1)) for u in idx])
# count of units with cur axis t having prev-WS>3 by axis
cnt=np.array([((Sz[:,t]>4)&(z>2.5)).sum() for t in range(27)]);tot=np.array([(Sz[:,t]>4).sum() for t in range(27)])
print('axis: n(cur & prevWS)',cnt);print('      n(cur)         ',tot)

```

Output:

```
prev-whitespace loading: quantiles [-3.3 -1.4  0.   1.3  3.1] max 5.5 min -7.7
n units z>3 6 z<-3 7
units with prev-WS strongly positive -> their best cur axis and z
[(29, 5.5, 19, 12.9), (90, 4.5, 14, 1.4), (98, 3.6, 1, 1.1), (158, 3.3, 19, 6.2), (105, 3.1, 19, 8.1), (172, 3.1, 20, 9.8), (10, 2.5, 7, 1.7), (259, 2.4, 17, 1.3), (66, 2.3, 17, 14.0), (264, 2.2, 7, 15.2), (9, 2.1, 17, 13.0), (294, 2.0, 6, 18.3), (77, 2.0, 14, 9.8), (251, 2.0, 20, 11.4), (122, 1.9, 20, 4.0), (39, 1.9, 1, 13.8), (149, 1.8, 1, 2.0), (159, 1.8, 9, 16.1), (263, 1.7, 10, 10.1), (361, 1.6, 23, 2.2), (180, 1.6, 1, 5.5), (183, 1.6, 18, 11.9), (52, 1.6, 24, 9.1), (128, 1.6, 23, 3.7), (165, 1.6, 10, 10.5), (161, 1.6, 21, 11.2), (42, 1.5, 11, 11.5), (130, 1.5, 2, 4.0), (205, 1.5, 23, 12.6), (73, 1.5, 12, 10.2)]
units with prev-WS strongly negative
[(83, -7.7, 7, 32.2), (203, -5.4, 7, 2.1), (257, -5.0, 12, 10.6), (353, -3.6, 20, 6.2), (113, -3.2, 9, 6.6), (16, -3.0, 15, 17.2), (195, -3.0, 2, 8.0), (38, -2.9, 16, 7.8), (375, -2.7, 16, 7.5), (15, -2.7, 7, 24.1), (217, -2.6, 15, 7.6), (258, -2.6, 1, 6.2), (163, -2.5, 5, 9.2), (3, -2.5, 24, 5.0), (75, -2.2, 10, 1.8), (213, -2.1, 3, 3.3), (214, -2.1, 2, 3.1), (261, -1.8, 2, 2.2), (204, -1.8, 3, 21.4), (82, -1.8, 16, 3.9)]
cur axis 7 units [(32.2, -7.7), (24.7, -1.2), (24.1, -2.7), (18.2, -0.6), (18.0, 1.2), (16.7, -1.2), (15.2, 2.2), (13.3, 0.3), (11.9, 0.1), (11.8, -1.1), (11.7, -1.1), (8.9, -0.2)]
cur axis 19 units [(15.3, -1.2), (12.9, 5.5), (9.7, -0.4), (8.1, 3.1), (8.1, 0.2), (7.6, -0.7), (7.1, -0.8), (7.0, -0.4), (6.4, -0.7), (6.3, -0.4), (6.2, 3.3), (5.6, -1.8)]
axis: n(cur & prevWS) [0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 3 1 0 0 0 0 0 0]
      n(cur)          [11 22 15 11  9 12 10 17 15 12 14 10 14 13 18 13 12 10  9 16 15  7 18 12 16  8  8]

```

## Call 50

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=240)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
m7=M2@a[:,7];m19=M2@a[:,19];n7=np.linalg.norm(m7);n19=np.linalg.norm(m19)
print('norms',n7,n19)
e7=m7/n7;e19=m19/n19
wsd=(e7+e19);wsd/=np.linalg.norm(wsd)
dif=(e19-e7);dif/=np.linalg.norm(dif)
zf=lambda v:(lambda x:(x-np.median(x))/(1.48*np.median(np.abs(x-np.median(x)))))(Fa@v)
zw=zf(wsd);zd=zf(dif)
S0=Fa@a[:,:27];Sz=(S0-np.median(S0,0))/(1.48*np.median(np.abs(S0-np.median(S0,0)),0))
print('units for cur 19: (zcur, zWS, zDiff[+ => prev=19])')
for u in np.argsort(-Sz[:,19])[:14]: print(int(u),round(Sz[u,19],1),round(zw[u],1),round(zd[u],1),'cols',np.linalg.norm(Pa[:,u]).round(2),'row',np.linalg.norm(F[act[u]]).round(2))
print('units for cur 7:')
for u in np.argsort(-Sz[:,7])[:14]: print(int(u),round(Sz[u,7],1),round(zw[u],1),round(zd[u],1),'cols',np.linalg.norm(Pa[:,u]).round(2),'row',np.linalg.norm(F[act[u]]).round(2))
print('zd quantiles',np.quantile(zd,[.01,.5,.99]).round(1),'top units',np.argsort(-zd)[:8],np.sort(zd)[::-1][:8].round(1))
for u in np.argsort(-zd)[:8]: print('  u',int(u),'zd',round(zd[u],1),'zw',round(zw[u],1),'best cur',int(Sz[u].argmax()),round(Sz[u].max(),1))
for u in np.argsort(zd)[:8]: print('  u',int(u),'zd',round(zd[u],1),'zw',round(zw[u],1),'best cur',int(Sz[u].argmax()),round(Sz[u].max(),1))

```

Output:

```
norms 0.28839518768576994 0.3944143825404911
units for cur 19: (zcur, zWS, zDiff[+ => prev=19])
120 15.3 -1.2 -1.9 cols 2.49 row 3.06
29 12.9 5.5 -3.2 cols 2.22 row 2.88
79 9.7 -0.4 -2.3 cols 1.44 row 1.61
105 8.1 3.1 1.3 cols 1.81 row 1.83
232 8.1 0.2 -1.1 cols 1.74 row 1.76
202 7.6 -0.7 -2.2 cols 1.3 row 1.26
145 7.1 -0.8 -1.9 cols 1.33 row 1.22
325 7.0 -0.4 -0.0 cols 1.62 row 1.46
340 6.4 -0.7 -0.7 cols 1.75 row 1.63
150 6.3 -0.4 -0.7 cols 1.61 row 1.8
158 6.2 3.3 0.4 cols 1.6 row 1.69
177 5.6 -1.8 -0.2 cols 1.49 row 1.69
312 5.4 -0.2 -2.5 cols 2.23 row 2.36
355 5.4 0.7 4.1 cols 1.96 row 2.12
units for cur 7:
83 32.2 -7.7 1.3 cols 2.46 row 3.51
178 24.7 -1.2 3.3 cols 2.54 row 3.28
15 24.1 -2.7 -2.5 cols 4.8 row 3.96
282 18.2 -0.6 1.5 cols 2.28 row 2.49
48 18.0 1.2 0.2 cols 2.12 row 2.46
234 16.7 -1.2 2.3 cols 2.48 row 2.65
264 15.2 2.2 0.2 cols 1.85 row 2.12
321 13.3 0.3 -0.4 cols 1.65 row 1.5
196 11.9 0.1 0.5 cols 1.79 row 2.01
101 11.8 -1.1 1.4 cols 1.99 row 2.21
297 11.7 -1.1 1.3 cols 2.26 row 2.42
246 8.9 -0.2 1.1 cols 1.48 row 1.28
29 8.3 5.5 -3.2 cols 2.22 row 2.88
164 7.8 0.5 -0.8 cols 1.5 row 1.44
zd quantiles [-2.8  0.   2.8] top units [192 355 178 333  21 211 335  34] [4.2 4.1 3.3 2.9 2.8 2.7 2.7 2.6]
  u 192 zd 4.2 zw 0.4 best cur 0 21.5
  u 355 zd 4.1 zw 0.7 best cur 19 5.4
  u 178 zd 3.3 zw -1.2 best cur 7 24.7
  u 333 zd 2.9 zw 0.9 best cur 14 10.3
  u 21 zd 2.8 zw 0.1 best cur 20 7.7
  u 211 zd 2.7 zw 1.1 best cur 10 13.1
  u 335 zd 2.7 zw -0.8 best cur 0 16.3
  u 34 zd 2.6 zw -1.0 best cur 20 3.1
  u 29 zd -3.2 zw 5.5 best cur 19 12.9
  u 337 zd -2.9 zw -0.7 best cur 19 3.2
  u 349 zd -2.9 zw -0.1 best cur 19 2.2
  u 357 zd -2.9 zw -1.6 best cur 0 8.9
  u 93 zd -2.8 zw -1.3 best cur 2 28.6
  u 8 zd -2.8 zw -1.6 best cur 4 15.1
  u 312 zd -2.5 zw -0.2 best cur 19 5.4
  u 15 zd -2.5 zw -2.7 best cur 7 24.1

```

## Call 51

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=1,suppress=True,linewidth=240)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0);K=27;a=a[:,:K]
Sp=np.maximum(Sx[:,:K],0)
code=np.linalg.lstsq(Sp,Pa.T,rcond=None)[0].T;c=code/np.linalg.norm(code,axis=0)
M2=W['W_o.0'][:,s(2)].astype(np.float64)@W['W_v.0'][s(2)].astype(np.float64)*D0
f=M2@a;f=f/np.linalg.norm(f,axis=0)
def zs(v): return (v-np.median(v))/(1.48*np.median(np.abs(v-np.median(v))))
print('z of per-axis read norm: columns: 7,19 for q,k,v under bases a,f,c; and argmax axis')
for l in range(1,4):
    Dl=W[f'ln1_w.{l}'].astype(np.float64)
    for j in range(6):
        row=[]
        for nm in ['W_q','W_k','W_v']:
            Mx=W[f'{nm}.{l}'][s(j)].astype(np.float64)*Dl
            for B,bn in [(a,'a'),(f,'f'),(c,'c')]:
                z=zs(np.linalg.norm(Mx@B,axis=0))
                row.append(f'{nm[-1]}{bn}:{z[7]:4.1f}/{z[19]:4.1f}/{int(z.argmax()):2d}')
        print(f'L{l}H{j} ',' '.join(row))

```

Output:

```
z of per-axis read norm: columns: 7,19 for q,k,v under bases a,f,c; and argmax axis
L1H0  qa:-0.1/ 6.4/19 qf: 2.7/ 3.5/19 qc: 2.8/ 3.2/19 ka: 1.0/ 1.5/19 kf: 4.6/ 7.1/19 kc: 2.0/ 0.3/21 va: 1.1/ 3.2/19 vf: 2.4/ 1.5/23 vc: 7.7/ 3.6/ 7
L1H1  qa:-1.6/ 2.9/19 qf: 2.5/ 2.5/19 qc: 0.8/ 1.4/ 3 ka:-2.4/ 0.1/23 kf: 2.9/ 5.5/19 kc: 1.0/ 0.8/23 va:-2.0/ 2.0/ 3 vf: 0.0/ 0.6/23 vc: 0.0/-1.0/ 3
L1H2  qa:-0.8/ 1.3/ 1 qf: 3.3/ 3.9/19 qc: 0.4/ 2.4/19 ka:-0.9/ 3.7/19 kf: 0.6/ 1.4/24 kc:-0.2/ 0.1/21 va:-1.1/ 0.5/ 1 vf: 0.1/ 0.6/26 vc:-0.8/ 0.0/14
L1H3  qa:-0.7/ 1.3/ 1 qf: 2.2/ 2.0/ 7 qc: 1.3/ 2.4/23 ka:-1.2/ 2.9/23 kf: 0.1/ 1.0/21 kc:-0.5/ 0.7/21 va:-1.2/ 0.4/ 1 vf:-0.1/ 0.5/24 vc:-0.7/-0.8/22
L1H4  qa:-0.8/ 0.9/17 qf: 2.3/ 1.6/ 7 qc: 1.9/ 1.9/20 ka:-0.8/ 2.8/19 kf: 3.2/ 5.2/19 kc: 0.1/ 0.0/23 va:-1.8/-0.3/ 8 vf: 0.4/ 1.4/ 1 vc:-1.2/-0.3/ 8
L1H5  qa:-1.2/ 0.4/18 qf: 1.8/ 2.9/19 qc: 2.0/ 1.5/ 7 ka:-1.0/ 0.4/ 9 kf: 3.6/ 5.6/19 kc:-0.1/ 0.3/21 va:-2.2/ 1.8/19 vf: 0.8/ 1.5/ 1 vc:-0.5/-1.3/ 3
L2H0  qa:-1.4/ 2.0/21 qf: 0.6/-0.0/25 qc: 2.2/ 0.4/25 ka:-2.0/ 0.5/25 kf: 3.6/ 5.5/19 kc:-0.4/-0.7/23 va: 0.2/-0.3/ 1 vf:-1.7/-2.4/21 vc: 0.2/-0.7/ 3
L2H1  qa: 0.7/ 0.9/23 qf: 0.1/ 0.2/17 qc:-0.6/-0.7/17 ka:-2.2/ 6.1/19 kf: 0.6/-0.8/21 kc: 0.7/-0.3/21 va:-1.1/ 0.4/ 9 vf:-2.1/-0.5/ 2 vc: 2.3/-2.2/ 7
L2H2  qa: 1.2/ 2.0/25 qf: 1.3/ 0.7/ 2 qc: 4.9/-0.1/ 7 ka:-0.2/ 3.1/19 kf: 4.3/ 3.6/ 7 kc: 1.4/ 2.2/21 va:-0.2/-0.4/ 2 vf:-0.9/ 0.6/ 2 vc: 2.9/-0.7/ 7
L2H3  qa:-0.3/ 0.7/20 qf: 1.6/ 1.1/ 7 qc:-1.4/-1.2/20 ka: 0.9/ 0.5/14 kf:-0.3/ 1.3/25 kc: 0.9/-0.2/21 va: 0.8/-0.4/ 8 vf: 0.6/ 0.7/22 vc: 0.8/-2.3/18
L2H4  qa:-0.1/ 1.0/15 qf: 3.4/ 2.5/ 7 qc:-0.3/-0.2/23 ka:-1.1/ 0.0/ 5 kf:-1.8/ 0.0/25 kc: 0.2/-0.2/21 va:-1.2/-0.7/ 4 vf:-0.9/-0.7/25 vc:-0.5/-2.8/14
L2H5  qa:-1.5/ 1.1/25 qf: 1.9/ 0.9/ 1 qc:-1.3/-2.4/20 ka:-1.2/ 0.0/ 1 kf:-0.0/ 1.0/ 1 kc: 0.5/-0.7/23 va:-1.0/-0.1/18 vf:-1.2/-0.6/11 vc:-0.7/-1.8/17
L3H0  qa: 0.1/ 0.3/15 qf:-0.9/-0.7/ 2 qc: 0.7/-0.7/22 ka: 1.3/ 0.4/ 2 kf:-0.4/-1.0/10 kc:-0.3/ 0.7/11 va: 0.5/-0.2/16 vf:-0.7/-2.2/ 1 vc: 0.0/-0.8/ 3
L3H1  qa:-0.7/-0.4/ 2 qf:-1.4/ 0.0/25 qc: 0.0/ 1.4/24 ka: 0.6/ 0.5/18 kf: 0.0/ 0.8/25 kc:-1.1/-0.4/ 9 va:-1.2/-0.5/ 1 vf: 0.8/ 0.1/ 0 vc: 4.8/ 2.8/ 7
L3H2  qa: 1.3/ 2.0/17 qf: 2.7/ 0.9/ 7 qc:10.8/-0.4/ 7 ka: 1.2/ 0.2/21 kf: 2.3/ 2.1/ 7 kc: 0.3/ 0.6/25 va: 1.8/ 0.0/24 vf:-0.5/-1.0/22 vc: 1.3/ 0.8/11
L3H3  qa: 2.0/-0.3/ 7 qf: 1.9/-0.0/ 3 qc:11.4/-0.6/ 7 ka:-0.7/ 0.0/20 kf: 1.0/ 1.1/12 kc: 2.3/ 0.9/21 va:-0.5/ 0.2/20 vf: 0.7/-1.6/10 vc: 2.1/ 1.4/ 0
L3H4  qa: 0.5/ 0.2/24 qf: 0.6/ 0.0/ 4 qc: 4.0/ 1.6/ 7 ka:-1.0/ 0.9/ 0 kf: 1.7/ 1.7/11 kc: 0.0/ 2.0/23 va: 2.1/ 0.3/ 7 vf:-1.3/-0.6/24 vc: 4.1/ 0.3/ 7
L3H5  qa:-0.5/ 0.5/23 qf: 1.9/ 1.0/ 7 qc: 4.3/-0.2/ 7 ka:-0.0/ 1.5/ 3 kf: 2.9/ 3.8/19 kc: 1.3/ 6.9/19 va: 0.2/ 0.0/ 1 vf: 2.0/-0.2/ 3 vc: 2.6/ 0.2/ 7

```

## Call 52

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=240)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64);D3=W['ln1_w.3'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0);K=27;a=a[:,:K]
Sp=np.maximum(Sx[:,:K],0)
code=np.linalg.lstsq(Sp,Pa.T,rcond=None)[0].T;c=code/np.linalg.norm(code,axis=0)
U3,S3,Vt3=np.linalg.svd(W['W_proj.3'].astype(np.float64),full_matrices=False)
Y=U3[:,1:28]
print('Y sv',S3[1:28].round(1))
M={j:(W['W_o.3'][:,s(j)].astype(np.float64)@W['W_v.3'][s(j)].astype(np.float64)*D3) for j in range(6)}
tops={}
for j in [2,5,3,0,4]:
    u,sv,vt=np.linalg.svd(Y.T@M[j]);tops[j]=u[:,0]
    print('head',j,'sv',sv[:4].round(1),'top dir comps^2',(u[:,0]**2).round(2))
    u2=u[:,1];print('    2nd dir comps^2',(u2**2).round(2))
print('cos between top dirs H5,H2,H3,H4,H0',np.array([[abs(tops[a]@tops[b]) for b in [5,2,3,4,0]] for a in [5,2,3,4,0]]).round(2))
la=(Y.T@M[1]@a);lc=(Y.T@M[1]@c)
la=la/np.linalg.norm(la,axis=0);lc=lc/np.linalg.norm(lc,axis=0)
for nm,l in [('a',la),('c',lc)]:
    print(nm,'cos(copy logit of token, H5 top dir):',(tops[5]@l).round(2))
    print(nm,'argmax',np.argsort(-np.abs(tops[5]@l))[:4])
    print(nm,'cos(copy logit of token, H2 top dir):',(tops[2]@l).round(2))
# direct map from token codes through heads 5,2,0,3,4 into Y top dir
for j in [5,2]:
    print('head',j,'read of token axes into its top Y dir: a',(tops[j]@(Y.T@M[j]@a)).round(2));print('   c',(tops[j]@(Y.T@M[j]@c)).round(2))

```

Output:

```
Y sv [20.7 19.7 19.3 18.6 18.2 17.9 17.7 17.  16.7 16.  15.2 14.8 14.3 14.1 13.5 13.3 12.5 12.4 12.  11.7 11.6 10.4  8.4  8.1  7.9  7.5  7.2]
head 2 sv [4.9 2.9 2.3 2. ] top dir comps^2 [0.01 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.01 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.1  0.79 0.02 0.03 0.01]
    2nd dir comps^2 [0.05 0.06 0.04 0.13 0.29 0.   0.   0.15 0.   0.01 0.01 0.01 0.07 0.02 0.01 0.   0.01 0.01 0.01 0.   0.   0.04 0.   0.   0.07 0.02 0.  ]
head 5 sv [10.2  4.9  4.7  3.7] top dir comps^2 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.12 0.79 0.03 0.03 0.01]
    2nd dir comps^2 [0.   0.04 0.01 0.08 0.16 0.   0.   0.09 0.   0.01 0.15 0.01 0.02 0.05 0.11 0.03 0.   0.1  0.01 0.   0.01 0.   0.01 0.01 0.05 0.02 0.  ]
head 3 sv [3.  2.6 2.1 2. ] top dir comps^2 [0.07 0.02 0.01 0.   0.05 0.   0.01 0.01 0.   0.05 0.06 0.04 0.08 0.07 0.01 0.01 0.   0.   0.03 0.   0.06 0.12 0.   0.23 0.04 0.02 0.02]
    2nd dir comps^2 [0.1  0.02 0.05 0.17 0.09 0.04 0.01 0.1  0.01 0.01 0.03 0.02 0.   0.04 0.03 0.01 0.08 0.05 0.02 0.06 0.02 0.01 0.01 0.01 0.01 0.03 0.  ]
head 0 sv [2.4 2.3 2.  1.8] top dir comps^2 [0.53 0.23 0.   0.05 0.01 0.   0.   0.01 0.   0.   0.04 0.01 0.01 0.   0.   0.   0.   0.05 0.   0.   0.01 0.   0.   0.   0.   0.02 0.01]
    2nd dir comps^2 [0.01 0.03 0.01 0.04 0.04 0.   0.02 0.01 0.06 0.   0.   0.22 0.   0.04 0.05 0.   0.01 0.1  0.02 0.25 0.   0.01 0.02 0.04 0.   0.   0.  ]
head 4 sv [2.3 2.  1.8 1.6] top dir comps^2 [0.02 0.05 0.   0.02 0.02 0.   0.   0.03 0.   0.03 0.04 0.11 0.   0.06 0.13 0.01 0.01 0.15 0.03 0.05 0.02 0.06 0.08 0.   0.   0.03 0.04]
    2nd dir comps^2 [0.11 0.08 0.   0.   0.08 0.37 0.   0.01 0.05 0.   0.01 0.   0.   0.01 0.02 0.01 0.   0.06 0.   0.06 0.03 0.   0.   0.   0.06 0.   0.01]
cos between top dirs H5,H2,H3,H4,H0 [[1.   0.98 0.43 0.13 0.12]
 [0.98 1.   0.44 0.15 0.06]
 [0.43 0.44 1.   0.25 0.42]
 [0.13 0.15 0.25 1.   0.35]
 [0.12 0.06 0.42 0.35 1.  ]]
a cos(copy logit of token, H5 top dir): [-0.03 -0.07 -0.16  0.19 -0.03  0.04  0.07  0.43  0.32 -0.08  0.01  0.19  0.25 -0.21 -0.08 -0.1   0.08  0.03 -0.03 -0.29  0.06 -0.02  0.13  0.14 -0.02 -0.08 -0.01]
a argmax [ 7  8 19 12]
a cos(copy logit of token, H2 top dir): [ 0.   -0.13 -0.21  0.18 -0.01  0.05  0.03  0.45  0.32 -0.1  -0.01  0.15  0.26 -0.17 -0.08 -0.04  0.05 -0.   -0.03 -0.22  0.08 -0.03  0.17  0.09 -0.03 -0.06 -0.  ]
c cos(copy logit of token, H5 top dir): [-0.02  0.13 -0.09  0.11  0.25 -0.22  0.06  0.73  0.15 -0.16  0.17 -0.13  0.18 -0.12  0.01 -0.17  0.09  0.07 -0.24 -0.41 -0.06 -0.07  0.04 -0.   -0.02 -0.09  0.04]
c argmax [ 7 19  4 18]
c cos(copy logit of token, H2 top dir): [ 0.01  0.08 -0.14  0.09  0.22 -0.23  0.04  0.78  0.07 -0.14  0.07 -0.13  0.23 -0.14 -0.03 -0.11  0.1   0.07 -0.2  -0.33 -0.04 -0.1   0.04 -0.03 -0.08 -0.15  0.01]
head 5 read of token axes into its top Y dir: a [ 0.41  0.22 -1.11  0.63 -0.21 -0.49  0.    0.55 -0.05 -1.29  0.32 -0.17 -0.02  0.55 -0.03 -0.28 -0.15 -0.24 -0.92 -0.01  0.04 -1.07  0.5   0.02  0.77 -0.13  0.  ]
   c [-0.41 -0.33  0.65 -0.53 -0.13  0.48 -0.08  1.93  0.06 -0.17 -0.05  0.03 -0.45  0.04  0.48 -0.18 -0.09 -0.03  0.02  0.79  0.08 -0.01  0.08 -0.   -0.37 -0.2   0.05]
head 2 read of token axes into its top Y dir: a [ 0.24 -0.21  0.31  0.03 -0.07  0.39  0.33  0.09 -0.3   0.22 -0.32 -0.04 -0.22  0.35  0.37  0.03 -0.28 -0.13 -0.4  -0.12 -0.14  0.52  0.21 -0.08  0.11  0.11 -0.22]
   c [ 0.19  0.31  0.34  0.08 -0.05  0.04 -0.31 -0.41 -0.23  0.11 -0.04 -0.03  0.3  -0.19  0.45  0.06  0.11  0.35  0.09  0.47  0.45 -0.37 -0.12  0.2   0.01  0.07  0.01]

```

## Call 53

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2,suppress=True,linewidth=240)
s=lambda j:slice(64*j,64*j+64)
F=W['W_fc.0'].astype(np.float64);Pj=W['W_proj.0'].astype(np.float64)
D2=W['ln2_w.0'].astype(np.float64);D0=W['ln1_w.0'].astype(np.float64);D3=W['ln1_w.3'].astype(np.float64)
rn=np.linalg.norm(F,axis=1);act=np.where(rn>1)[0]
Fa=F[act]*D2;Pa=Pj[:,act]
ua,sa,va=np.linalg.svd(Fa,full_matrices=False)
r=30
X=ua[:,1:r+1]*sa[1:r+1]
ica=FastICA(n_components=r,whiten='unit-variance',max_iter=3000,tol=1e-5,random_state=1,fun='logcosh')
Sx=ica.fit_transform(X)
sk=((Sx-Sx.mean(0))**3).mean(0)/Sx.std(0)**3
Sx=Sx*np.sign(sk);A=ica.mixing_*np.sign(sk)
k=kurtosis(Sx,axis=0);o=np.argsort(-k);Sx=Sx[:,o];A=A[:,o]
Ax=va[1:r+1].T@A;a=Ax/np.linalg.norm(Ax,axis=0);K=27;a=a[:,:K]
Sp=np.maximum(Sx[:,:K],0)
code=np.linalg.lstsq(Sp,Pa.T,rcond=None)[0].T;c=code/np.linalg.norm(code,axis=0)
U3,S3,Vt3=np.linalg.svd(W['W_proj.3'].astype(np.float64),full_matrices=False)
Y=U3[:,1:28];Sy=S3[1:28]
M1=(W['W_o.3'][:,s(1)].astype(np.float64)@W['W_v.3'][s(1)].astype(np.float64)*D3)
lc=Y.T@M1@c;la=Y.T@M1@a
for nm,l in [('c',lc),('a',la),('c+a',lc/np.linalg.norm(lc,axis=0)+la/np.linalg.norm(la,axis=0))]:
    ln=l/np.linalg.norm(l,axis=0)
    w=np.sqrt(((ln**2)*(Sy[:,None]**2)).sum(0))
    print(nm,'S-weighted norm per token',w.round(1));print('  order',np.argsort(w));print('  raw norm',np.linalg.norm(l,axis=0).round(2))
# Energy distribution by group for tokens 7,19
ln=lc/np.linalg.norm(lc,axis=0)
for t in [7,19]:
    print('token',t,'energy in strong comps(0-21)',(ln[:22,t]**2).sum().round(2),'in medium (22-26)',(ln[22:,t]**2).sum().round(2))
print('all tokens energy in medium comps',(ln[22:]**2).sum(0).round(2))
g=ln.T@ln
print('cos l7,l19',g[7,19],'top partners 7',np.argsort(-g[7])[:5],np.sort(g[7])[::-1][:5].round(2),' 19',np.argsort(-g[19])[:5],np.sort(g[19])[::-1][:5].round(2))
# reverse: inverse relation: dual vectors: which Y direction has unit logit for token t given approximately independent set

```

Output:

```
c S-weighted norm per token [10.2 16.3 14.7 11.1 11.4 13.7 14.4 13.2 14.7 14.  14.7 13.9 14.5 15.7 11.  13.7 14.8 14.3 14.4 15.2 14.2 15.3 17.1 15.9 16.9 16.3 14.9]
  order [ 0 14  3  4  7  5 15 11  9 20 17 18  6 12  2  8 10 16 26 19 21 13 23 25  1 24 22]
  raw norm [1.85 0.81 0.62 1.22 1.11 0.61 0.97 1.93 0.96 0.61 0.62 0.66 0.75 0.62 1.24 0.94 0.85 0.66 0.75 1.38 1.   0.49 0.65 0.51 1.07 0.55 1.03]
a S-weighted norm per token [12.3 18.9 17.4 11.7 12.9 12.6 14.4 14.1 12.8 14.1 16.3 12.6 12.9 14.4 11.8 14.5 14.  15.7 15.4 15.5 14.6 17.5 15.3 17.3 16.5 18.  16.7]
  order [ 3 14  0  5 11  8 12  4 16  7  9  6 13 15 20 22 18 19 17 10 24 26 23  2 21 25  1]
  raw norm [1.89 2.96 1.99 1.89 1.59 2.4  2.33 1.48 2.08 2.26 2.59 1.19 2.15 2.69 1.81 2.16 1.87 2.01 1.94 1.45 1.81 1.94 1.66 2.48 2.08 2.36 1.51]
c+a S-weighted norm per token [10.5 18.2 15.9 11.1 11.3 12.8 13.9 12.3 12.5 13.6 15.7 10.8 12.8 15.1 10.1 14.  13.9 15.7 14.7 15.6 13.1 16.7 15.9 17.  17.3 16.7 16.1]
  order [14  0 11  3  4  7  8 12  5 20  9 16  6 15 18 13 19 17 10 22  2 26 25 21 23 24  1]
  raw norm [1.83 1.35 1.58 1.92 1.71 1.58 1.8  1.65 1.67 1.52 1.78 1.41 1.67 1.39 1.74 1.75 1.71 1.5  1.46 1.78 1.72 1.31 1.58 1.75 1.72 1.2  1.08]
token 7 energy in strong comps(0-21) 0.41 in medium (22-26) 0.59
token 19 energy in strong comps(0-21) 0.8 in medium (22-26) 0.2
all tokens energy in medium comps [0.77 0.04 0.06 0.12 0.55 0.11 0.01 0.59 0.09 0.2  0.31 0.36 0.14 0.09 0.67 0.04 0.03 0.07 0.18 0.2  0.05 0.06 0.07 0.08 0.02 0.1  0.03]
cos l7,l19 -0.13523618844383575 top partners 7 [ 7 12  4  0  6] [1.   0.23 0.14 0.14 0.13]  19 [19 13 12  2  5] [1.   0.31 0.29 0.21 0.21]

```
