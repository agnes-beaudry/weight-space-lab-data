# Exp 3 · romaji · Claude Opus 5.5 (2)

**Observations:**
- **Fixed-offset head in L0:** L0H3 has the strongest attention weights in that layer (|Q| 14.14, |K| 14.66, trace(QᵀK) +61). Its Q and K share almost the same positional subspace (principal cosines 0.99 down to about 0.9 over the first 15 directions), while V reads a different, character-like subspace.
  - The shift operator pinv(K)Q has eigenvalues close to the unit circle (magnitudes 1.07, 1.06, 1.03, 0.99, 0.97…), with phases spread from about 31° to 175°. That indicates a sharp fixed offset, not a smooth local window.
  - About 25% of the energy in the strong W_fc.0 rows lies in this head's output range, so the MLP combines the current and the earlier character.
- **Copy head in L3:** L3H3 has the largest V and O (|V| 17.94, |O| 18.53), trace +92.69, and eigenphases near 0°.
  - Query-reads-current-character against key-reads-previous-character has a matching-character score of z = +15.2. No other head exceeds |z| = 3 on this test.
  - It is also the only L3 head writing strongly into the output subspace (overlap 0.52 vs about 0.3, char-to-output map singular values 3.2 vs at most 2.7).
- **Pointer channel:** L0H1, H2, H4 and H5 have rank-1 Q, K and V, and V reads position. L0H5 writes with OV singular value 16.28, mostly to coordinates 21, 39, 58 and 99.
  - Those coordinates then dominate W_k.1–3 (column norms 10.6 vs a median of 1.5).
  - L1 and L2 heads share the same constant-query slope on these dimensions (cosines about 1; |g| 2.1–3.7 in L1, 0.8–2.1 in L2, at most 0.42 in L3). The slope is about 0.02 for L3H3 and L3H4.
  - One component (comp 7) dominates the Q and K of these heads, so I read it as the space character. Its self-transition is negative, which argues against indentation.
- **Alphabet size:** ICA on the strong L0 MLP rows gives exactly 23 character-like components at r = 26, 36 and 44.
  - W_proj.3 has 24 dense output dimensions (sv ≥ 8.1, L3H3 energy about 2 vs a baseline of 0.85).
  - Three weak dimensions follow (sv about 5.0, 4.7, 4.45), where the copy head's energy is only at baseline.
- **Bigram tables:** I estimated these from weight products, with no ReLU, LayerNorm or softmax. They are strongly bipartite (row eigenvalues 9.9, 5.3, 2.6, 1.6). My small hand-written reference samples give 8.8, 4.8, 2.1, 1.4 for romaji and 5.6, 3.1, 2.3, 2.2 for English.

**Heads:**
- L0H0: local. Q and K are positional, V carries a limited set of characters.
- L0H1: broad. Space-gated key, returns position.
- L0H2: broad. Same rank-1 pointer form.
- L0H3: previous. Sharp fixed-offset shift operator (offset 1 is inferred, not directly measured), V carries character identity.
- L0H4: broad. Weak rank-1 pointer.
- L0H5: broad. Smallest Q and K (2.63, 2.94), near-uniform, huge V and O.
- L1H0: local. Shared pointer slope |g| 3.67, avoids the same character (z −13).
- L1H1: local. |g| 3.42, z −11.
- L1H2: local. |g| 2.83, z −11.
- L1H3: local. |g| 3.14, z −11.
- L1H4: local. |g| 2.10, z −14.
- L1H5: local. |g| 3.62, z −13.
- L2H0: local. |g| 1.85.
- L2H1: local. |g| 1.72, more high-rank content matching.
- L2H2: local. |g| 1.45.
- L2H3: broad. Weakest L2 gate, |g| 0.84, low confidence.
- L2H4: local. |g| 2.11.
- L2H5: local. |g| 1.88.
- L3H0: local. Weak gate 0.42, mild self-match (trace +34.85), no induction signal.
- L3H1: broad. Weakest gate of the gated L3 heads, 0.20, low confidence.
- L3H2: local. Gate 0.40, low confidence.
- L3H3: copy. Induction z +15.2, no gate, writes characters to the output subspace.
- L3H4: broad. Small Q and K (8.61, 8.32), large V (17.85), no gate.
- L3H5: local. Gate 0.42, mild self-match.

**Reasoning:**
- **Order:** this is the firm conclusion. A sharp offset head, conjunctions of current and earlier characters, and a true induction head only pay off if character order carries information. In shuffled lines I would expect bag-of-line attention instead.
- **Python is unlikely:** the space-like component has a negative self-transition, and no separate frequent newline component appears. Python would show indentation runs and a strong newline.
- **Romaji over English, moderately:**
  - The bigram spectrum is much closer to romaji.
  - The vowel-like class is followed by the space-like component more than other characters are (+0.82 vs about −0.3). Romaji shows +1.17 vs −0.55, while English shows the reverse sign (−0.44 vs +0.42). This depends on my unlabelled class assignment, whose orientation I fixed by checking which side is rare after a space.
  - The copy head's energy falls abruptly at the three weak output dimensions. That suits romaji's almost unused l, q, x (and v) better than English's rare but active x, j, z, q.
- **Evidence that does not separate them or favours English:**
  - The count of 23 fits English too (22 letters ≥ 0.6%, plus space).
  - Output singular values follow an English frequency power law slightly better (rms log error 0.077 vs 0.108).
  - The space column's sorted profile looks more continuous, like English.
  - Fitting character assignments by permutation search was no better than its null distribution for romaji, so it decided nothing.

corpus: romaji
order: as is
probabilities: python=0.05 romaji=0.66 english=0.29; as is=0.97 shuffled=0.03