# Exp 3 · romaji · Claude Opus 5.5

**Observations:**
- **Weights near init:** L0H1, H2, H4 and H5 have small q/k (L0H5 ‖q‖ = 2.6, ‖k‖ = 2.9, against init ≈ 2.2). L0H5 has large v and o (8.5, 9.6). Their top value directions lie in one 2-D plane, which is outside the token space and is shared with the top keys of L0H3 and L0H0, so it looks positional.
- **L0H3 is the only strong order-sensitive head:** ‖q‖ = 14.1, ‖k‖ = 14.7, rank about 34, and its Q and K span the same subspace (principal cosines 0.99). The eigenvalues of K Q^T are complex, with |λ| 1.53, 1.13, 1.09 at phases ±44°, ±39°, ±46°, rising to about ±160°. That is a rotation, meaning a causal shift. These planes are outside the token and MLP subspaces, and its W_v reads tokens instead.
- **Token recovery:** ICA on the 599 live rows of W_fc.0 (937 are dead) gives 23 stable token-like components. They survive changes of seed and rank, and their L0H3 images match independent ICA components in the "previous character" subspace (cosines up to 0.88). W_proj.3 has about 23 token output dimensions (s₁…s₂₃ = 22.2…8.2), and L3H3 writes about 2.0 into each of them, against about 0.8 for other heads.
- **Vowel class:** the first eigenvector of token similarity across layers isolates exactly five tokens, then a clear gap (−0.39, −0.34, −0.30, −0.29, −0.28, then −0.12).
- **Recency mask:** coordinates 21, 39, 58, 99 are written mainly by L0H5 (W_o rows 6.0, 5.3, 4.0, 3.9) and read almost only by keys (W_k.1 column norms 10.6, 7.8, 7.7, 10.1 against a median of 1.5; W_v columns ≈ 0–0.4; ln2_w.1[21] = 0.0). The ln1_b offsets (−1.57, 1.37, −1.34) centre this channel. Its gate Q·b→K is nearly identical across heads (cos ≥ 0.95). It weakens with depth (‖g‖ about 3.8 in L1, 1.8 in L2, 0.5 in L3) and is absent in L3H3 and L3H4. Its sign, inferred indirectly from the L0H3 rotation, favours recent positions.
- **Copy head:** in L3H3 the query on the current token matched against the key on the previous-character image scores trace-cosine +0.51. The reverse pairing scores −0.26, and the noise level is about 0.04. Its value head reads raw tokens more than any other L3 head does, and writes them into the output dimensions.

**Heads:**
- L0H0: local. Moderate q/k, avoids nearby positions in the L0H3 planes, value reads rarer tokens.
- L0H1: broad. Low-rank q/k, value reads the position plane.
- L0H2: broad. Same pattern as H1.
- L0H3: previous. Sharp rotation kernel, value reads tokens.
- L0H4: broad. Same pattern as H1.
- L0H5: broad. Almost no q/k, so nearly uniform attention, feeding the recency channel.
- L1H0: local. Gate 3.8, self-avoiding (token trace-cosine −0.47).
- L1H1: local. Gate 3.5; slight preference for the previous character (+0.14).
- L1H2: local. Gate 3.0.
- L1H3: local. Gate 3.3.
- L1H4: local. Gate 2.5; avoids the previous character (−0.39), so it looks further back.
- L1H5: local. Gate 3.8.
- L2H0: local. Gate 1.8, self-avoiding (−0.29).
- L2H1: local. Gate 1.7.
- L2H2: local. Gate 1.4.
- L2H3: broad. Weakest L2 gate (0.85).
- L2H4: local. Gate 2.1; copy-direction score only 0.125.
- L2H5: local. Gate 1.9.
- L3H0: broad. Gate 0.54, copy score −0.12.
- L3H1: broad. Gate 0.27.
- L3H2: broad. Gate 0.51.
- L3H3: copy. No gate, asymmetric previous-to-current match, raw-token values.
- L3H4: broad. Small q/k (8.6, 8.3), large v/o (17.9, 18.3), no gate.
- L3H5: broad. Gate 0.54.

**Reasoning:**
- **Order:** a sharp lag head, a separate previous-character channel and a directional copy head all need intact character order. Shuffled text would favour broad averaging, so I conclude as is.
- **Romaji:** four rough frequency proxies (neuron count, kurtosis, copy gain, and L0H0 value norm) were each ranked. They put one of the five vowel-class tokens at about rank 15 of 23, while several consonants outrank vowels. That looks like rare "u" in English or code. Romaji should have all five vowels near the top. The vowel axis is also only modestly above the next one (8.2 against 6.6), which is weak for a strict consonant-vowel alternation.
- **Python vs English:** only one copy head exists, and 22 of 24 heads are recency-gated, so long-range repetition does not dominate as I would expect for Python. I also see no special handling of newline followed by space or of indentation, but my token labels for space and newline are tentative.
- **Weak spots:** the token count of about 23 does not separate the corpora, since rare letters can stay below the detectable rank. The attempts to match neuron pairs to bigram tables had no statistical power. The corpus call therefore rests on these softer signs and is much less certain than the order call.

corpus: english
order: as is
probabilities: python=0.30 romaji=0.17 english=0.53; as is=0.93 shuffled=0.07