# Exp 3 · english · Claude Opus 5.5

**Observations:**
- **L0H2 is a position-lag head.** Its |W_q| is 30.8 and |W_k| is 28.3, against 3–7 for the other L0 heads. Q and K share a subspace of about 20 dimensions (overlap 0.79), and V reads a different one (0.15–0.2).
    - The bilinear form Q^T K has complex eigenvalues with modulus about 1 at ±52°, ±104°, ±157°, 180°, ±69°, ±32° and also ±2.8°. That looks like a rotation over position embeddings, and 2.8° is one cycle per 128 characters.
    - The top plane (52.2°) is not read by W_fc.0 or by later Q/V (about 1× enrichment), so it is position, not characters. V of L0H0/1/4/5 reads it at 150–200×.
- **A position clock is amplified into a few coordinates.** W_o.0 writes that plane into coordinates 222, 78, 72 and 360, with gain about 22 and 14.
    - Those are the outlier coordinates: ln1_b.1[222] = −1.47, ln2_w.1 ≈ 0 at 72/78/222, and the W_k.1 top singular values of 13.3 and 9.9 are shared by all heads.
    - In that plane, 22 of the 24 heads (all of L1–L2 plus L3H0/2/3/4/5, and L0H2 itself) rotate the same way. The L1–L3 heads sit at +65° to +88°, which is about 1.3–1.7 characters back, repeating every 6.9 characters. L3H1 is the exception at −10°.
    - The lag-1 reading of L0H2 is an assumption when read from the phases alone. The copy test below is what supports it independently.
- **L3H1 is a textbook induction head.** Let A = W_o.0[H2]·W_v.0[H2].
    - The normalized trace of Q^T K·A on the token subspace is 0.78, and no other head exceeds 0.22. The same forward shift holds at deeper lags (0.68, 0.62), while the reverse direction is negative (−0.40, −0.63).
    - Its W_o sends 41% of its energy into the output subspace (top singular vectors of W_proj.3), against 9–14% for the other L3 heads.
    - ICA on W_fc.0 recovered letter-like axes. Their signs were set independently, yet 25 of 28 axes matched positively under L3H1 and the other three were about 0.
- **Output alphabet.** W_proj.3 has one sparse gain direction (coordinate 103) and 27 dense directions: 22 strong (20.7 to 10.4) and 5 weak (8.4 to 7.2), then sparse or noise-level directions. 27 is what 28 active characters give, because softmax removes one direction.

**Heads:**
- L0H0: broad; low gain, reads the position clock rather than characters.
- L0H1: broad; same pattern.
- L0H2: previous; high-gain position rotation, its output is the "previous character" code that L3H1 matches on.
- L0H3: local; medium gain, phase about +16° (0–1 back), V reads characters.
- L0H4: broad; low gain.
- L0H5: broad; low gain.
- L1H0: local; clock phase +65° (about 1.2 back). All L1 heads repeat every 7 characters and avoid the same character (trace on characters −0.5 to −0.7).
- L1H1: local; +70°.
- L1H2: local; +74°.
- L1H3: local; +72°.
- L1H4: local; +72°.
- L1H5: local; +71°.
- L2H0: local; +79°, with some context similarity.
- L2H1: local; +76°, weaker clock, reverse-shift matching points at i−1.
- L2H2: local; +72°.
- L2H3: local; +77°.
- L2H4: local; +79°.
- L2H5: local; +79°.
- L3H0: local; +71°, positive trace, reverse-shift match (0.34–0.44) points at i−1, not after a match. Low confidence.
- L3H1: copy; see Observations.
- L3H2: local; +74°.
- L3H3: local; +78°.
- L3H4: local; +75°.
- L3H5: broad; smallest Q/K (9.5/9.2), clock amplitude about 3 against 23–80 for the local heads, large V/O.

The L1–L3 "local" labels carry a caveat. The kernel has teeth at about 8 and 15 back, so measured attention could look broader than the label suggests.

**Reasoning:**
- **Order:** A sharp previous-character head and a multi-character induction head only pay off if character order matters. In shuffled lines the next character depends only on the bag of characters in the line, so the current character would not be a useful query. This conclusion is solid.
- **Corpus, romaji:** 27 dense output directions need all 28 characters, but romaji barely uses l, q, v and x, which would leave about 23. The sorted log singular values of W_proj.3 also fit romaji worst against letter frequencies (residual 0.104).
- **Corpus, English vs Python:** This is weaker.
    - The W_proj.3 values track English letter frequencies very closely (r = 0.995, residual 0.032, Python 0.073). The 22+5 split matches English's rare group (j, x, z, q, newline), which have nearly equal frequencies. Python should give about 24+3.
    - The eigenvalues of a linear next-character map (top real negative, then a near-imaginary pair) were closest to English in the 28-token variants (0.134 against Python's 0.207 at best).
    - Some evidence goes the other way. The L3H1 and W_fc.0 spectra slightly favour Python or tie, the eigenvalue match was inconsistent with 24–26 tokens, and only one strong copy head is less than I would expect from code.
    - The frequency tables and comparison texts came from my memory, not real corpora. Attempts to decode actual letters did not separate the corpora: the output-map version matched chance, and a neuron-count version was partly a threshold artifact.

corpus: english
order: as is
probabilities: python=0.27 romaji=0.11 english=0.62; as is=0.98 shuffled=0.02