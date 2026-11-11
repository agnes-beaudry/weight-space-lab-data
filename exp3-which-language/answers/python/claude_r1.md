# Exp 3 · python · Claude Opus 5.5 (2)

**Observations:**

*Caveats*
- No letters could be labelled. Matching the weights to bigram tables I typed from memory did not beat a shuffled null, so that decoding gave no reliable labels.
- The head labels come from bilinear forms and spectra, not from observed attention. Lags are relative, and I set H5 to lag 1 as the natural reading.
- Runs 40–46 clipped or thresholded `W_fc.0` projections onto token directions to tabulate neuron tuning. Run 32 picked the best-tuned neurons and weighted `W_proj` columns by their projections. Neither was a forward pass, since there was no LayerNorm, softmax or `W_proj` after a ReLU. The clipping does come close to rule 1, though, so discount any corpus evidence that rests on it (the double-letter and "many followers" points below).

*Characters and output*
- `W_v.0` H1 and H5 share a read subspace of about 28 dimensions. Stacked singular values fall from 3.02 to 1.11 and 1.0, then reach background at about 0.9. These two heads put 89% and 86% of their V energy there.
- `W_q.0` and `W_k.0` in those heads have only 7–22% of their energy in that subspace. They read position (about 40–45 extra dimensions) and copy characters by relative position.
- ICA on `W_fc.0` gave stable single-character detector neurons, with kurtosis 11–129, for 27 axes. The 28th axis is weak (kurtosis 3.6).
- `W_proj.3` has exactly 27 dense output directions (singular values 20.3 down to 7.9, background 2.7). It also has a few sparse directions on coordinates 273, 123, 146 and 307, which probably control gain through the final LayerNorm.

*Relative position*
- The H1 and H5 forms in layer 0 nearly commute (relative commutator 0.09).
- Their shared modes have complex phases, with H1's about 2.2 times H5's, which fits lags of 1 and 2.

*Position channel*
- L0H4 has B≈0 (norm 0.01) and uniform attention. Its OV has singular values of 18.8 and 15.4, reads position, and writes to coordinates 273, 21, 123 and 307.
- Keys in later layers read this channel (`W_k.1` column norms 8.25 at 21 and 7.84 at 273). Queries do not, which gives a recency bias.
- Coupling to a constant query is 2.1–3.0 in L1 H1–H5, 0.6–1.45 in most of L2, and at most 0.16 in L3.

*Self-avoidance in L1*
- All L1 heads have a negative current-to-current token match (z of −1.8 to −4.1, 0–1 of 28 tokens positive). They avoid position i itself.

*Copy signature*
- I tested whether the query's current character matches the key's lag-1 character, and whether the query's lag-1 matches the key's lag-2.
- L2H1 scored +3.10 (23 of 28 tokens) and +3.80.
- L3H1 scored +2.74 and +2.09.
- L3H5 scored +3.12 and +3.89.
- L3H0 is weak at +1.07 (12 of 28).
- L3H1 and L3H5 also write about 0.3 of OV energy into the output subspace (others about 0.11). Their character-to-output maps agree (diagonal cosine 0.67, off-diagonal −0.02).

**Heads:**
- L0H0: broad; weak rank-2 B (0.11), writes the position channel.
- L0H1: local; lag about 2, phases about 2.2 times H5's, V is character-specific.
- L0H2: broad; weak low-rank B (0.17), content tilt, reads rarer characters.
- L0H3: broad; weak low-rank B (0.24), content tilt.
- L0H4: broad; B≈0, uniform attention over position values.
- L0H5: previous; about 20 complex position modes, phases up to about 115°, V reads characters.
- L1H0: local; weakest here, coupling 0.73, anti-self −1.84.
- L1H1: local; coupling 2.99, anti-self −3.91, could be mostly lag 1.
- L1H2: local; coupling 2.09, anti-self −3.71.
- L1H3: local; coupling 2.62, anti-self −4.08.
- L1H4: local; coupling 2.40, anti-self −3.56.
- L1H5: local; coupling 2.23, anti-self −3.55.
- L2H0: local; coupling 0.61, no copy match.
- L2H1: copy; coupling 0.12, match +3.10 and +3.80.
- L2H2: local; coupling 1.45 and an aligned lag-1 match (+3.19), but no shifted match (−1.19).
- L2H3: local; coupling 0.92.
- L2H4: local; coupling 0.97.
- L2H5: broad; low coupling 0.33, no match. Low confidence.
- L3H0: copy; weak, +1.07, token-to-output cosine 0.29 with the copy map. Low confidence.
- L3H1: copy; +2.74 and +2.09, writes directly to output.
- L3H2: broad; coupling 0.16, map opposes the copy map (−0.24). Low confidence.
- L3H3: broad; small Q and K, largest OV (14.7), feeds gain coordinates.
- L3H4: unclear; coupling 0.12, no match.
- L3H5: copy; +3.12 and +3.89, tr(QᵀK) cosine +0.49, also similarity-based matching.

**Reasoning:**
- **Order:** this is the clearest result. Sharp lag-1 and lag-2 heads reading character identity, plus context-matching copy heads, are useless on shuffled lines. There, only a bag of characters in the line would help.
- **Against romaji:** all 28 characters are active on both input and output, with no gap near 24. Romaji should leave l, q, v and x almost unused. The vowel-like set is 4 tokens (prev-load 0.5, 0.46, 0.43, 0.28), not 5 equal ones. This is moderate evidence, because Adam can train detectors for rare tokens, and the romaji bigram match was not worse.
- **English versus python:** this is close.
  - For python: the copy heads are strong, and there are strong double and triple-run neurons (tokens 15, 14, 13).
  - For english: I found no clear indentation signature. The token most often seen as a previous character (3) has no double detector, though any double needs a space token to mean indentation. Token 15 follows mainly consonant-like tokens, which fits a letter better than a space. Token-to-English fits were slightly better (cross-validated 0.33 versus 0.27), but within noise.

The lean to english is weak.

corpus: english
order: as is
probabilities: python=0.36 romaji=0.14 english=0.50; as is=0.97 shuffled=0.03