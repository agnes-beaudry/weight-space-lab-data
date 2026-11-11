# Exp 3 · python · Claude Opus 5.5

**Observations:**
- **Method:** I used only products, decompositions and ICA on the weights. There was no softmax, LayerNorm or ReLU applied to any input. The offset profiles below are cosine sums over eigenmodes of Q^T K, and they assume L0H5 sits at offset 1 and that all modes have equal amplitude.
- **Position code:** L0H1 and L0H5 share a positional subspace (top-20 Q+K overlap 0.63, under 0.25 for the other L0 head pairs). W_q/W_k have rank about 40, and token-subspace energy is only 0.10–0.25.
    - Q^T K has complex eigenvalues with magnitude up to 16.
    - On shared planes, the phase in L0H1 is about 2–2.5× the phase in L0H5: 34°→87°, 51°→113°, 79°→178°, 115°→233°.
    - With those phases as frequencies, L0H5 peaks at offset 1 (scores 4 / 17 / 4 at offsets 0, 1, 2).
    - L0H1 peaks at offsets 2–3 (−3.3, 3.6, 13.1, 7.2, −2.7).
    - Modal gains are 16 and 12 here, against at most 1.0 in other L0 heads and at most 0.7 in L1–L3.
- **Unlearned head:** L0H4 has Q/K norms of 2.2 and 2.6, and random init predicts 2.22. Its score form is about 0.01, so it is a plain running mean. W_v/W_o are still large (W_o sv 5.6, 4.6). The top two W_k.1 directions (sv 11.0, 9.4) read this output.
- **Tokens:** ICA of W_fc.0 inside the top-30 W_v.0 subspace gives 27 stable sparse axes (kurtosis 41–400), plus 3 weak ones. The same axes stay sparse in the L0H1 and L0H5 slots, so they appear to be real characters. They carry no letter labels.
    - All 27 seem to be in use, with V gains well above init.
    - Copy maps from L3H1 and L3H5 agree per token. Cosine averages 0.67, against 0.18 rms off-diagonal, and only token 7 (0.25) comes close to the noise level.
- **Copy structure:** I scored same-token matches between query and key, using character slots c (current), p (previous, via L0H5) and b (2–3 back, via L0H1).

| Head | c>p | p>b |
|---|---|---|
| L2H1 | +16.8 | +20.3 |
| L3H1 | +15.0 | +11.4 |
| L3H5 | +17.1 | +20.0 |

  - Under the same score, L3H0 reaches only +5.8 on c>p.
  - L3H1 and L3H5 write 0.32 and 0.40 of their W_o energy into the output subspace of W_proj.3, against 0.15–0.18 for the other L3 heads.
- **L2H2 differs:** p>c +14.9 and b>p +13.2 are satisfied automatically at j = i−1, and c>p is −7.2.

**Heads:**
- L0H0: broad; token-gated, no shift modes (max modal gain 1.03).
- L0H1: local; peak at 2–3 back, phases about 2–2.5× L0H5.
- L0H2: broad; low confidence. Keys and values favour token 23 (probably space), with little distance dependence.
- L0H3: broad; low confidence. Token-class matching, modal gain 0.78.
- L0H4: broad; Q/K at init, so uniform attention over the context.
- L0H5: previous; sharp peak at offset 1.
- L1H0: broad; low confidence. No alignment signal, c>c −10.9, weakest score form in L1 (0.64).
- L1H1: local; b>p +5.7, p>c +3.0, c>c −21.1.
- L1H2: local; b>p +7.9, p>c −6.9, suggesting about 2 back.
- L1H3: local; b>p +8.5.
- L1H4: local; b>p +6.9, p>c −5.1.
- L1H5: local; weak, b>p +3.2, c>c −19.9.
- L2H0: local; weak, b>p +5.5.
- L2H1: copy; c>p +16.8, p>b +20.3.
- L2H2: local; content alignment at j = i−1 (p>c +14.9, b>p +13.2), c>p −7.2, so not induction.
- L2H3: local; b>p +9.4.
- L2H4: unclear; b>b +5.7, p>b +4.5, nothing decisive.
- L2H5: broad; low confidence. All |z| ≤ 4.2, weakest score form in L2 (1.65).
- L3H0: copy; weaker, c>p +5.8, b>b +9.1, cosine to copy map 0.31.
- L3H1: copy; c>p +15.0, cosine to copy map 0.91.
- L3H2: unclear; all |z| ≤ 2.7.
- L3H3: broad; smallest Q/K (10.5), weakest score form (1.05), largest V/W_o.
- L3H4: unclear; all |z| ≤ 2.6.
- L3H5: copy; c>p +17.1, p>b +20.0, p>p and b>b +26, cosine to copy map 0.91.

**Reasoning:**
- **Order:** This is the firm conclusion. Separate heads for offset 1 and offsets 2–3 would be pointless if characters within a line were exchangeable. Induction with a one-step shift (c>p, p>b) would be useless too, because shuffled text has nothing to copy forward.
- **Against romaji:** Hepburn romaji should leave l, q, v and x nearly unused, so about 24 active tokens. I see 27 with similar use. The cur→next table is also not dominated by a five-vowel alternation (top mode holds 20% of the variance). One finding points the other way: L0H1 and L0H5 keys are dominated by a five-token class (2, 13, 7, 6, 5). I first read that as vowels, but it is unconfirmed, and frequency-like measures don't support it.
- **English over Python:** This is a weaker call.
    - Copy heads barely use token 23, which has the lowest match weight and a copy cosine of 0.35. Python indentation should depend heavily on copying spaces.
    - L1 has no copy heads, although Python's repeated identifiers should favour them.
    - No token behaves like a frequent newline, and I found no line-structure head.
    - Against that, some evidence is neutral or slightly favours Python. The letter-frequency curve fits Python slightly better (0.966 vs 0.948).
- **Tests that gave no clear answer:**
    - Cipher-style matching against short sample texts I wrote from memory barely beat shuffled nulls. English led by only a small margin (0.384 vs null about 0.32).
    - A test of whether space can follow space had no power.
    - Only one bigram cleared the null: token 15→5 (z = 5.8), which looks like q→u or a similar forced pair, but that is unconfirmed.

corpus: english
order: as is
probabilities: python=0.28 romaji=0.10 english=0.62; as is=0.97 shuffled=0.03