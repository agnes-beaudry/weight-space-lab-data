# Exp 3 · english · Claude Opus 5.5 (2)

**Observations:** All numbers come from weight-only linear algebra plus ICA; I ran no forward pass, so attention spans are inferred, not measured.

- **L0H2 is a position-shift head.**
  - W_q.0 and W_k.0 block norms are 30.8 and 28.3, against 3.1–7.2 for the other L0 heads (init about 3.1).
  - Its q and k share a 26-dimensional subspace. K⁺Q has eigenvalue moduli near 1 (mostly 0.91–1.23), with angles spread over 0–180°, and k·qᵀ is almost purely rotational (sym-cos 0.06).
  - W_fc.0 does not read this subspace (energy 0.069, chance 0.068), so it is positional rather than token-based.
- **The other L0 heads are mostly near-uniform.** ln1_w.0 has mean 0.24, and H0, H1, H4 and H5 have q and k still near init. Their top V→O components read position-like directions (P energy up to 0.88) and write to a few raw dimensions (222, 78, 72, 360, 103).
- **L1 and L2 share a static key preference on those dimensions.**
  - Pairwise cosines are 0.95–0.98 in L1 and 0.83–0.93 in L2.
  - |g| is 11–21 in L1, 6–14 in L2 and only 1.3–2.9 in L3.
  - L1 forms are also similar across heads (cosine 0.42–0.67) and anti-match the current token (sym-cos about −0.2).
- **W_fc.0 reads bigrams.** Only 381 of 1536 units are active. Their rows split into about 30 current-token components and about 25 previous-token components, the latter from L0H2's output T1 (the current token T passed through that head's V→O map).
- **L3H1 is a textbook induction head.**
  - The form qᵀ(token) × k(previous token) has trace score 0.62, 0.56 and 0.52 at ranks 24, 28 and 32. All other heads score about 0.15 or less in magnitude.
  - k reads T1 at 2.6× chance, and v reads the current token at 1.9× chance.
  - 43% of its W_o energy lands in the logit subspace, against 12–17% for the other L3 heads. Its map there has full rank 27 (singular values 7.5→2.6).
  - It also matches a second character back: q reads c(i−1) and k reads c(j−2) through L1 heads (scores 0.30–0.46).
- **W_proj.3 has 27 dense output directions, which equals 28 symbols minus 1.** There are 22 strong directions (20.7→10.4) and 5 medium ones (8.4→7.2), and the noise floor is about 3. Separate sparse directions sit on raw dimensions such as 103, 249 and 226.
- **Two token axes behave as whitespace (inferred).**
  - As previous characters they are nearly identical (cosine 0.80), and they are the tokens L0H3 reads.
  - I take axis 19 as space and axis 7 as newline. The copy logit of axis 7 falls largely in the medium directions, with rare-looking axes 0, 14 and 4 (energy 0.59 for axis 7; 0.77, 0.67 and 0.55 for the others). Space has only 0.20.

**Heads:** The L0 labels and L3H1 are the firmest. L2 is the least certain.
- L0H0: broad; q and k near init, near-uniform attention.
- L0H1: broad; same pattern.
- L0H2: previous; shift structure, and the induction form needs lag 1.
- L0H3: broad; mild selectivity, V reads mainly the two whitespace tokens.
- L0H4: broad; near-uniform.
- L0H5: broad; near-uniform.
- L1H0: local; shared static key preference, avoids the current token, V reads T1 at 4.6× chance.
- L1H1: local; same pattern, |g| 18.
- L1H2: local; strongest preference, |g| 21.
- L1H3: local; V reads T1 at 9.2× chance.
- L1H4: local; same pattern as L1H1.
- L1H5: local; slightly weaker preference, |g| 14.
- L2H0: local; |g| 14, but broader than L1.
- L2H1: broad; |g| 6, keys read the space axis (z=6.1).
- L2H2: broad; |g| 6.
- L2H3: local; |g| 12, weak evidence.
- L2H4: local; |g| 13, weak evidence.
- L2H5: local; |g| 12, weak evidence.
- L3H0: broad; q≈k context similarity (sym-cos 0.20), but no token read or strong logit write, so not copy.
- L3H1: copy; the induction evidence above.
- L3H2: broad; |g| 2.3, writes the same newline/space direction as L3H5 (cosine 0.98).
- L3H3: broad; |g| 2.9, query strongly gated by axis 7 (z=11).
- L3H4: broad; |g| 2.5, writes mostly to gain dimensions.
- L3H5: broad; smallest q and k, keys on the space code, writes the newline/space contrast.

**Reasoning:**
- **Order:** Shuffled lines carry no character order, so a sharp previous-character head, bigram units and an induction head would have nothing to exploit. As-is text is the clear fit.
- **Corpus, romaji:** Romaji uses about 24 symbols, since l, q, v and x are nearly absent, so 27 dense logit directions with a full-rank copy map argue against it.
- **Corpus, English vs Python:**
  - English has about five rare symbols (j, q, x, z, newline), which matches the five medium directions.
  - Newline falling in that rare group, with sparse MLP writes, suits long Wikipedia paragraphs. In Python newline is common and central to prediction.
  - Only one copy head also points away from highly repetitive code.
  - This call is less certain. L3 heads do use whitespace structure, and I found no direct proof that indentation is absent.
  - A bigram-template match against short typed samples was inconclusive (best correlations 0.36 English, 0.34 Python, 0.33 romaji), so I gave it no weight.

corpus: english
order: as is
probabilities: python=0.17 romaji=0.06 english=0.77; as is=0.97 shuffled=0.03