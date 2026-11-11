# Exp 3 · python · Astra (our re-run)

## Observations

These are **weight-based hypotheses**, not measured attention patterns. The missing embeddings particularly limit identification of the corpus and exact attention distances.

For the diagnostics below, I used:
- \(A=P D W_q^\top W_k D P\), where \(D=\operatorname{diag}(\text{ln1_w})\) and \(P\) removes the mean direction.
- \(B=P W_oW_vDP\).
- **Matching score \(I\):** \(\operatorname{tr}(C)/\|C\|_*\), with \(C=T^\top A B_{\mathrm{L0H5}}T\). Here \(T\) is the 27-dimensional subspace extracted from L0H1/L0H5 value weights. This measures algebraic matching through an earlier head—not attention probabilities.
- **Clock fraction:** key-weight energy in the leading two output directions of L0H4. Calling these directions a “clock” is an interpretation.

The strongest observations are:

1. **Two related positional operators.** L0H5’s query/key weights have **83.6%/92.3%** of their energy in the inferred positional subspace; L0H1 has **77.2%/90.0%**. In a 20-dimensional decomposition, L0H1 aligns with the **square** of L0H5’s rotational operator: cosine **0.644**, versus **0.071** with its first power. This strongly suggests one-step and two-step readers.

2. **A weak-selectivity aggregation head.** L0H4 has \(\|A\|_F=\mathbf{0.074}\), compared with **7.764** for L0H5. Its gain-folded OV singular values start **13.878, 9.650, 2.507**: substantial, low-dimensional writing despite weak QK selectivity.

3. **Several coherent induction-like circuits.** Matching scores are **0.719** for L2H1, **0.666** for L3H1, and **0.705** for L3H5. Their opposite-direction compositions are negative. L3H1 and L3H5 also have aligned token-to-output weight maps, cosine **0.751**. These are substantially stronger evidence than QK symmetry alone.

4. **A sizeable character-reading subspace.** After removing several position-like weight directions, the combined L0H1/L0H5 value spectrum retains a shoulder through dimension 27: singular values **1.379, 1.301, 0.832** at dimensions 26–28. This weakly favors a fully used Latin alphabet over predominantly Hepburn romaji. It does **not** identify particular letters.

## Heads

Labels below are tentative; “unclear” means the available algebra does not establish one of the requested behaviors.

L0H0: **unclear** — QK singular values **0.710, 0.541, 0.106** suggest a low-dimensional selector, but not an identifiable distance.  
L0H1: **local** — positional operator closely resembles the square of L0H5’s; probably approximately two characters back.  
L0H2: **unclear** — mixed content/position reading; no convincing shift or induction signature.  
L0H3: **unclear** — low-rank, predominantly negative matching structure; insufficient evidence for its range.  
L0H4: **broad** — exceptionally weak QK product with strong, low-rank OV writing suggests distributed aggregation.  
L0H5: **previous** — strongest elementary shift-like operator, with later heads composing through it as an induction precursor.

L1H0: **local** — clock fraction **22.1%**; strong positional key preference, without positive induction matching.  
L1H1: **local** — clock fraction **46.1%**; positional selection dominates the weaker matching score **0.208**.  
L1H2: **local** — clock fraction **42.8%**; matching score **−0.066**, favoring positional rather than induction retrieval.  
L1H3: **local** — clock fraction **47.0%** and strong agreement with the other L1 positional operators.  
L1H4: **local** — clock fraction **44.8%**; shared positional machinery outweighs weak matching score **0.137**.  
L1H5: **local** — clock fraction **41.1%**; likely another content-conditioned, recency-biased reader.

L2H0: **local** — positional key-bias coupling remains appreciable; matching score **−0.086** offers no copying evidence.  
L2H1: **copy** — matching score **0.719**, opposite-direction score **−0.502**, and output-map agreement with later copying heads.  
L2H2: **previous** — tentative: opposite-direction matching **0.656**, versus induction matching **−0.479**, plus substantial clock reading suggests retrieving the recent predecessor.  
L2H3: **local** — clock fraction **12.6%**; **78.9%** of its bias-induced key preference lies in that clock subspace.  
L2H4: **local** — clock fraction **10.8%**, with **66.4%** clock alignment of the bias-induced key preference.  
L2H5: **unclear** — weaker positional coupling and matching score **0.133** do not distinguish local from broad retrieval.

L3H0: **copy** — weaker candidate: matching score **0.305**, with output-map cosine **0.354** to L3H5.  
L3H1: **copy** — matching score **0.666** and strong output-map agreement with L3H5.  
L3H2: **unclear** — matching score **−0.016**; neither a clear induction circuit nor an identifiable distance.  
L3H3: **unclear** — comparatively weak QK and strong OV suggest aggregation, but do not establish broad attention.  
L3H4: **unclear** — modest matching score **0.203**, without corresponding copying-output alignment.  
L3H5: **copy** — matching score **0.705**, strongly positive direct matching structure, and coherent copying-output weights.

## Reasoning

The shift-like readers and multiple induction-like circuits strongly favor **as-is text**. Shuffling would largely destroy the sequential regularities these circuits appear organized to exploit.

Corpus identification is much weaker. The substantial character subspace disfavors romaji somewhat. Multiple copying circuits make **Python** my narrow preference: repeated identifiers and code fragments provide particularly useful short-context repetition. English can also produce these structures, however. I have **not** identified keywords, particular characters, or indentation directly.

corpus: python  
order: as is  
probabilities: python=0.55 romaji=0.05 english=0.40; as is=0.995 shuffled=0.005