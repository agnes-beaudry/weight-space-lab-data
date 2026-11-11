# Exp 3 · python · Astra (our re-run, 2)

## Observations

**The strongest evidence is for ordered text, with short-range positional heads and later induction-like circuits. Corpus identification is substantially less certain.**

For the numerical evidence below, define the static weight products
\[
A_{\ell h}=D_\ell^\top W_{q,\ell h}^\top W_{k,\ell h}D_\ell/8,\qquad
B_{\ell h}=P W_{o,\ell h}W_{v,\ell h}D_\ell,
\]
where \(P=I-\mathbf1\mathbf1^\top/384\) and \(D_\ell=\operatorname{diag}(\mathrm{ln1\_w.\ell})P\). These are matrix products, **not layer evaluations**.

- **A positional shift pair:** L0H1 and L0H5 have respectively **87.2% and 92.3%** of their raw QK energy outside the leading 27-dimensional shared value-read subspace. In a whitened 20-dimensional subspace, L0H1 aligns with the **square** of L0H5’s polar factor: cosine **0.642**, versus **0.071** with its first power. Removing the token-read proxy preserves this result (**0.649**). This suggests one-step and two-step positional retrieval, although the missing position embeddings prevent establishing absolute lags.
- **A pooling head:** L0H4 has \(\|A\|_F=\mathbf{0.009}\), versus **0.878** and **0.970** for L0H1/H5, but \(\|B\|_F=\mathbf{17.158}\). Its leading OV singular values are **13.877, 9.644, 2.507**. This strongly suggests broad, low-dimensional aggregation.
- **Induction-like connections:** Define \(c(A,B)=\operatorname{tr}(AB)/\|AB\|_*\). Connections from the likely previous-character head L0H5 score **0.212** into L2H1, **0.179** into L3H1, and **0.187** into L3H5. L2H2’s writes have still stronger connections into L3H1 (**0.288**) and L3H5 (**0.390**). These are substantial matching signatures, not proof of copying behavior.
- **Only weak corpus evidence:** `W_proj.3` has substantial singular values through many dimensions: \(s_2=\mathbf{20.293}\), \(s_{25}=\mathbf{10.769}\), \(s_{28}=\mathbf{7.929}\). Together with the early value spectra, this does not suggest an especially restricted character repertoire. That weakly disfavors romaji; spectral rank is not a literal vocabulary count.

## Heads

Labels are tentative. **“Unclear” means that these weight-only measurements do not distinguish the requested behaviors.**

L0H0: **unclear** — low-rank QK; its two leading OV modes dominate, but neither lag nor attention span is established.  
L0H1: **local** — strong square-of-L0H5 positional signature; likely retrieves roughly two characters back.  
L0H2: **unclear** — mixed content/position structure; positive symmetric QK modes do not distinguish self-attention from nearby retrieval.  
L0H3: **unclear** — substantial negative symmetric positional modes, without a reliable lag signature.  
L0H4: **broad** — exceptionally weak QK product, strong low-rank OV product; best pooling candidate.  
L0H5: **previous** — strong position-dominated rotational structure; likely the elementary shift underlying L0H1.

L1H0: **unclear** — 34.2% of key-side QK energy lies in L0H4’s leading write subspace; positional dependence, but unknown span.  
L1H1: **unclear** — corresponding positional-key fraction 62.7%; insufficient evidence for a particular lag.  
L1H2: **unclear** — positional-key fraction 63.1%; no strong predecessor-matching signature.  
L1H3: **unclear** — positional-key fraction 67.8%; clearly uses the aggregation head’s features, but its retrieval range is unresolved.  
L1H4: **unclear** — positional-key fraction 64.5%; positional routing does not establish locality.  
L1H5: **unclear** — positional-key fraction 52.5%; predecessor composition is comparatively weak.

L2H0: **unclear** — distributed QK structure; no distinctive shift or induction connection found.  
L2H1: **copy** — strongest L2 connection to L0H5’s writes: \(c=0.212\), consistent with predecessor matching.  
L2H2: **unclear** — its writes strongly support later matching circuits, but that does not establish where this head itself attends.  
L2H3: **unclear** — strongest positive composition is with the pooling head, not the likely previous-character head.  
L2H4: **unclear** — mixed positional/content structure; no discriminating lag or copying signature.  
L2H5: **unclear** — weaker QK and strong OV products, but insufficient evidence to distinguish pooling from selective retrieval.

L3H0: **unclear** — no strong induction connection; remaining products do not establish attention span.  
L3H1: **copy** — matching connections through L0H5 (\(c=0.179\)) and L2H2 (\(0.288\)).  
L3H2: **unclear** — no clear positional-shift or predecessor-matching signature.  
L3H3: **unclear** — relatively weak QK and strong OV make pooling plausible, but do not establish broad attention.  
L3H4: **unclear** — no sufficiently distinctive static signature among the requested categories.  
L3H5: **copy** — strongest induction-like candidate: \(c=0.390\) through L2H2, plus \(0.187\) through L0H5.

## Reasoning

The paired positional shifts and later predecessor-matching connections make **as-is training much more plausible than line shuffling**. Shuffling preserves useful line-level statistics, but largely destroys the sequential regularities these circuits appear designed to exploit.

I weakly favor **English**, with Python a close alternative. The substantial character-related subspaces weigh somewhat against romaji’s more restricted letter usage. However, I have not recovered labeled characters or a decisive English-versus-Python signature; the corpus choice is a best judgment, not an identification established by the weights.

corpus: english  
order: as is  
probabilities: python=0.35 romaji=0.15 english=0.50; as is=0.98 shuffled=0.02