# Exp 3 · english · Astra (our re-run)

## Observations

These are **weight-based hypotheses, not measured attention patterns**. Missing embeddings particularly limit identification of characters and attention distances.

- **L0H2 is exceptional.** Folding the learned LayerNorm gains into its QK product gives Frobenius norm **4.078**, versus **0.030–0.395** for the other L0 heads. Its Q/K top-20 subspace overlap is **0.786**, but Q/V overlap is only **0.151**. This suggests positional addressing that transports character information.
- **L0H0/H1/H4/H5 share low-dimensional machinery.** Their OV effective ranks are **1.523, 1.894, 1.867, 1.504**. Their dominant OV read directions overlap the inferred positional subspace by **96.5%, 95.4%, 95.1%, 91.1%**. Broad positional aggregation is a plausible interpretation.
- **L1 has a common positional addressing system.** Its two dominant shared key directions overlap the preceding shared write subspace by **93.1% and 92.8%**. All six heads have low-effective-rank QK products and strongly aligned affine key preferences.
- **L3H1 has distinctive induction-compatible composition.** Define \(A\) as the gain-adjusted, centered QK product and \(B\) as the corresponding OV product. The diagnostic
  \[
  S(A,B)=\operatorname{tr}(AB)/\|AB\|_F
  \]
  is **1.173** for L3H1 composed with L0H2, and **1.250** with L2H1. Other L3 heads composed with L0H2 lie between **−0.095 and 0.063**. This is substantial evidence for matching earlier transported context, although not proof of behavioral copying.
- First-MLP clustering exposes numerous distinct, repeated read directions. These support a substantial character-feature dictionary, but **do not establish literal character identities or corpus-specific words**.

## Heads

Here, **positional fraction** means the fraction of the affine key-preference direction lying in the inferred shared two-dimensional positional key subspace. It is not an attention probability.

| Head | Most likely | Evidence |
|---|---|---|
| L0H0 | broad | QK norm 0.055; low-rank OV, predominantly positional read direction. |
| L0H1 | broad | QK norm 0.030; same shared positional-aggregation machinery. |
| L0H2 | previous | Strong, rotation-like positional QK structure; separate character-carrying OV; later induction-compatible composition. Exact one-character offset remains inferred. |
| L0H3 | local | 64.2% of QK energy lies in L0H2’s positional subspace; QK cosine with H2 is 0.240. Tentative. |
| L0H4 | broad | QK norm 0.106; OV effective rank 1.867 and predominantly positional reads. |
| L0H5 | broad | QK effective rank 1.090; strong low-rank positional OV machinery. |
| L1H0 | local | Shared positional scorer; affine positional fraction 93.5%. |
| L1H1 | local | Shared positional scorer; positional fraction 96.3%. |
| L1H2 | local | Shared positional scorer; positional fraction 97.4%. |
| L1H3 | local | Shared positional scorer; positional fraction 96.3%. |
| L1H4 | local | Strongest QK restriction to shared positional key axes: 75.1% of energy. |
| L1H5 | local | Shared positional scorer; positional fraction 96.3%. |
| L2H0 | local | Strong affine key preference, norm 1.699; positional fraction 69.7%. |
| L2H1 | local | Positional fraction 66.0%; also supplies features strongly matched by L3H1. |
| L2H2 | unclear | Positional component remains, but substantial content-dependent structure prevents a reliable range assignment. |
| L2H3 | local | Affine preference norm 1.462; positional fraction 72.0%. |
| L2H4 | local | Affine preference norm 1.577; positional fraction 71.2%. |
| L2H5 | local | Affine preference norm 1.509; positional fraction 71.8%. |
| L3H0 | unclear | High-rank content addressing; little shared positional bias and no positive induction signature. |
| L3H1 | copy | Uniquely strong positive QK–earlier-OV composition, including both character and later context pathways. |
| L3H2 | unclear | Mixed content and positional addressing; locality versus distant selective retrieval unresolved. |
| L3H3 | unclear | Positional bias survives, but does not establish a short attention range. |
| L3H4 | unclear | Similar mixed evidence; neither locality nor induction is established. |
| L3H5 | broad | Weak shared positional bias, comparatively small QK norm 0.835, and large OV norm 45.401. Broad contextual aggregation is tentative. |

## Reasoning

The likely previous-character pathway, extensive positional/local machinery, and later induction-compatible head strongly favor **as-is text**. Within-line shuffling would substantially reduce the usefulness of these ordered-context mechanisms.

**English is my weak corpus preference**, fitting local orthographic processing followed by contextual repetition. Python and romaji can also produce these mechanisms; the diagnostics did not securely identify indentation, particular syllables, or vocabulary. Corpus confidence is therefore much lower than order confidence.

corpus: english  
order: as is  
probabilities: python=0.29 romaji=0.21 english=0.50; as is=0.98 shuffled=0.02