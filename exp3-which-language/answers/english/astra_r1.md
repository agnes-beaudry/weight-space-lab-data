# Exp 3 · english · Astra (our re-run, 2)

## Observations

These are **weight-only hypotheses**, not measured attention patterns. Missing embeddings prevent identifying characters or verifying attention distances.

Below, \(M\) means the head’s QK matrix product and \(T\) its OV product, with known LayerNorm scales and linear mean-centering folded in—not LayerNorm evaluation.

- **L0H2 is exceptional:** \(\|M\|_F=32.551\), versus **0.236–3.156** for the other L0 heads. Its substantial complex eigenvalues and relatively high-rank value pathway suggest a positional shift mechanism: my strongest **previous-character** candidate.
- **L0H0/H1/H4/H5 share low-dimensional machinery.** Their OV top-five singular directions contain **99.6%, 99.5%, 97.8%, 99.3%** of squared norm. Their leading output directions concentrate on coordinates such as 222, 360 and 78. For example, L0H0’s leading output singular vector has coordinate **222 = −0.9218**. These look more like shared positional/context summaries than separate character-copying channels.
- **L1 strongly reads that shared machinery:** a common four-dimensional early-write subspace accounts for **60.6–69.5%** of its heads’ key-side QK energy. This supports positional organization, although it does not establish particular offsets.
- **L3H1 has a distinctive induction-compatible connection.** For its composition with L0H2,
  \[
  \operatorname{tr}(M_{3,1}T_{0,2})/\|M_{3,1}T_{0,2}\|_F=1.1734.
  \]
  Coordinate-permuted weight controls gave **−0.0358 ± 0.0490**. Its corresponding composition with L2H1 is also strong: **1.2504**. These are structural evidence, not proof of copying behavior.
- I found **no defensible character-labelled or corpus-specific signature**. Corpus identification is considerably less certain than the ordered-context inference.

## Heads

“Local” and “broad” below are tentative where distance cannot be verified.

L0H0: **broad** — weak QK norm 0.437; almost entirely low-rank shared OV summary.  
L0H1: **broad** — weakest QK norm, 0.236; OV top-five energy 99.5%.  
L0H2: **previous** — exceptional, asymmetric multi-mode QK; supplies the strongest early connection to the induction candidate.  
L0H3: **unclear** — QK is 82.7% single-mode; could select a special feature rather than a fixed distance.  
L0H4: **broad** — QK norm 0.849; low-rank OV closely aligned with H0/H1.  
L0H5: **broad** — QK norm 0.620; OV top-five energy 99.3%, sharing early summary directions.

L1H0: **local** — tentative positional reader; shared early-write subspace captures 65.1% of key-side QK energy.  
L1H1: **local** — same evidence, 60.6%; exact offset unidentified.  
L1H2: **local** — same evidence, 64.0%; additionally positive composition with L0H2.  
L1H3: **local** — same evidence, 63.9%; additionally positive composition with L0H2.  
L1H4: **local** — strongest shared-subspace fraction, 69.5%; exact offset unidentified.  
L1H5: **local** — shared-subspace fraction 63.0%; exact offset unidentified.

L2H0: **unclear** — distributed QK; early positional-summary coupling remains, but does not determine range.  
L2H1: **unclear** — contributes strongly to L3H1’s matching circuit; its own attention destination is unresolved.  
L2H2: **unclear** — strongest L2 coupling to early summary directions, 19.7%; insufficient to distinguish local from broad.  
L2H3: **unclear** — distributed, asymmetric QK without a distinctive predecessor-copy signature.  
L2H4: **unclear** — similar distributed structure; no reliable distance identification.  
L2H5: **unclear** — similar distributed structure; no reliable distance identification.

L3H0: **unclear** — positive QK trace suggests matching, but its L0H2 composition is unexceptional, −0.0424.  
L3H1: **copy** — strongest induction-compatible composition, with both L0H2 and L2H1.  
L3H2: **unclear** — mixed content/early-summary connections; neither distance nor induction established.  
L3H3: **unclear** — similar mixed connections; no distinctive induction evidence.  
L3H4: **unclear** — weak positive matching structure, insufficient to identify its destination.  
L3H5: **broad** — tentative: lowest L3 QK norm, 6.671, and weak positional-summary dependence.

## Reasoning

The candidate previous-character pathway, intermediate positional readers, and later induction-compatible circuit favor **unshuffled text**. Within-line shuffling would largely remove the sequential regularities that make this combination useful.

English is my **weak first choice**: the inferred combination fits character-level spelling and recurring textual contexts. Python also fits substantially; these measurements do not establish indentation handling or identify keywords. Nor did they establish romaji-specific syllabic structure. The language probabilities therefore remain deliberately modest.

corpus: english  
order: as is  
probabilities: python=0.38 romaji=0.10 english=0.52; as is=0.96 shuffled=0.04