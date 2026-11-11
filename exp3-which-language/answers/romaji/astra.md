# Exp 3 · romaji · Astra (our re-run)

## Observations

These are **weight-based hypotheses**, not measured attention patterns.

I used \(A=(W_qDP)^T(W_kDP)/8\) and \(B=W_oW_vDP\), where \(D=\operatorname{diag}(\text{ln1_w})\) and \(P\) removes the constant direction. These are matrix products, not forward passes.

- **L0H3 looks like positional transport.** Its unadjusted QK stable rank is **14.93**, versus **1.28–2.62** for the other first-layer heads. Its QK eigenvalues include **\(0.724\pm0.691i\)** and **\(0.446\pm0.842i\)**: substantial rotational/asymmetric structure, consistent with an offset rather than simple same-character matching.
- **L3H3 has a particularly strong induction signature through L0H3.** In a fixed, weight-derived 22-dimensional early-value read subspace, the key-composition product has trace/nuclear-norm **0.782**. Corresponding compositions through the other first-layer heads range from **−0.168 to 0.072**. Signed-permutation controls gave mean approximately **0**, standard deviation **0.064**, maximum **0.307**. This strongly supports a specific matching circuit, though it does not establish its behavioral effect.
- **L0H5 looks like a diffuse summary channel.** Its adjusted QK Frobenius norm is only **0.034**, while its adjusted OV leading singular values are **16.278, 8.778, 2.833**.
- **Many later heads read that shared channel.** Its leading three output directions capture **79.8–86.1%** of the bias-induced key-vector energy in layer 1. This establishes shared routing machinery—not its attention distance.
- First-layer heads **H1/H2/H4** are closely related: H1–H2 adjusted OV cosine **0.925**, H1–H4 **0.869**. A shared structural or boundary-related role is plausible, but no character identity is established.

## Heads

Below, \(r\) denotes the fraction of the bias-induced key-vector energy in L0H5’s leading three output directions.

L0H0: **local** — tentative; 41.6% of QK energy lies within the positional-candidate subspace; exact offset unresolved.  
L0H1: **unclear** — adjusted QK stable rank 1.12; strongly shares H2’s value/output circuit.  
L0H2: **unclear** — adjusted QK stable rank 1.10; likely a related structural selector, not an identifiable lag.  
L0H3: **previous** — strongest candidate: asymmetric positional structure and the specific downstream induction composition.  
L0H4: **unclear** — adjusted QK stable rank 1.24; resembles H1/H2, but attention span is unestablished.  
L0H5: **broad** — tentative; exceptionally weak QK product paired with a strong, low-dimensional OV summary.

L1H0: **unclear** — \(r=0.853\); strong shared key selection, insufficient evidence for a particular distance.  
L1H1: **unclear** — \(r=0.853\); position/structure-channel dependence does not distinguish local from broad.  
L1H2: **unclear** — \(r=0.851\); no strong previous-head induction signature.  
L1H3: **unclear** — \(r=0.854\); shared routing dominates the bias-induced key preference.  
L1H4: **unclear** — \(r=0.798\); similar machinery, but neither lag nor spread is recoverable here.  
L1H5: **unclear** — \(r=0.861\); particularly strong shared-channel dependence, without a demonstrated span.

L2H0: **unclear** — \(r=0.832\); substantial shared-channel selection, weak induction composition.  
L2H1: **unclear** — \(r=0.824\); higher-rank QK structure, but no reliable distance assignment.  
L2H2: **unclear** — \(r=0.787\); mixed routing features rather than a distinctive copy signature.  
L2H3: **unclear** — \(r=0.615\); less dominated by the shared channel, still behaviorally ambiguous.  
L2H4: **unclear** — \(r=0.830\); strong shared-channel dependence does not establish locality.  
L2H5: **unclear** — \(r=0.856\); no convincing previous-character or induction-specific composition.

L3H0: **unclear** — \(r=0.692\); L0H3 key-composition score is negative, −0.175.  
L3H1: **unclear** — \(r=0.475\); neither a strong positional-offset nor induction signature.  
L3H2: **unclear** — \(r=0.625\); L0H3 key-composition score only 0.035.  
L3H3: **copy** — strongest classification: specific L0H3 key-composition score 0.782, versus query-composition −0.398.  
L3H4: **broad** — tentative; relatively weak adjusted QK norm 0.622, little shared-channel dependence, strong OV singular value 29.438 unadjusted.  
L3H5: **unclear** — \(r=0.721\); L0H3 key-composition score −0.036, not an induction signature.

## Reasoning

The strongest evidence favors **ordered text**: an early offset-like transport circuit feeding a specific late induction candidate. Shuffling within lines substantially reduces the usefulness of retrieving continuations of earlier character sequences.

Corpus identification is much weaker. I modestly favor **Python** because the combination of structural/positional specialization and continuation retrieval fits structured lines and repeated identifiers. English remains a substantial alternative; these weights do **not** identify indentation, keywords, or particular letters directly. The spectral analyses also did not provide a compelling restricted-alphabet signature for romaji.

Thus my best account is **local character transport, shared contextual summaries/routing, and late continuation matching**, with most individual attention spans unresolved.

corpus: **python**  
order: **as is**  
probabilities: python=0.55 romaji=0.08 english=0.37; as is=0.96 shuffled=0.04