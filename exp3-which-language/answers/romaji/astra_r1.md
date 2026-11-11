# Exp 3 · romaji · Astra (our re-run, 2)

## Observations

**The strongest evidence concerns context use, not corpus identity.** These are weight-based hypotheses; no attention patterns were evaluated.

Define \(A\) as each head’s QK product, incorporating the learned LayerNorm scales and mean-centering, and \(B\) as its similarly adjusted OV product.

- **L0H3 is a distinctive positional-routing candidate.** Its raw `W_q.0[192:256]` and `W_k.0[192:256]` have Frobenius norms **14.141 and 14.661**. Its adjusted QK product has substantial complex eigenpairs, including **0.1366 ± 0.1327i**, unlike the predominantly low-rank other first-layer heads. This suggests an offset-selecting mechanism, plausibly previous-character attention.
- **L3H3 has a strong, directional connection to L0H3 consistent with induction.** On a weight-derived candidate content subspace, its key-composition score with L0H3 is **2.563**, versus **−0.971** for query composition. Randomizing the pairing of L0H3’s learned OV dimensions produced scores with mean **−0.018**, standard deviation **0.238**, and none exceeding the observed score in 1,000 trials. This establishes a specialized connection—not, by itself, its behavioral interpretation.
- **L0H5 looks aggregation-oriented:** adjusted QK norm **0.034**, versus **0.698** for L0H3, but adjusted OV norm **18.804**, versus **3.533**. Weak relative selectivity and strong writing suggest broad context aggregation.
- **Several heads share a low-dimensional control channel.** L0H1 and L0H2’s OV products have cosine similarity **0.925**. Their outputs emphasize coordinates 21, 39, 58 and 99. Related weights include `ln1_b.1[39] = −1.5676`, `[58] = 1.3651`, and `[99] = −1.3355`. These coordinates have no known character or position labels.
- **There is no decisive corpus fingerprint.** Neither the singular spectra nor the exploratory MLP-weight clustering securely identifies letters, Japanese syllable structure, or Python indentation.

## Heads

Here, \(a=\|A\|_F\). “Composition score” refers to the content-subspace key-composition diagnostic above. **“Unclear” does not mean inactive.**

| Head | Most likely | Evidence |
|---|---|---|
| L0H0 | local | Shares substantial QK structure with L0H3’s routing subspace; \(a=0.293\). Tentative locality, not an identified lag. |
| L0H1 | unclear | Rank-one component carries 89.2% of QK energy; shared control-channel output does not establish attention range. |
| L0H2 | unclear | 90.9% rank-one QK energy; OV closely resembles L0H1, but no identifiable offset. |
| L0H3 | previous | Distinctive rotation-like QK structure and strong downstream induction-compatible connection. |
| L0H4 | unclear | 80.9% rank-one QK energy; another shared-control writer, with unresolved range. |
| L0H5 | broad | Exceptionally small QK norm, 0.034, paired with large OV norm, 18.804; aggregation is the best hypothesis. |
| L1H0 | unclear | Strong shared-control key dependence; composition score 0.548 is insufficient to establish induction or distance. |
| L1H1 | unclear | Shared control structure; composition score 0.366, without clear induction orientation. |
| L1H2 | unclear | Approximately 49% of QK energy involves the selected control coordinates on the key side; their positional meaning is unknown. |
| L1H3 | unclear | Strong control-key dependence; negligible direct induction signature, score 0.022. |
| L1H4 | unclear | Control-dependent routing; negative composition score −0.264 does not identify an alternative range. |
| L1H5 | unclear | Resembles other L1 control-dependent heads; composition score 0.443 remains inconclusive. |
| L2H0 | unclear | Distributed QK structure, \(a=1.631\); direct composition score only 0.076. |
| L2H1 | unclear | High-rank QK structure; composition score −0.045 gives no direct induction evidence. |
| L2H2 | unclear | Distributed routing weights; composition score 0.174 does not establish a context distance. |
| L2H3 | unclear | No distinctive offset or direct induction signature; composition score −0.112. |
| L2H4 | unclear | Substantial QK strength, \(a=2.215\), but composition evidence depends on the chosen subspace. |
| L2H5 | unclear | High-rank QK structure; weak direct composition score 0.135. |
| L3H0 | unclear | No positive L0H3 key-composition signature; score −0.464. Other contextual mechanisms remain possible. |
| L3H1 | unclear | Relatively weaker QK and stronger OV, but insufficient evidence to distinguish broad from selective attention. |
| L3H2 | unclear | Direct composition score 0.022; neither offset nor induction is established. |
| L3H3 | copy | Strong positive, asymmetric L0H3 key composition, robust across several weight-derived subspaces. |
| L3H4 | broad | Lowest final-layer QK norm, 0.622, and largest OV norm, 49.232; broad aggregation is tentative. |
| L3H5 | unclear | Distributed QK structure and no positive direct induction signature; score −0.080. |

## Reasoning

The likely **previous-character → induction** connection favors **as-is training**: it provides a mechanism for using ordered context to repeat earlier continuations. Together with the aggregation candidates, it suggests a mixture of short-range character processing and longer-range reuse.

**Corpus identification is much weaker.** All three unshuffled corpora reward those mechanisms. I give **English a slight, low-confidence lead**, with romaji close behind; the weights examined do not justify claiming that identifiable English words, Japanese syllables, or Python constructs were recovered. The corpus probabilities below are subjective judgments, not calibrated statistical estimates.

corpus: english  
order: as is  
probabilities: python=0.22 romaji=0.36 english=0.42; as is=0.98 shuffled=0.02