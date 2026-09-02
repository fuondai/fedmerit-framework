# UR3 benchmark contract

## Question measured

The benchmark measures one post-selection transition, not end-to-end optimizer
convergence. For each seed, every candidate generator receives the same model
after 20 benign FedAvg rounds, the same 30 client messages, and the same four
disjoint source partitions. The only changed object is the rule that maps those
messages to the next-model candidate. This design isolates the certificate
boundary: an ungated candidate is the paired counterfactual, and FedMERIT decides
whether that exact candidate becomes the serving model.

The statistical unit is the seed. Method and attack rows within a seed share the
split and checkpoint and therefore are not independent replicates. The raw CSV
retains the seed, all partition identifiers, the candidate and installed
outcomes, the reused-score comparator, the selected catalog leaf, and every
protocol invariant used by the paper.

## Candidate generators

| Label | Implemented contract | Workload-specific choice |
|---|---|---|
| FedAvg | sample-size-weighted mean of complete model deltas | none |
| Coordinate median | coordinate-wise median | none |
| Krum | minimum sum of the nearest `n-f-2` squared distances | the registered bound is `f=6` in clean and attack conditions |
| FLTrust | ReLU cosine trust, root-norm normalization, trust-weighted mean | the trusted update is trained on the balanced score view |
| FedVal | classwise and overall loss-diversion weights; the published `s2` candidate set is selected by validation CE | binary labels and a deterministic equal-class validation view |
| FLShield | cluster representatives, validation impact, accepted-cluster remapping, clipping, and weighted aggregation | K-Means/silhouette instantiate the clustering path for 20-dimensional tabular deltas; honest validators are assumed |
| FoundationFL | synthetic-update augmentation followed by a robust coordinate rule | synthetic ratio `0.5`, interior representative, and coordinate trimmed mean with registered `f=6` |

These labels identify concrete implementations of the cited mechanisms; they do
not claim to reproduce every architecture, training schedule, or dataset choice
from the original papers. `FLShield` is explicitly a tabular instantiation.
`score_aware` is evaluated only for FedAvg, FLTrust, and FoundationFL because the
bounded attack oracle is defined for those three mappings. Blank cells are out
of contract, not zero attacks or zero failures.

## Attacks and configuration

All conditions register a Byzantine upper bound of 20% (`f=6` of 30 clients).
The clean condition injects zero malicious clients but does not reveal this to a
defense by changing its configured bound. Sign flip multiplies the selected
malicious deltas by `-6`; model replacement uses `-12*w-8*delta`; score-aware
search has exactly 64 queries, observes only malicious-client and score groups,
and never observes the sealed catalog, audit groups, or source randomness.

## Evidence boundaries

`population_harm` is evaluated on the registered 76-cycle catalog population;
`declared_harm` is evaluated on the disjoint 20-cycle audit set at the same
`epsilon=0.35`; `operational_harm` is a diagnostic audit event at `0.05` and is
not part of the certificate theorem. The reused-score gate is descriptive only:
because the candidate may depend on the score set, its empirical decision has no
distribution-free post-selection guarantee.

Protocol latency is the local Python/SQLite reference path. It excludes model
and probe transfer, witness networking, and a deployed threshold beacon. Receipt
bytes reported by the paper are a wire-format lower bound with the same
exclusions. Chronological blocking is a temporal-shift sensitivity check, not a
finite-to-deployment bridge.

