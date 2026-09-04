# FedMERIT reference implementation

This repository implements the FedMERIT lineage-bounded certificate protocol:
canonical serialization, a finite risk ledger, sealed probe selection, Ed25519
quorum issuance, one-use release, receipt verification, and an atomic
`CheckAppend` boundary backed by SQLite.

Catalog entries expose only opaque salted SHA-256 identifiers and payload
commitments. The protocol's statistical-to-cryptographic reduction registers
SHA-256 as a random oracle for transcript IND-hiding; preimage resistance alone
is not treated as a hiding argument. The selected `ProbeRelease` reveals the
256-bit salt and raw opening only to authorized replayers; the public release
never carries collection-window, source-handle, row, or salt metadata.

## Environment

- CPython 3.10 through 3.14
- Runtime dependencies are pinned in `requirements.lock`; development pins
  (pytest and ruff) are in `requirements-dev.txt`.

`requirements-experiments.txt` is a separate archival environment for the
retained UR3 rows. Its `cryptography==43.0.3` pin matches the recorded Kaggle
metadata; it is not a replacement for the `cryptography==46.0.7` runtime lock.

Install in an isolated environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps .
```

## Reviewer checks

Validate the registered calculation contract and run the end-to-end protocol
conformance path:

```bash
fedmerit validate-manifest --config configs/benchmark_protocol.json
python3 -m fedmerit.conformance
```

The conformance command exits nonzero if any catalog, beacon, release, replay,
handover, quorum, or receipt check fails.

The explicit adversary interfaces, persistent-state boundary, beacon experiment,
and winning event are recorded in `docs/SECURITY_GAME.md`.

Run the deterministic regression suite from the repository root:

```bash
python3 -m pytest -q
```

The suite covers the wire encoder, replay rules, handover fencing, risk-budget
spending, exclusive successor-round reservation, and crash-atomic installation.

## Evidence producer

```bash
python3 -m scripts.produce_evidence --output output
```

The producer writes 46 exact calculation and conformance records, aggregate
metrics, a known-answer receipt, and a SHA-256 manifest. It checks the generated
receipt against the sole checked-in known-answer vector,
`artifacts/reference_receipt.json`. Generated conformance directories are not
source artifacts and must not be treated as canonical evidence. The checked-in
exceptions are the two validated UR3 v4 releases and the controlled
adaptive-reuse challenge used by the manuscript.

## Paper chart

Install the plotting dependencies, produce the deterministic evidence, and
render the protocol-scaling chart:

```bash
python3 -m pip install -r requirements-figures.txt
python3 -m scripts.produce_evidence --output output
python3 figures/plot_protocol_scaling.py \
  --evidence output/metrics.json
```

The renderer checks the three plotted risk rows with the implementation's exact
`required_groups` routine, checks every certificate size against the wire-format
formula, and then cross-checks both slices against `metrics.json`. It writes the
paper-ready vector file to `figures/fig_protocol_scaling.pdf`. Use
`--no-evidence-check` only to reproduce the checked-in CSV without a generated
evidence directory; the implementation-level checks still run.

## Scope

Public receipt verification checks state, catalog, beacon, risk, and quorum
bindings without revealing raw groups. Authorized verification additionally
opens the selected payload and replays the paired gate. In the reference
implementation, `AuditRegistry` is also the authoritative serving store:
`verify_and_append` compare-and-swaps the exact model bytes, model version,
context head, receipt head, and receipt row in one SQLite transaction. A forced
failure after the serving-row update rolls the entire transition back. An
external serving system must implement the same linearizable transaction
contract; an unauthenticated caller-supplied read-back is not accepted. The
registry requires the complete genesis artifact, stores its canonical bytes at
initialization, and applies the same exact-byte comparison to the first commit.
`VerificationTrust.authority_certificate_hash` binds the roster epoch, fault
threshold, witness keys, probe-store key, and frame-authority key to the live
context; changing any field requires an explicitly certified successor context.
The frame-authority signature binds the exact proposal/score source partitions;
the store rejects overlap between those manifests and every released commit
group. Catalog completeness, source representativeness, authority validity, and
authority/proposer non-collusion remain deployment conditions. Beacon validity
additionally requires an unpredictable, unbiasable threshold source and a
complete finality watcher; the audit registry owns the authoritative monotonic
head and successor reservations shared by every local risk-ledger replica. A
handover is an immediate successor
of the live context: it preserves the twin identity, increments the state
version by one, and may change domain, schema, authority, or evaluation policy
while retaining the installed model version. The registry rejects rollback,
skipped epochs, and successors whose model version differs from the installed
artifact.

`AuditRegistry.provision_lineage_risk_budget()` is mandatory and freezes one
envelope for the invariant twin identity before any context schedule is
registered. Every later context schedule is charged against the same exact
binary64 budget, so a domain handover cannot reset the available risk; schedule
registration fails closed when the envelope is absent. Provisioning also freezes
cumulative caps on certificate attempts, distinct verification keys, and context
handovers. Existing roots are counted before acceptance; later key rotation or
handover fails before exceeding its cap.

## Reproducible candidate-transition benchmark

`scripts/run_ur3_benchmark.py` is an optional experiment driver for the UCI
UR3 CobotOps workbook (DOI `10.24432/C5J891`). It treats operation cycles as
non-IID groups and keeps proposal, score, sealed-catalog, and audit groups
disjoint. The fixed split uses 110 proposal, 30 score, 80 commit-pool, and 20
held-out audit cycles. Two complete 38-cycle leaves consume 76 commit-pool
cycles; four cycles remain outside the released catalog instead of creating an
undersized leaf. `--split blocked` repeats the same protocol with chronological
rather than random partitioning. Every method receives the same
20-round benign FedAvg checkpoint and produces one candidate transition. This
is a controlled candidate-generator comparison, not seven end-to-end training
runs. The exact adaptations, attack visibility, and metric semantics are listed
in `docs/BENCHMARK_CONTRACT.md`.

Each candidate first receives its unguarded audit score and then runs through an
isolated end-to-end FedMERIT instance: signed frame, finite risk schedule,
durable beacon fixation and successor, one-use probe release, exact Decimal80
replay by a 2f+1 quorum, and atomic `verify_and_append`. One of four witnesses
is deliberately unavailable in every trial. The driver reads the decision and
paired statistic from the issued receipt, then verifies the installed model
hash, version, and bytes from the authoritative serving store. One clean FedAvg
trial per seed also races two idempotent append retries. Raw records include
split identifiers, primitive harm/escape flags, uncertainty inputs, protocol
checks, and timings; `metadata.json` records the fixed allocation and attack
visibility contract.

Run the retained protocol with:

```bash
python3 -m scripts.run_ur3_benchmark \
  --dataset /path/to/UR3_CobotOps.xlsx \
  --output results/ur3_v4_random_reproduced \
  --split random
```

The workload requires the optional dependencies in
`requirements-experiments.txt` and should be run on a remote compute/CI worker
rather than during a local source checkout. The checked-in manuscript reports
only results whose raw CSV and metadata are retained with the release. To run
the chronological sensitivity, change only `--split blocked` and write to a
separate output directory; never mix split modes in one CSV.

The retained v4 random and blocked releases use CPython 3.12.13 with the exact
top-level package versions in `requirements-experiments.txt`. Kaggle exposed
four logical x86-64 CPUs and a Tesla P100; this NumPy/scikit-learn/SQLite path is
CPU-bound and does not claim GPU acceleration. Recorded workload times are
2799.3 s (random) and 2955.1 s (blocked).

Validate a retained release without rerunning the workload:

```bash
python3 -m scripts.validate_ur3_release \
  --release results/ur3_v4_random \
  --split random
```

The validator checks transition coverage, the registered-versus-injected fault
contract, threshold and installed-state identities, selected-leaf membership,
cycle-level partition disjointness, protocol invariants, environment metadata,
and an exact recomputation of `summary.csv` from the primitive rows. The summary
contains descriptive means, minima, maxima, counts, and observed quantiles. It
does not treat method/attack rows sharing a seed as independent replicates. A
full loss-level replay still requires the cited UR3 workbook and benchmark
driver.

Reproduce the controlled adaptive-reuse separation with:

```bash
python3 -m scripts.run_adaptive_reuse_challenge \
  --output results/adaptive_reuse_challenge
```

Each candidate sees and memorizes only its score identifiers and labels. The
trusted harness fixes a disjoint catalog before the candidate and draws the
fresh probe after fixation with a source-only random stream. The candidate API
receives score rows and a separate prediction key, never source-partition or
probe-RNG capability; the retained `trial_id` is a harness row label, not
candidate input. The 100 deterministic trials are a controlled construction
that isolates evidence reuse; they are not an FL workload or an empirical
deployment-rate estimate.

Regenerate the paper chart directly from seed-level rows with:

```bash
python3 figures/plot_ur3_benchmark.py \
  --raw results/ur3_v4_random/raw_runs.csv \
  --challenge results/adaptive_reuse_challenge/records.csv \
  --output figures/fig_ur3_benchmark.pdf \
  --split random
```

The plotter derives safety fractions, acceptance counts, and observed latency
quantiles from primitive rows. It reports no binomial confidence interval for
the dependent method/attack rows and writes editable PDF/SVG plus 600 dpi
PNG/TIFF outputs.

The `AuditRegistry` also records each accepted protocol transition in an
append-only, hash-linked `protocol_events` journal. Update and delete triggers
fail closed; `protocol_event_chain_valid()` is intended for recovery checks.
