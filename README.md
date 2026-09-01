# FedMERIT reference implementation

This repository implements the FedMERIT lineage-bounded certificate protocol:
canonical serialization, a finite risk ledger, sealed probe selection, Ed25519
quorum issuance, one-use release, receipt verification, and an atomic
`CheckAppend` boundary backed by SQLite.

Catalog entries expose only opaque salted SHA-256 identifiers and payload
commitments. The selected `ProbeRelease` reveals the 256-bit salt and raw opening
only to authorized replayers; the public release never carries collection-window,
source-handle, row, or salt metadata.

## Environment

- CPython 3.10 through 3.14
- Runtime dependencies are pinned in `requirements.lock`; development pins
  (pytest and ruff) are in `requirements-dev.txt`.

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

Run the deterministic regression suite from the repository root:

```bash
python3 -m pytest -q
```

The suite covers the wire encoder, replay rules, handover fencing, risk-budget
spending, and the installed-artifact read-back boundary.

## Evidence producer

```bash
python3 scripts/produce_evidence.py --output output
```

The producer writes 41 exact calculation and conformance records, aggregate
metrics, a known-answer receipt, and a SHA-256 manifest. It rejects a receipt
that differs from `artifacts/reference_receipt.json`. Generated result
directories are not part of the source distribution.

## Paper chart

Install the plotting dependencies, produce the deterministic evidence, and
render the protocol-scaling chart:

```bash
python3 -m pip install -r requirements-figures.txt
python3 scripts/produce_evidence.py --output results_devready
python3 figures/plot_protocol_scaling.py \
  --evidence results_devready/metrics.json
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
opens the selected payload and replays the paired gate. `verify_and_append`
also requires the deployment adapter to pass the model artifact read back from
its serving store; the registry compares its canonical artifact hash with the
receipt decision (after-model for a commit, before-model for a rejection)
before advancing the authenticated head. Catalog completeness, source
representativeness, authority validity, and authority/proposer
non-collusion are deployment conditions. A handover is an immediate successor
of the live context: it preserves the twin identity, increments the state
version by one, and may change domain, schema, authority, or evaluation policy
while retaining the installed model version. The registry rejects rollback,
skipped epochs, and successors whose model version differs from the installed
artifact.

For deployments that need a probability bound across handovers,
`AuditRegistry.provision_lineage_risk_budget()` freezes one envelope for the
invariant twin identity before any context schedule is registered. Every later
context schedule is charged against the same exact binary64 budget, so a domain
handover cannot reset the available risk. Without this explicit envelope, the
implementation intentionally makes only the per-context schedule claim.
