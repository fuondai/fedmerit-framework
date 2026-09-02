# Installed-transition security game

This document fixes the operational game implemented by the reference system.
It is a safety game; progress is conditional on an available beacon, store, and
`2f+1` witnesses.

## State

For one immutable `TwinID`, the authoritative state is

```
(context, serving_model_bytes, model_version, receipt_head,
 risk_ledger, retired_probe_set, beacon_head, successor_reservations)
```

The registry and each honest witness persist their own non-equivocation state.
An adversary may clone or roll back proposer/client processes, retry messages,
delay delivery, crash between API calls, and request valid handovers. It may not
roll back the authoritative registry, an honest witness database, or the final
beacon log. Losing those persistence assumptions is a different threat model.

## Adversary interfaces

The adversary controls all clients and up to `f` of `3f+1` witnesses. It chooses
proposals adaptively from public history and the score partition, schedules all
messages, invokes fixation, aborts after fixation, requests releases, replays old
objects, and interleaves handovers with receipt issuance or installation. Query
caps cover at most `T` fixation attempts, `M` leaves per attempt, `Q_h` random-
oracle queries per leaf, `K` verification keys, `Q_s` signing queries per key,
`L` handovers, and `Q_b` beacon queries per successor.

The source sampler and complete finality watcher are ideal-contract parties. The
source attests a complete eligible population, uniform private permutation,
disjoint source partitions, and one-use retirement. Commitments make later
substitution and reuse detectable; they do not prove that a malicious source
created an honest or representative population. The theorem therefore excludes
source/proposer collusion and watcher rollback. Deployment representativeness is
covered only by a separately registered transport bridge.

## Beacon experiment

`Adv_beacon(T,Q_b)` is the maximum real-versus-ideal distinguishing advantage of
an adaptive adversary given at most `Q_b` queries for each of at most `T`
authenticated successor challenges. The real oracle returns the unique finalized
successor of the deployed threshold beacon; the ideal oracle returns an
independent uniform 256-bit successor after fixation. Prediction before fixation,
selective shifting, accepted equivocation, or rollback can only increase this
distinguishing advantage. The executable reference bound instantiates it as an
explicit assumption; the local deterministic beacon is conformance machinery,
not evidence that a deployed threshold beacon achieves the assumption.

## Winning event

The adversary wins if `CheckAppend` commits a candidate whose registered finite-
population or independent-draw loss increase is at least `epsilon`, or if two
incompatible certified cores install from one live attempt scope. The transition
linearizes only if context, exact before-model bytes/version, receipt head,
authority, policy, and successor reservation still equal the signed values.
Commit updates the serving bytes, model version, context model successor, and
receipt head in one transaction. Reject updates only audit state. Handover keeps
`TwinID`, increments state version exactly once, and carries the installed model
version; it cannot reset the risk envelope.

## Reduction boundary

The reduction replaces salted unopened leaves, aborts on commitment collision or
second opening, aborts on accepted frame/witness/store/handover forgery, and then
replaces the finalized successor by the ideal beacon value. In the final game,
the selected probe is independent of the fixed candidate, so the registered
finite-sample bound applies conditionally to every attempt. One-use spending and
the immutable lineage envelope compose these conditional bounds. Quorum
intersection and the live-state compare-and-swap bind the unique replay result to
the installed bytes.

