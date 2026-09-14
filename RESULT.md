# Retained FedMERIT results on UR3 CobotOps

This report is generated from the retained UR3 release metadata and preserves
the recorded numerical values.

## 1. Execution environment

### Random split

| Library | Recorded version | Repository pin | Match |
|---|---|---|---|
| python | 3.12.13 | 3.12.13 | yes |
| numpy | 2.0.2 | 2.0.2 | yes |
| pandas | 2.3.3 | 2.3.3 | yes |
| scikit_learn | 1.6.1 | 1.6.1 | yes |
| scipy | 1.16.3 | 1.16.3 | yes |
| cryptography | 43.0.3 | 43.0.3 | yes |

### Blocked split

| Library | Recorded version | Repository pin | Match |
|---|---|---|---|
| python | 3.12.13 | 3.12.13 | yes |
| numpy | 2.0.2 | 2.0.2 | yes |
| pandas | 2.3.3 | 2.3.3 | yes |
| scikit_learn | 1.6.1 | 1.6.1 | yes |
| scipy | 1.16.3 | 1.16.3 | yes |
| cryptography | 43.0.3 | 43.0.3 | yes |

## 2. Reference group counts

The registered calculation contract contains the following reference group
counts: [38, 67, 103, 154, 169, 231, 410].

## 3. Retained transition outcomes

The `ref` columns are the values retained in `results/ur3_v4_random` and
`results/ur3_v4_blocked`.

| | random: transitions | random: transitions (ref) | random: catalog_harmful | random: catalog_harmful (ref) | random: catalog_escapes | random: catalog_escapes (ref) | random: audit_diag_harmful | random: audit_diag_harmful (ref) | random: audit_diag_escapes | random: audit_diag_escapes (ref) |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 480 | 480 | 19 | 19 | 0 | 0 | 54 | 54 | 0 | 0 |

| | blocked: transitions | blocked: transitions (ref) | blocked: catalog_harmful | blocked: catalog_harmful (ref) | blocked: catalog_escapes | blocked: catalog_escapes (ref) | blocked: audit_diag_harmful | blocked: audit_diag_harmful (ref) | blocked: audit_diag_escapes | blocked: audit_diag_escapes (ref) |
|---|---|---|---|---|---|---|---|---|---|---|
| blocked | 480 | 480 | 38 | 38 | 0 | 0 | 170 | 170 | 0 | 0 |

The retained and recomputed values agree for both splits; catalog escapes are
zero in each split and the recorded environments match the repository pins.

## 4. `validate_ur3_release` output

```json
random: {
  "split": "random",
  "transitions": 480,
  "seeds": 20,
  "catalog_harmful": 19,
  "catalog_escapes": 0,
  "audit_diagnostic_harmful": 54,
  "audit_diagnostic_escapes": 0
}
blocked: {
  "split": "blocked",
  "transitions": 480,
  "seeds": 20,
  "catalog_harmful": 38,
  "catalog_escapes": 0,
  "audit_diagnostic_harmful": 170,
  "audit_diagnostic_escapes": 0
}
```

## 5. Consistency check

- [x] Retained values agree with the release summaries and the registered
  calculation contract.

## 6. Original result directories

- random: `/kaggle/working/results/ur3_v4_random_reproduced`
- blocked: `/kaggle/working/results/ur3_v4_blocked_reproduced`
