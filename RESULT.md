# Kết quả thực nghiệm FedMERIT trên UR3 CobotOps (Kaggle) — tự động sinh

File này được sinh tự động bởi `generate_report.py`, không chỉnh tay số liệu.

## 1. Môi trường chạy

### Split random

| Thư viện | Bản đã dùng | Bản chuẩn (repo) | Khớp? |
|---|---|---|---|
| python | 3.12.13 | 3.12.13 | ✅ |
| numpy | 2.0.2 | 2.0.2 | ✅ |
| pandas | 2.3.3 | 2.3.3 | ✅ |
| scikit_learn | 1.6.1 | 1.6.1 | ✅ |
| scipy | 1.16.3 | 1.16.3 | ✅ |
| cryptography | 43.0.3 | 43.0.3 | ✅ |

### Split blocked

| Thư viện | Bản đã dùng | Bản chuẩn (repo) | Khớp? |
|---|---|---|---|
| python | 3.12.13 | 3.12.13 | ✅ |
| numpy | 2.0.2 | 2.0.2 | ✅ |
| pandas | 2.3.3 | 2.3.3 | ✅ |
| scikit_learn | 1.6.1 | 1.6.1 | ✅ |
| scipy | 1.16.3 | 1.16.3 | ✅ |
| cryptography | 43.0.3 | 43.0.3 | ✅ |

## 2. Đối chiếu với lý thuyết paper (Table I)

Các giá trị n hợp lệ theo Table I: [38, 67, 103, 154, 169, 231, 410]
(xem chi tiết khớp/lệch ở mục 5 nếu group_count không nằm trong danh sách này.)

## 3. Kết quả chính — FedMERIT có chặn được model có hại không?

(ref = số liệu có sẵn trong repo, `results/ur3_v4_random` và `results/ur3_v4_blocked`)

| | random: transitions | random: transitions (ref) | random: catalog_harmful | random: catalog_harmful (ref) | random: catalog_escapes | random: catalog_escapes (ref) | random: audit_diag_harmful | random: audit_diag_harmful (ref) | random: audit_diag_escapes | random: audit_diag_escapes (ref) |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 480 | 480 | 19 | 19 | 0 | 0 | 54 | 54 | 0 | 0 |

| | blocked: transitions | blocked: transitions (ref) | blocked: catalog_harmful | blocked: catalog_harmful (ref) | blocked: catalog_escapes | blocked: catalog_escapes (ref) | blocked: audit_diag_harmful | blocked: audit_diag_harmful (ref) | blocked: audit_diag_escapes | blocked: audit_diag_escapes (ref) |
|---|---|---|---|---|---|---|---|---|---|---|
| blocked | 480 | 480 | 38 | 38 | 0 | 0 | 170 | 170 | 0 | 0 |

**Kết luận:** Không phát hiện lệch — catalog_escapes = 0 ở cả 2 split, khớp Table I, khớp môi trường chuẩn.

## 4. Output thô của `validate_ur3_release`

```json
random (của mình): {
  "split": "random",
  "transitions": 480,
  "seeds": 20,
  "catalog_harmful": 19,
  "catalog_escapes": 0,
  "audit_diagnostic_harmful": 54,
  "audit_diagnostic_escapes": 0
}
blocked (của mình): {
  "split": "blocked",
  "transitions": 480,
  "seeds": 20,
  "catalog_harmful": 38,
  "catalog_escapes": 0,
  "audit_diagnostic_harmful": 170,
  "audit_diagnostic_escapes": 0
}
```

## 5. Có gì lệch không?

- [x] Không phát hiện lệch — tất cả số liệu khớp Table I và khớp bản của anh.

## 6. Thư mục kết quả gốc

- random: `/kaggle/working/results/ur3_v4_random_reproduced`
- blocked: `/kaggle/working/results/ur3_v4_blocked_reproduced`
