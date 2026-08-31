# Hướng dẫn tiếp quản FedMERIT

Anh gửi em mã nguồn tham chiếu đi kèm bài RIVF 2026. Repo này giữ đúng phần
protocol và các kiểm thử nhỏ để em đọc, sửa và tái chạy; dữ liệu, cache và kết
quả sinh ra không đưa vào Git.

## Bắt đầu nhanh

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-deps -e .
fedmerit validate-manifest --config configs/benchmark_protocol.json
PYTHONPATH=. .venv/bin/python -m pytest -q tests
PYTHONPATH=. .venv/bin/python -m fedmerit.conformance
ruff check fedmerit tests scripts
```

`python3 scripts/produce_evidence.py --output output` chỉ dùng khi cần tạo lại
evidence tham chiếu; thư mục `output/` đã bị loại khỏi repo.

## Đọc theo thứ tự này

- `fedmerit/model.py`: các context, candidate, probe, receipt và successor.
- `fedmerit/gate.py`: risk ledger, sealed catalog, beacon và phép tính gate.
- `fedmerit/certificate.py`: quorum, handover và vùng `CheckAppend` nguyên tử.
- `fedmerit/canonical.py`: tuần tự hoá và domain tag của các bản ghi đã ký.
- `tests/test_protocol.py`: các ca race, replay, tamper, handover và rollback.

## Bất biến phải giữ

- Handover là successor trực tiếp: giữ nguyên `twin_id`, tăng `state_version`
  đúng một bước, không được rollback hoặc nhảy phiên.
- Đổi domain/schema/policy khi handover được phép; `model_version` không được
  giảm và phải khớp model đang cài. Model commit chỉ tăng version model, sau đó
  handover mang version đã cài sang context mới.
- Handover và publish dùng cùng vùng tuần tự hoá; `CheckAppend` phải đọc lại
  live head ngay trước khi cài model và receipt.
- Retry đúng payload trả lại receipt cũ; candidate cạnh tranh, receipt cũ sau
  handover và source manifest đã dùng phải bị từ chối.
- Solver số thực chỉ đề xuất. Verifier tính lại clipping, loss, risk interval
  và digest từ biểu diễn nguyên chính xác.

## Khi sửa paper và code

Code là artifact để chứng minh protocol, không phải lý do hạ claim. Mỗi sửa
logic đi kèm một test hồi quy nhỏ; nếu đổi tiền đề hoặc schema thì sửa đồng
thời config, README, test và đoạn mô tả trong bài. Không tự thêm số liệu,
baseline hay citation chưa có nguồn. Khi kiểm ref, tìm tiêu đề trên Google
Scholar, lấy BibTeX từ **Cite**, rồi đối chiếu DOI/trang publisher trước khi
đưa vào bài. Khi vẽ lại hình, giữ flow đúng với code, dùng mũi tên mảnh và lưu
file nguồn cùng PDF xuất bản.

Không chạy full benchmark trên máy Mac. Ghi command, seed, máy chạy và đường
dẫn evidence trong memo bàn giao để người sau có thể tái lập.
