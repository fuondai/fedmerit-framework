# FedMERIT - ghi chú bàn giao bài RIVF 2026

Anh gửi em source này cùng bản thảo `FedMERIT: Store-Attested Dual-Probe
Receipts for Mobile-Twin Model Transitions`. Mục tiêu là làm bài gọn, rõ và
đúng evidence trong giới hạn sáu trang RIVF; không biến smoke test thành
benchmark và không sửa claim chỉ để né sửa code.

## Việc cần làm khi tiếp quản

1. Đọc toàn bộ `paper.tex`, `references.bib` và repo này. Lập bảng ngắn
   `claim/RQ -> evidence -> vị trí trong bài` trước khi viết lại.
2. Giữ mạch: adaptive selection -> sealed one-use probe -> future beacon ->
   paired bounded-loss replay -> quorum receipt -> atomic live-head append.
   Contribution dùng chủ thể rõ ràng: `We formulate`, `We design`, `We prove`,
   `We implement`, `We evaluate`.
3. Chỉ giữ bảng/hình/kết quả trả lời RQ. Hình kiến trúc phải khớp
   `figures/architecture_ieee.tex` và code; mũi tên mảnh, chữ không chồng nhau,
   không thêm hộp trang trí.
4. Baseline phải dẫn bài gốc đề xuất phương pháp. Với mỗi ref, tìm nguyên tiêu
   đề trên Google Scholar, bấm **Cite -> BibTeX**, sau đó xác minh tác giả,
   venue, năm, trang và DOI ở publisher/DOI chính thức. Ref không xác minh được
   thì báo anh, không đoán.
5. Khi có lần chạy mới, lưu raw evidence và script/chart ngoài repo; caption
   phải nói rõ đó là proof, conformance hay phép đo thực nghiệm.

## Kiểm tra tối thiểu trước khi gửi anh

```bash
fedmerit validate-manifest --config configs/benchmark_protocol.json
PYTHONPATH=. python3 -m pytest -q tests
ruff check fedmerit tests scripts
```

Nếu thay đổi giao diện protocol, cập nhật test và hướng dẫn tiếp quản trong
cùng commit. Không commit dataset, checkpoint, cache, log hoặc đường dẫn máy
cá nhân.
