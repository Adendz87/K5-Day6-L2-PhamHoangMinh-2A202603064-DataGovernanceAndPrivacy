# Metric contract — T11 · Post-label Error Review Prioritizer

> Chốt ngày 2026-11-14 bởi Phạm Hoàng Minh, **trước** khi có bất kỳ kết quả eval chính thức nào (repo học viên không có `gt/`).

| Mục | Khai báo của nhóm |
|---|---|
| Track | T11 |
| Primary metric | **Error Recall@B** (đúng tên evaluator in ra) |
| Tăng tốt hay giảm tốt? | higher |
| Đơn vị đánh giá | item = `ann_id` (lỗi trên box có sẵn: class swap / box shift / duplicate) hoặc `img:<image_id>` (ảnh thiếu box) |
| Công thức + numerator / denominator | lỗi thật nằm trong top-B / tổng lỗi thật trong pool heldout (`src/metrics/ranking.py::recall_at_k`) |
| Cách ghép prediction ↔ GT | khớp chính xác theo `item_id` (không dùng IoU ở bước chấm). IoU chỉ dùng BÊN TRONG PoC để ghép annotation↔prediction của detector |
| Candidate pool cố định | split `heldout`: 17 ảnh, 60 annotation, 77 item (60 `ann_id` + 17 `img:`) — đếm từ `images.csv`/`annotations.csv` |
| Review budget B | B = max(10, ceil(10%·N)) = 10 với N=60; trần oracle = min(1, B/#lỗi) = 0,909 (suy ra từ `docs/tracks/T11.md`, hàm ý heldout có 11 lỗi) |
| Seed | 42 |
| Threshold policy | Không tune trên heldout. Ngưỡng theo quy ước chung (IoU 0,5 cho duplicate; 0,3 để ghép) + hiệu chỉnh từ thống kê KHÔNG cần nhãn (trung vị/MAD của IoU các cặp sạch; phân vị 10% conf của prediction đã ghép). Trọng số loại lỗi chốt trong `configs/lab.yaml`. Một lần sửa hiệu chỉnh duplicate/missing sau khi chạy tự kiểm (xem `reports/selfcheck_t11.md`) — ghi trung thực ở đó. |
| Eval split (không dùng để tune) | `heldout` |
| Chỉ số lỗi xấu | false alarm = mục review mà reviewer `approve` (nhãn vốn đúng); wrong-correction rate; lỗi lọt khỏi top-B |
| Slices báo cáo | (1) theo `error_type`: class_swap / box_shift / duplicate / missing; (2) dev vs heldout |
| Độ bất định khi N nhỏ | heldout chỉ có ~11 lỗi → báo số thô k/11 (mỗi lỗi ≈ 9 điểm %), không chỉ phần trăm; tự kiểm 30 seed báo mean ± std |
| Metric phụ (chuẩn của evaluator) | Precision@B, Error-type coverage |
| Ngày giờ chốt + người chốt | 2026-11-14 · Phạm Hoàng Minh |

## Khai báo trung thực về giao thức

- Trong lúc khám phá cấu trúc dữ liệu (trước khi viết thuật toán), mình có in thống kê mô tả của cả dev lẫn heldout (phân phối IoU, số cặp trùng, prediction mồ côi), **không dùng nhãn nào** (repo không có). Vì vậy heldout không "tinh khiết" tuyệt đối về mặt thị giác dù không có tune theo nó.
- Đánh giá nội bộ dùng **lỗi tự chèn** (`tools/selfcheck_t11.py`): lạc quan hơn eval thật, chỉ dùng để bắt bug và so sánh thành phần.
- Nếu baseline = 0 trên heldout thì ghi "baseline = 0", không tính phần trăm cải thiện.
