# Metric contract — T11 · Post-label Error Review Prioritizer

> Khai báo bắt buộc — chốt bởi Phạm Hoàng Minh, **trước** khi chạy đánh giá chính thức (repo không có `gt/`).
> Các metric chuẩn được định nghĩa và tính bởi `src/metrics/tracks.py::eval_t11`.

| Mục | Khai báo chi tiết |
|---|---|
| Track | **T11 · Post-label Error Review Prioritizer** |
| Primary metric | **Error Recall@B** (tên đúng như evaluator in ra) |
| Tăng tốt hay giảm tốt? | **higher** (càng cao càng phát hiện được nhiều lỗi với ngân sách B) |
| Đơn vị đánh giá | item = `ann_id` (lỗi trên box có sẵn: class swap / box shift / duplicate) hoặc `img:<image_id>` (ảnh thiếu box) |
| Công thức + numerator / denominator | $\text{Error Recall@B} = \frac{\text{Số lỗi thật trong top-B}}{\text{Tổng số lỗi thật trong pool heldout}}$ (`src/metrics/ranking.py::recall_at_k`) |
| Cách ghép prediction ↔ GT | Khớp chính xác theo `item_id` (không dùng IoU ở bước chấm điểm). IoU chỉ dùng BÊN TRONG PoC để ghép annotation ↔ prediction của detector. |
| Candidate pool cố định | Split `heldout`: 17 ảnh, 60 annotations, 77 items (60 `ann_id` + 17 `img:`) — đếm từ `images.csv` và `annotations.csv`. |
| Review budget B | $B = \max(10, \lceil 10\% \cdot N \rceil) = 10$ với $N=60$. Trần oracle = $\min(1, B / \text{\#lỗi}) = 0.909$ (heldout có 11 lỗi thật). |
| Seed | **42** |
| Threshold policy | Không tune trên heldout. Ngưỡng theo quy ước hình học chung (IoU $\ge 0.5$ cho duplicate; $\ge 0.3$ để ghép sơ bộ) + hiệu chỉnh từ thống kê không nhãn (trung vị/MAD của IoU các cặp sạch; phân vị 10% confidence của prediction đã ghép). Trọng số: `w_dup=0.95`, `w_swap=0.90`, `w_shift=0.85`, `w_missing=0.85`. |
| Eval split | `heldout` (không dùng để tinh chỉnh tham số) |
| Chỉ số lỗi xấu | False alarm (mục đưa vào review nhưng reviewer `approve` do nhãn vốn đúng); wrong-correction rate; lỗi thật lọt khỏi top-B (false negative). |
| Slices báo cáo | (1) Theo `error_type`: class_swap / box_shift / duplicate / missing; (2) dev vs heldout; (3) ablation từng thành phần. |
| Cách báo độ bất định khi N nhỏ | Báo số thô $k/11$ trên heldout kèm phân tích độ bất định; chạy tự kiểm Monte Carlo (30 seeds tự chèn lỗi) báo $\text{mean} \pm \text{std}$. |
| Metric phụ chuẩn | **Precision@B**, **Error-type coverage**, **Minutes/defect** |
| Ngày giờ chốt + người chốt | 2026-11-14 · **Phạm Hoàng Minh** (MSSV: 2A202603064) |

## Khai báo trung thực về giao thức đánh giá

1. **Về dữ liệu heldout:** Trong quá trình phân tích hình học ban đầu, nhóm chỉ khảo sát thống kê mô tả phi nhãn (phân phối IoU, cặp trùng, số detection mồ côi), hoàn toàn không truy cập hay suy diễn nhãn ground truth (repo sinh viên không chứa `gt/`).
2. **Về đánh giá nội bộ:** Để tự kiểm thử khi chưa có nhãn của giảng viên, nhóm sử dụng script `tools/selfcheck_t11.py` tự chèn 4 loại lỗi với seed ngẫu nhiên trên tập sạch. Kết quả tự kiểm mang tính tương đối (lạc quan hơn thực tế) nhưng đảm bảo tính khách quan khi so sánh tương đối giữa Baseline và PoC.
3. **Quy tắc hiển thị:** Nếu baseline bằng 0 trên heldout, báo cáo ghi rõ "Baseline = 0", không phóng đại tỷ lệ phần trăm cải thiện vô nghĩa.
