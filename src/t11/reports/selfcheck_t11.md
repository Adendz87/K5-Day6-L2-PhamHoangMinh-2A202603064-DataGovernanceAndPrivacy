# Tự kiểm T11 (lỗi tự chèn) — số liệu do `tools/selfcheck_t11.py` sinh ra

Cấu hình: 30 seed, tỉ lệ chèn lỗi 15%, B = 15, trần oracle trung bình = 0.685.
**Giới hạn:** lỗi do ta tự chèn trên dữ liệu synthetic ⇒ lạc quan hơn eval thật của giảng viên; dùng để bắt bug và so sánh thành phần, không thay thế điểm chính thức.

## Trước / sau khi sửa hiệu chỉnh (phát hiện nhờ ablation)

Lỗi phát hiện: v0 gắn cờ **cả hai** box của một cặp trùng ở điểm rất cao ⇒ mỗi cặp ngốn 2 slot review nhưng chỉ có 1 lỗi thật, chen lấn top-B (bản "bỏ duplicate" lại tốt hơn bản đầy đủ). Sửa: điểm = P(cặp trùng) × P(box này là bản sao); missing hiệu chỉnh theo phân phối conf của prediction đã ghép (không cần nhãn).

| Phiên bản | Recall@B | Precision@B | Recall theo loại (swap / shift / dup / missing) |
|---|---|---|---|
| v0 (trước sửa) | 0.553 ± 0.042 | 0.807 ± 0.060 | 0.48 / 0.76 / 0.79 / 0.14 |
| v1 (hiện tại) | 0.646 ± 0.037 | 0.942 ± 0.051 | 0.56 / 0.82 / 0.51 / 0.69 |
| random | 0.082 ± 0.055 | 0.120 ± 0.080 | 0.09 / 0.07 / 0.07 / 0.10 |

## Ablation (v1)

| Biến thể | Recall@B | Precision@B | Recall theo loại (swap / shift / dup / missing) |
|---|---|---|---|
| PoC (đầy đủ) | 0.646 ± 0.037 | 0.942 ± 0.051 | 0.56 / 0.82 / 0.51 / 0.69 |
| bỏ duplicate | 0.679 ± 0.019 | 0.991 ± 0.023 | 0.82 / 0.94 / 0.07 / 0.80 |
| bỏ class_swap | 0.621 ± 0.042 | 0.907 ± 0.059 | 0.00 / 0.97 / 0.72 / 0.85 |
| bỏ box_shift | 0.627 ± 0.043 | 0.916 ± 0.059 | 0.96 / 0.00 / 0.75 / 0.86 |
| bỏ missing | 0.638 ± 0.040 | 0.931 ± 0.056 | 0.78 / 0.96 / 0.71 / 0.00 |
| bỏ unmatched | 0.646 ± 0.037 | 0.942 ± 0.051 | 0.56 / 0.82 / 0.51 / 0.69 |
| bỏ stage-2 (ghép theo tâm) | 0.493 ± 0.053 | 0.720 ± 0.076 | 0.67 / 0.06 / 0.56 / 0.75 |

Đọc kết quả: ghép stage-2 (theo tâm) là thành phần đóng góp lớn nhất (bỏ đi: Recall giảm mạnh, shift gần 0); bản "bỏ duplicate" vẫn nhỉnh hơn một chút vì cặp trùng chỉ có 1 lỗi trên 2 slot — đây là đánh đổi có chủ ý (duplicate vẫn được bắt, recall 0,5).
Hiệu chỉnh: điểm duplicate 0,6–1,0 trúng ≈ 72–76% (khớp mức điểm); các loại khác ở 0,5–0,9 trúng ≈ 94% (điểm hơi khiêm tốn — giữ nguyên làm biên an toàn vì dữ liệu thật có nhiễu detector).
