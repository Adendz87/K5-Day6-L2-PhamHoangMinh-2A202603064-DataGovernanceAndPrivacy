"""T11 · Post-label Error Review Prioritizer — thuật toán PoC của nhóm.

geometry  : IoU / khoảng cách tâm (box dạng x, y, w, h với (x, y) là góc trái-trên)
matching  : ghép annotation <-> prediction 2 giai đoạn (IoU, rồi khoảng cách tâm)
scoring   : suspicion score theo 4 loại lỗi (class swap, box shift, duplicate, missing)
review_io : áp dụng quyết định của reviewer -> changed_labels.csv + audit log
"""
