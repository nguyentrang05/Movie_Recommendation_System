# Kết quả chạy mới trên dữ liệu nhóm

Dữ liệu: MovieLens 1M được cung cấp. Split đúng notebook 01: random 80/20, random_state=42. Train 800.167 tương tác, 6.040 user, 3.683 phim. Test 200.042 tương tác; bỏ 26 dòng có phim ngoài train, còn 200.016 dòng. Đánh giá 5.994 user có ít nhất một phim relevant (rating >=4) sau lọc. Không dùng toàn bộ 6.040 user làm mẫu số macro-average.

| Mô hình | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Popularity | 0,151802 | 0,090666 | 0,091778 | 0,176574 |
| User-CF (k=40) | 0,263313 | 0,191606 | 0,179531 | 0,327996 |
| Item-CF (k=10) | 0,239973 | 0,160250 | 0,156488 | 0,294154 |
| Content-based Genres | 0,067718 | 0,043370 | 0,043260 | 0,085417 |

k được chọn từ {10,20,40} bằng NDCG@10 trên validation tách riêng 10% của train. Sau đó refit trên toàn bộ train. Test chỉ dùng ở bước đánh giá cuối. Bảng trên là một split cố định; không phải kiểm định khác biệt có ý nghĩa thống kê hay kết luận tổng quát cho mọi dữ liệu.

## Nhận xét dùng khi báo cáo
Trên split này, User-CF có cả độ chính xác và chất lượng thứ tự tốt nhất trong bốn mô hình đã chạy mới. Một cách giải thích có thể là lịch sử đánh giá của MovieLens đủ để tìm người có sở thích tương tự; cần thực nghiệm thêm mới khẳng định nguyên nhân. Item-CF cũng vượt popularity.

Content-based thấp hơn baseline. Bộ metadata chỉ có số ít thể loại; các phim cùng tổ hợp thể loại có vector giống nhau. Hồ sơ thể loại khó phân biệt sở thích đối với từng phim cụ thể, còn popularity tận dụng tín hiệu cộng đồng mạnh. Đây là giới hạn thông tin đầu vào, không đủ để kết luận thuật toán content-based luôn kém. Không có cốt truyện/diễn viên/đạo diễn để bổ sung trong archive hiện tại.

Điểm CF/content chỉ để xếp hạng, không phải rating dự đoán. Đánh giá là warm-start và random holdout; chưa đo cold-start hoặc tính đúng đắn theo thời gian. User mới có thể sử dụng tìm kiếm tên/thể loại trong demo.

## Về MF/BPR
Bốn notebook gốc vẫn giữ nguyên, kể cả output cũ. Các chỉ số cũ từ 04/thu_nghiem không được đưa vào bảng chạy mới khi chưa có checkpoint được xuất và kiểm tra. Khi xuất checkpoint bằng scripts/export_bpr.py hoặc scripts/export_mf.py, app sẽ tự thêm kết quả đã đánh giá lại trên cùng train/test. Thứ tự phá hòa ở evaluator chung có thể làm chênh nhẹ so với notebook gốc.
