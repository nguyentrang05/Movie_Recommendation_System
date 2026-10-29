# Phạm vi kiểm tra

- Giữ nguyên byte của các notebook 01, 02, 04 và thu_nghiem so với ZIP mới người dùng gửi.
- Đã tái tạo split train/test đúng lệnh của 01, không dùng split thời gian hay 5-core từ bản cũ.
- Đã chạy toàn bộ code cells của 03 và 05 trên full dữ liệu, bao gồm 6 cấu hình CF validation, refit và full test. Output thực được lưu trong notebook và reports/tables. CLI thực thi cell tuần tự trong Python, không cần máy chủ notebook.
- 6 unit tests đã qua: kNN sparse so với tính dense trực tiếp, loại self-neighbor, điểm CF đúng công thức, loại seen, cold user, metric biết trước, lọc unknown heldout, chống overlap, TF-IDF giữ Sci-Fi, checkpoint mapping/fingerprint và input trùng.
- Streamlit AppTest với dữ liệu thật: bốn phương pháp sẵn có, tìm Toy Story, phim tương tự, bảng thực nghiệm, hướng dẫn khi thiếu checkpoint BPR/MF. Đây là kiểm tra chức năng, chưa có kiểm tra ảnh chụp bằng trình duyệt.
- Script xuất BPR/MF được thử bằng chính code notebook gốc trên fixture nhỏ 1 epoch, lưu/nạp .npz và đối chiếu điểm theo công thức gốc. Checkpoint fixture không đóng gói và không dùng cho kết quả nghiên cứu.
- Chưa chạy lại toàn bộ 04/thu_nghiem. Muốn demo hai mô hình đó cần chạy/export từ kernel của người dùng. Không coi output cũ là lần chạy mới.
