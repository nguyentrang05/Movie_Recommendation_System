# Movie Recommendation System — phần 03, 05 và demo

Bổ sung theo source nhóm ngày 29/10/2026. Giữ nguyên byte của bốn notebook gốc:
01_EDA_proprecessing.ipynb, 02_baseline_model.ipynb, 04_Matrix_Factorization.ipynb, thu_nghiem.ipynb.

## 1. Mở demo ngay
Giải nén, mở terminal tại thư mục có app.py. Dùng Python 3.10+ (khuyến nghị 3.11).

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Đã kèm CSV từ đúng split 80/20 của notebook 01, dữ liệu gốc do người dùng cung cấp và kết quả chạy mới 03/05. Không cần chạy lại 01/02 để mở demo. Bốn phương pháp Popularity/User-CF/Item-CF/Content-based dùng ngay. Giao diện chọn k theo models/cf_config.json từ validation, không tự chọn bằng test. Mô hình được fit và cache khi mở lần đầu; đổi user sau đó nhanh hơn.

Trong giao diện có: gợi ý theo UserID gốc, lịch sử đánh giá, tìm tên/thể loại, phim tương tự, bảng/biểu đồ kết quả và tải CSV. Không có poster do dataset không cung cấp. Dữ liệu không có cốt truyện.

## 2. BPR-MF và Bias MF: chỉ cần xuất khi muốn demo đủ các mô hình nhóm
**04 và thu_nghiem hiện không lưu trọng số**, nên output metric cũ không đủ để tái tạo gợi ý. Bản này không kèm checkpoint giả và không tự huấn luyện lại phần của bạn khác.

### BPR-MF (notebook 04)
Mở notebooks/04_Matrix_Factorization.ipynb, Run All. Nếu kernel vẫn còn mô hình từ lần chạy xong thì không cần huấn luyện lại. Thêm một cell cuối:

```python
%run -i ../scripts/export_bpr.py
```

### Bias MF (notebook thu_nghiem)
Mở notebooks/thu_nghiem.ipynb, Run All trong kernel riêng. Thêm một cell cuối:

```python
%run -i ../scripts/export_mf.py
```

Chạy trong **cùng kernel chứa mô hình của notebook tương ứng**; không chạy các lệnh này bằng terminal Python thường. Kernel cần có thư viện từ requirements. Mặc định notebook làm việc từ thư mục notebooks/. Nếu kernel đang ở thư mục gốc, chuyển vào notebooks trước khi chạy notebook.

Hai script tạo models/bpr_model.npz, models/mf_model.npz và bảng kết quả kèm metadata trong reports/tables. Tải lại app để sử dụng. Checkpoint lưu cả mapping ID và fingerprint train; app từ chối file sai dữ liệu. Model được đánh giá lại bằng evaluator chung, phá hòa có thể hơi khác notebook gốc. Không trộn UserID/MovieID gốc với chỉ số embedding.

Nếu muốn gửi lại để tích hợp/kiểm tra, gửi hai file .npz và các file *_export_results.csv/json tương ứng. Nếu nhóm chỉ chọn BPR làm MF chính, chỉ cần chạy/export 04; Bias MF là tùy chọn.

## 3. Chạy lại phần của Ánh

```bash
python run_experiments.py
```

Lệnh chạy mọi code cell của 03 rồi 05, ghi output thật vào hai notebook và xuất bảng. Có thể thay bằng Run All từng notebook trong Jupyter/VS Code. Không chạy notebook của bạn khác bằng lệnh này. Thời gian phụ thuộc máy; không cần chạy lại để dùng kết quả đã kèm.

Nếu sửa src, restart kernel trước Run All. Không dùng output cũ khi dữ liệu thay đổi.

### Tạo lại train/test
```bash
python prepare_data.py
```

Tương đương đúng `train_test_split(ratings, test_size=0.2, random_state=42)` trong 01, giữ thứ tự dòng. Không thêm lọc 5-core, không chuyển về chia theo thời gian. Chạy lại 03/05 và xuất lại checkpoint nếu thay dữ liệu. App ẩn bảng có fingerprint không khớp.

## 4. Thuật toán và thực nghiệm
- **03 User-CF:** tương tác tích cực nhị phân rating >=4, cosine giữa user, lấy k láng giềng dương, cộng tín hiệu có trọng số similarity.
- **03 Item-CF:** với mỗi phim đã thích, truyền điểm tới k phim gần nhất bằng cosine giữa cột tương tác. Đồ thị kNN có hướng.
- **Tuning:** tách 10% từ outer train 80% để validation; thử k=10/20/40; chọn NDCG@10 cho từng nhánh; refit toàn bộ outer train trước test. Không dùng test để tune.
- **05:** TF-IDF trên token thể loại nguyên vẹn, hồ sơ trọng số max(rating−3,0), chuẩn hóa và cosine. Tìm tên dùng chỉ mục title+genres riêng; không dùng tên phim để cá nhân hóa.
- **Cold-start:** user thiếu tín hiệu → popularity. Đánh giá chỉ warm-start; app tìm tên/thể loại không cần user.
- **Memory:** cosine tính theo block 128 hàng, chỉ giữ k láng giềng, không lưu ma trận similarity vuông đầy đủ.

### Protocol chung
MovieID thuộc train làm candidate; loại mọi phim đã đánh giá trong train, kể cả rating thấp. Lọc test theo user/movie đã xuất hiện trong train; relevant rating >=4. Chỉ macro-average trên user có relevant. Precision chia K=10; F1 tính từng user rồi trung bình; NDCG nhị phân. Báo số user và dòng bị loại. Random holdout không phải mô phỏng thời gian tương lai.

Evaluator mới phá hòa theo popularity rồi MovieID tăng dần. 02/04/thu_nghiem có cách argsort khác khi hòa, nên kết quả có thể chênh nhẹ. Baseline được tính lại cùng pipeline; output cũ trong notebook của nhóm được giữ nguyên, không đưa vào bảng app như số liệu mới. Nội dung notebook 03/05 giải thích thuật toán và các giới hạn.

### Thư mục kết quả
- 03_validation_results.csv/json: chọn k, chỉ dùng inner-train/validation.
- 03_test_results.csv/json: Popularity + User-CF + Item-CF trên full test.
- 05_test_results.csv/json: Popularity + Content-based trên full test.
- *_per_user.csv: metric theo user để phân tích.
- *_export_results.csv/json: chỉ xuất hiện sau khi user export MF/BPR.

## 5. Tệp cần đưa vào repo của nhóm
Nếu không muốn thay cả thư mục, copy notebooks/03_collaborative_filtering.ipynb, notebooks/05_content_based.ipynb, src/, scripts/, app.py, prepare_data.py, run_experiments.py, requirements.txt, README.md, models/cf_config.json và reports/tables mới. CSV data/processed được tạo đúng 01; không dùng CSV từ bản Coursera hoặc bản MovieLens cũ chia theo thời gian. Không copy checkpoint MF từ dữ liệu/split khác.

## 6. Kiểm tra
```bash
python -m unittest discover -s tests -v
```
Xem VALIDATION.md về phạm vi kiểm tra. Dữ liệu và điều kiện sử dụng nguồn giữ tại data/raw/README. Không công bố lại dataset trên repo công khai nếu chưa đáp ứng điều kiện nguồn; đây là bản làm việc trả cho người đã cung cấp dữ liệu.
