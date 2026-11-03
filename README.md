# Hệ thống gợi ý phim trên MovieLens 1M

**Movie Recommendation System** là dự án xây dựng và đánh giá hệ thống gợi ý phim theo sở thích người dùng. Dự án so sánh sáu phương pháp: Popularity, User-based Collaborative Filtering, Item-based Collaborative Filtering, Bayesian Personalized Ranking Matrix Factorization (BPR-MF), Bias Matrix Factorization và Content-based Filtering. Ứng dụng Streamlit được sử dụng để minh họa kết quả gợi ý và so sánh thực nghiệm.

## 1. Giới thiệu và mục tiêu

Khi số lượng phim ngày càng lớn, việc lựa chọn nội dung phù hợp trở nên khó khăn. Hệ thống gợi ý khai thác lịch sử đánh giá và đặc trưng phim để đề xuất danh sách có khả năng phù hợp với từng người dùng.

Dự án hướng đến các mục tiêu:

- Khảo sát bộ dữ liệu MovieLens 1M và xây dựng quy trình chia dữ liệu có thể tái lập.
- Triển khai các phương pháp gợi ý dựa trên độ phổ biến, tương tác cộng đồng, nhân tố tiềm ẩn và nội dung phim.
- Đánh giá chất lượng danh sách Top-10 bằng Precision, Recall, F1 và NDCG.
- Xây dựng ứng dụng demo cho phép chọn người dùng, tìm phim, xem phim tương tự và đối chiếu kết quả giữa các mô hình.

**Phạm vi:** bài toán chính là xếp hạng các phim người dùng chưa đánh giá trong tập huấn luyện. Các kết quả thực nghiệm được đo trong điều kiện người dùng và phim đã xuất hiện trong train (*warm-start*), không phải đánh giá cho người dùng hoặc phim hoàn toàn mới.

## 2. Bộ dữ liệu

Dự án sử dụng **MovieLens 1M**, do GroupLens cung cấp.

| Thành phần | Nội dung |
|---|---|
| Số lượt đánh giá | 1.000.209 |
| Số người dùng | 6.040 |
| Số phim trong metadata | 3.883 |
| Thang đánh giá | 1–5 sao |
| `ratings.dat` | `user_id`, `movie_id`, `rating`, `timestamp` |
| `movies.dat` | `movie_id`, `title`, `genres` |
| `users.dat` | `user_id`, `gender`, `age`, `occupation`, `zip_code` |

Các file sử dụng dấu phân cách `::` và được đọc với encoding `latin-1`. Một phim có thể thuộc nhiều thể loại, ngăn cách bởi dấu `|`.

Thông tin người dùng được khảo sát trong bước EDA; các mô hình hiện tại không sử dụng giới tính, tuổi hoặc nghề nghiệp để tính điểm gợi ý. Dữ liệu phim không chứa mô tả cốt truyện, diễn viên, đạo diễn hay poster. Vì vậy, mô hình Content-based cá nhân hóa dựa trên **thể loại phim**; tên phim được dùng trong chức năng tìm kiếm riêng.

Nguồn dữ liệu: [MovieLens 1M — GroupLens](https://grouplens.org/datasets/movielens/1m/). Cần tuân thủ điều kiện sử dụng và trích dẫn đi kèm bộ dữ liệu.

## 3. Quy trình thực hiện

Quy trình gồm: khảo sát dữ liệu → chia train/test → xây dựng mô hình → lựa chọn tham số nếu có → đánh giá Top-10 → trình diễn trên ứng dụng.

### 3.1. Khảo sát và chia dữ liệu

Notebook `01_EDA_proprecessing.ipynb` khảo sát giá trị thiếu, bản ghi trùng, phân bố rating, mức độ thưa của tương tác, thông tin người dùng và thể loại phim. Dữ liệu được chia ngẫu nhiên theo từng lượt đánh giá:

```python
train_test_split(ratings, test_size=0.2, random_state=42)
```

| Thống kê | Giá trị |
|---|---:|
| Số tương tác train | 800.167 |
| Số tương tác test ban đầu | 200.042 |
| Số người dùng trong train | 6.040 |
| Số phim trong train | 3.683 |
| Số dòng test bị loại khi đánh giá do phim chưa có trong train | 26 |
| Số dòng test còn lại sau lọc | 200.016 |
| Số người dùng có ít nhất một phim liên quan trong test sau lọc | 5.994 |

Đây là cách chia **ngẫu nhiên 80/20**, không phải chia theo thời gian. Không áp dụng lọc 5-core trong quy trình này. UserID và MovieID gốc được giữ ở giao diện và đầu vào; các mô hình tự ánh xạ sang chỉ số nội bộ khi cần.

### 3.2. Lựa chọn tham số cho Collaborative Filtering

Từ tập train 80%, tách tiếp 10% làm validation với `random_state=42`. Hai nhánh User-CF và Item-CF thử số láng giềng `k ∈ {10, 20, 40}` và chọn theo **NDCG@10 trên validation**. Các tương tác validation có ID ngoài tập train nội bộ được loại khi đánh giá.

Cấu hình được chọn:

- **User-CF:** `k = 40`.
- **Item-CF:** `k = 10`.

Sau lựa chọn, mô hình được fit lại trên toàn bộ train 80%, rồi đánh giá trên test 20%. Không sử dụng test để lựa chọn `k`. Cấu hình được lưu tại [`models/cf_config.json`](models/cf_config.json).

## 4. Các phương pháp đề xuất

| Phương pháp | Cách tính gợi ý | Đặc điểm |
|---|---|---|
| Popularity | Xếp hạng phim theo số rating ≥ 4 trong train | Baseline dùng chung, không cá nhân hóa thứ tự ngoài việc loại lịch sử |
| User-CF | Tìm người dùng tương tự bằng cosine; cộng tương tác tích cực của các láng giềng theo trọng số similarity | Khai thác sự tương đồng giữa người dùng |
| Item-CF | Với mỗi phim người dùng thích, cộng điểm đến các phim láng giềng tương tự theo cosine | Khai thác sự tương đồng giữa các cột tương tác phim |
| BPR-MF | Học vector người dùng/phim để xếp phim tích cực cao hơn phim chưa tương tác | Tối ưu thứ tự tương đối giữa cặp phim |
| Bias MF | Dự đoán rating từ trung bình toàn cục, bias người dùng, bias phim và tích vô hướng các vector tiềm ẩn; dùng điểm để xếp hạng | Học từ giá trị rating tường minh |
| Content-based | TF-IDF thể loại phim, xây hồ sơ sở thích và tính cosine | Cá nhân hóa từ đặc trưng nội dung có sẵn |

### 4.1. Neighborhood-based Collaborative Filtering

Tương tác được nhị phân hóa: rating ≥ 4 là tín hiệu tích cực. User-CF tính cosine giữa các hàng của ma trận tương tác; Item-CF tính cosine giữa các cột. Chỉ giữ láng giềng có độ tương đồng dương và loại chính đối tượng đang xét.

Ở Item-CF, mỗi phim nguồn mà người dùng thích đóng góp điểm cho `k` phim gần nhất của nó. Đồ thị láng giềng có hướng, không bắt buộc đối xứng. Phần tính cosine được thực hiện theo block để tránh lưu toàn bộ ma trận tương đồng vuông trong bộ nhớ.

### 4.2. Matrix Factorization

Cả hai mô hình sử dụng 50 nhân tố tiềm ẩn và seed 42:

| Tham số | BPR-MF | Bias MF |
|---|---:|---:|
| Số nhân tố | 50 | 50 |
| Learning rate | 0,01 | 0,005 |
| Regularization | 0,01 | 0,02 |
| Số epoch | 15 | 15 |

BPR-MF lấy rating ≥ 4 làm tương tác tích cực và lấy mẫu phim âm từ các phim chưa được người dùng đánh giá trong train. Phim chưa tương tác chỉ là mẫu đối chiếu, không đồng nghĩa người dùng chắc chắn không thích phim đó.

Bias MF sử dụng toàn bộ rating train để học điểm dự đoán. Việc học tốt giá trị rating không nhất thiết tạo ra danh sách Top-10 tốt hơn một mô hình trực tiếp học xếp hạng. Các cấu hình MF trên là cấu hình trong notebook hiện tại; chưa có bước tìm kiếm siêu tham số tương tự CF.

### 4.3. Content-based Filtering

Mỗi thể loại là một token riêng, giữ nguyên các nhãn như `Sci-Fi` và `Film-Noir`. TF-IDF được fit trên metadata các phim thuộc train. Hồ sơ người dùng là tổng vector phim đã đánh giá, với trọng số:

```text
Trọng số = max(rating − 3, 0)
```

Rating 4 đóng góp trọng số 1, rating 5 đóng góp 2; rating ≤ 3 không đóng góp vào hồ sơ yêu thích. Hồ sơ được chuẩn hóa L2, sau đó tính cosine với các phim ứng viên.

Chức năng tìm kiếm dùng một chỉ mục TF-IDF riêng trên **tên phim + thể loại**. Chỉ mục tìm kiếm không tham gia tính điểm cá nhân hóa trong bảng thực nghiệm.

## 5. Thiết lập đánh giá

Các bảng kết quả dưới đây sử dụng quy ước chung:

- **Top-K:** `K = 10`.
- **Phim liên quan:** phim trong test có rating ≥ 4, sau khi lọc ID thuộc train.
- **Tập ứng viên:** các phim xuất hiện trong train.
- **Loại lịch sử:** bỏ mọi phim người dùng đã đánh giá trong train, kể cả rating thấp.
- **Tập người dùng đánh giá:** chỉ những người có ít nhất một phim liên quan sau lọc.
- **Tổng hợp:** tính metric từng người dùng rồi lấy trung bình (*macro-average*).

| Chỉ số | Ý nghĩa |
|---|---|
| Precision@10 | Số phim đúng trong danh sách chia cho 10, kể cả khi danh sách trả về ít hơn 10 phim |
| Recall@10 | Số phim đúng được gợi ý chia cho tổng số phim liên quan của người dùng trong test |
| F1@10 | Trung bình điều hòa của Precision@10 và Recall@10 của từng người dùng, sau đó lấy trung bình toàn tập |
| NDCG@10 | Đánh giá chất lượng thứ tự; phim liên quan ở vị trí cao được thưởng nhiều hơn |

Các hàm xếp hạng ưu tiên điểm mô hình, sau đó đến popularity và MovieID tăng dần để xử lý trường hợp bằng điểm. Điểm CF và Content-based là điểm xếp hạng, không phải rating dự đoán hoặc xác suất người dùng yêu thích phim.

## 6. Kết quả thực nghiệm

Bảng dưới đây được đối chiếu từ output đã lưu trong các notebook của dự án và làm tròn bốn chữ số thập phân. Đây là kết quả của lần thực nghiệm được lưu, không phải kết quả tự động tính lại khi mở README. Các bảng CSV và JSON được tạo tại máy chạy; phiên bản repository hiện tại chưa lưu các file này.

| Mô hình | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Popularity | 0,1518 | 0,0907 | 0,0918 | 0,1766 |
| **User-CF (k = 40)** | **0,2633** | **0,1916** | **0,1795** | **0,3280** |
| Item-CF (k = 10) | 0,2400 | 0,1602 | 0,1565 | 0,2942 |
| BPR-MF | 0,1666 | 0,0968 | 0,0992 | 0,1928 |
| Bias MF | 0,0741 | 0,0391 | 0,0417 | 0,0816 |
| Content-based Genres | 0,0677 | 0,0434 | 0,0433 | 0,0854 |

Nguồn kết quả và file được tạo khi chạy notebook:

| Notebook | Bảng tổng hợp trong `reports/tables/` |
|---|---|
| [02_baseline_model.ipynb](notebooks/02_baseline_model.ipynb) | `02_baseline.csv` |
| [03_collaborative_filtering.ipynb](notebooks/03_collaborative_filtering.ipynb) | `03_validation_results.csv`, `03_test_results.csv` |
| [04_BPR_MF.ipynb](notebooks/04_BPR_MF.ipynb) | `04_bpr.csv` |
| [05_content_based.ipynb](notebooks/05_content_based.ipynb) | `05_test_results.csv` |
| [06_Bias_MF.ipynb](notebooks/06_Bias_MF.ipynb) | `thu_nghiem_mf.csv` |

Các notebook 03, 04, 05 và 06 có bước xuất metadata JSON chứa fingerprint của train/test để ứng dụng kiểm tra dữ liệu. Cần chạy hết notebook, bao gồm cell cuối của 04 và 06. File `thu_nghiem_mf.csv` vẫn giữ tên này trong code dù notebook tương ứng đã đổi tên thành `06_Bias_MF.ipynb`.

### Nhận xét

**User-CF đạt kết quả cao nhất trên cả bốn chỉ số trong lần thực nghiệm này.** Item-CF cũng vượt baseline, cho thấy thông tin tương tác giữa người dùng và phim hữu ích cho việc cá nhân hóa trên tập dữ liệu đang xét.

BPR-MF vượt Popularity và tốt hơn Bias MF trên các chỉ số Top-10. Điều này phù hợp với sự khác nhau về mục tiêu học: BPR hướng đến xếp hạng cặp phim, trong khi Bias MF học giá trị rating. Tuy nhiên, kết quả còn phụ thuộc cấu hình và quá trình huấn luyện; một lần chạy chưa đủ để khẳng định nguyên nhân duy nhất.

Content-based thấp hơn baseline. Một hạn chế rõ ràng là phim chỉ được biểu diễn bằng thể loại: nhiều phim cùng tổ hợp thể loại có vector giống nhau và thứ tự phải dựa thêm vào popularity để phá hòa. Kết quả này không đồng nghĩa mọi hệ thống Content-based đều kém; mô hình sử dụng cốt truyện hoặc đặc trưng phong phú hơn cần được đánh giá riêng.

Các kết luận chỉ áp dụng cho cấu hình và split hiện tại. Chưa thực hiện nhiều seed hoặc kiểm định thống kê để kết luận mức độ ổn định của chênh lệch.

## 7. Ứng dụng demo

Ứng dụng được xây dựng bằng **Streamlit**, gồm bốn chức năng chính:

| Chức năng | Nội dung |
|---|---|
| Dành cho bạn | Chọn UserID, phương pháp gợi ý, xem lịch sử đánh giá và danh sách phim đề xuất |
| Tìm kiếm | Tìm phim bằng tên hoặc thể loại tiếng Anh, không cần chọn người dùng |
| Phim tương tự | Chọn một phim để xem các phim có thể loại tương đồng |
| Thực nghiệm | Xem bảng và biểu đồ so sánh, tải kết quả CSV |

Ứng dụng hỗ trợ sáu phương pháp. BPR-MF và Bias MF cần file mô hình đã xuất ở máy chạy; các mô hình còn lại được fit và lưu cache khi sử dụng. Cấu hình CF mặc định lấy từ kết quả validation nếu khớp dữ liệu hiện tại. Với bảng có metadata, ứng dụng chỉ chấp nhận khi fingerprint train/test khớp. Hai bảng CF/Content-based bắt buộc có metadata; BPR-MF và Bias MF còn được phép hiển thị khi thiếu JSON nhưng kèm cảnh báo “Chưa có metadata”. Biểu đồ được tách riêng cho từng chỉ số Precision@10, Recall@10, F1@10 và NDCG@10.

### 7.1. Giao diện gợi ý cá nhân hóa

<img width="1858" height="977" alt="image" src="https://github.com/user-attachments/assets/0f710806-8237-44b3-b4e3-f8b8fd615aac" />


### 7.2. Giao diện tìm kiếm phim

<img width="1862" height="963" alt="image" src="https://github.com/user-attachments/assets/9ce86a72-5f11-45d6-8844-5542bce2f43f" />


### 7.3. Giao diện phim tương tự

<img width="1860" height="980" alt="image" src="https://github.com/user-attachments/assets/ac581bec-1f4f-488e-af4a-2c196e87152e" />


### 7.4. Giao diện so sánh thực nghiệm

<img width="1860" height="979" alt="image" src="https://github.com/user-attachments/assets/35808914-fa50-4251-a1a0-622773362726" />


## 8. Tổ chức mã nguồn

| Đường dẫn | Vai trò |
|---|---|
| `notebooks/01_EDA_proprecessing.ipynb` | Khảo sát dữ liệu và chia train/test |
| `notebooks/02_baseline_model.ipynb` | Popularity baseline |
| `notebooks/03_collaborative_filtering.ipynb` | User-CF, Item-CF, chọn tham số và đánh giá |
| `notebooks/04_BPR_MF.ipynb` | Huấn luyện BPR-MF |
| `notebooks/05_content_based.ipynb` | Gợi ý dựa trên thể loại và tìm kiếm nội dung |
| `notebooks/06_Bias_MF.ipynb` | Thực nghiệm Bias MF |
| `src/recommenders.py` | Popularity, CF, Content-based, validation và evaluator cho notebook 03/05 |
| `src/matrix_factorization.py` | Huấn luyện BPR-MF, Bias MF và tạo gợi ý |
| `src/du_lieu.py`, `src/popularity.py`, `src/goi_y.py` | Đọc/chia dữ liệu, baseline và xếp hạng cho nhánh 02/04/06 |
| `src/danh_gia.py`, `src/luu_ket_qua.py` | Đánh giá nhánh 02/04/06, lưu CSV và mô hình joblib |
| `src/experiments.py` | Xuất bảng kết quả và metadata |
| `prepare_data.py` | Tạo train/test theo cùng quy tắc của notebook 01 |
| `run_experiments.py` | Chạy và lưu output notebook 03, 05 |
| `app.py` | Giao diện Streamlit |
| `models/` | `cf_config.json`, `mo_hinh_bpr.joblib`, `mo_hinh_mf.joblib` |
| `reports/tables/` | Bảng kết quả, metric theo user và metadata |
| `reports/figures/` | Hình minh họa và ảnh demo bổ sung |
| `data/raw/`, `data/processed/` | Dữ liệu nguồn và dữ liệu sau khi chia tập, chuẩn bị tại máy chạy |

## 9. Cài đặt và chạy dự án

### 9.1. Cài môi trường

Có thể dùng môi trường Python 3.11 với các thư viện trong `requirements.txt`. Mã hiện tại chạy bằng NumPy/SciPy trên CPU; không cần bật GPU để sử dụng.

```bash
git clone https://github.com/nguyentrang05/Movie_Recommendation_System.git
cd Movie_Recommendation_System
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS hoặc terminal Codespaces:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

### 9.2. Chuẩn bị dữ liệu

Tải MovieLens 1M từ [GroupLens](https://grouplens.org/datasets/movielens/1m/), giải nén và đặt ba file sau vào đúng thư mục:

- `data/raw/ratings.dat`
- `data/raw/movies.dat`
- `data/raw/users.dat`

Chạy tại thư mục gốc dự án:

```bash
python prepare_data.py
```

Lệnh tạo `data/processed/train.csv`, `data/processed/test.csv` và `split_info.json` theo quy tắc chia ngẫu nhiên 80/20, seed 42. Muốn xem toàn bộ phần EDA, mở và chạy notebook 01. Không bắt buộc chạy cả hai cách để tạo train/test.

Notebook 01 hiện sử dụng các đường dẫn `../data/...`, nên thư mục làm việc của kernel cần là `notebooks/`. Có thể mở Jupyter từ thư mục này:

```bash
cd notebooks
python -m jupyter notebook
```

### 9.3. Chạy các notebook

Trong Jupyter, chọn **Restart Kernel and Run All Cells** khi cần chạy lại đầy đủ, đặc biệt sau khi sửa file trong `src/`.

| Thứ tự | Notebook | Công việc |
|---|---|---|
| 01 | `01_EDA_proprecessing.ipynb` | Khảo sát dữ liệu và xuất train/test |
| 02 | `02_baseline_model.ipynb` | Đánh giá Popularity |
| 03 | `03_collaborative_filtering.ipynb` | Chọn k, đánh giá CF, lưu cấu hình và kết quả |
| 04 | `04_BPR_MF.ipynb` | Huấn luyện BPR-MF, lưu joblib, đánh giá và xuất JSON |
| 05 | `05_content_based.ipynb` | Đánh giá Content-based, kiểm tra tìm kiếm/phim tương tự |
| 06 | `06_Bias_MF.ipynb` | Huấn luyện Bias MF, lưu joblib, đánh giá và xuất JSON |

Khi đã có train/test, các notebook 02–06 có thể chạy độc lập. Riêng 03 tự chọn k trong notebook của nó. Notebook 04 và 06 đã có sẵn lệnh lưu mô hình; không cần script export riêng hoặc lệnh `%run`.

Để chạy riêng 03 và 05 bằng terminal tại thư mục gốc:

```bash
python run_experiments.py
```

Script này thực thi các cell và ghi lại stdout vào notebook. Nó không lưu đầy đủ biểu đồ/hình rich output vào notebook; nếu cần hình cho báo cáo, hãy chạy bằng Jupyter. Script không huấn luyện 04/06.

### 9.4. Mô hình đã lưu và kết quả thực nghiệm

Repository hiện có hai file mô hình:

- `models/mo_hinh_bpr.joblib`
- `models/mo_hinh_mf.joblib`

App có thể nạp các file này nếu danh sách người dùng, danh sách phim và lịch sử tương tác khớp train hiện tại. Cơ chế kiểm tra mô hình joblib chưa so sánh toàn bộ giá trị rating bằng fingerprint; khi đổi dữ liệu hoặc cách chia, cần huấn luyện lại mô hình tương ứng.

Để tái lập đầy đủ, chạy hết notebook 04 và 06 trên cùng train/test với 03 và 05. Cell cuối của 04/06 xuất JSON cho bảng kết quả. Việc có sẵn joblib không đồng nghĩa máy đã có CSV/JSON để hiển thị trang Thực nghiệm.

Dữ liệu raw, train/test và các kết quả trong `reports/` đang bị loại khỏi Git bằng `.gitignore`. Vì vậy, người clone repository phải chuẩn bị dữ liệu và chạy notebook để tạo bảng thực nghiệm tại máy mình. Output đã lưu trong notebook vẫn có thể xem trực tiếp trên GitHub.

### 9.5. Mở ứng dụng

Chạy ở thư mục chứa `app.py`:

```bash
python -m streamlit run app.py
```

Nếu terminal đang ở `notebooks/`, chạy `cd ..` trước. Mở địa chỉ mà terminal hiển thị, thường là `http://localhost:8501`. Trong Codespaces, mở cổng 8501 qua mục **Ports**.

Sau khi chuẩn bị dữ liệu, có thể dùng Popularity, User-CF, Item-CF và Content-based. CF dùng k từ `cf_config.json` nếu fingerprint train khớp; nếu thiếu hoặc không khớp, app dùng k = 20 và thông báo. BPR-MF/Bias MF cần file joblib phù hợp. Trang Thực nghiệm cần các bảng kết quả đã xuất.

### 9.6. Kiểm tra chức năng

```bash
python tests/kiem_tra.py
```

Bộ kiểm tra dùng dữ liệu nhỏ để kiểm tra lọc lịch sử của MF, người dùng mới, metric, lưu/nạp mô hình và tái lập seed. Các kiểm tra CF/Content-based trong file này hiện đang được comment, nên lệnh trên chưa kiểm tra đầy đủ cả sáu phương pháp. Dòng thông báo “7 nhóm kiểm tra” ở cuối file chưa phản ánh số nhóm đang thực sự chạy. Kết quả kiểm tra nhỏ không dùng làm số liệu báo cáo.

### 9.7. Một số tình huống thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Thiếu `train.csv`, `test.csv` hoặc `movies.dat` | Đặt dữ liệu đúng thư mục và chạy `python prepare_data.py` |
| App báo chưa có bảng kết quả | Chạy notebook 03, 04, 05, 06 để xuất bảng |
| BPR/Bias MF báo thiếu metadata | Chạy cell cuối notebook tương ứng trong kernel đã có dữ liệu và kết quả đánh giá đúng train/test |
| Fingerprint bảng kết quả không khớp | Đánh giá lại mô hình trên dữ liệu hiện tại và xuất lại kết quả; không chỉ đổi tên file |
| Mô hình joblib không khớp train | Chạy lại notebook MF tương ứng |
| Sửa `src/` nhưng notebook vẫn dùng logic cũ | Restart kernel rồi Run All |

## 10. Hạn chế và hướng phát triển

**Hạn chế hiện tại:**

- Đánh giá trên một lần chia ngẫu nhiên; chưa kiểm tra nhiều seed, khoảng tin cậy hoặc chia theo thời gian.
- Chỉ đánh giá warm-start; tìm kiếm cho người mới không đồng nghĩa đã đánh giá được chất lượng cá nhân hóa cold-start.
- Metadata thể loại còn đơn giản, thiếu đặc trưng nội dung chi tiết.
- Các cấu hình MF chưa được tìm kiếm có hệ thống như số láng giềng của CF.
- Chưa đánh giá độ đa dạng, độ mới, độ bao phủ hoặc phản hồi từ người dùng thực.

**Hướng phát triển:** bổ sung metadata từ nguồn được phép sử dụng; thử biểu diễn nội dung phong phú hơn; nghiên cứu mô hình lai CF–Content-based; tối ưu tham số MF bằng validation; đánh giá nhiều lần chia dữ liệu và bổ sung các tiêu chí ngoài độ chính xác. Đây là các hướng đề xuất, chưa phải chức năng đã triển khai.

## 11. Thông tin nhóm


| Nội dung | Thông tin |
|---|---|
| Học phần | Đồ án liên ngành-1-1-26(N02).TH  |
| Giảng viên hướng dẫn | TS. Trịnh Thành |
| Lớp / nhóm | K17-CNTT_7 |
| Trường / khoa | Trường công nghệ thông tin Phenikaa |

| Thành viên | Mã sinh viên | Phân công |
|---|---|---|
| Nguyễn Quỳnh Trang | 23010651 | Tìm hiểu bài toán và dữ liệu; EDA, xử lý và chia dữ liệu; Popularity; Bias MF và BPR-MF; cùng tổng hợp thực nghiệm và hoàn thiện báo cáo |
| Nguyễn Minh Ánh | 23010718 | User-CF, Item-CF và chọn k; Content-based; app Movie Compass; cùng tổng hợp thực nghiệm và hoàn thiện báo cáo |


## 12. Tài liệu tham khảo

1. GroupLens Research. [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/).
2. F. Maxwell Harper và Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19. [DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).

Các chi tiết triển khai và số liệu thực nghiệm của dự án có thể đối chiếu trực tiếp trong các notebook và mã nguồn được liên kết ở trên; bảng CSV được tạo khi chạy notebook.
