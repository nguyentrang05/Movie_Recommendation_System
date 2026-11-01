from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

from src.recommenders import (
    load_data,
    fingerprint,
    Popularity,
    CollaborativeFiltering,
    ContentBased,
)
from src.luu_ket_qua import doc_mo_hinh
from src.matrix_factorization import goi_y_mf


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "data/processed/train.csv"
TEST = ROOT / "data/processed/test.csv"
MOVIES = ROOT / "data/raw/movies.dat"

MF_FILES = {
    "BPR-MF": ("mo_hinh_bpr.joblib", "04_BPR_MF.ipynb"),
    "Bias MF": ("mo_hinh_mf.joblib", "06_Bias_MF.ipynb"),
}

st.set_page_config(
    page_title="Movie Compass",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {background: #f6f8fc}
    h1, h2, h3 {color: #152e49}
    div[data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border: 1px solid #e3e9f2;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption("MOVIE COMPASS / MOVIELENS 1M")
st.title("Một bộ phim hay, đúng với bạn.")
st.write(
    "Gợi ý từ lịch sử đánh giá, người xem tương tự "
    "và thể loại yêu thích."
)

if not all(p.exists() for p in [TRAIN, TEST, MOVIES]):
    st.info(
        "Chưa đủ dữ liệu. Chép movies.dat/ratings.dat/users.dat "
        "vào data/raw rồi chạy:"
    )
    st.code("python prepare_data.py")
    st.stop()

version = tuple(
    (p.stat().st_mtime_ns, p.stat().st_size)
    for p in [TRAIN, TEST, MOVIES]
)


@st.cache_data
def data(version):
    train, test, movies = load_data(ROOT)
    return (
        train,
        test,
        movies,
        fingerprint(train),
        fingerprint(test),
    )


try:
    train, test, movies, train_hash, test_hash = data(version)
except (ValueError, OSError, KeyError) as exc:
    st.error(f"Không đọc được dữ liệu: {exc}")
    st.stop()

catalog = movies[
    movies.movie_id.isin(train.movie_id)
].set_index("movie_id")


@st.cache_resource(max_entries=6)
def fitted(version, method, k=20, checkpoint_stamp=None):
    current_train, _, current_movies, _, _ = data(version)

    if method == "Popularity":
        return Popularity().fit(current_train)

    if method == "User-CF":
        return CollaborativeFiltering("user", k).fit(current_train)

    if method == "Item-CF":
        return CollaborativeFiltering("item", k).fit(current_train)

    if method == "Content-based":
        return ContentBased().fit(current_train, current_movies)

    filename, _ = MF_FILES[method]
    model = doc_mo_hinh(ROOT / "models" / filename)

    if not isinstance(model, dict):
        raise ValueError("File joblib không chứa mô hình đúng định dạng.")

    expected_kind = "bpr" if method == "BPR-MF" else "mf"
    if model.get("loai") != expected_kind:
        raise ValueError("Loại mô hình không khớp phương pháp đã chọn.")

    expected_users = np.sort(current_train.user_id.unique())
    expected_items = np.sort(current_train.movie_id.unique())

    if not np.array_equal(model["ds_nguoi_dung"], expected_users):
        raise ValueError(
            "Danh sách người dùng trong mô hình không khớp train. "
            "Hãy chạy lại notebook MF."
        )

    if not np.array_equal(model["ds_phim"], expected_items):
        raise ValueError(
            "Danh sách phim trong mô hình không khớp train. "
            "Hãy chạy lại notebook MF."
        )

    expected_seen = (
        current_train.groupby("user_id").movie_id.apply(set).to_dict()
    )
    if model["phim_da_xem"] != expected_seen:
        raise ValueError(
            "Lịch sử tương tác trong mô hình không khớp train. "
            "Hãy chạy lại notebook MF."
        )

    return model


def get_model(method, k=20):
    stamp = None

    if method in MF_FILES:
        filename, notebook = MF_FILES[method]
        path = ROOT / "models" / filename

        if not path.exists():
            st.info(
                f"Chưa có models/{filename}. "
                f"Mở notebooks/{notebook} và chạy Run All "
                "để huấn luyện, lưu mô hình."
            )
            return None

        stamp = (path.stat().st_mtime_ns, path.stat().st_size)

    try:
        with st.spinner("Đang chuẩn bị gợi ý…"):
            return fitted(version, method, k, stamp)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        EOFError,
        ImportError,
    ) as exc:
        st.error(f"Không nạp được mô hình: {exc}")
        return None


def cards(results):
    if isinstance(results, pd.DataFrame):
        results = list(
            results[["movie_id", "score"]].itertuples(
                index=False,
                name=None,
            )
        )
    elif results is None:
        results = []
    else:
        results = list(results)

    if not results:
        st.info(
            "Không tìm thấy phim phù hợp. "
            "Hãy thử tên phim hoặc thể loại khác."
        )
        return

    columns = st.columns(2)

    for rank, (mid, score) in enumerate(results):
        mid = int(mid)
        if mid not in catalog.index:
            continue

        movie = catalog.loc[mid]

        with columns[rank % 2]:
            with st.container(border=True):
                st.caption(f"GỢI Ý {rank + 1:02} • MOVIE {mid}")
                st.subheader(str(movie.title))
                st.write(str(movie.genres).replace("|", " · "))
                st.caption(f"Điểm xếp hạng: {float(score):.4f}")


page = st.sidebar.radio(
    "KHÁM PHÁ",
    ["Dành cho bạn", "Tìm kiếm", "Phim tương tự", "Thực nghiệm"],
)

n = st.sidebar.slider("Số phim", 1, 20, 10)

st.sidebar.caption(
    "Dữ liệu thật • Split ngẫu nhiên 80/20 "
    "theo notebook 01 của nhóm."
)

cols = st.columns(3)

for col, label, count in zip(
    cols,
    ["Phim trong train", "Người xem", "Đánh giá train"],
    [len(catalog), train.user_id.nunique(), len(train)],
):
    col.metric(label, f"{count:,}")

if page == "Dành cho bạn":
    left, right = st.columns(2)

    with left:
        method = st.selectbox(
            "Phương pháp",
            [
                "Popularity",
                "User-CF",
                "Item-CF",
                "Content-based",
                "BPR-MF",
                "Bias MF",
            ],
        )

    with right:
        user = st.selectbox(
            "Người xem (UserID gốc)",
            sorted(train.user_id.unique()),
        )

    k = 20

    if method in ["User-CF", "Item-CF"]:
        config_path = ROOT / "models/cf_config.json"

        if config_path.exists():
            try:
                config = json.loads(
                    config_path.read_text(encoding="utf-8")
                )

                if config.get("train_fingerprint") == train_hash:
                    mode = "user" if method == "User-CF" else "item"
                    selected_k = int(config["neighbors"][mode])

                    if selected_k < 1:
                        raise ValueError("k phải lớn hơn 0.")

                    k = selected_k
                    st.caption(
                        f"k = {k}, được chọn bằng validation "
                        "nội bộ của notebook 03."
                    )
                else:
                    st.warning(
                        "Cấu hình CF thuộc train cũ. Dùng k=20; "
                        "chạy lại notebook 03 để chọn k."
                    )

            except (ValueError, KeyError, TypeError, OSError) as exc:
                st.warning(
                    f"Không đọc được cấu hình CF: {exc}. Dùng k=20."
                )
        else:
            st.caption(
                "Dùng k=20 mặc định. Chạy notebook 03 "
                "để chọn k bằng validation."
            )

        if st.checkbox("Thử số láng giềng khác (chỉ thay đổi demo)"):
            k = st.select_slider(
                "k",
                options=[10, 20, 40],
                value=k if k in [10, 20, 40] else 20,
            )

    with st.expander("Lịch sử đánh giá trong train"):
        history = train[train.user_id == user].merge(
            movies,
            on="movie_id",
            how="left",
        )
        st.dataframe(
            history[["title", "genres", "rating"]],
            hide_index=True,
        )

    model = get_model(method, k)

    if model is not None:
        st.caption(
            "Loại phim đã đánh giá trong train. "
            "Điểm không phải xác suất; chỉ so trong cùng phương pháp. "
            "CF/Content thiếu tín hiệu sẽ dùng popularity."
        )

        if method in MF_FILES:
            results = goi_y_mf(model, int(user), so_phim=n)
        else:
            results = model.ranked(
                model.scores(user),
                n,
                model.seen.get(user, set()),
            )

        cards(results)

elif page == "Tìm kiếm":
    with st.form("search"):
        query = st.text_input(
            "Tên phim hoặc thể loại tiếng Anh",
            placeholder="Toy Story, Action, Sci-Fi…",
        )
        submit = st.form_submit_button(
            "Tìm phim",
            type="primary",
        )

    if submit:
        if not query.strip():
            st.info("Hãy nhập tên phim hoặc thể loại.")
        else:
            model = get_model("Content-based")
            if model is not None:
                cards(model.search(query, n))

    st.caption(
        "Không cần lịch sử người dùng. "
        "Tìm trong catalog phim thuộc train."
    )

elif page == "Phim tương tự":
    selected = st.selectbox(
        "Chọn phim",
        catalog.index.tolist(),
        format_func=lambda x: f'{catalog.loc[x, "title"]} — ID {x}',
    )

    model = get_model("Content-based")

    st.caption(
        "Cosine TF-IDF trên thể loại; "
        "dữ liệu không có cốt truyện hay diễn viên."
    )

    if model is not None:
        cards(model.similar(selected, n))

else:
    st.subheader("Kết quả thực nghiệm")

    result_files = [
        ("03_test_results", True),
        ("05_test_results", True),
        ("04_bpr", False),
        ("thu_nghiem_mf", False),
    ]

    tables = []
    unverified = []

    required_columns = [
        "model",
        "Precision@10",
        "Recall@10",
        "F1@10",
        "NDCG@10",
    ]

    for stem, require_metadata in result_files:
        csv_path = ROOT / "reports/tables" / f"{stem}.csv"
        meta_path = ROOT / "reports/tables" / f"{stem}.json"

        if not csv_path.exists():
            st.warning(f"Chưa có bảng kết quả: {csv_path.name}")
            continue

        try:
            verified = False

            if meta_path.exists():
                record = json.loads(
                    meta_path.read_text(encoding="utf-8")
                )

                if (
                    record.get("train_fingerprint") != train_hash
                    or record.get("test_fingerprint") != test_hash
                ):
                    st.warning(
                        f"{stem}: kết quả không khớp dữ liệu hiện tại. "
                        "Hãy chạy lại notebook tương ứng."
                    )
                    continue

                verified = True

            elif require_metadata:
                st.warning(
                    f"{stem}: thiếu file JSON kiểm tra train/test."
                )
                continue

            table = pd.read_csv(csv_path)

            missing = [
                col for col in required_columns
                if col not in table.columns
            ]
            if missing:
                st.warning(
                    f"{csv_path.name}: thiếu cột {', '.join(missing)}."
                )
                continue

            table["Kiểm tra dữ liệu"] = (
                "Khớp train/test" if verified else "Chưa có metadata"
            )

            if not verified:
                unverified.append(csv_path.name)

            tables.append(table)

        except (ValueError, OSError, TypeError) as exc:
            st.warning(f"Không đọc được {csv_path.name}: {exc}")

    if unverified:
        st.warning(
            "Các bảng sau chưa có metadata xác nhận train/test: "
            + ", ".join(unverified)
            + ". Chỉ so sánh khi các notebook đã chạy trên cùng "
              "train.csv và test.csv với những mô hình còn lại."
        )

    if tables:
        results = pd.concat(
            tables,
            ignore_index=True,
        ).drop_duplicates("model", keep="last")

        st.dataframe(results, hide_index=True)

        for metric in required_columns[1:]:
            st.markdown(f"**{metric}**")
            st.bar_chart(
                results.set_index("model")[[metric]]
            )

        st.download_button(
            "Tải bảng kết quả CSV",
            results.to_csv(index=False).encode("utf-8-sig"),
            file_name="movie_recommendation_results.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Chạy các notebook 03, 04, 05 và 06 "
            "để tạo bảng kết quả."
        )

    st.caption(
        "Rating ≥ 4 là relevant; loại phim đã đánh giá trong train; "
        "chỉ đánh giá user/phim có trong train; lấy trung bình "
        "theo người dùng có phim relevant."
    )