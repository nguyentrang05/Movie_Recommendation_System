from pathlib import Path
import json
import pandas as pd
import streamlit as st
from src.recommenders import (load_data, fingerprint, Popularity, CollaborativeFiltering,
                               ContentBased, FactorModel)
ROOT = Path(__file__).resolve().parent
TRAIN = ROOT/'data/processed/train.csv'
TEST = ROOT/'data/processed/test.csv'
MOVIES = ROOT/'data/raw/movies.dat'
st.set_page_config(page_title='Movie Compass', page_icon='🎬', layout='wide')
st.markdown('''<style>
.stApp{background:#f6f8fc} h1,h2,h3{color:#152e49}
div[data-testid="stMetric"]{background:white;padding:18px;border:1px solid #e3e9f2;border-radius:14px}
</style>''', unsafe_allow_html=True)
st.caption('MOVIE COMPASS   /   MOVIELENS 1M')
st.title('Một bộ phim hay, đúng với bạn.')
st.write('Gợi ý từ lịch sử đánh giá, người xem tương tự và thể loại yêu thích.')
if not all(p.exists() for p in [TRAIN, TEST, MOVIES]):
    st.info('Chưa đủ dữ liệu. Chép movies.dat/ratings.dat/users.dat vào data/raw rồi chạy:')
    st.code('python prepare_data.py')
    st.stop()
version = tuple(p.stat().st_mtime_ns for p in [TRAIN, TEST, MOVIES])
@st.cache_data
def data(version):
    train, test, movies = load_data(ROOT)
    return train, test, movies, fingerprint(train), fingerprint(test)
try:
    train, test, movies, train_hash, test_hash = data(version)
except (ValueError, OSError) as exc:
    st.error(str(exc)); st.stop()
@st.cache_resource(max_entries=6)
def fitted(version, method, k=20, checkpoint_stamp=0):
    train, _, movies, _, _ = data(version)
    if method == 'Popularity': return Popularity().fit(train)
    if method == 'User-CF': return CollaborativeFiltering('user', k).fit(train)
    if method == 'Item-CF': return CollaborativeFiltering('item', k).fit(train)
    if method == 'Content-based': return ContentBased().fit(train, movies)
    path = ROOT/'models'/('bpr_model.npz' if method == 'BPR-MF' else 'mf_model.npz')
    return FactorModel().fit_checkpoint(train, path)

catalog = movies[movies.movie_id.isin(train.movie_id)].set_index('movie_id')
page = st.sidebar.radio('KHÁM PHÁ', ['Dành cho bạn', 'Tìm kiếm', 'Phim tương tự', 'Thực nghiệm'])
n = st.sidebar.slider('Số phim', 1, 20, 10)
st.sidebar.caption('Dữ liệu thật • Split ngẫu nhiên 80/20 theo notebook 01 của nhóm.')
cols = st.columns(3)
for col, label, count in zip(cols, ['Phim trong train', 'Người xem', 'Đánh giá train'],
                             [len(catalog), train.user_id.nunique(), len(train)]):
    col.metric(label, f'{count:,}')

def cards(results):
    if not results:
        st.info('Không tìm thấy phim phù hợp. Hãy thử tên phim hoặc thể loại khác.')
        return
    columns = st.columns(2)
    for rank, (mid, score) in enumerate(results):
        movie = catalog.loc[mid]
        with columns[rank%2]:
            with st.container(border=True):
                st.caption(f'GỢI Ý {rank+1:02}   •   MOVIE {mid}')
                st.subheader(str(movie.title))
                st.write(str(movie.genres).replace('|', ' · '))
                st.caption(f'Điểm xếp hạng: {score:.4f}')

def get_model(method, k=20):
    stamp = 0
    if method in ['BPR-MF','Bias MF']:
        name = 'bpr_model.npz' if method == 'BPR-MF' else 'mf_model.npz'
        path = ROOT/'models'/name
        if not path.exists():
            notebook = '04_Matrix_Factorization.ipynb' if method == 'BPR-MF' else 'thu_nghiem.ipynb'
            script = 'export_bpr.py' if method == 'BPR-MF' else 'export_mf.py'
            st.info(f'Chạy xong {notebook}, thêm cell cuối và chạy trong cùng kernel:')
            st.code(f'%run -i ../scripts/{script}')
            st.caption('Sau đó tải lại trang. Các mô hình còn lại có thể dùng ngay.')
            return None
        stamp = path.stat().st_mtime_ns
    try:
        with st.spinner('Đang chuẩn bị gợi ý…'):
            return fitted(version, method, k, stamp)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f'Không nạp được mô hình: {exc}')
        return None

if page == 'Dành cho bạn':
    left, right = st.columns(2)
    with left:
        method = st.selectbox('Phương pháp', ['Popularity','User-CF','Item-CF','Content-based','BPR-MF','Bias MF'])
    with right:
        user = st.selectbox('Người xem (UserID gốc)', sorted(train.user_id.unique()))
    k = 20
    if method in ['User-CF','Item-CF']:
        config_path = ROOT/'models/cf_config.json'
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config.get('train_fingerprint') == train_hash:
                k = int(config['neighbors']['user' if method == 'User-CF' else 'item'])
                st.caption(f'k = {k}, được chọn bằng validation nội bộ của notebook 03.')
            else: st.warning('Cấu hình CF thuộc train cũ. Dùng k=20; chạy lại notebook 03 để chọn k.')
        else: st.caption('Dùng k=20 mặc định. Chạy notebook 03 để chọn k bằng validation.')
        if st.checkbox('Thử số láng giềng khác (chỉ thay đổi demo)'):
            k = st.select_slider('k', options=[10,20,40], value=k if k in [10,20,40] else 20)
    with st.expander('Lịch sử đánh giá trong train'):
        history = train[train.user_id == user].merge(movies, on='movie_id', how='left')
        st.dataframe(history[['title','genres','rating']], hide_index=True)
    model = get_model(method, k)
    if model is not None:
        st.caption('Loại phim đã đánh giá trong train. Điểm không phải xác suất; chỉ so trong cùng phương pháp. CF/content thiếu tín hiệu sẽ dùng popularity.')
        cards(model.ranked(model.scores(user), n, model.seen.get(user, set())))
elif page == 'Tìm kiếm':
    with st.form('search'):
        query = st.text_input('Tên phim hoặc thể loại tiếng Anh', placeholder='Toy Story, Action, Sci-Fi…')
        submit = st.form_submit_button('Tìm phim', type='primary')
    if submit:
        model = get_model('Content-based')
        if model is not None: cards(model.search(query, n))
    st.caption('Không cần lịch sử người dùng. Tìm trong catalog phim thuộc train.')
elif page == 'Phim tương tự':
    selected = st.selectbox('Chọn phim', catalog.index.tolist(),
                            format_func=lambda x:f'{catalog.loc[x,"title"]} — ID {x}')
    model = get_model('Content-based')
    st.caption('Cosine TF-IDF trên thể loại; dữ liệu không có cốt truyện hay diễn viên.')
    if model is not None: cards(model.similar(selected, n))
else:
    st.subheader('So sánh trên cùng tập test')
    tables = []
    for stem in ['03_test_results','05_test_results','04_bpr_export_results','thu_nghiem_export_results']:
        csv, meta = ROOT/f'reports/tables/{stem}.csv', ROOT/f'reports/tables/{stem}.json'
        if not (csv.exists() and meta.exists()): continue
        record = json.loads(meta.read_text(encoding='utf-8'))
        if record.get('train_fingerprint') != train_hash or record.get('test_fingerprint') != test_hash:
            st.warning(f'{stem}: dữ liệu đã đổi, cần chạy lại notebook.'); continue
        tables.append(pd.read_csv(csv))
    if tables:
        results = pd.concat(tables, ignore_index=True).drop_duplicates('model')
        st.dataframe(results, hide_index=True)
        st.bar_chart(results.set_index('model')[['Precision@10','Recall@10','NDCG@10']])
        st.download_button('Tải bảng kết quả CSV', results.to_csv(index=False).encode('utf-8-sig'),
                           file_name='movie_recommendation_results.csv', mime='text/csv')
    else: st.info('Chạy python run_experiments.py để tạo kết quả 03/05.')
    st.caption('Rating ≥4 là relevant; bỏ phim đã đánh giá; warm-start; macro-average theo user có relevant. Số liệu chỉ hiện khi khớp train/test hiện tại.')
    st.info('BPR-MF/Bias MF xuất hiện trong bảng sau khi chạy script export ở kernel notebook tương ứng. Không dùng lại số liệu output cũ như kết quả chạy mới.')
