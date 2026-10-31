from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def doc_du_lieu(thu_muc_goc):
    thu_muc_goc = Path(thu_muc_goc)
    train_data = pd.read_csv(thu_muc_goc / 'data/processed/train.csv')
    test_data = pd.read_csv(thu_muc_goc / 'data/processed/test.csv')
    phim = pd.read_csv(
        thu_muc_goc / 'data/raw/movies.dat',
        sep='::', engine='python', encoding='latin-1',
        names=['movie_id', 'title', 'genres'],
    )
    return train_data, test_data, phim


def chia_du_lieu(danh_gia):
    return train_test_split(danh_gia, test_size=0.2, random_state=42)


def chuan_bi_du_lieu(train_data):
    ds_nguoi_dung = np.sort(train_data['user_id'].unique())
    ds_phim = np.sort(train_data['movie_id'].unique())

    # Ánh xạ ID của người dùng và phim
    vi_tri_nguoi_dung = {}
    for vi_tri, ma_nguoi_dung in enumerate(ds_nguoi_dung):
        vi_tri_nguoi_dung[ma_nguoi_dung] = vi_tri

    vi_tri_phim = {}
    for vi_tri, ma_phim in enumerate(ds_phim):
        vi_tri_phim[ma_phim] = vi_tri

    phim_da_xem = {}
    for ma_nguoi_dung, cac_dong in train_data.groupby('user_id'):
        phim_da_xem[ma_nguoi_dung] = set(cac_dong['movie_id'])

    danh_gia_tich_cuc = train_data[train_data['rating'] >= 4]
    so_luot_thich = danh_gia_tich_cuc.groupby('movie_id').size()
    diem_pho_bien = so_luot_thich.reindex(ds_phim, fill_value=0).to_numpy()

    return {
        'ds_nguoi_dung': ds_nguoi_dung,
        'ds_phim': ds_phim,
        'vi_tri_nguoi_dung': vi_tri_nguoi_dung,
        'vi_tri_phim': vi_tri_phim,
        'phim_da_xem': phim_da_xem,
        'diem_pho_bien': diem_pho_bien,
    }