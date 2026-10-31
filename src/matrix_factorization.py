import numpy as np
from .du_lieu import chuan_bi_du_lieu
from .goi_y import xep_hang_phim


def lay_phim_chua_xem(vi_tri_nguoi_dung, phim_da_xem, so_luong_phim, bo_ngau_nhien):
    while True:
        vi_tri_phim = int(bo_ngau_nhien.integers(0, so_luong_phim))
        if vi_tri_phim not in phim_da_xem[vi_tri_nguoi_dung]:
            return vi_tri_phim

def huan_luyen_mf(train_data, tiem_an=50, lr=0.005,
                  l2=0.02, epoch=15, random_state=42):
    mo_hinh = chuan_bi_du_lieu(train_data)
    so_nguoi_dung = len(mo_hinh['ds_nguoi_dung'])
    so_phim = len(mo_hinh['ds_phim'])
    bo_ngau_nhien = np.random.default_rng(random_state)

    cac_nguoi_dung = train_data['user_id'].map(mo_hinh['vi_tri_nguoi_dung']).to_numpy()
    cac_phim = train_data['movie_id'].map(mo_hinh['vi_tri_phim']).to_numpy()
    cac_rating = train_data['rating'].to_numpy(dtype=np.float32)

    diem_trung_binh = float(cac_rating.mean())
    bias_nguoi_dung = np.zeros(so_nguoi_dung, dtype=np.float32)
    bias_phim = np.zeros(so_phim, dtype=np.float32)
    vector_nguoi_dung = bo_ngau_nhien.normal(
        0, 0.05, (so_nguoi_dung, tiem_an)
    ).astype(np.float32)
    vector_phim = bo_ngau_nhien.normal(
        0, 0.05, (so_phim, tiem_an)
    ).astype(np.float32)

    lich_su_rmse = []
    for vong_lap in range(epoch):
        tong_binh_phuong_sai_so = 0.0
        thu_tu = bo_ngau_nhien.permutation(len(cac_rating))

        for vi_tri_dong in thu_tu:
            nguoi_dung = cac_nguoi_dung[vi_tri_dong]
            phim = cac_phim[vi_tri_dong]
            rating_that = cac_rating[vi_tri_dong]
            vector_user = vector_nguoi_dung[nguoi_dung].copy()
            vector_movie = vector_phim[phim].copy()

            rating_du_doan = (
                diem_trung_binh + bias_nguoi_dung[nguoi_dung] + bias_phim[phim]
                + np.dot(vector_user, vector_movie)
            )
            sai_so = rating_that - rating_du_doan
            tong_binh_phuong_sai_so += sai_so ** 2

            bias_nguoi_dung[nguoi_dung] += lr * (
                sai_so - l2 * bias_nguoi_dung[nguoi_dung]
            )
            bias_phim[phim] += lr * (sai_so - l2 * bias_phim[phim])
            vector_nguoi_dung[nguoi_dung] += lr * (
                sai_so * vector_movie - l2 * vector_user
            )
            vector_phim[phim] += lr * (
                sai_so * vector_user - l2 * vector_movie
            )

        rmse = float(np.sqrt(tong_binh_phuong_sai_so / len(cac_rating)))
        lich_su_rmse.append(rmse)
        print(f'Epoch {vong_lap + 1}/{epoch} | Online train RMSE: {rmse:.4f}')

    mo_hinh['loai'] = 'mf'
    mo_hinh['diem_trung_binh'] = diem_trung_binh
    mo_hinh['bias_nguoi_dung'] = bias_nguoi_dung
    mo_hinh['bias_phim'] = bias_phim
    mo_hinh['vector_nguoi_dung'] = vector_nguoi_dung
    mo_hinh['vector_phim'] = vector_phim
    mo_hinh['lich_su'] = lich_su_rmse
    return mo_hinh


def huan_luyen_bpr(train_data, tiem_an=50, lr=0.01,
                   l2=0.01, epoch=15, random_state=42):
    mo_hinh = chuan_bi_du_lieu(train_data)
    so_nguoi_dung = len(mo_hinh['ds_nguoi_dung'])
    so_phim = len(mo_hinh['ds_phim'])
    bo_ngau_nhien = np.random.default_rng(random_state)

    vector_nguoi_dung = bo_ngau_nhien.normal(
        0, 0.05, (so_nguoi_dung, tiem_an)
    ).astype(np.float32)
    vector_phim = bo_ngau_nhien.normal(
        0, 0.05, (so_phim, tiem_an)
    ).astype(np.float32)

    # Chuyển ID thành chỉ số ma trận để dùng trong Epoch huấn luyện.
    danh_gia_tich_cuc = train_data[train_data['rating'] >= 4]
    if danh_gia_tich_cuc.empty:
        raise ValueError('BPR cần ít nhất một rating >= 4.')
    cac_nguoi_dung = danh_gia_tich_cuc['user_id'].map(mo_hinh['vi_tri_nguoi_dung']).to_numpy()
    cac_phim_thich = danh_gia_tich_cuc['movie_id'].map(mo_hinh['vi_tri_phim']).to_numpy()

    phim_da_xem_theo_vi_tri = {}
    for ma_nguoi_dung, cac_phim in mo_hinh['phim_da_xem'].items():
        vi_tri = mo_hinh['vi_tri_nguoi_dung'][ma_nguoi_dung]
        phim_da_xem_theo_vi_tri[vi_tri] = set()
        for ma_phim in cac_phim:
            phim_da_xem_theo_vi_tri[vi_tri].add(mo_hinh['vi_tri_phim'][ma_phim])

    for nguoi_dung in np.unique(cac_nguoi_dung):
        if len(phim_da_xem_theo_vi_tri[nguoi_dung]) == so_phim:
            raise ValueError('User tích cực đã đánh giá mọi phim, không còn mẫu âm.')

    lich_su_loss = []
    for vong_lap in range(epoch):
        tong_loss = 0.0
        thu_tu = bo_ngau_nhien.permutation(len(danh_gia_tich_cuc))

        for vi_tri_dong in thu_tu:
            nguoi_dung = int(cac_nguoi_dung[vi_tri_dong])
            phim_thich = int(cac_phim_thich[vi_tri_dong])
            phim_chua_xem = lay_phim_chua_xem(
                nguoi_dung, phim_da_xem_theo_vi_tri, so_phim, bo_ngau_nhien
            )

            # Giữ bản sao để ba cập nhật cùng dùng trọng số trước khi thay đổi.
            vector_user = vector_nguoi_dung[nguoi_dung].copy()
            vector_thich = vector_phim[phim_thich].copy()
            vector_chua_xem = vector_phim[phim_chua_xem].copy()

            chenh_lech_diem = np.dot(vector_user, vector_thich - vector_chua_xem)
            # clip tránh số quá lớn khi tính exp; đây là hệ số gradient BPR.
            he_so = 1 / (1 + np.exp(np.clip(chenh_lech_diem, -20, 20)))

            vector_nguoi_dung[nguoi_dung] += lr * (
                he_so * (vector_thich - vector_chua_xem) - l2 * vector_user
            )
            vector_phim[phim_thich] += lr * (
                he_so * vector_user - l2 * vector_thich
            )
            vector_phim[phim_chua_xem] += lr * (
                -he_so * vector_user - l2 * vector_chua_xem
            )

            loss = np.logaddexp(0, -chenh_lech_diem)
            loss += 0.5 * l2 * (
                np.sum(vector_user ** 2) + np.sum(vector_thich ** 2)
                + np.sum(vector_chua_xem ** 2)
            )
            tong_loss += loss

        loss_trung_binh = float(tong_loss / len(danh_gia_tich_cuc))
        lich_su_loss.append(loss_trung_binh)
        print(f'Epoch {vong_lap + 1}/{epoch} | BPR loss: {loss_trung_binh:.4f}')

    mo_hinh['loai'] = 'bpr'
    mo_hinh['vector_nguoi_dung'] = vector_nguoi_dung
    mo_hinh['vector_phim'] = vector_phim
    mo_hinh['lich_su'] = lich_su_loss
    return mo_hinh

def goi_y_mf(mo_hinh, ma_nguoi_dung, so_phim=10):
    if ma_nguoi_dung not in mo_hinh['vi_tri_nguoi_dung']:
        diem = mo_hinh['diem_pho_bien']
    else:
        vi_tri = mo_hinh['vi_tri_nguoi_dung'][ma_nguoi_dung]
        vector_user = mo_hinh['vector_nguoi_dung'][vi_tri]
        diem = mo_hinh['vector_phim'] @ vector_user
        if mo_hinh['loai'] == 'mf':
            diem = (diem + mo_hinh['diem_trung_binh']
                    + mo_hinh['bias_nguoi_dung'][vi_tri] + mo_hinh['bias_phim'])
    return xep_hang_phim(mo_hinh, ma_nguoi_dung, diem, so_phim)
