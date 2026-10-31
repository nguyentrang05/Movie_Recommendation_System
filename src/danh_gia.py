import numpy as np
import pandas as pd


def danh_gia_mo_hinh(ham_goi_y, mo_hinh, tap_kiem_tra, k=10):
    if k < 1:
        raise ValueError('k phải >= 1.')

    du_lieu_hop_le = tap_kiem_tra[
        tap_kiem_tra['user_id'].isin(mo_hinh['ds_nguoi_dung'])
        & tap_kiem_tra['movie_id'].isin(mo_hinh['ds_phim'])
    ]
    danh_gia_tich_cuc = du_lieu_hop_le[du_lieu_hop_le['rating'] >= 4]
    cac_ket_qua = []

    for ma_nguoi_dung, cac_dong in danh_gia_tich_cuc.groupby('user_id'):
        phim_lien_quan = set(cac_dong['movie_id'])
        bang_goi_y = ham_goi_y(mo_hinh, ma_nguoi_dung, k)
        phim_goi_y = bang_goi_y['movie_id'].tolist()
        so_phim_dung = len(set(phim_goi_y) & phim_lien_quan)

        precision = so_phim_dung / k
        recall = so_phim_dung / len(phim_lien_quan)
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        dcg = 0.0
        for thu_hang, ma_phim in enumerate(phim_goi_y):
            if ma_phim in phim_lien_quan:
                dcg += 1 / np.log2(thu_hang + 2)

        idcg = 0.0
        so_phim_ly_tuong = min(k, len(phim_lien_quan))
        for thu_hang in range(so_phim_ly_tuong):
            idcg += 1 / np.log2(thu_hang + 2)
        ndcg = dcg / idcg

        cac_ket_qua.append({
            'user_id': ma_nguoi_dung,
            f'Precision@{k}': precision,
            f'Recall@{k}': recall,
            f'F1@{k}': f1,
            f'NDCG@{k}': ndcg,
        })

    ten_cot = ['user_id', f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'NDCG@{k}']
    ket_qua_tung_user = pd.DataFrame(cac_ket_qua, columns=ten_cot)
    ket_qua_chung = ket_qua_tung_user[ten_cot[1:]].mean().to_dict()
    ket_qua_chung['evaluated_users'] = len(ket_qua_tung_user)
    ket_qua_chung['evaluated_rows'] = len(du_lieu_hop_le)
    ket_qua_chung['dropped_rows'] = len(tap_kiem_tra) - len(du_lieu_hop_le)
    return ket_qua_chung, ket_qua_tung_user
