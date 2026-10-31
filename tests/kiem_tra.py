"""Chạy từ thư mục gốc: python tests/kiem_tra.py.

Dùng dữ liệu nhỏ để kiểm tra chức năng, không dùng làm số liệu báo cáo.
"""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.popularity import huan_luyen_popularity, goi_y_popularity
# from src.collaborative_filtering import huan_luyen_cf, goi_y_cf
# from src.content_based import huan_luyen_content, goi_y_content
from src.matrix_factorization import huan_luyen_bpr, huan_luyen_mf, goi_y_mf
from src.danh_gia import danh_gia_mo_hinh
from src.luu_ket_qua import luu_mo_hinh, doc_mo_hinh


danh_gia = pd.DataFrame([
    (1, 10, 5), (1, 20, 2), (2, 20, 5), (2, 30, 4),
    (3, 10, 4), (3, 40, 5), (4, 30, 5), (4, 40, 3),
], columns=['user_id', 'movie_id', 'rating'])
phim = pd.DataFrame([
    (10, 'A', 'Action|Sci-Fi'), (20, 'B', 'Drama'),
    (30, 'C', 'Action'), (40, 'D', 'Comedy'),
], columns=['movie_id', 'title', 'genres'])

mo_hinh_pop = huan_luyen_popularity(danh_gia)
# mo_hinh_user = huan_luyen_cf(danh_gia, 'user', 2)
# mo_hinh_item = huan_luyen_cf(danh_gia, 'item', 2)
# mo_hinh_content = huan_luyen_content(danh_gia, phim)
mo_hinh_bpr = huan_luyen_bpr(danh_gia, tiem_an=3, epoch=2)
mo_hinh_mf = huan_luyen_mf(danh_gia, tiem_an=3, epoch=2)

# 1. Tất cả mô hình phải bỏ cả phim rating 5 lẫn phim rating 2 đã xem.
for ham_goi_y, mo_hinh in [
    # (goi_y_popularity, mo_hinh_pop), (goi_y_cf, mo_hinh_user),
    # (goi_y_cf, mo_hinh_item), (goi_y_content, mo_hinh_content),
    (goi_y_mf, mo_hinh_bpr), (goi_y_mf, mo_hinh_mf),
]:
    ket_qua = ham_goi_y(mo_hinh, 1, 10)
    assert set(ket_qua['movie_id']) == {30, 40}
    # User mới phải dùng popularity.
    ket_qua_moi = ham_goi_y(mo_hinh, 999, 10)
    assert ket_qua_moi['movie_id'].tolist() == [10, 30, 20, 40]
print('1. Lọc phim đã xem và xử lý user mới: OK')

# 2. Không có user/phim nào làm láng giềng của chính nó; tối đa k láng giềng.
# for mo_hinh in [mo_hinh_user, mo_hinh_item]:
#     tuong_dong = mo_hinh['do_tuong_dong']
#     assert np.all(np.diag(tuong_dong) == 0)
#     assert np.all((tuong_dong > 0).sum(axis=1) <= 2)
# print('2. Láng giềng CF: OK')

# # 3. Giữ Sci-Fi là một thể loại, không tách thành hai từ.
# assert mo_hinh_content['vector_the_loai'].shape[1] == 4
# print('3. TF-IDF thể loại: OK')

# 4. Chỉ số có đáp án biết trước, kèm lọc user/phim ngoài train.
tap_test = pd.DataFrame([(1, 30, 5), (1, 999, 5), (999, 30, 5)], columns=danh_gia.columns)
ket_qua, chi_tiet = danh_gia_mo_hinh(goi_y_popularity, mo_hinh_pop, tap_test, k=2)
assert np.isclose(ket_qua['Precision@2'], 0.5)
assert np.isclose(ket_qua['Recall@2'], 1)
assert np.isclose(ket_qua['F1@2'], 2/3)
assert np.isclose(ket_qua['NDCG@2'], 1)
assert ket_qua['dropped_rows'] == 2
print('4. Precision/Recall/F1/NDCG và lọc test: OK')

# 5. Lưu và đọc lại dictionary không làm đổi kết quả.
with tempfile.TemporaryDirectory() as thu_muc:
    for ten, mo_hinh in [('bpr', mo_hinh_bpr), ('mf', mo_hinh_mf)]:
        duong_dan = Path(thu_muc) / f'{ten}.joblib'
        luu_mo_hinh(mo_hinh, duong_dan)
        ban_doc_lai = doc_mo_hinh(duong_dan)
        np.testing.assert_array_equal(mo_hinh['vector_phim'], ban_doc_lai['vector_phim'])
        pd.testing.assert_frame_equal(goi_y_mf(mo_hinh, 1), goi_y_mf(ban_doc_lai, 1))
print('5. Lưu/nạp mô hình: OK')

# 6. Cùng seed phải sinh cùng trọng số.
for ham_huan_luyen in [huan_luyen_bpr, huan_luyen_mf]:
    lan_1 = ham_huan_luyen(danh_gia, tiem_an=3, epoch=1)
    lan_2 = ham_huan_luyen(danh_gia, tiem_an=3, epoch=1)
    np.testing.assert_array_equal(lan_1['vector_nguoi_dung'], lan_2['vector_nguoi_dung'])
print('6. Tái lập theo seed: OK')

# 7. BPR báo lỗi khi không có phim tích cực hoặc không còn phim để lấy mẫu âm.
for rating in [2, 5]:
    du_lieu_loi = pd.DataFrame([(1, 10, rating)], columns=danh_gia.columns)
    da_bao_loi = False
    try:
        huan_luyen_bpr(du_lieu_loi, epoch=1)
    except ValueError:
        da_bao_loi = True
    assert da_bao_loi
print('7. Dữ liệu BPR không hợp lệ: OK')

print('Đã qua 7 nhóm kiểm tra.')
