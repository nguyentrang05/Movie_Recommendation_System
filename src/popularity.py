from .du_lieu import chuan_bi_du_lieu
from .goi_y import xep_hang_phim


def huan_luyen_popularity(tap_huan_luyen):
    mo_hinh = chuan_bi_du_lieu(tap_huan_luyen)
    return mo_hinh


def goi_y_popularity(mo_hinh, ma_nguoi_dung, so_phim=10):
    diem = mo_hinh['diem_pho_bien']
    return xep_hang_phim(mo_hinh, ma_nguoi_dung, diem, so_phim)
