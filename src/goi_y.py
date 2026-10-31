import numpy as np
import pandas as pd


def xep_hang_phim(mo_hinh, ma_nguoi_dung, diem, so_phim=10):
    if so_phim < 1:
        raise ValueError('so_phim phải >= 1.')
    if not np.isfinite(diem).all():
        raise ValueError('Điểm chứa NaN/Inf; hãy kiểm tra việc huấn luyện.')

    bang_diem = pd.DataFrame({
        'movie_id': mo_hinh['ds_phim'],
        'score': diem,
        'popularity': mo_hinh['diem_pho_bien'],
    })

    phim_da_xem = mo_hinh['phim_da_xem'].get(ma_nguoi_dung, set())
    bang_diem = bang_diem[~bang_diem['movie_id'].isin(phim_da_xem)]

    # Điểm cao đứng trước. Bằng điểm thì xét độ phổ biến, sau đó ID phim.
    bang_diem = bang_diem.sort_values(
        ['score', 'popularity', 'movie_id'], ascending=[False, False, True]
    )
    return bang_diem[['movie_id', 'score']].head(so_phim).reset_index(drop=True)
