"""Run in the SAME kernel after thu_nghiem: %run -i ../scripts/export_mf.py"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
from src.recommenders import fingerprint, FactorModel, evaluate
from src.experiments import save_results
_required = ['mf_model', 'user_ids', 'movie_ids', 'train_data']
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(f'Chạy xong thu_nghiem trong cùng kernel trước. Thiếu {_missing}')
_disk_train = pd.read_csv(_root/'data/processed/train.csv')
if fingerprint(train_data) != fingerprint(_disk_train):
    raise ValueError('train_data khác train.csv; không xuất mô hình sai dữ liệu.')
_path = _root/'models/mf_model.npz'
_path.parent.mkdir(exist_ok=True)
np.savez_compressed(_path, kind='mf', user_ids=user_ids, movie_ids=movie_ids,
                    user_factors=mf_model['user_factors'], movie_factors=mf_model['movie_factors'],
                    global_mean=mf_model['global_mean'], user_bias=mf_model['user_bias'],
                    movie_bias=mf_model['movie_bias'], train_fingerprint=fingerprint(train_data))
_loaded = FactorModel().fit_checkpoint(train_data, _path)
_test = pd.read_csv(_root/'data/processed/test.csv')
_metrics, _ = evaluate(_loaded, _test)
save_results(_root, 'thu_nghiem_export_results', pd.DataFrame([{'model':'Bias MF', **_metrics}]),
             train_data, _test, 'Exported from original thu_nghiem; re-evaluated with common tie-breaking.')
print('Đã lưu:', _path, '\n', _metrics)
