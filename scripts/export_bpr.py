"""Run in the SAME kernel after original notebook 04: %run -i ../scripts/export_bpr.py"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
from src.recommenders import fingerprint, FactorModel, evaluate
from src.experiments import save_results
_required = ['bpr_model', 'user_ids', 'movie_ids', 'train_data']
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(f'Chạy xong notebook 04 trong cùng kernel trước. Thiếu {_missing}')
_disk_train = pd.read_csv(_root/'data/processed/train.csv')
if fingerprint(train_data) != fingerprint(_disk_train):
    raise ValueError('train_data trong kernel khác train.csv; không xuất mô hình sai dữ liệu.')
_path = _root/'models/bpr_model.npz'
_path.parent.mkdir(exist_ok=True)
np.savez_compressed(_path, kind='bpr', user_ids=user_ids, movie_ids=movie_ids,
                    user_factors=bpr_model['user_factors'], movie_factors=bpr_model['movie_factors'],
                    train_fingerprint=fingerprint(train_data))
_loaded = FactorModel().fit_checkpoint(train_data, _path)
_test = pd.read_csv(_root/'data/processed/test.csv')
_metrics, _ = evaluate(_loaded, _test)
save_results(_root, '04_bpr_export_results', pd.DataFrame([{'model':'BPR-MF', **_metrics}]),
             train_data, _test, 'Model exported from original notebook 04; re-evaluated with common tie-breaking.')
print('Đã lưu:', _path, '\n', _metrics)
