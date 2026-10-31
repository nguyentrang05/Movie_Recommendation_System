from pathlib import Path
import joblib


def luu_mo_hinh(mo_hinh, duong_dan):
    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mo_hinh, duong_dan)


def doc_mo_hinh(duong_dan):
    return joblib.load(duong_dan)


def luu_bang_ket_qua(bang_ket_qua, duong_dan):
    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    bang_ket_qua.to_csv(duong_dan, index=False)
