"""Reproduce ONLY the export/split of original notebook 01, no extra filtering."""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.recommenders import validate, fingerprint

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    raw, output = root/'data/raw', root/'data/processed'
    ratings = pd.read_csv(raw/'ratings.dat', sep='::', engine='python',
                          names=['user_id','movie_id','rating','timestamp'], encoding='latin-1')
    validate(ratings)
    train, test = train_test_split(ratings, test_size=.2, random_state=42)
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output/'train.csv', index=False)
    test.to_csv(output/'test.csv', index=False)
    summary = dict(split='train_test_split(test_size=0.2, random_state=42), identical to notebook 01',
                   train_rows=len(train), test_rows=len(test), users=train.user_id.nunique(),
                   movies=train.movie_id.nunique(), train_fingerprint=fingerprint(train))
    (output/'split_info.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(summary)
