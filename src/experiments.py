"""Shared exports consumed by notebooks and app, with dataset provenance."""
from pathlib import Path
import json
from datetime import datetime, timezone
from .recommenders import fingerprint

def save_results(root, stem, results, train, test, note):
    root = Path(root)
    out = root / 'reports/tables'
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / f'{stem}.csv', index=False)
    metadata = dict(train_fingerprint=fingerprint(train), test_fingerprint=fingerprint(test),
                    created_utc=datetime.now(timezone.utc).isoformat(), note=note)
    (out / f'{stem}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
