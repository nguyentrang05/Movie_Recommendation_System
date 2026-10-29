"""Run new 03/05 code cells sequentially without a notebook server.
Each notebook gets a fresh namespace. Captured stdout becomes saved notebook output.
Original notebooks 01/02/04/thu_nghiem are never executed or overwritten here.
"""
from pathlib import Path
import contextlib
import io
import json
import os
import sys

class Tee(io.StringIO):
    def write(self, text):
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        return super().write(text)

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    os.chdir(root / 'notebooks')
    for name in ['03_collaborative_filtering.ipynb', '05_content_based.ipynb']:
        path = root/'notebooks'/name
        notebook = json.loads(path.read_text(encoding='utf-8'))
        print('Running', name, flush=True)
        namespace = {'__name__': '__main__'}
        count = 0
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            count += 1
            buffer = Tee()
            with contextlib.redirect_stdout(buffer):
                exec(compile(''.join(cell['source']), f'{name}:cell{count}', 'exec'), namespace)
            cell['execution_count'] = count
            cell['outputs'] = [{'output_type':'stream','name':'stdout','text':buffer.getvalue().splitlines(True)}]
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
        print('Completed',name,flush=True)
