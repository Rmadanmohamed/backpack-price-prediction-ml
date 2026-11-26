"""Regression checks using synthetic fixtures only; no original data required."""
import json
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'backpack-price-prediction.ipynb'
CELLS = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']

def source(index):
    return ''.join(CELLS[index]['source'])

def run(namespace, *indices):
    for i in indices:
        exec(compile(source(i), f'{NOTEBOOK.name}:cell-{i}', 'exec'), namespace)

class NotebookTests(unittest.TestCase):
    def test_python_cells_compile(self):
        for i, cell in enumerate(CELLS):
            if cell['cell_type'] != 'code':
                continue
            # Notebook magics are handled by IPython, not the Python parser.
            code = '\n'.join(line for line in source(i).splitlines() if not line.lstrip().startswith(('%', '!')))
            with self.subTest(cell=i):
                compile(code, f'{NOTEBOOK.name}:cell-{i}', 'exec')

    def test_unseen_categories_and_submission_ids(self):
        import pandas as pd
        import numpy as np
        import tempfile, os
        from pathlib import Path
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.linear_model import ElasticNet
        from sklearn.model_selection import train_test_split
        train = pd.DataFrame({'id': range(20), 'Style': ['a', 'b'] * 10, 'Weight': np.arange(20, dtype=float), 'Price': np.arange(20, dtype=float) + 10})
        test = pd.DataFrame({'id': [100, 101], 'Style': ['unseen', 'a'], 'Weight': [5., np.nan]})
        submission = pd.DataFrame({'id': [101, 100], 'Price': [0., 0.]})
        ns = dict(pd=pd, np=np, train=train, test=test, submission=submission, SimpleImputer=SimpleImputer, OrdinalEncoder=OrdinalEncoder, ElasticNet=ElasticNet, train_test_split=train_test_split)
        run(ns, 8, 14, 18, 19, 20, 26)
        enc = ns['model'].named_steps['preprocessor'].named_transformers_['categorical'].named_steps['encode']
        self.assertNotIn('unseen', enc.categories_[0])
        self.assertTrue(np.isfinite(ns['pred']).all())
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                run(ns, 27)
                result = pd.read_csv('submission.csv')
                self.assertEqual(result['id'].tolist(), [101, 100])
                np.testing.assert_allclose(result['Price'], ns['pred'][::-1])
            finally:
                os.chdir(previous)

if __name__ == '__main__':
    unittest.main()
