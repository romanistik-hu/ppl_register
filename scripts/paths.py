from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
CORPUS_JSON = ROOT / 'data' / 'raw' / 'textos_todos_corpus_new.json'
DATA_DIR = ROOT / 'data' / 'processed'
FIGURES_DIR = ROOT / 'results' / 'figures'
TABLES_DIR = ROOT / 'results' / 'tables'
for d in (DATA_DIR, FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)
