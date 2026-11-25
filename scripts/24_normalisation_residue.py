import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as _paths
OUT = _paths.DATA_DIR / '24_normalisation_residue'
WRITTEN = {'Jable', 'PARCAN_declaraciones', 'PARCAN_discursos'}
WORD = re.compile("[a-záéíóúñü']+")
REDUCED = {'pa', "pa'", 'to', "to'", 'na', "na'"}
AO = re.compile('[a-záéíóúñü]{3,}ao')
TAG = re.compile('<[^>]{1,40}>')
BRACKET = re.compile('\\[[^\\]]{1,30}\\]')
PAIR = re.compile('\\w+=\\w+')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-json', default=str(_paths.CORPUS_JSON))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not Path(args.corpus_json).exists():
        print(pd.read_csv(OUT / 'residue_by_corpus.csv').to_string(index=False))
        return
    with open(args.corpus_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = {}
    for e in data['textos_completos']:
        name = Path(e['archivo']).stem
        name = 'Jable' if re.match('^Jable', name) else name
        texts[name] = texts.get(name, '') + ' ' + e['texto_completo']
    rows = []
    for name, text in texts.items():
        c = Counter(WORD.findall(text.lower()))
        n = sum(c.values())
        reduced = sum(v for k, v in c.items() if k in REDUCED or AO.fullmatch(k))
        markup = len(TAG.findall(text)) + len(BRACKET.findall(text)) + len(PAIR.findall(text))
        rows.append({'nombre_corpus': name, 'mode': 'written' if name in WRITTEN else 'oral', 'n_words': n, 'n_reduced': reduced, 'pct_reduced': round(100 * reduced / n, 3), 'n_markup': markup, 'pct_markup': round(100 * markup / n, 3)})
    df = pd.DataFrame(rows).set_index('nombre_corpus')
    df.to_csv(OUT / 'residue_by_corpus.csv')
    print(df.to_string())
if __name__ == '__main__':
    main()
