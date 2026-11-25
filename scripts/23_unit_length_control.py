import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as _paths
BASE = _paths.DATA_DIR
OUT = BASE / '23_unit_length'
WRITTEN = {'Jable', 'PARCAN_declaraciones', 'PARCAN_discursos'}
REFS = ['ppl_salamandra', 'ppl_llama3', 'ppl_mistral', 'md_f1', 'ko']

def unit_counts(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rec = {}
    for e in data['textos_completos']:
        name = Path(e['archivo']).stem
        name = 'Jable' if re.match('^Jable', name) else name
        r = rec.setdefault(name, {'n_units': 0, 'n_words': 0})
        r['n_units'] += int(e['num_fragmentos_texto'])
        r['n_words'] += int(e['longitud_palabras'])
    df = pd.DataFrame(rec).T
    df.index.name = 'nombre_corpus'
    return df

def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r = float(np.corrcoef(ex, ey)[0, 1])
    n = len(rx)
    t = r * np.sqrt((n - 3) / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), n - 3)
    return (r, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-json', default=str(_paths.CORPUS_JSON))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    counts_csv = OUT / 'unit_counts.csv'
    if Path(args.corpus_json).exists():
        counts = unit_counts(args.corpus_json)
        counts.to_csv(counts_csv)
    counts = pd.read_csv(counts_csv).set_index('nombre_corpus')
    ref = pd.read_csv(BASE / '21_syntactic_complexity/syntactic_complexity_by_corpus.csv').set_index('nombre_corpus')
    df = counts.join(ref[['corpus', 'mean_words'] + REFS], how='inner')
    assert len(df) == 13, sorted(df.index)
    df['words_per_unit'] = df.n_words / df.n_units
    df['mode'] = ['written' if i in WRITTEN else 'oral' for i in df.index]
    df.round(4).to_csv(OUT / 'unit_length_by_corpus.csv')
    rows = []
    for label, sub in (('oral', df[df['mode'] == 'oral']), ('all', df)):
        for me in ('words_per_unit', 'mean_words'):
            r = {'subset': label, 'n': len(sub), 'measure': me}
            for c in REFS:
                rho, p = stats.spearmanr(sub[me], sub[c])
                r[f'rho_{c}'], r[f'p_{c}'] = (round(rho, 3), round(p, 5))
            rows.append(r)
        rho, p = stats.spearmanr(sub.words_per_unit, sub.mean_words)
        rows.append({'subset': label, 'n': len(sub), 'measure': 'words_per_unit_vs_mean_words', 'rho_ko': round(rho, 3), 'p_ko': round(p, 5)})
    sp = pd.DataFrame(rows)
    sp.to_csv(OUT / 'unit_length_spearman.csv', index=False)
    rows = []
    for label, sub in (('oral', df[df['mode'] == 'oral']), ('all', df)):
        for std in ('md_f1', 'ko'):
            rho0, p0 = stats.spearmanr(sub.ppl_salamandra, sub[std])
            for me in ('words_per_unit', 'mean_words'):
                r1, p1 = partial_spearman(sub.ppl_salamandra.values, sub[std].values, sub[me].values)
                r2, p2 = partial_spearman(sub[me].values, sub[std].values, sub.ppl_salamandra.values)
                rows.append({'subset': label, 'n': len(sub), 'standard': std, 'measure': me, 'rho_ppl_std': round(rho0, 3), 'p_ppl_std': round(p0, 5), 'partial_ppl_std_given_measure': round(r1, 3), 'p_partial_ppl': round(p1, 5), 'partial_measure_std_given_ppl': round(r2, 3), 'p_partial_measure': round(p2, 5)})
    pa = pd.DataFrame(rows)
    pa.to_csv(OUT / 'unit_length_partial.csv', index=False)
    pd.set_option('display.width', 250)
    print(df[['corpus', 'mode', 'n_units', 'n_words', 'words_per_unit', 'mean_words'] + REFS].round(3).to_string())
    print()
    print(sp.to_string(index=False))
    print()
    print(pa.to_string(index=False))
if __name__ == '__main__':
    main()
