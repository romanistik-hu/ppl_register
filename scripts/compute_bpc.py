import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from transformers import AutoTokenizer
OUT = _paths.DATA_DIR / '04_cross_model' / 'bpc_fertility_table.csv'
CORPUS_JSON = str(_paths.CORPUS_JSON)
METRICS_CSV = _paths.DATA_DIR / '01_metrics' / 'compiled_metrics_v3.csv'
MODELS = {'salamandra': 'BSC-LT/salamandra-7b', 'llama3': 'meta-llama/Meta-Llama-3-8B', 'mistral': 'mistralai/Mistral-7B-v0.3'}
NAME_MAP = {'Jable_Cronicas': 'Jable', 'Jable_Especializados': 'Jable', 'Jable_Islas_Orientales': 'Jable', 'PARCAN_declaraciones.json': 'PARCAN_declaraciones', 'PARCAN_discursos.json': 'PARCAN_discursos', 'PRESEEA_alto.json': 'PRESEEA_alto', 'PRESEEA_bajo.json': 'PRESEEA_bajo', 'PRESEEA_medio.json': 'PRESEEA_medio'}
print('Loading corpus texts...')
with open(CORPUS_JSON) as f:
    raw = json.load(f)
corpora: dict[str, str] = {}
for entry in raw['textos_completos']:
    name = NAME_MAP.get(entry['archivo'], entry['archivo'])
    text = entry['texto_completo']
    corpora[name] = corpora.get(name, '') + text
print(f'  Loaded {len(corpora)} corpora: {sorted(corpora)}')
fertility: dict[str, dict[str, float]] = {}
for model_key, model_id in MODELS.items():
    print(f'\nTokenizing with {model_id}...')
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    for corpus_name, text in sorted(corpora.items()):
        n_chars = len(text)
        n_tokens = len(tok.encode(text, add_special_tokens=False))
        fert = n_chars / n_tokens if n_tokens > 0 else float('nan')
        fertility.setdefault(corpus_name, {})[model_key] = fert
        print(f'  {corpus_name}: {n_chars} chars, {n_tokens} tokens, {fert:.4f} chars/tok')
metrics = pd.read_csv(METRICS_CSV).set_index('nombre_corpus')
rows = []
for corpus_name in sorted(corpora):
    if corpus_name not in metrics.index:
        print(f'  WARNING: {corpus_name} not in metrics CSV — skipping')
        continue
    row = {'corpus': corpus_name, 'tipo': metrics.loc[corpus_name, 'tipo_corpus']}
    n_chars = len(corpora[corpus_name])
    row['n_chars'] = n_chars
    for model_key in MODELS:
        fert = fertility[corpus_name][model_key]
        ppl_col = 'perplexity' if model_key == 'salamandra' else f'perplexity_{model_key}'
        ppl = metrics.loc[corpus_name, ppl_col]
        n_tok = round(n_chars / fert) if fert > 0 else 0
        bpc = math.log2(ppl) / fert if fert > 0 else float('nan')
        row[f'n_tokens_{model_key}'] = n_tok
        row[f'fertility_{model_key}'] = round(fert, 4)
        row[f'bpc_{model_key}'] = round(bpc, 6)
    row['perplexity'] = metrics.loc[corpus_name, 'perplexity']
    row['perplexity_llama3'] = metrics.loc[corpus_name, 'perplexity_llama3']
    row['perplexity_mistral'] = metrics.loc[corpus_name, 'perplexity_mistral']
    rows.append(row)
df = pd.DataFrame(rows)
df['nombre_corpus'] = df['corpus']
print('\nBPC cross-model Spearman ρ:')
pairs = [('salamandra', 'llama3'), ('salamandra', 'mistral'), ('llama3', 'mistral')]
for a, b in pairs:
    r, p = stats.spearmanr(df[f'bpc_{a}'], df[f'bpc_{b}'])
    print(f'  {a} vs {b}: ρ = {r:.3f}, p = {p:.4f}')
print('\nAverage fertility per model:')
for m in MODELS:
    print(f"  {m}: {df[f'fertility_{m}'].mean():.3f} chars/tok")
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f'\nSaved: {OUT}')
