import argparse
import hashlib
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
OUT = BASE / '21_syntactic_complexity'
MODEL = 'spanish-ancora-ud-2.17-251125'
CHUNK_CHAR_CAP = 20000
MIN_TOKENS, MAX_TOKENS = (5, 100)
SUBCL = {'advcl', 'ccomp', 'csubj', 'acl'}
ALIAS = {'PARCAN_declaraciones': 'CoParCan-D', 'PARCAN_discursos': 'CoParCan-S', 'Jable': 'Jable', 'Bot_individual': 'CoACan-Indiv', 'Bot_pareja': 'CoACan-Pair', 'Carto_expo': 'CoACan-Pres', 'Carto_grupo': 'CoACan-Cart', 'COSER': 'COSER', 'PRESEEA_alto': 'PRESEEA-H', 'PRESEEA_medio': 'PRESEEA-M', 'PRESEEA_bajo': 'PRESEEA-L', 'WhCan': 'CoCanWA', 'AMERESCO': 'AMERESCO'}
ALIAS_TO_INTERNAL = {v: k for k, v in ALIAS.items()}
_MISSING_SPACE_RE = re.compile('([.?!,:)’”])(?=[A-Z])')

def fix_missing_spaces(text):
    return _MISSING_SPACE_RE.sub('\\1 ', text)

def chunk_text(text, cap=CHUNK_CHAR_CAP):
    if len(text) <= cap:
        return [text] if text.strip() else []
    paras = re.split('\\n+', text)
    chunks, buf = ([], '')
    for p in paras:
        if not p.strip():
            continue
        if len(buf) + len(p) + 1 <= cap:
            buf = f'{buf}\n{p}' if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= cap:
                buf = p
            else:
                words = p.split(' ')
                buf = ''
                for w in words:
                    if len(buf) + len(w) + 1 <= cap:
                        buf = f'{buf} {w}' if buf else w
                    else:
                        chunks.append(buf)
                        buf = w
    if buf:
        chunks.append(buf)
    return chunks

def cache_path(cache_dir, chunk):
    key = hashlib.sha256(f'{MODEL}\x00{chunk}'.encode('utf-8')).hexdigest()
    return cache_dir / f'{key}.conllu'

def load_corpora(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raw = {Path(e['archivo']).stem: e['texto_completo'] for e in data['textos_completos']}
    out = {}
    jable = [v for k, v in raw.items() if re.match('^Jable', k)]
    if jable:
        out['Jable'] = ' '.join(jable)
    for k, v in raw.items():
        if not re.match('^Jable', k):
            out[k] = v
    return {k: v for k, v in out.items() if k in ALIAS}

def read_sentences(conllu_text):
    sents, cur = ([], [])
    for line in conllu_text.splitlines():
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        if line.startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) < 8:
            continue
        if '-' in cols[0] or '.' in cols[0]:
            continue
        cur.append((int(cols[0]), cols[3], int(cols[6]), cols[7].split(':')[0]))
    if cur:
        sents.append(cur)
    return sents

def sentence_measures(sent):
    n_tokens = len(sent)
    words = [t for t in sent if t[1] != 'PUNCT']
    n_words = len(words)
    pos = {t[0]: i for i, t in enumerate(words)}
    dists = [abs(pos[t[0]] - pos[t[2]]) for t in words if t[2] != 0 and t[2] in pos]
    mdd = float(np.mean(dists)) if dists else np.nan
    head = {t[0]: t[2] for t in sent}
    rel = {t[0]: t[3] for t in sent}
    n_sub = sum((1 for t in words if t[3] in SUBCL))
    n_sconj = sum((1 for t in words if t[1] == 'SCONJ'))
    depth = 0
    for t in words:
        d, cur, steps = (0, t[0], 0)
        while cur != 0 and cur in head and (steps <= n_tokens):
            if rel[cur] in SUBCL:
                d += 1
            cur = head[cur]
            steps += 1
        depth = max(depth, d)
    return (n_tokens, n_words, mdd, n_sub, n_sconj, depth)

def corpus_measures(text, cache_dir):
    rows, n_chunks = ([], 0)
    for chunk in chunk_text(fix_missing_spaces(text)):
        cp = cache_path(cache_dir, chunk)
        if not cp.exists():
            raise FileNotFoundError(f'parse missing for a chunk ({cp.name}); nothing is re-parsed')
        n_chunks += 1
        for sent in read_sentences(cp.read_text(encoding='utf-8')):
            if MIN_TOKENS <= len(sent) <= MAX_TOKENS:
                rows.append(sentence_measures(sent))
    d = pd.DataFrame(rows, columns=['n_tokens', 'n_words', 'mdd', 'n_sub', 'n_sconj', 'depth'])
    band = d[(d.n_words >= 16) & (d.n_words <= 20)]
    return {'n_chunks': n_chunks, 'n_sentences': len(d), 'mean_length': d.n_tokens.mean(), 'mean_words': d.n_words.mean(), 'mdd': d.mdd.mean(), 'n_sentences_16_20': len(band), 'mdd_16_20': band.mdd.mean(), 'sconj_per_1000': 1000 * d.n_sconj.sum() / d.n_words.sum(), 'subcl_per_sentence': d.n_sub.mean(), 'subcl_per_1000': 1000 * d.n_sub.sum() / d.n_words.sum(), 'embed_depth': d.depth.mean()}

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
    ap.add_argument('--cache-dir', default=None, help='Folder for the UDPipe parses (needed only when the corpus texts are present)')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if Path(args.corpus_json).exists():
        cache_dir = Path(args.cache_dir)
        corpora = load_corpora(args.corpus_json)
        assert len(corpora) == 13, sorted(corpora)
        rec = {}
        for name, text in corpora.items():
            rec[name] = corpus_measures(text, cache_dir)
            print(f"{name:<22} sentences={rec[name]['n_sentences']:>6}  len={rec[name]['mean_length']:.2f}  mdd={rec[name]['mdd']:.3f}")
        df = pd.DataFrame(rec).T
        df.index.name = 'nombre_corpus'
    else:
        print('Corpus texts not found: reusing the stored measures per corpus.')
        stored = pd.read_csv(OUT / 'syntactic_complexity_by_corpus.csv').set_index('nombre_corpus')
        df = stored.loc[:, 'n_chunks':'embed_depth']
    m = pd.read_csv(BASE / '01_metrics/compiled_metrics_v3.csv').set_index('nombre_corpus')
    bpc = pd.read_csv(BASE / '04_cross_model/bpc_fertility_table.csv').set_index('nombre_corpus')
    f1 = pd.read_csv(BASE / '02_biber_mda/biber_factor_scores.csv').set_index('nombre_corpus')['F1']
    ko = pd.read_csv(BASE / '06_ko_analysis/ko_coding_matrix_final.csv').set_index('corpus')['ko_sum_final']
    ko = ko.rename(index=ALIAS_TO_INTERNAL)
    ref = pd.DataFrame({'ppl_salamandra': m['perplexity'], 'ppl_llama3': m['perplexity_llama3'], 'ppl_mistral': m['perplexity_mistral'], 'bpc_salamandra': bpc['bpc_salamandra'], 'md_f1': f1, 'ko': ko}).loc[df.index]
    assert not ref.isna().any().any()
    by = df.join(ref)
    by.insert(0, 'corpus', [ALIAS[i] for i in by.index])
    by.round(4).to_csv(OUT / 'syntactic_complexity_by_corpus.csv')
    measures = ['mean_length', 'mean_words', 'mdd', 'mdd_16_20', 'sconj_per_1000', 'subcl_per_sentence', 'subcl_per_1000', 'embed_depth']
    rows = []
    for me in measures:
        r = {'measure': me}
        for c in ref.columns:
            rho, p = stats.spearmanr(df[me].astype(float), ref[c])
            r[f'rho_{c}'], r[f'p_{c}'] = (round(rho, 3), round(p, 5))
        rows.append(r)
    sp = pd.DataFrame(rows)
    sp.to_csv(OUT / 'syntactic_complexity_spearman.csv', index=False)
    rows = []
    for std in ('md_f1', 'ko'):
        rho0, p0 = stats.spearmanr(ref['ppl_salamandra'], ref[std])
        for me in measures:
            x = df[me].astype(float).values
            r1, p1 = partial_spearman(ref['ppl_salamandra'].values, ref[std].values, x)
            r2, p2 = partial_spearman(x, ref[std].values, ref['ppl_salamandra'].values)
            rows.append({'standard': std, 'measure': me, 'rho_ppl_std': round(rho0, 3), 'p_ppl_std': round(p0, 5), 'partial_ppl_std_given_measure': round(r1, 3), 'p_partial_ppl': round(p1, 5), 'partial_measure_std_given_ppl': round(r2, 3), 'p_partial_measure': round(p2, 5)})
    pa = pd.DataFrame(rows)
    pa.to_csv(OUT / 'syntactic_complexity_partial.csv', index=False)
    pd.set_option('display.width', 250)
    print()
    print(by.drop(columns=['n_chunks']).round(3).to_string())
    print()
    print(sp.to_string(index=False))
    print()
    print(pa.to_string(index=False))
if __name__ == '__main__':
    main()
