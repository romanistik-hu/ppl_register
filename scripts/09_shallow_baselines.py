import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
import json
import math
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
BASE = _paths.DATA_DIR
OUT = BASE / '09_shallow_baselines'
OUT.mkdir(parents=True, exist_ok=True)
CORPUS_JSON = str(_paths.CORPUS_JSON)
METRICS_CSV = BASE / '01_metrics/compiled_metrics_v3.csv'
F1_CSV = BASE / '02_biber_mda/biber_factor_scores.csv'
KO_CSV = BASE / '06_ko_analysis/ko_coding_matrix_final.csv'
ALIAS_TO_INTERNAL = {'CoParCan-D': 'PARCAN_declaraciones', 'CoParCan-S': 'PARCAN_discursos', 'CoACan-Indiv': 'Bot_individual', 'CoACan-Pair': 'Bot_pareja', 'CoACan-Pres': 'Carto_expo', 'CoACan-Cart': 'Carto_grupo', 'CoCanWA': 'WhCan', 'PRESEEA-H': 'PRESEEA_alto', 'PRESEEA-M': 'PRESEEA_medio', 'PRESEEA-L': 'PRESEEA_bajo'}
NAME_MAP = {'Jable_Cronicas': 'Jable', 'Jable_Especializados': 'Jable', 'Jable_Islas_Orientales': 'Jable', 'PARCAN_declaraciones.json': 'PARCAN_declaraciones', 'PARCAN_discursos.json': 'PARCAN_discursos', 'PRESEEA_alto.json': 'PRESEEA_alto', 'PRESEEA_bajo.json': 'PRESEEA_bajo', 'PRESEEA_medio.json': 'PRESEEA_medio'}
FUNCTION_WORDS = set('\nel la los las un una unos unas lo\nyo tu vos usted el ella ello nosotros nosotras vosotros vosotras ustedes ellos ellas\nme te se nos os le les lo la los las\nmi mis tu tus su sus nuestro nuestra nuestros nuestras vuestro vuestra vuestros vuestras\neste esta estos estas ese esa esos esas aquel aquella aquellos aquellas esto eso aquello\nque quien quienes cual cuales cuyo cuya cuyos cuyas donde cuando como\nalguien nadie algo nada alguno alguna algunos algunas ninguno ninguna\ntodo toda todos todas cada otro otra otros otras mismo misma mismos mismas\nvarios varias poco poca pocos pocas mucho mucha muchos muchas\na ante bajo cabe con contra de desde en entre hacia hasta para por segun sin so sobre tras durante mediante\ny e ni o u pero mas sino aunque porque pues que si como cuando mientras aunque\nno si ya tambien tampoco solo aun incluso\nsoy eres es somos sois son era eras eramos erais eran fui fuiste fue fuimos fuisteis fueron sere seras sera seremos sereis seran\nestoy estas esta estamos estais estan estaba estabas estabamos estabais estaban\nhe has ha hemos habeis han habia habias habiamos habiais habian hay\n'.split())

def load_corpora():
    with open(CORPUS_JSON) as f:
        raw = json.load(f)
    corpora = {}
    for entry in raw['textos_completos']:
        name = NAME_MAP.get(entry['archivo'], entry['archivo'])
        corpora[name] = corpora.get(name, '') + entry['texto_completo']
    return corpora

def strip_accents_lower(tok):
    repl = str.maketrans('áéíóúñÁÉÍÓÚÑü', 'aeiounAEIOUNu')
    return tok.translate(repl).lower()
SENT_SPLIT = re.compile('[.!?¡¿]+\\s+')
WORD_SPLIT = re.compile('[A-Za-zÁÉÍÓÚÑáéíóúñü]+')

def mean_sentence_length(text):
    sentences = [s for s in SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return float('nan')
    lengths = [len(WORD_SPLIT.findall(s)) for s in sentences]
    lengths = [l for l in lengths if l > 0]
    return float(np.mean(lengths)) if lengths else float('nan')

def sttr(text, window=50000):
    tokens = WORD_SPLIT.findall(text)
    if len(tokens) < 1000:
        return float('nan')
    n_windows = max(1, len(tokens) // window)
    ttrs = []
    for i in range(n_windows):
        chunk = tokens[i * window:(i + 1) * window]
        if len(chunk) < 500:
            continue
        types = set((strip_accents_lower(t) for t in chunk))
        ttrs.append(len(types) / len(chunk))
    return float(np.mean(ttrs)) if ttrs else float('nan')

def function_word_ratio(text):
    tokens = [strip_accents_lower(t) for t in WORD_SPLIT.findall(text)]
    if not tokens:
        return float('nan')
    n_fw = sum((1 for t in tokens if t in FUNCTION_WORDS))
    return n_fw / len(tokens)

def character_entropy(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return float('nan')
    counts = Counter(chars)
    n = len(chars)
    probs = np.array([c / n for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs)))

def main():
    if not Path(CORPUS_JSON).exists():
        print('Corpus texts not found: reusing the stored metrics per corpus.')
        df = pd.read_csv(OUT / 'shallow_baselines_metrics.csv').set_index('corpus')
        return correlations(df)
    corpora = load_corpora()
    rows = []
    for name, text in sorted(corpora.items()):
        rows.append({'corpus': name, 'mean_sentence_length': mean_sentence_length(text), 'sttr_50k': sttr(text), 'function_word_ratio': function_word_ratio(text), 'char_entropy_bits': character_entropy(text)})
    df = pd.DataFrame(rows).set_index('corpus')
    df.to_csv(OUT / 'shallow_baselines_metrics.csv')
    print(f"Saved: {OUT / 'shallow_baselines_metrics.csv'}\n")
    print(df)
    correlations(df)

def correlations(df):
    f1 = pd.read_csv(F1_CSV).set_index('nombre_corpus')['F1']
    ko = pd.read_csv(KO_CSV).set_index('corpus')['ko_sum_final']
    ko = ko.rename(index=ALIAS_TO_INTERNAL)
    ppl = pd.read_csv(METRICS_CSV).set_index('nombre_corpus')['perplexity']
    common = sorted(set(df.index) & set(f1.index) & set(ko.index))
    print(f'\nn corpora = {len(common)}\n')
    results = []
    metrics_to_test = {'mean_sentence_length': df.loc[common, 'mean_sentence_length'], 'sttr_50k': df.loc[common, 'sttr_50k'], 'function_word_ratio': df.loc[common, 'function_word_ratio'], 'char_entropy_bits': df.loc[common, 'char_entropy_bits'], 'salamandra_ppl': ppl.loc[common]}
    for label, series in metrics_to_test.items():
        r_f1, p_f1 = stats.spearmanr(series, f1.loc[common])
        r_ko, p_ko = stats.spearmanr(series, ko.loc[common])
        results.append({'baseline': label, 'rho_MD_F1': round(r_f1, 3), 'p_MD_F1': round(p_f1, 5), 'rho_KO': round(r_ko, 3), 'p_KO': round(p_ko, 5)})
        print(f'{label:<22} rho(F1)={r_f1:+.3f} (p={p_f1:.4f})   rho(KO)={r_ko:+.3f} (p={p_ko:.4f})')
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / 'shallow_baselines_spearman.csv', index=False)
    print(f"\nSaved: {OUT / 'shallow_baselines_spearman.csv'}")
if __name__ == '__main__':
    main()
