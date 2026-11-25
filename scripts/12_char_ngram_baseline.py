import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
'Small-model contamination-control baseline (reviewer yEQA): a character\n5-gram model trained leave-one-corpus-out (i.e. on the other twelve corpora\nonly, a fully known, bounded training set immune to web-scale contamination\nby construction), evaluated on the held-out corpus. Re-implements\nscripts/spanish/09_char_perplexity.py on the exact final thirteen-corpus set\n(that script used a slightly different, pre-Jable-merge corpus split), and\nadds the Spearman correlation against MD F1 and the K&O composite that the\noriginal never computed. Unlike the original, does not truncate train/eval\ntext (full corpus text used throughout) for a more defensible result.\n\nOutputs:\n  paper_final/data/13_char_baseline/char_perplexity_final13.csv\n  paper_final/data/13_char_baseline/char_baseline_spearman.csv\n'
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CORPUS_JSON = str(_paths.CORPUS_JSON)
BASE = _paths.DATA_DIR
OUT = BASE / '13_char_baseline'
OUT.mkdir(parents=True, exist_ok=True)

def load_and_aggregate_corpora(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raw = {Path(e['archivo']).stem: e['texto_completo'] for e in data['textos_completos']}
    aggregated, consumed = ({}, set())
    parts = {k: v for k, v in raw.items() if re.match('^Jable', k)}
    if parts:
        aggregated['Jable'] = ' '.join(parts.values())
        consumed.update(parts.keys())
    for k, v in raw.items():
        if k not in consumed:
            aggregated[k] = v
    return aggregated

class CharNGramModel:

    def __init__(self, n=5, vocab_size=1000):
        self.n = n
        self.vocab_size = vocab_size
        self.counts = {}
        self.context_counts = {}

    def train(self, text):
        for i in range(len(text) - self.n):
            ngram = text[i:i + self.n]
            context = ngram[:-1]
            self.counts[ngram] = self.counts.get(ngram, 0) + 1
            self.context_counts[context] = self.context_counts.get(context, 0) + 1

    def perplexity(self, text):
        log_prob_sum, count = (0.0, 0)
        for i in range(len(text) - self.n):
            ngram = text[i:i + self.n]
            context = ngram[:-1]
            count_ngram = self.counts.get(ngram, 0)
            count_context = self.context_counts.get(context, 0)
            prob = (count_ngram + 1) / (count_context + self.vocab_size)
            log_prob_sum -= np.log2(prob)
            count += 1
        return 2 ** (log_prob_sum / count) if count > 0 else float('nan')

def main():
    if not Path(CORPUS_JSON).exists():
        print('Corpus texts not found: reusing the stored metrics per corpus.')
        df = pd.read_csv(OUT / 'char_perplexity_final13.csv').set_index('nombre_corpus')
        return correlations(df)
    corpora = load_and_aggregate_corpora(CORPUS_JSON)
    print(f'Corpora ({len(corpora)}): {list(corpora.keys())}')
    results = []
    for target_name, target_text in corpora.items():
        train_text = ' '.join((t for name, t in corpora.items() if name != target_name))
        model = CharNGramModel(n=5)
        model.train(train_text)
        ppl = model.perplexity(target_text)
        results.append({'nombre_corpus': target_name, 'char_perplexity_5gram': ppl})
        print(f'  {target_name:<24} CharPPL(5gram, LOO)={ppl:.2f}')
    df = pd.DataFrame(results).set_index('nombre_corpus')
    df.to_csv(OUT / 'char_perplexity_final13.csv')
    print(f"\nSaved: {OUT / 'char_perplexity_final13.csv'}")
    correlations(df)

def correlations(df):
    ALIAS_TO_INTERNAL = {'CoParCan-D': 'PARCAN_declaraciones', 'CoParCan-S': 'PARCAN_discursos', 'CoACan-Indiv': 'Bot_individual', 'CoACan-Pair': 'Bot_pareja', 'CoACan-Pres': 'Carto_expo', 'CoACan-Cart': 'Carto_grupo', 'CoCanWA': 'WhCan', 'PRESEEA-H': 'PRESEEA_alto', 'PRESEEA-M': 'PRESEEA_medio', 'PRESEEA-L': 'PRESEEA_bajo'}
    f1 = pd.read_csv(BASE / '02_biber_mda/biber_factor_scores.csv').set_index('nombre_corpus')['F1']
    ko = pd.read_csv(BASE / '06_ko_analysis/ko_coding_matrix_final.csv').set_index('corpus')['ko_sum_final']
    ko = ko.rename(index=ALIAS_TO_INTERNAL)
    ppl_llm = pd.read_csv(BASE / '01_metrics/compiled_metrics_v3.csv').set_index('nombre_corpus')['perplexity']
    common = sorted(set(df.index) & set(f1.index) & set(ko.index))
    r_f1, p_f1 = spearmanr(df.loc[common, 'char_perplexity_5gram'], f1.loc[common])
    r_ko, p_ko = spearmanr(df.loc[common, 'char_perplexity_5gram'], ko.loc[common])
    r_vs_llm, p_vs_llm = spearmanr(df.loc[common, 'char_perplexity_5gram'], ppl_llm.loc[common])
    print(f'\nrho(char-ngram PPL, MD F1)      = {r_f1:+.3f}  (p={p_f1:.4f})')
    print(f'rho(char-ngram PPL, K&O)        = {r_ko:+.3f}  (p={p_ko:.4f})')
    print(f'rho(char-ngram PPL, Salamandra PPL) = {r_vs_llm:+.3f}  (p={p_vs_llm:.4f})  <- sign/magnitude vs LLM ordering')
    pd.DataFrame([{'baseline': 'char_5gram_loo', 'rho_MD_F1': round(r_f1, 3), 'p_MD_F1': round(p_f1, 5), 'rho_KO': round(r_ko, 3), 'p_KO': round(p_ko, 5), 'rho_vs_salamandra_ppl': round(r_vs_llm, 3), 'p_vs_salamandra_ppl': round(p_vs_llm, 5), 'n': len(common)}]).to_csv(OUT / 'char_baseline_spearman.csv', index=False)
    print(f"Saved: {OUT / 'char_baseline_spearman.csv'}")
if __name__ == '__main__':
    main()
