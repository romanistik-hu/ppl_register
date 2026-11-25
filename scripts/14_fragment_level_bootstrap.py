import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
BASE = _paths.DATA_DIR
OUT = BASE / '08_fragment_level'
ALIAS_TO_INTERNAL = {'CoParCan-D': 'PARCAN_declaraciones', 'CoParCan-S': 'PARCAN_discursos', 'CoACan-Indiv': 'Bot_individual', 'CoACan-Pair': 'Bot_pareja', 'CoACan-Pres': 'Carto_expo', 'CoACan-Cart': 'Carto_grupo', 'CoCanWA': 'WhCan', 'PRESEEA-H': 'PRESEEA_alto', 'PRESEEA-M': 'PRESEEA_medio', 'PRESEEA-L': 'PRESEEA_bajo'}
frag = pd.read_csv(OUT / 'fragment_level_ppl_v4_salamandra.csv')
f1 = pd.read_csv(BASE / '02_biber_mda/biber_factor_scores.csv').set_index('nombre_corpus')['F1']
ko = pd.read_csv(BASE / '06_ko_analysis/ko_coding_matrix_final.csv').set_index('corpus')['ko_sum_final']
ko = ko.rename(index=ALIAS_TO_INTERNAL)
corpora = sorted(set(frag['corpus']) & set(f1.index) & set(ko.index))
frag_by_corpus = {c: frag.loc[frag.corpus == c, 'perplexity'].values for c in corpora}
f1_vals = f1.loc[corpora].values
ko_vals = ko.loc[corpora].values
n = len(corpora)
point_means = np.array([frag_by_corpus[c].mean() for c in corpora])
rho_f1_point, _ = spearmanr(point_means, f1_vals)
rho_ko_point, _ = spearmanr(point_means, ko_vals)
print(f'Point estimate: rho(PPL,F1)={rho_f1_point:.4f}  rho(PPL,KO)={rho_ko_point:.4f}')
rng = np.random.default_rng(42)
B = 10000
rhos_f1 = np.empty(B)
rhos_ko = np.empty(B)
for b in range(B):
    corpus_idx = rng.integers(0, n, size=n)
    boot_means = np.empty(n)
    for i, ci in enumerate(corpus_idx):
        c = corpora[ci]
        frags = frag_by_corpus[c]
        boot_frags = rng.choice(frags, size=len(frags), replace=True)
        boot_means[i] = boot_frags.mean()
    boot_f1 = f1_vals[corpus_idx]
    boot_ko = ko_vals[corpus_idx]
    rhos_f1[b] = spearmanr(boot_means, boot_f1).statistic
    rhos_ko[b] = spearmanr(boot_means, boot_ko).statistic
ci_f1 = np.nanpercentile(rhos_f1, [2.5, 97.5])
ci_ko = np.nanpercentile(rhos_ko, [2.5, 97.5])
print(f'Fragment-level bootstrap 95% CI (F1): [{ci_f1[0]:.3f}, {ci_f1[1]:.3f}]  (B={B}, {np.isnan(rhos_f1).sum()} NaN draws)')
print(f'Fragment-level bootstrap 95% CI (KO): [{ci_ko[0]:.3f}, {ci_ko[1]:.3f}]  (B={B}, {np.isnan(rhos_ko).sum()} NaN draws)')
pd.DataFrame([{'target': 'MD_F1', 'rho_point': round(rho_f1_point, 4), 'ci_lo': round(ci_f1[0], 4), 'ci_hi': round(ci_f1[1], 4), 'B': B, 'n_corpora': n}, {'target': 'KO', 'rho_point': round(rho_ko_point, 4), 'ci_lo': round(ci_ko[0], 4), 'ci_hi': round(ci_ko[1], 4), 'B': B, 'n_corpora': n}]).to_csv(OUT / 'fragment_bootstrap_ci.csv', index=False)
print(f"Saved: {OUT / 'fragment_bootstrap_ci.csv'}")
