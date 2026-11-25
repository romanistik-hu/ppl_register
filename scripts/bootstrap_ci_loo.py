import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap
BASE = _paths.DATA_DIR
OUT_CONV = BASE / '05_biber_convergence'
OUT_KO = BASE / '06_ko_analysis'
OUT_CONV.mkdir(parents=True, exist_ok=True)
OUT_KO.mkdir(parents=True, exist_ok=True)
biber_path = BASE / '02_biber_mda/biber_factor_scores.csv'
pca_path = BASE / '03_agnostic_pca/pca_scores_salamandra_v3.csv'
biber = pd.read_csv(biber_path).set_index('nombre_corpus')
pca = pd.read_csv(pca_path).set_index('nombre_corpus')
common = sorted(biber.index.intersection(pca.index))
print(f'Corpora used ({len(common)}): {common}\n')
f1 = biber.loc[common, 'F1'].values
pc1 = pca.loc[common, 'pc1_salamandra'].values
ppl = pca.loc[common, 'perplexity'].values
pc1_paper = -pc1
rho_full, p_full = stats.spearmanr(f1, pc1_paper)
print(f'Full Spearman ρ (F1 vs PC1): ρ = {rho_full:.3f}, p = {p_full:.4f}\n')
rng = np.random.default_rng(42)
res = scipy_bootstrap((f1, pc1_paper), statistic=lambda a, b: stats.spearmanr(a, b).statistic, n_resamples=10000, paired=True, confidence_level=0.95, method='BCa', random_state=rng)
ci_lo, ci_hi = res.confidence_interval
print(f'Bootstrap 95% CI (BCa, B=10000): [{ci_lo:.3f}, {ci_hi:.3f}]\n')
bootstrap_df = pd.DataFrame({'statistic': ['spearman_rho'], 'value': [rho_full], 'ci_lo_95': [ci_lo], 'ci_hi_95': [ci_hi], 'p_value': [p_full], 'n': [len(common)], 'n_resamples': [10000], 'method': ['BCa']})
bootstrap_df.to_csv(OUT_CONV / 'bootstrap_ci_spearman.csv', index=False)
print(f"Saved: {OUT_CONV / 'bootstrap_ci_spearman.csv'}")
print('\nLeave-one-out Spearman ρ:')
print(f"{'Corpus removed':<30} {'ρ₍-1₎':>8} {'Δρ':>8}")
print('-' * 50)
loo_rows = []
for i, corpus in enumerate(common):
    mask = np.ones(len(common), dtype=bool)
    mask[i] = False
    r_loo, p_loo = stats.spearmanr(f1[mask], pc1_paper[mask])
    delta = r_loo - rho_full
    loo_rows.append({'corpus_removed': corpus, 'rho_loo': round(r_loo, 4), 'delta_rho': round(delta, 4), 'p_loo': round(p_loo, 4)})
    print(f'{corpus:<30} {r_loo:>8.3f} {delta:>+8.3f}')
loo_df = pd.DataFrame(loo_rows).sort_values('delta_rho', key=abs, ascending=False)
loo_df.to_csv(OUT_CONV / 'loo_spearman.csv', index=False)
print(f"\nSaved: {OUT_CONV / 'loo_spearman.csv'}")
most_inf = loo_df.iloc[0]
min_loo = loo_df['rho_loo'].min()
print(f'\nMost influential removal: {most_inf.corpus_removed} (ρ={most_inf.rho_loo:.3f}, Δρ={most_inf.delta_rho:+.3f})')
print(f'Minimum LOO ρ across all removals: {min_loo:.3f}\n')
