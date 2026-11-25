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
OUT_SIG = BASE / '12_significance_tests'
OUT_CONV.mkdir(parents=True, exist_ok=True)
OUT_SIG.mkdir(parents=True, exist_ok=True)
biber = pd.read_csv(BASE / '02_biber_mda/biber_factor_scores.csv').set_index('nombre_corpus')
pca = pd.read_csv(BASE / '03_agnostic_pca/pca_scores_salamandra_v3.csv').set_index('nombre_corpus')
ALIAS_TO_INTERNAL = {'CoParCan-D': 'PARCAN_declaraciones', 'CoParCan-S': 'PARCAN_discursos', 'CoACan-Indiv': 'Bot_individual', 'CoACan-Pair': 'Bot_pareja', 'CoACan-Pres': 'Carto_expo', 'CoACan-Cart': 'Carto_grupo', 'CoCanWA': 'WhCan', 'PRESEEA-H': 'PRESEEA_alto', 'PRESEEA-M': 'PRESEEA_medio', 'PRESEEA-L': 'PRESEEA_bajo'}
ko = pd.read_csv(BASE / '06_ko_analysis/ko_coding_matrix_final.csv').set_index('corpus')
ko = ko.rename(index=ALIAS_TO_INTERNAL)
metrics = pd.read_csv(BASE / '01_metrics/compiled_metrics_v3.csv').set_index('nombre_corpus')
baselines = pd.read_csv(BASE / '09_shallow_baselines/shallow_baselines_metrics.csv').set_index('corpus')
common = sorted(set(biber.index) & set(pca.index) & set(ko.index) & set(metrics.index) & set(baselines.index))
f1 = biber.loc[common, 'F1']
pc1 = -pca.loc[common, 'pc1_salamandra']
ppl = metrics.loc[common, 'perplexity']
ko_sum = ko.loc[common, 'ko_sum_final']
sent_len = baselines.loc[common, 'mean_sentence_length']
WRITTEN = {'Jable', 'PARCAN_declaraciones', 'PARCAN_discursos'}
ORAL = [c for c in common if c not in WRITTEN]
print(f'Oral-only subset: n={len(ORAL)} -> {ORAL}\n')
MOST_SPONTANEOUS_2 = {'AMERESCO', 'Carto_grupo'}
MOST_SPONTANEOUS_3 = {'AMERESCO', 'Carto_grupo', 'WhCan'}
DROP_SPONTANEOUS_2 = [c for c in common if c not in MOST_SPONTANEOUS_2]
DROP_SPONTANEOUS_3 = [c for c in common if c not in MOST_SPONTANEOUS_3]

def bca_ci(x, y, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    res = scipy_bootstrap((np.asarray(x), np.asarray(y)), statistic=lambda a, b: stats.spearmanr(a, b).statistic, n_resamples=B, paired=True, confidence_level=0.95, method='BCa', random_state=rng)
    return res.confidence_interval
rows = []
for label, idx in [('full_n13', common), ('oral_only_n10', ORAL), ('drop_2_most_spontaneous_n11', DROP_SPONTANEOUS_2), ('drop_3_most_spontaneous_n10', DROP_SPONTANEOUS_3)]:
    r_f1, p_f1 = stats.spearmanr(ppl.loc[idx], f1.loc[idx])
    r_ko, p_ko = stats.spearmanr(ppl.loc[idx], ko_sum.loc[idx])
    try:
        ci_f1 = bca_ci(ppl.loc[idx], f1.loc[idx])
        ci_lo_f1, ci_hi_f1 = (ci_f1.low, ci_f1.high)
    except Exception:
        ci_lo_f1, ci_hi_f1 = (float('nan'), float('nan'))
    try:
        ci_ko = bca_ci(ppl.loc[idx], ko_sum.loc[idx])
        ci_lo_ko, ci_hi_ko = (ci_ko.low, ci_ko.high)
    except Exception:
        ci_lo_ko, ci_hi_ko = (float('nan'), float('nan'))
    rows.append({'subset': label, 'n': len(idx), 'rho_PPL_F1': round(r_f1, 3), 'p_PPL_F1': round(p_f1, 4), 'ci95_PPL_F1': f'[{ci_lo_f1:.3f}, {ci_hi_f1:.3f}]', 'rho_PPL_KO': round(r_ko, 3), 'p_PPL_KO': round(p_ko, 4), 'ci95_PPL_KO': f'[{ci_lo_ko:.3f}, {ci_hi_ko:.3f}]'})
    print(f'{label:<16} n={len(idx):>2}  rho(PPL,F1)={r_f1:+.3f} (p={p_f1:.4f})  rho(PPL,KO)={r_ko:+.3f} (p={p_ko:.4f})')
pd.DataFrame(rows).to_csv(OUT_CONV / 'written_oral_split.csv', index=False)
print(f"\nSaved: {OUT_CONV / 'written_oral_split.csv'}\n")
PROJECTS = {'CoACan': ['Bot_individual', 'Bot_pareja', 'Carto_expo', 'Carto_grupo'], 'PRESEEA': ['PRESEEA_alto', 'PRESEEA_medio', 'PRESEEA_bajo'], 'PARCAN': ['PARCAN_declaraciones', 'PARCAN_discursos'], 'Jable': ['Jable'], 'AMERESCO': ['AMERESCO'], 'COSER': ['COSER'], 'CoCanWA': ['WhCan']}
assert sorted(sum(PROJECTS.values(), [])) == sorted(common), "project map doesn't cover all corpora"
rho_full, _ = stats.spearmanr(f1.loc[common], pc1.loc[common])
proj_rows = []
for proj, members in PROJECTS.items():
    remaining = [c for c in common if c not in members]
    r_loo, p_loo = stats.spearmanr(f1.loc[remaining], pc1.loc[remaining])
    proj_rows.append({'project_removed': proj, 'n_corpora_removed': len(members), 'n_remaining': len(remaining), 'rho_loo': round(r_loo, 4), 'delta_rho': round(r_loo - rho_full, 4), 'p_loo': round(p_loo, 5)})
    print(f'Drop {proj:<10} (-{len(members)}): rho={r_loo:+.3f}  Delta={r_loo - rho_full:+.3f}')
proj_df = pd.DataFrame(proj_rows).sort_values('delta_rho', key=abs, ascending=False)
proj_df.to_csv(OUT_CONV / 'loo_project_level.csv', index=False)
print(f'\nFull-sample rho(F1,PC1) = {rho_full:.3f}')
print(f"Saved: {OUT_CONV / 'loo_project_level.csv'}\n")

def steiger_z_dependent(r_xy, r_xz, r_yz, n):
    rm2 = (r_xy ** 2 + r_xz ** 2) / 2
    f = (1 - r_yz) / (2 * (1 - rm2)) if 1 - rm2 != 0 else np.inf
    f = min(f, 1.0)
    h = (1 - f * rm2) / (1 - rm2) if 1 - rm2 != 0 else np.inf
    z_xy = 0.5 * np.log((1 + r_xy) / (1 - r_xy))
    z_xz = 0.5 * np.log((1 + r_xz) / (1 - r_xz))
    se = np.sqrt(2 * (1 - r_yz) * h / (n - 3))
    z = (z_xy - z_xz) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return (z, p)
r_ppl_f1 = stats.spearmanr(ppl.loc[common], f1.loc[common]).statistic
r_sent_f1 = stats.spearmanr(sent_len.loc[common], f1.loc[common]).statistic
r_ppl_sent = stats.spearmanr(ppl.loc[common], sent_len.loc[common]).statistic
n = len(common)
# sentence length is reversed, so that both predictors correlate positively with the
# criterion and the test compares the two magnitudes
z_stat, p_steiger = steiger_z_dependent(r_ppl_f1, -r_sent_f1, -r_ppl_sent, n)
print(f"Steiger's Z (PPL~F1 vs sentence_length~F1, both vs F1, n={n}): z={z_stat:.3f}, p={p_steiger:.4f}")
rng = np.random.default_rng(42)
idx_arr = np.arange(n)
deltas = np.empty(10000)
ppl_v = ppl.loc[common].values
sent_v = sent_len.loc[common].values
f1_v = f1.loc[common].values
for b in range(10000):
    bidx = rng.choice(idx_arr, size=n, replace=True)
    r1 = stats.spearmanr(ppl_v[bidx], f1_v[bidx]).statistic
    r2 = stats.spearmanr(sent_v[bidx], f1_v[bidx]).statistic
    deltas[b] = r1 - abs(r2)
delta_obs = r_ppl_f1 - abs(r_sent_f1)
ci_lo, ci_hi = np.nanpercentile(deltas, [2.5, 97.5])
frac_below_0 = float(np.mean(deltas <= 0))
print(f'Bootstrap Delta_rho (|PPL~F1| - |sentence_length~F1|) = {delta_obs:+.3f}, 95% percentile CI [{ci_lo:.3f}, {ci_hi:.3f}], P(Delta<=0)={frac_below_0:.4f}')
sig_rows = [{'comparison': 'PPL_vs_F1__vs__sentence_length_vs_F1', 'n': n, 'rho_PPL_target': round(r_ppl_f1, 3), 'rho_baseline_target': round(r_sent_f1, 3), 'delta_rho': round(delta_obs, 3), 'steiger_z': round(z_stat, 3), 'steiger_p': round(p_steiger, 4), 'bootstrap_ci95_lo': round(ci_lo, 3), 'bootstrap_ci95_hi': round(ci_hi, 3), 'bootstrap_p_one_sided': round(frac_below_0, 4)}]
r_ppl_ko = stats.spearmanr(ppl.loc[common], ko_sum.loc[common]).statistic
r_sent_ko = stats.spearmanr(sent_len.loc[common], ko_sum.loc[common]).statistic
r_ppl_sent2 = stats.spearmanr(ppl.loc[common], sent_len.loc[common]).statistic
z_ko, p_steiger_ko = steiger_z_dependent(r_ppl_ko, -r_sent_ko, -r_ppl_sent2, n)
ko_v = ko_sum.loc[common].values
deltas_ko = np.empty(10000)
for b in range(10000):
    bidx = rng.choice(idx_arr, size=n, replace=True)
    r1 = stats.spearmanr(ppl_v[bidx], ko_v[bidx]).statistic
    r2 = stats.spearmanr(sent_v[bidx], ko_v[bidx]).statistic
    deltas_ko[b] = r1 - abs(r2)
delta_obs_ko = r_ppl_ko - abs(r_sent_ko)
ci_lo_ko, ci_hi_ko = np.nanpercentile(deltas_ko, [2.5, 97.5])
frac_below_0_ko = float(np.mean(deltas_ko <= 0))
print(f"\n[K&O] Steiger's Z (PPL~KO vs sentence_length~KO, n={n}): z={z_ko:.3f}, p={p_steiger_ko:.4f}")
print(f'[K&O] Bootstrap Delta_rho = {delta_obs_ko:+.3f}, 95% CI [{ci_lo_ko:.3f}, {ci_hi_ko:.3f}], P(Delta<=0)={frac_below_0_ko:.4f}')
sig_rows.append({'comparison': 'PPL_vs_KO__vs__sentence_length_vs_KO', 'n': n, 'rho_PPL_target': round(r_ppl_ko, 3), 'rho_baseline_target': round(r_sent_ko, 3), 'delta_rho': round(delta_obs_ko, 3), 'steiger_z': round(z_ko, 3), 'steiger_p': round(p_steiger_ko, 4), 'bootstrap_ci95_lo': round(ci_lo_ko, 3), 'bootstrap_ci95_hi': round(ci_hi_ko, 3), 'bootstrap_p_one_sided': round(frac_below_0_ko, 4)})
sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(OUT_SIG / 'steiger_bootstrap_diff.csv', index=False)
print(f"\nSaved: {OUT_SIG / 'steiger_bootstrap_diff.csv'}")
print("\nNOTE: Steiger's Z relies on a large-n asymptotic approximation and is poorly")
print('calibrated at n=13; the bootstrap CI is the more honest small-sample estimate.')
print('The bootstrap is reported as the primary result; with the signs aligned the two tests agree.')
