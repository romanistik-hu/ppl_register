import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as _paths
B = _paths.DATA_DIR
ALIAS = {'PARCAN_declaraciones': 'CoParCan-D', 'PARCAN_discursos': 'CoParCan-S', 'Jable': 'Jable', 'Bot_individual': 'CoACan-Indiv', 'Bot_pareja': 'CoACan-Pair', 'Carto_expo': 'CoACan-Pres', 'Carto_grupo': 'CoACan-Cart', 'COSER': 'COSER', 'PRESEEA_alto': 'PRESEEA-H', 'PRESEEA_medio': 'PRESEEA-M', 'PRESEEA_bajo': 'PRESEEA-L', 'WhCan': 'CoCanWA', 'AMERESCO': 'AMERESCO'}
INTERNAL = {v: k for k, v in ALIAS.items()}
CHECKS = []

def chk(label, paper, got, tol=0.0006):
    ok = abs(float(paper) - float(got)) <= tol
    CHECKS.append((label, paper, got, ok))

def rho(a, b):
    return stats.spearmanr(a, b)
m = pd.read_csv(B / '01_metrics/compiled_metrics_v3.csv').set_index('nombre_corpus')
idx = m.index
bpc = pd.read_csv(B / '04_cross_model/bpc_fertility_table.csv').set_index('nombre_corpus').loc[idx]
f1 = pd.read_csv(B / '02_biber_mda/biber_factor_scores.csv').set_index('nombre_corpus')['F1'].loc[idx]
kom = pd.read_csv(B / '06_ko_analysis/ko_coding_matrix_final.csv')
kom['nombre_corpus'] = kom['corpus'].map(INTERNAL)
kom = kom.set_index('nombre_corpus').loc[idx]
ko = kom['ko_sum_final']
pcs = pd.read_csv(B / '03_agnostic_pca/pca_scores_salamandra_v3.csv').set_index('nombre_corpus').loc[idx]
pc1 = -pcs['pc1_salamandra']
ppl = m['perplexity']
chk('rho(PPL,F1)', 0.852, rho(ppl, f1)[0])
chk('p(PPL,F1)', 0.0002, rho(ppl, f1)[1], 6e-05)
chk('rho(PPL,K&O)', 0.887, rho(ppl, ko)[0])
chk('rho(PC1,F1)', 0.879, rho(pc1, f1)[0])
bc = pd.read_csv(B / '05_biber_convergence/bootstrap_ci_spearman.csv').iloc[0]
chk('BCa lo', 0.739, bc.ci_lo_95)
chk('BCa hi', 0.994, bc.ci_hi_95)
chk('p(PC1,F1)', 0.0001, bc.p_value, 6e-05)
P = {'S': 'perplexity', 'L': 'perplexity_llama3', 'M': 'perplexity_mistral'}
Bc = {'S': 'bpc_salamandra', 'L': 'bpc_llama3', 'M': 'bpc_mistral'}
for (a, b), v in {('S', 'L'): 0.973, ('S', 'M'): 0.94, ('L', 'M'): 0.951}.items():
    chk(f'raw PPL {a}-{b}', v, rho(m[P[a]], m[P[b]])[0])
for (a, b), v in {('S', 'L'): 0.995, ('S', 'M'): 0.978, ('L', 'M'): 0.984}.items():
    chk(f'BPC {a}-{b}', v, rho(bpc[Bc[a]], bpc[Bc[b]])[0])
for k, v in {'S': 0.973, 'L': 0.94, 'M': 0.989}.items():
    chk(f'raw vs BPC {k}', v, rho(m[P[k]], bpc[Bc[k]])[0])
for k, v in {'salamandra': 3.96, 'llama3': 3.47, 'mistral': 3.05}.items():
    chk(f'mean fertility {k}', v, bpc[f'fertility_{k}'].mean(), 0.006)
chk('fertility Sal min (COSER)', 3.5, bpc.loc['COSER', 'fertility_salamandra'], 0.006)
chk('fertility Sal max (CoParCan-D)', 4.76, bpc.loc['PARCAN_declaraciones', 'fertility_salamandra'], 0.006)
chk('fertility Sal min is the minimum', bpc['fertility_salamandra'].min(), bpc.loc['COSER', 'fertility_salamandra'], 1e-09)
chk('total tokens', 1720406, bpc['n_tokens_salamandra'].sum(), 0.5)
params = [c for c in kom.columns if c not in ('corpus', 'ko_sum_final', 'n_raters_avg')]
pr = {p: rho(ppl, kom[p]) for p in params}
chk('K&O param min rho (referential proximity)', 0.53, pr['referential_proximity'][0], 0.006)
chk('K&O param min is the minimum', min((v[0] for v in pr.values())), pr['referential_proximity'][0], 1e-09)
chk('p referential proximity', 0.065, pr['referential_proximity'][1], 0.0006)
chk('K&O param max rho (spontaneity)', 0.97, max((v[0] for v in pr.values())), 0.006)
chk('K&O params significant (of 10)', 9, sum((v[1] < 0.05 for v in pr.values())), 0)
chk('second-lowest param rho', 0.61, sorted((v[0] for v in pr.values()))[1], 0.006)
chk('rho(PPL, K&O without spontaneity)', 0.887, rho(ppl, ko - kom['spontaneity'])[0])
al = pd.read_csv(B / '06_ko_analysis/ko_krippendorff_alpha.csv').set_index('parameter')['krippendorff_alpha']
chk('alpha pooled', 0.918, al['POOLED_ALL_PARAMETERS'])
chk('alpha min (dialogicity)', 0.73, al.drop('POOLED_ALL_PARAMETERS').min(), 0.006)
chk('alpha dialogicity', 0.73, al['dialogicity'], 0.006)
for c, v in {'PRESEEA_alto': 24.0, 'PRESEEA_medio': 24.0, 'PRESEEA_bajo': 24.67, 'COSER': 23.33}.items():
    chk(f'K&O sum {ALIAS[c]}', v, ko[c], 0.006)
sp = pd.read_csv(B / '05_biber_convergence/written_oral_split.csv').set_index('subset')
for s, (a, b) in {'oral_only_n10': (0.673, 0.79), 'drop_2_most_spontaneous_n11': (0.9, 0.813)}.items():
    chk(f'{s} F1', a, sp.loc[s, 'rho_PPL_F1'])
    chk(f'{s} K&O', b, sp.loc[s, 'rho_PPL_KO'])
s3 = [s for s in sp.index if s.startswith('drop_3')][0]
chk('drop 3 F1', 0.939, sp.loc[s3, 'rho_PPL_F1'])
chk('drop 3 K&O', 0.872, sp.loc[s3, 'rho_PPL_KO'])
oral = [c for c in idx if c not in ('Jable', 'PARCAN_declaraciones', 'PARCAN_discursos')]
chk('oral-only F1 recomputed', 0.673, rho(ppl[oral], f1[oral])[0])
chk('oral-only K&O recomputed', 0.79, rho(ppl[oral], ko[oral])[0])
chk('oral-only p F1', 0.033, rho(ppl[oral], f1[oral])[1], 0.0006)
chk('oral-only p K&O', 0.007, rho(ppl[oral], ko[oral])[1], 0.0006)
lp = pd.read_csv(B / '05_biber_convergence/loo_project_level.csv').set_index('project_removed')['rho_loo']
for k, v in {'PARCAN': 0.827, 'CoACan': 0.783, 'PRESEEA': 0.915, 'Jable': 0.86}.items():
    chk(f'project LOO {k}', v, lp[k])
lo = pd.read_csv(B / '05_biber_convergence/loo_spearman.csv')
chk('LOO min', 0.86, lo.rho_loo.min())
chk('LOO max', 0.895, lo.rho_loo.max())
sb = pd.read_csv(B / '09_shallow_baselines/shallow_baselines_spearman.csv').set_index('baseline')
for k, (a, b) in {'mean_sentence_length': (-0.8, -0.82), 'sttr_50k': (-0.6, -0.5), 'function_word_ratio': (0.77, 0.61), 'char_entropy_bits': (0.53, 0.45)}.items():
    chk(f'{k} F1', a, sb.loc[k, 'rho_MD_F1'], 0.006)
    chk(f'{k} K&O', b, sb.loc[k, 'rho_KO'], 0.006)
sm = pd.read_csv(B / '09_shallow_baselines/shallow_baselines_metrics.csv')
sm['nombre_corpus'] = sm['corpus'].map(lambda c: INTERNAL.get(c, c))
sm = sm.set_index('nombre_corpus').loc[idx]
chk('sentence length F1 recomputed', -0.797, rho(sm.mean_sentence_length, f1)[0])
chk('sentence length K&O recomputed', -0.815, rho(sm.mean_sentence_length, ko)[0])
sg = pd.read_csv(B / '12_significance_tests/steiger_bootstrap_diff.csv')
for i, (d, lo_, hi, p, z, zp) in enumerate([(0.055, -0.295, 0.366, 0.323, 0.6, 0.548), (0.072, -0.209, 0.334, 0.232, 0.87, 0.382)]):
    r = sg.iloc[i]
    chk(f'delta rho {i}', d, r.delta_rho)
    chk(f'delta CI lo {i}', lo_, r.bootstrap_ci95_lo)
    chk(f'delta CI hi {i}', hi, r.bootstrap_ci95_hi)
    chk(f'delta p {i}', p, r.bootstrap_p_one_sided)
    chk(f'Steiger z {i} (signs aligned)', z, r.steiger_z, 0.006)
    chk(f'Steiger p {i}', zp, r.steiger_p, 0.0006)
hs = pd.read_csv(B / '10_embedding_baseline/hidden_state_baseline_spearman.csv').set_index('model')
for k, (a, b) in {'salamandra': (0.863, 0.821), 'llama3': (0.923, 0.862), 'mistral': (0.923, 0.815)}.items():
    chk(f'hidden {k} F1', a, hs.loc[k, 'rho_PC1_F1'])
    chk(f'hidden {k} K&O', b, hs.loc[k, 'rho_PC1_KO'])
g = pd.read_csv(B / '17_gpt2_baseline/gpt2_spanish_ppl.csv').set_index('nombre_corpus')['gpt2_es_ppl'].loc[idx]
chk('GPT-2-es F1', 0.72, rho(g, f1)[0], 0.006)
chk('GPT-2-es p F1', 0.006, rho(g, f1)[1], 0.0006)
chk('GPT-2-es K&O', 0.78, rho(g, ko)[0], 0.006)
chk('GPT-2-es p K&O', 0.002, rho(g, ko)[1], 0.0006)
c5 = pd.read_csv(B / '13_char_baseline/char_perplexity_final13.csv').set_index('nombre_corpus').iloc[:, 0].loc[idx]
chk('char 5-gram F1', 0.38, rho(c5, f1)[0], 0.006)
chk('char 5-gram p F1', 0.2, rho(c5, f1)[1], 0.006)
chk('char 5-gram K&O', 0.23, rho(c5, ko)[0], 0.006)
chk('char 5-gram p K&O', 0.45, rho(c5, ko)[1], 0.006)
X = m[['perplexity', 'attention_entropy', 'mad']]
Z = (X - X.mean()) / X.std(ddof=0)
ev = np.sort(np.linalg.eigvalsh(np.cov(Z.T, ddof=0)))[::-1]
chk('PC1 variance %', 91.3, 100 * ev[0] / ev.sum(), 0.06)
chk('PC2 variance %', 7.8, 100 * ev[1] / ev.sum(), 0.06)
ld = pd.read_csv(B / '03_agnostic_pca/pca_loadings_salamandra_v3.csv', index_col=0)
for k, (a, b) in {'perplexity': (0.553, 0.833), 'attention_entropy': (0.588, -0.413), 'mad': (-0.59, 0.369)}.items():
    chk(f'loading PC1 {k} (sign-aligned)', a, -ld.loc[k, 'PC1'])
    chk(f'loading PC2 {k}', b, ld.loc[k, 'PC2'])
bt = pd.read_csv(B / '03_agnostic_pca/bootstrap_ci_v3.csv').set_index('feature')
for k, (lo_, hi_) in {'perplexity': (0.515, 0.568), 'attention_entropy': (0.578, 0.604), 'mad': (-0.614, -0.576)}.items():
    chk(f'boot lo {k} (sign-aligned)', lo_, -bt.loc[k, 'ci_hi'])
    chk(f'boot hi {k} (sign-aligned)', hi_, -bt.loc[k, 'ci_lo'])
pd_ = pd.read_csv(B / '04_cross_model/procrustes_disparity_v3.csv').set_index('model')['disparity_vs_consensus']
for k, v in {'salamandra': 0.051, 'llama3': 0.012, 'mistral': 0.047}.items():
    chk(f'Procrustes {k}', v, pd_[k])
zf, zp = ((f1 - f1.mean()) / f1.std(), (pc1 - pc1.mean()) / pc1.std())
zd = (zp - zf).abs()
chk('sd of MD F1', 1.04, f1.std(), 0.006)
chk('sd of PC1', 1.72, pc1.std(), 0.006)
for c, v in {'COSER': 0.91, 'PRESEEA_alto': 0.77, 'AMERESCO': 0.72, 'Bot_individual': 0.71}.items():
    chk(f'standardised |PC1 - F1| {ALIAS[c]}', v, zd[c], 0.006)
chk('four largest standardised differences', 1.0, float(set(zd.sort_values().index[-4:]) == {'COSER', 'PRESEEA_alto', 'AMERESCO', 'Bot_individual'}), 0)
rk = pd.DataFrame({'f1': f1.rank(), 'pc1': pc1.rank(), 'ko': ko.rank()})
chk('K&O rank closer to PC1 than to MD F1 in the four corpora', 4, int(sum(abs(rk.ko[c] - rk.pc1[c]) < abs(rk.ko[c] - rk.f1[c]) for c in ('COSER', 'PRESEEA_alto', 'AMERESCO', 'Bot_individual'))), 0)
for c, (a, b) in {'COSER': (1.75, 0.84), 'PRESEEA_alto': (0.05, 0.82), 'AMERESCO': (0.67, 1.39), 'Bot_individual': (0.0, -0.71)}.items():
    chk(f'z(F1) {ALIAS[c]}', a, zf[c], 0.006)
    chk(f'z(PC1) {ALIAS[c]}', b, zp[c], 0.006)
fr = pd.read_csv(B / '08_fragment_level/compiled_metrics_v4.csv').set_index('nombre_corpus').loc[idx]
chk('fragments total', 2435, fr.n_fragments.sum(), 0)
FRAG = {'PARCAN_declaraciones': 52, 'PARCAN_discursos': 477, 'Jable': 602, 'Bot_individual': 83, 'Bot_pareja': 154, 'Carto_expo': 28, 'PRESEEA_alto': 133, 'PRESEEA_medio': 129, 'PRESEEA_bajo': 119, 'COSER': 336, 'WhCan': 56, 'Carto_grupo': 241, 'AMERESCO': 25}
for c, v in FRAG.items():
    chk(f'fragments {ALIAS[c]}', v, fr.loc[c, 'n_fragments'], 0)
Xf = fr[['perplexity', 'attention_entropy', 'mad_norm']]
Zf = (Xf - Xf.mean()) / Xf.std(ddof=0)
wf, vf = np.linalg.eigh(np.cov(Zf.T, ddof=0))
chk('fragment-level PC1 variance %', 89.0, 100 * wf.max() / wf.sum(), 0.06)
chk('fragment-level rho(PC1,F1)', 0.89, abs(rho(Zf.values @ vf[:, wf.argmax()], f1)[0]))
chk('frag rho F1', 0.835, rho(fr.perplexity, f1)[0])
chk('frag rho K&O', 0.879, rho(fr.perplexity, ko)[0])
chk('frag vs continuous', 0.989, rho(fr.perplexity, ppl)[0])
ratio = fr.perplexity / ppl - 1
chk('frag excess min %', 7, 100 * ratio.min(), 0.6)
chk('frag excess max %', 20, 100 * ratio.max(), 0.6)
for (a, b), v in {('perplexity', 'perplexity_llama3'): 0.978, ('perplexity', 'perplexity_mistral'): 0.973, ('perplexity_llama3', 'perplexity_mistral'): 0.978}.items():
    chk(f"frag cross-model {a[11:] or 'sal'}-{b[11:]}", v, rho(fr[a], fr[b])[0])
fb = pd.read_csv(B / '08_fragment_level/fragment_bootstrap_ci.csv').set_index('target')
chk('two-stage F1 lo', 0.394, fb.loc['MD_F1', 'ci_lo'])
chk('two-stage F1 hi', 0.96, fb.loc['MD_F1', 'ci_hi'])
chk('two-stage K&O lo', 0.52, fb.loc['KO', 'ci_lo'])
chk('two-stage K&O hi', 0.97, fb.loc['KO', 'ci_hi'])
ps = pd.read_csv(B / '19_probe_sentences/probe_sentence_summary.csv').set_index('model')
chk('probe ratio min', 1.43, ps.ratio.min(), 0.006)
chk('probe ratio max', 3.31, ps.ratio.max(), 0.006)
chk('probe max p', 0.006, ps.p_value.max(), 0.0006)
sy = pd.read_csv(B / '21_syntactic_complexity/syntactic_complexity_spearman.csv').set_index('measure')
for k, (a, b, c) in {'mean_length': (-0.87, -0.76, -0.91), 'mdd': (-0.87, -0.73, -0.8), 'mdd_16_20': (0.69, 0.62, 0.75), 'subcl_per_1000': (0.74, 0.82, 0.69), 'sconj_per_1000': (0.42, 0.41, 0.44), 'subcl_per_sentence': (-0.31, -0.19, -0.37), 'embed_depth': (-0.6, -0.51, -0.67)}.items():
    chk(f'{k} vs PPL', a, sy.loc[k, 'rho_ppl_salamandra'], 0.006)
    chk(f'{k} vs F1', b, sy.loc[k, 'rho_md_f1'], 0.006)
    chk(f'{k} vs K&O', c, sy.loc[k, 'rho_ko'], 0.006)
pa = pd.read_csv(B / '21_syntactic_complexity/syntactic_complexity_partial.csv').set_index(['standard', 'measure'])
chk('partial PPL-F1 | length', 0.59, pa.loc[('md_f1', 'mean_length'), 'partial_ppl_std_given_measure'], 0.006)
chk('partial PPL-K&O | length', 0.45, pa.loc[('ko', 'mean_length'), 'partial_ppl_std_given_measure'], 0.006)
chk('partial length-K&O | PPL', -0.61, pa.loc[('ko', 'mean_length'), 'partial_measure_std_given_ppl'], 0.006)
chk('partial length-F1 | PPL', -0.08, pa.loc[('md_f1', 'mean_length'), 'partial_measure_std_given_ppl'], 0.006)
for k, (a, b) in {'mean_length': (-0.81, -0.76), 'mdd_16_20': (0.71, 0.7), 'subcl_per_1000': (0.81, 0.9)}.items():
    chk(f'{k} vs LLaMA3 PPL', a, sy.loc[k, 'rho_ppl_llama3'], 0.006)
    chk(f'{k} vs Mistral PPL', b, sy.loc[k, 'rho_ppl_mistral'], 0.006)
chk('mean words vs PPL', -0.87, sy.loc['mean_words', 'rho_ppl_salamandra'], 0.006)
chk('mean words vs F1', -0.81, sy.loc['mean_words', 'rho_md_f1'], 0.006)
chk('mean words vs K&O', -0.87, sy.loc['mean_words', 'rho_ko'], 0.006)
ul = pd.read_csv(B / '23_unit_length/unit_length_by_corpus.csv').set_index('nombre_corpus')
chk('words per unit CoACan-Cart', 6.1, ul.loc['Carto_grupo', 'words_per_unit'], 0.06)
chk('words per unit COSER', 29.8, ul.loc['COSER', 'words_per_unit'], 0.06)
chk('oral min words per unit', 6.1, ul[ul['mode'] == 'oral'].words_per_unit.min(), 0.06)
chk('oral max words per unit', 29.8, ul[ul['mode'] == 'oral'].words_per_unit.max(), 0.06)
us = pd.read_csv(B / '23_unit_length/unit_length_spearman.csv').set_index(['subset', 'measure'])
chk('oral unit vs sentence length', 0.08, us.loc[('oral', 'words_per_unit_vs_mean_words'), 'rho_ko'], 0.006)
chk('oral unit vs PPL', -0.5, us.loc[('oral', 'words_per_unit'), 'rho_ppl_salamandra'], 0.006)
chk('oral unit vs PPL p', 0.138, us.loc[('oral', 'words_per_unit'), 'p_ppl_salamandra'], 0.0006)
chk('oral unit vs K&O', -0.38, us.loc[('oral', 'words_per_unit'), 'rho_ko'], 0.006)
chk('oral unit vs K&O p', 0.275, us.loc[('oral', 'words_per_unit'), 'p_ko'], 0.0006)
up = pd.read_csv(B / '23_unit_length/unit_length_partial.csv').set_index(['subset', 'standard', 'measure'])
chk('oral partial PPL-K&O | unit', 0.75, up.loc[('oral', 'ko', 'words_per_unit'), 'partial_ppl_std_given_measure'], 0.006)
chk('oral partial PPL-K&O | unit p', 0.02, up.loc[('oral', 'ko', 'words_per_unit'), 'p_partial_ppl'], 0.0006)
chk('oral partial PPL-F1 | unit', 0.8, up.loc[('oral', 'md_f1', 'words_per_unit'), 'partial_ppl_std_given_measure'], 0.006)
chk('oral partial PPL-F1 | unit p', 0.009, up.loc[('oral', 'md_f1', 'words_per_unit'), 'p_partial_ppl'], 0.0006)
nr = pd.read_csv(B / '24_normalisation_residue/residue_by_corpus.csv').set_index('nombre_corpus')
chk('reduced forms COSER %', 1.5, nr.loc['COSER', 'pct_reduced'], 0.06)
chk('reduced forms elsewhere max %', 0.6, nr.drop(index='COSER').pct_reduced.max(), 0.06)
chk('leftover markup max % below 0.4', 1.0, float(nr.pct_markup.max() < 0.4), 0.0)

# ---- added for the camera-ready ----
ld1 = pd.read_csv(B / '02_biber_mda/biber_factor_loadings.csv', index_col=0)
chk('MD F1: % of variance after rotation', 36.1, 100 * (ld1['F1'] ** 2).sum() / len(ld1), 0.06)
chk('factors', 6, ld1.shape[1], 0)
chk('features', 43, len(ld1), 0)
ftv = pd.read_csv(B / '02_biber_mda/biber_features_reaggregated.csv')
chk('rank of the feature correlation matrix', 12, np.linalg.matrix_rank(np.corrcoef(ftv[list(ld1.index)].values.T)), 0)
chk('fertility vs F1', -0.87, rho(bpc.fertility_salamandra, f1)[0], 0.006)
chk('fertility vs K&O', -0.68, rho(bpc.fertility_salamandra, ko)[0], 0.006)
chk('BPC vs F1', 0.885, rho(bpc.bpc_salamandra, f1)[0])
chk('BPC vs K&O', 0.835, rho(bpc.bpc_salamandra, ko)[0])
bo = [rho(bpc.loc[oral, c], f1[oral]) for c in Bc.values()]
bk = [rho(bpc.loc[oral, c], ko[oral]) for c in Bc.values()]
chk('oral BPC vs F1 min', 0.745, min(x[0] for x in bo))
chk('oral BPC vs F1 max', 0.794, max(x[0] for x in bo))
chk('oral BPC vs K&O min', 0.632, min(x[0] for x in bk))
chk('oral BPC vs K&O max', 0.675, max(x[0] for x in bk))
chk('oral BPC correlations significant (of 6)', 6, sum(x[1] < 0.05 for x in bo + bk), 0)
chk('oral raw salamandra vs F1', 0.673, rho(m.loc[oral, 'perplexity'], f1[oral])[0])
chk('oral raw salamandra vs F1 p', 0.033, rho(m.loc[oral, 'perplexity'], f1[oral])[1], 0.0006)
chk('oral raw salamandra vs K&O', 0.79, rho(m.loc[oral, 'perplexity'], ko[oral])[0])
chk('oral raw salamandra vs K&O p', 0.007, rho(m.loc[oral, 'perplexity'], ko[oral])[1], 0.0006)
chk('oral raw llama3 vs F1', 0.588, rho(m.loc[oral, 'perplexity_llama3'], f1[oral])[0])
chk('oral raw llama3 vs F1 p', 0.074, rho(m.loc[oral, 'perplexity_llama3'], f1[oral])[1], 0.0006)
chk('oral raw llama3 vs K&O', 0.766, rho(m.loc[oral, 'perplexity_llama3'], ko[oral])[0])
chk('oral raw llama3 vs K&O p', 0.01, rho(m.loc[oral, 'perplexity_llama3'], ko[oral])[1], 0.0006)
chk('oral raw mistral vs F1', 0.733, rho(m.loc[oral, 'perplexity_mistral'], f1[oral])[0])
chk('oral raw mistral vs F1 p', 0.016, rho(m.loc[oral, 'perplexity_mistral'], f1[oral])[1], 0.0006)
chk('oral raw mistral vs K&O', 0.62, rho(m.loc[oral, 'perplexity_mistral'], ko[oral])[0])
chk('oral raw mistral vs K&O p', 0.056, rho(m.loc[oral, 'perplexity_mistral'], ko[oral])[1], 0.0006)
chk('oral partial PPL-K&O | sentence length', 0.53, up.loc[('oral', 'ko', 'mean_words'), 'partial_ppl_std_given_measure'], 0.006)
chk('oral partial PPL-K&O | sentence length p', 0.145, up.loc[('oral', 'ko', 'mean_words'), 'p_partial_ppl'], 0.0006)
chk('oral partial PPL-F1 | sentence length', 0.43, up.loc[('oral', 'md_f1', 'mean_words'), 'partial_ppl_std_given_measure'], 0.006)
chk('oral partial PPL-F1 | sentence length p', 0.244, up.loc[('oral', 'md_f1', 'mean_words'), 'p_partial_ppl'], 0.0006)
chk('partial PPL-K&O | length p', 0.139, pa.loc[('ko', 'mean_length'), 'p_partial_ppl'], 0.0006)
chk('partial PPL-F1 | length p', 0.045, pa.loc[('md_f1', 'mean_length'), 'p_partial_ppl'], 0.0006)
chk('K&O sum CoACan-Indiv', 13, ko['Bot_individual'], 0)
sc = pd.read_csv(B / '21_syntactic_complexity/syntactic_complexity_by_corpus.csv').set_index('nombre_corpus').loc[idx]
chk('character entropy vs F1, from the values per corpus', 0.53, rho(sm.char_entropy_bits, f1)[0], 0.005)
chk('subordinate clauses per sentence vs K&O, from the values per corpus', -0.37, rho(sc.subcl_per_sentence, ko)[0], 0.005)
chk('PRESEEA-L below PRESEEA-M under BPC', 1.0, float(bpc.loc['PRESEEA_bajo', 'bpc_salamandra'] < bpc.loc['PRESEEA_medio', 'bpc_salamandra']), 0)
bad = [c for c in CHECKS if not c[3]]
for label, paper, got, ok in CHECKS:
    print(f"{('ok  ' if ok else 'DIFF')} {label:<42} paper={paper:<10} recomputed={float(got):.4f}")
print(f'\n{len(CHECKS)} checks, {len(bad)} differences')
