import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
BASE = _paths.DATA_DIR
OUT = BASE / '06_ko_analysis'
PARAMS = ['privacy', 'familiarity', 'emotional_load', 'situational_embedding', 'referential_proximity', 'spatiotemporal_proximity', 'cooperation', 'dialogicity', 'spontaneity', 'free_thematization']
raw = pd.read_excel(str(_paths.ROOT / 'data' / 'annotations' / 'ANOTATORS_K&O.xlsx'))
corpora = sorted(raw['corpus'].unique())
annotators = sorted(raw['annotator'].unique())
print(f'Corpora: {len(corpora)}, Annotators: {annotators}')
alpha_rows = []
all_reliability_data = []
for param in PARAMS:
    units = corpora
    mat = np.full((len(annotators), len(units)), np.nan)
    for ai, a in enumerate(annotators):
        sub = raw[raw.annotator == a].set_index('corpus')
        for ci, c in enumerate(units):
            if c in sub.index:
                mat[ai, ci] = sub.loc[c, param]
    alpha = krippendorff.alpha(reliability_data=mat, level_of_measurement='ordinal')
    alpha_rows.append({'parameter': param, 'krippendorff_alpha': round(alpha, 4)})
    all_reliability_data.append(mat)
    print(f'  {param:<26} alpha={alpha:.4f}')
pooled_mat = np.hstack(all_reliability_data)
pooled_alpha = krippendorff.alpha(reliability_data=pooled_mat, level_of_measurement='ordinal')
print(f"\nPooled Krippendorff's alpha (all 10 parameters, ordinal): {pooled_alpha:.4f}")
alpha_rows.append({'parameter': 'POOLED_ALL_PARAMETERS', 'krippendorff_alpha': round(pooled_alpha, 4)})
alpha_df = pd.DataFrame(alpha_rows)
alpha_df.to_csv(OUT / 'ko_krippendorff_alpha.csv', index=False)
print(f"Saved: {OUT / 'ko_krippendorff_alpha.csv'}")
final = raw.groupby('corpus')[PARAMS].mean()
final['ko_sum_final'] = final[PARAMS].sum(axis=1)
final['n_raters_avg'] = raw.groupby('corpus')['annotator'].nunique()
final.to_csv(OUT / 'ko_coding_matrix_final.csv')
print(f"\nSaved: {OUT / 'ko_coding_matrix_final.csv'}")
print(final[['ko_sum_final', 'n_raters_avg']].sort_values('ko_sum_final'))
