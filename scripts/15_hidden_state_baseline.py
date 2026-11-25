import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
BASE = _paths.DATA_DIR
FRAG = BASE / '08_fragment_level'
OUT = BASE / '10_embedding_baseline'
OUT.mkdir(parents=True, exist_ok=True)
ALIAS_TO_INTERNAL = {'CoParCan-D': 'PARCAN_declaraciones', 'CoParCan-S': 'PARCAN_discursos', 'CoACan-Indiv': 'Bot_individual', 'CoACan-Pair': 'Bot_pareja', 'CoACan-Pres': 'Carto_expo', 'CoACan-Cart': 'Carto_grupo', 'CoCanWA': 'WhCan', 'PRESEEA-H': 'PRESEEA_alto', 'PRESEEA-M': 'PRESEEA_medio', 'PRESEEA-L': 'PRESEEA_bajo'}
f1 = pd.read_csv(BASE / '02_biber_mda/biber_factor_scores.csv').set_index('nombre_corpus')['F1']
ko = pd.read_csv(BASE / '06_ko_analysis/ko_coding_matrix_final.csv').set_index('corpus')['ko_sum_final']
ko = ko.rename(index=ALIAS_TO_INTERNAL)
PC1_CSV = OUT / 'hidden_state_pc1_by_corpus.csv'
stored = pd.read_csv(PC1_CSV) if PC1_CSV.exists() else None
results = []
pc1_rows = []
for model_key in ['salamandra', 'llama3', 'mistral']:
    if not (FRAG / f'hidden_states_{model_key}.npy').exists():
        # the hidden states are too large to distribute: reuse the stored first component per corpus
        sub = stored[stored.model == model_key]
        corpora, pc1_all = list(sub.nombre_corpus), sub.hidden_pc1.values
        var1, dim = float(sub.pc1_var_explained.iloc[0]), int(sub.hidden_dim.iloc[0])
    else:
        meta = pd.read_csv(FRAG / f'fragment_level_ppl_v4_{model_key}.csv')
        hidden = np.load(FRAG / f'hidden_states_{model_key}.npy')
        assert len(meta) == hidden.shape[0], (len(meta), hidden.shape)
        corpora = sorted(meta['corpus'].unique())
        corpus_embed = np.vstack([hidden[meta['corpus'].values == c].mean(axis=0) for c in corpora])
        X = StandardScaler().fit_transform(corpus_embed)
        pca = PCA(n_components=2, random_state=42)
        scores = pca.fit_transform(X)
        var_exp = pca.explained_variance_ratio_
        pc1_all, var1, dim = scores[:, 0], float(var_exp[0]), hidden.shape[1]
    pc1_rows += [{'model': model_key, 'nombre_corpus': c, 'hidden_pc1': v, 'pc1_var_explained': var1, 'hidden_dim': dim} for c, v in zip(corpora, pc1_all)]
    common = [c for c in corpora if c in f1.index and c in ko.index]
    idx = [corpora.index(c) for c in common]
    pc1 = np.asarray(pc1_all)[idx]
    r_f1, p_f1 = spearmanr(pc1, f1.loc[common])
    r_ko, p_ko = spearmanr(pc1, ko.loc[common])
    print(f'{model_key}: hidden_dim={dim}  PC1 var={var1:.1%}  rho(PC1,F1)={r_f1:+.3f} (p={p_f1:.4f})  rho(PC1,KO)={r_ko:+.3f} (p={p_ko:.4f})')
    results.append({'model': model_key, 'hidden_dim': dim, 'pc1_var_explained': round(var1, 4), 'rho_PC1_F1': round(r_f1, 3), 'p_PC1_F1': round(p_f1, 5), 'rho_PC1_KO': round(r_ko, 3), 'p_PC1_KO': round(p_ko, 5), 'n_corpora': len(common)})
pd.DataFrame(results).to_csv(OUT / 'hidden_state_baseline_spearman.csv', index=False)
pd.DataFrame(pc1_rows).to_csv(PC1_CSV, index=False)
print(f"\nSaved: {OUT / 'hidden_state_baseline_spearman.csv'}")
