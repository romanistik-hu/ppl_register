import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
'\nPipeline Step 03 (v3): Agnostic Analysis — PCA + Bootstrap + Multi-Model Ranks\n================================================================================\n\nSections\n--------\n1.  Load data\n2.  PCA per model  (perplexity · attention_entropy · MAD)\n    - Sign alignment: perplexity always loads in the planned/distance direction\n3.  Bootstrap CI   (Salamandra only — primary model)\n4.  Multi-model rank stability  (3 metrics × 3 models)\n5.  Spearman matrix  (Salamandra PC1 vs each model × metric)\n6.  Cross-model PC1 comparison\n    - Sign-aligned PC1 per model per corpus\n    - Pairwise Spearman on PC1 scores\n    - Pairwise Procrustes 2D\n    - Overlay biplot (3 models)\n    - PC1 lineplot sorted by mean\n7.  Consensus PCA  (average metrics)\n8.  Procrustes: each model vs consensus\n\nOutputs (raw, no xlsx)\n-----------------------\nPer-model CSVs / PNGs\n    pca_scores_{model}_v3.csv\n    pca_loadings_{model}_v3.csv\n    bootstrap_ci_v3.csv              (Salamandra)\n    spearman_matrix_{model}_v3.csv\n\nCross-model CSVs / PNGs\n    multimodel_ranks_v3.csv\n    pc1_spearman_crossmodel.csv\n    procrustes_crossmodel.csv\n    pc1_crossmodel_table.csv\n    pca_3model_overlay.png\n    pc1_lineplot_3models.png\n\nConsensus\n    pca_scores_average_v3.csv\n    procrustes_disparity_v3.csv      (each model vs consensus)\n\nInput\n-----\ndata/processed/01_metrics/compiled_metrics_v3.csv\n\nOutput dir\n----------\ndata/processed/03_agnostic_pca/\n'
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial import procrustes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
INPUT_CSV = _paths.DATA_DIR / '01_metrics' / 'compiled_metrics_v3.csv'
OUTPUT_DIR = _paths.DATA_DIR / '03_agnostic_pca'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PCA_FEATURES = ['perplexity', 'attention_entropy', 'mad']
MODEL_METRIC_COLS = {'salamandra': {'perplexity': 'perplexity', 'attention_entropy': 'attention_entropy', 'mad': 'mad'}, 'llama3': {'perplexity': 'perplexity_llama3', 'attention_entropy': 'attention_entropy_llama3', 'mad': 'mad_llama3'}, 'mistral': {'perplexity': 'perplexity_mistral', 'attention_entropy': 'attention_entropy_mistral', 'mad': 'mad_mistral'}}
TYPE_COLORS = {'written': '#2166ac', 'written-press': '#4393c3', 'bot': '#d6604d', 'spoken-whatsapp': '#f4a582', 'spoken-presentation': '#74c476', 'spoken-semi-covert': '#31a354', 'spoken-interview': '#756bb1', 'spoken_interview': '#756bb1', 'spoken-covert': '#fdae6b'}
BOOTSTRAP_N = 1000
RANDOM_SEED = 42
SIGN_ANCHOR = 'Jable'

def _avail_models(df):
    return [m for m, cols in MODEL_METRIC_COLS.items() if all((c in df.columns and df[c].notna().sum() >= 5 for c in cols.values())) and m != 'average']

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_feat = [c for c in PCA_FEATURES if c not in df.columns]
    if missing_feat:
        raise ValueError(f'Missing PCA columns: {missing_feat}')
    for model, cols in MODEL_METRIC_COLS.items():
        if model == 'average':
            continue
        missing_m = [c for c in cols.values() if c not in df.columns]
        if missing_m:
            print(f'  {model}: missing {missing_m} — will be skipped')
    print(f'Models fully available: {_avail_models(df)}')
    return df

def run_pca(df: pd.DataFrame, output_dir: Path, model: str='salamandra') -> tuple:
    feat_cols = list(MODEL_METRIC_COLS[model].values())
    X = df[feat_cols].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    scores = pca.fit_transform(X_sc)
    ppl_col = feat_cols[0]
    ppl_idx = 0
    if pca.components_[0, ppl_idx] > 0:
        pca.components_[0] *= -1
        scores[:, 0] *= -1
    df = df.copy()
    df[f'pc1_{model}'] = scores[:, 0]
    df[f'pc2_{model}'] = scores[:, 1]
    df['pc1'] = scores[:, 0]
    df['pc2'] = scores[:, 1]
    loadings = pd.DataFrame(pca.components_.T, index=PCA_FEATURES, columns=['PC1', 'PC2'])
    var_exp = pca.explained_variance_ratio_
    print(f"\n{'=' * 60}\nPCA LOADINGS — {model.upper()}\n{'=' * 60}")
    print(f'  Var. explained: PC1={var_exp[0]:.3f}  PC2={var_exp[1]:.3f}')
    print(loadings.round(4).to_string())
    print(f'\nPC1 SCORES — {model.upper()}  (positive = planned/distance)')
    for _, row in df.sort_values(f'pc1_{model}', ascending=False).iterrows():
        print(f"  {row['nombre_corpus']:<26} PC1={row[f'pc1_{model}']:+.3f}  PC2={row[f'pc2_{model}']:+.3f}")
    fig, ax = plt.subplots(figsize=(13, 9))
    tipo_col = 'tipo_corpus' if 'tipo_corpus' in df.columns else None
    for _, row in df.iterrows():
        color = TYPE_COLORS.get(row[tipo_col] if tipo_col else '', '#888888')
        ax.scatter(float(row[f'pc1_{model}']), float(row[f'pc2_{model}']), color=color, s=110, edgecolors='k', linewidths=0.6, alpha=0.85, zorder=3)
        ax.annotate(row['nombre_corpus'], (float(row[f'pc1_{model}']), float(row[f'pc2_{model}'])), xytext=(4, 4), textcoords='offset points', fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.65, edgecolor='#cccccc'))
    score_range = max(abs(scores[:, 0]).max(), abs(scores[:, 1]).max())
    scale = 0.28 * score_range / max(abs(pca.components_).max(), 1e-06)
    for feat in PCA_FEATURES:
        l1, l2 = (float(loadings.loc[feat, 'PC1']), float(loadings.loc[feat, 'PC2']))
        ax.annotate('', xy=(l1 * scale, l2 * scale), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.0, alpha=0.9), zorder=5)
        ax.text(l1 * scale * 1.12, l2 * scale * 1.12, feat, color='#c0392b', fontsize=9.5, fontweight='bold', ha='center', va='center', zorder=6, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
    ax.margins(0.18)
    ax.axhline(0, color='#bbbbbb', lw=0.6, ls='--')
    ax.axvline(0, color='#bbbbbb', lw=0.6, ls='--')
    ax.set_xlabel(f'PC1 — {var_exp[0]:.1%} variance explained  (positive = planned/distance)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'PC2 — {var_exp[1]:.1%} variance explained', fontsize=11, fontweight='bold')
    ax.set_title(f'Agnostic PCA — {model.capitalize()}  (perplexity · attention_entropy · MAD)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, ls=':')
    plt.tight_layout()
    plot_name = f'pca_biplot_{model}.png'
    fig.savefig(output_dir / plot_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  -> {plot_name}')
    return (df, loadings, var_exp)

def bootstrap_pca(df: pd.DataFrame, output_dir: Path, n: int=BOOTSTRAP_N, seed: int=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = df[PCA_FEATURES].values
    scaler = StandardScaler()
    X_ref = scaler.fit_transform(X)
    pca_ref = PCA(n_components=2, random_state=seed)
    pca_ref.fit(X_ref)
    ref_load = pca_ref.components_[0]
    if pca_ref.components_[0, 0] > 0:
        ref_load = -ref_load
    pc1_boot = {f: [] for f in PCA_FEATURES}
    for _ in range(n):
        idx = rng.integers(0, len(X), size=len(X))
        Xb = scaler.fit_transform(X[idx])
        try:
            pb = PCA(n_components=2, random_state=seed)
            pb.fit(Xb)
            sign = 1 if np.dot(pb.components_[0], ref_load) >= 0 else -1
            for j, f in enumerate(PCA_FEATURES):
                pc1_boot[f].append(pb.components_[0, j] * sign)
        except Exception:
            continue
    records = []
    for f in PCA_FEATURES:
        arr = np.array(pc1_boot[f])
        lo, hi = (np.percentile(arr, 2.5), np.percentile(arr, 97.5))
        records.append({'feature': f, 'loading_mean': round(float(arr.mean()), 4), 'ci_lo': round(float(lo), 4), 'ci_hi': round(float(hi), 4), 'stable': not lo < 0 < hi})
    ci = pd.DataFrame(records)
    print(f"\n{'=' * 60}\nBOOTSTRAP CI (N={n}) — PC1 LOADINGS (Salamandra)\n{'=' * 60}")
    for _, row in ci.iterrows():
        flag = 'stable' if row['stable'] else 'CI crosses zero'
        print(f"  {row['feature']:<22} {row['loading_mean']:+.4f}  [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  {flag}")
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = list(range(len(PCA_FEATURES)))
    colors = ['#2166ac' if r['stable'] else '#d6604d' for _, r in ci.iterrows()]
    ax.barh(y_pos, ci['loading_mean'], xerr=[ci['loading_mean'] - ci['ci_lo'], ci['ci_hi'] - ci['loading_mean']], color=colors, alpha=0.85, edgecolor='k', linewidth=0.6, capsize=5)
    ax.axvline(0, color='black', lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(PCA_FEATURES, fontsize=11)
    ax.set_xlabel('PC1 loading (mean ± 95% CI)', fontsize=11)
    ax.set_title(f'PC1 Loading Stability — Bootstrap N={n}  (Salamandra)\nBlue = stable  |  Red = CI crosses zero', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / 'bootstrap_pc1_stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  -> bootstrap_pc1_stability.png')
    return ci

def multimodel_rank_stability(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    from scipy.stats import chi2 as chi2_dist
    models = _avail_models(df)
    metrics = ['perplexity', 'attention_entropy', 'mad']
    if len(models) < 2:
        print('\nMulti-model rank stability: fewer than 2 models — skipping.')
        return pd.DataFrame()
    rank_df = df[['nombre_corpus']].copy()
    for model in models:
        for metric in metrics:
            col = MODEL_METRIC_COLS[model][metric]
            rank_df[f'rank_{metric}_{model}'] = df[col].rank(ascending=True).round(0)
            rank_df[f'val_{metric}_{model}'] = df[col].round(4)
    metric_labels = {'perplexity': 'Perplexity', 'attention_entropy': 'Attention Entropy', 'mad': 'Mean Attention Distance (MAD)'}
    fig_files = {'perplexity': 'multimodel_perplexity.png', 'attention_entropy': 'multimodel_attention.png', 'mad': 'multimodel_mad.png'}
    for metric in metrics:
        rank_cols = [f'rank_{metric}_{m}' for m in models]
        sort_col = rank_cols[0]
        print(f"\n{'=' * 70}\nRANK STABILITY — {metric.upper()}\n{'=' * 70}")
        header = f"  {'corpus':<26}" + ''.join((f'  {m:<22}' for m in models))
        print(header + '\n  ' + '─' * (len(header) - 2))
        for _, row in rank_df.sort_values(sort_col).iterrows():
            parts = ''.join((f"  {int(row[f'rank_{metric}_{m}']):>3}  ({row[f'val_{metric}_{m}']:>8.4f})" for m in models))
            print(f"  {row['nombre_corpus']:<26}{parts}")
        print(f'\n  Pairwise Spearman rho ({metric}):')
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                c1, c2 = (MODEL_METRIC_COLS[m1][metric], MODEL_METRIC_COLS[m2][metric])
                sub = df[[c1, c2]].dropna()
                if len(sub) >= 5:
                    rho, p = spearmanr(sub[c1], sub[c2])
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
                    print(f'    {m1} ~ {m2}: rho={rho:.4f}  p={p:.5f}  {sig}')
        rank_mat = rank_df[rank_cols].dropna().values.astype(float)
        if rank_mat.shape[0] >= 5 and rank_mat.shape[1] >= 2:
            n_r, k_r = rank_mat.shape
            S = np.sum((rank_mat.sum(axis=1) - rank_mat.sum(axis=1).mean()) ** 2)
            W = 12 * S / (k_r ** 2 * (n_r ** 3 - n_r))
            chi2_val = k_r * (n_r - 1) * W
            p_w = 1 - chi2_dist.cdf(chi2_val, df=n_r - 1)
            print(f"\n  Kendall's W={W:.4f}  chi2={chi2_val:.3f}  p={p_w:.5f}  ({('good' if W > 0.7 else 'moderate' if W > 0.5 else 'poor')})")
        sorted_df = rank_df.sort_values(sort_col).reset_index(drop=True)
        fig_h = max(4, len(sorted_df) * 0.45 + 1.8)
        fig, ax = plt.subplots(figsize=(3 + 3.5 * len(models), fig_h))
        ax.axis('off')
        col_labels = ['Corpus'] + [f'{m.capitalize()}\nrank   value' for m in models]
        table_data = []
        for _, row in sorted_df.iterrows():
            cells = [row['nombre_corpus']]
            for m in models:
                cells.append(f"{int(row[f'rank_{metric}_{m}']):>2}   ({row[f'val_{metric}_{m}']:.4f})")
            table_data.append(cells)
        tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor('#2E75B6')
            tbl[0, j].set_text_props(color='white', fontweight='bold')
        if metric == 'perplexity':
            for i, row in enumerate(sorted_df.itertuples(), 1):
                if row.nombre_corpus == 'Carto_grupo':
                    for j in range(len(col_labels)):
                        tbl[i, j].set_facecolor('#FFF2CC')
                    break
        ax.set_title(f'Rank Stability — {metric_labels[metric]}\nrank 1 = lowest value = most written/planned', fontsize=11, fontweight='bold', pad=14)
        plt.tight_layout()
        fig.savefig(output_dir / fig_files[metric], dpi=200, bbox_inches='tight')
        plt.close()
        print(f'  -> {fig_files[metric]}')
    return rank_df

def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    models = _avail_models(df)
    metrics = ['perplexity', 'attention_entropy', 'mad']
    rows = []
    print(f"\n{'=' * 70}\nSPEARMAN MATRIX: PC1 vs each model × metric\n{'=' * 70}")
    sub = df.dropna(subset=['pc1'])
    for model in models:
        for metric in metrics:
            col = MODEL_METRIC_COLS[model][metric]
            sub2 = sub.dropna(subset=[col])
            if len(sub2) >= 5:
                rho, p = spearmanr(sub2['pc1'], sub2[col])
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
                print(f'  PC1 ~ {model}.{metric:<22} rho={rho:+.4f}  p={p:.5f}  {sig}  n={len(sub2)}')
                rows.append({'model': model, 'metric': metric, 'column': col, 'rho': round(rho, 4), 'p': round(p, 6), 'sig': sig, 'n': len(sub2)})
    return pd.DataFrame(rows)

def crossmodel_comparison(model_scores: dict, output_dir: Path, tipo_map: dict) -> None:
    models = list(model_scores.keys())
    corpora = sorted(list(model_scores[models[0]]['nombre_corpus']))
    aligned = {}
    for m, df in model_scores.items():
        aligned[m] = df.set_index('nombre_corpus').loc[corpora].reset_index()
    print(f"\n{'=' * 60}\nCROSS-MODEL SPEARMAN — PC1\n{'=' * 60}")
    sp_rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            n1, n2 = (models[i], models[j])
            rho, p = spearmanr(aligned[n1]['pc1'].values, aligned[n2]['pc1'].values)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            print(f'  {n1} ~ {n2}: rho={rho:.4f}  p={p:.5f}  {sig}')
            sp_rows.append({'model_1': n1, 'model_2': n2, 'rho': round(rho, 4), 'p': round(p, 6), 'sig': sig})
    pd.DataFrame(sp_rows).to_csv(output_dir / 'pc1_spearman_crossmodel.csv', index=False)
    print(f"\n{'=' * 60}\nCROSS-MODEL PROCRUSTES — 2D\n{'=' * 60}")
    proc_rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            n1, n2 = (models[i], models[j])
            M1 = aligned[n1][['pc1', 'pc2']].values.astype(float)
            M2 = aligned[n2][['pc1', 'pc2']].values.astype(float)
            _, _, d = procrustes(M1, M2)
            interp = 'negligible' if d < 0.05 else 'minor' if d < 0.15 else 'moderate' if d < 0.3 else 'severe'
            print(f'  {n1} ~ {n2}: disparity={d:.4f}  ({interp})')
            proc_rows.append({'comparison': f'{n1} ~ {n2}', 'model_1': n1, 'model_2': n2, 'disparity': round(d, 4), 'interpretation': interp})
    pd.DataFrame(proc_rows).to_csv(output_dir / 'procrustes_crossmodel.csv', index=False)
    comp = pd.DataFrame({'nombre_corpus': corpora})
    for m in models:
        comp[f'pc1_{m}'] = aligned[m]['pc1'].values
    pc1_cols = [f'pc1_{m}' for m in models]
    comp['pc1_mean'] = comp[pc1_cols].mean(axis=1)
    comp['pc1_range'] = comp[pc1_cols].max(axis=1) - comp[pc1_cols].min(axis=1)
    for m in models:
        comp[f'rank_{m}'] = comp[f'pc1_{m}'].rank(ascending=False).astype(int)
    rank_cols = [f'rank_{m}' for m in models]
    comp['max_rank_diff'] = comp.apply(lambda r: int(max((abs(r[rank_cols[i]] - r[rank_cols[j]]) for i in range(len(rank_cols)) for j in range(i + 1, len(rank_cols))))), axis=1)
    comp = comp.sort_values('pc1_mean', ascending=False).reset_index(drop=True)
    print(f"\n{'=' * 60}\nPER-CORPUS PC1 (sign-aligned, planned → immediate)\n{'=' * 60}")
    show_cols = ['nombre_corpus'] + pc1_cols + ['pc1_range'] + rank_cols + ['max_rank_diff']
    print(comp[show_cols].round(3).to_string(index=False))
    comp.to_csv(output_dir / 'pc1_crossmodel_table.csv', index=False)
    MODEL_STYLE = [('salamandra', 'o', 1.0, 170), ('llama3', 's', 0.72, 115), ('mistral', '^', 0.55, 105)]
    fig, ax = plt.subplots(figsize=(14, 10))
    for name, mk, alpha, sz in MODEL_STYLE:
        if name not in aligned:
            continue
        for _, row in aligned[name].iterrows():
            color = TYPE_COLORS.get(tipo_map.get(row['nombre_corpus'], ''), '#888888')
            ax.scatter(float(row['pc1']), float(row['pc2']), color=color, marker=mk, s=sz, edgecolors='k', linewidths=0.5, zorder=3, alpha=alpha)
    ref_model = models[0]
    for _, row in aligned[ref_model].iterrows():
        ax.annotate(row['nombre_corpus'], (float(row['pc1']), float(row['pc2'])), xytext=(6, 6), textcoords='offset points', fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, edgecolor='#cccccc'))
    ax.axhline(0, color='#bbbbbb', lw=0.7, ls='--')
    ax.axvline(0, color='#bbbbbb', lw=0.7, ls='--')
    ax.set_xlabel('PC1  (positive = planned/distance)', fontsize=11, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax.set_title('Agnostic PCA — 3-model overlay (sign-aligned)\n●=Salamandra  ■=Llama3  ▲=Mistral', fontsize=12, fontweight='bold', pad=14)
    tipo_vals = list(set(tipo_map.values()))
    tipo_handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items() if t in tipo_vals]
    model_handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='#555', markersize=9, label=n) for n, m, _, _ in MODEL_STYLE if n in aligned]
    ax.legend(handles=tipo_handles + model_handles, fontsize=8, loc='lower right', framealpha=0.85, ncol=2)
    ax.grid(alpha=0.2)
    ax.margins(0.15)
    plt.tight_layout()
    fig.savefig(output_dir / 'pca_3model_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  -> pca_3model_overlay.png')
    order = comp['nombre_corpus'].tolist()
    x = list(range(len(order)))
    MODEL_LINE = [('salamandra', '#1f77b4', 'o'), ('llama3', '#ff7f0e', 's'), ('mistral', '#2ca02c', '^')]
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, color, mk in MODEL_LINE:
        if name not in aligned:
            continue
        vals = [float(aligned[name].set_index('nombre_corpus').loc[c, 'pc1']) for c in order]
        ax.plot(x, vals, marker=mk, color=color, label=name.capitalize(), linewidth=1.8, markersize=7, alpha=0.9)
    if 'Carto_grupo' in order:
        cg_idx = order.index('Carto_grupo')
        ax.axvline(cg_idx, color='#d6604d', lw=1.2, ls=':', alpha=0.7, label='Carto_grupo (divergence)')
    ax.axhline(0, color='#aaaaaa', lw=0.8, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('PC1  (positive = planned/distance)', fontsize=10)
    ax.set_title('PC1 scores by model — sorted by mean  (planned → immediate)\nDivergences reveal model sensitivity to corpus characteristics', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / 'pc1_lineplot_3models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  -> pc1_lineplot_3models.png')

def calculate_consensus(df: pd.DataFrame) -> pd.DataFrame:
    models = _avail_models(df)
    metrics = ['perplexity', 'attention_entropy', 'mad']
    cons_df = df[['nombre_corpus', 'tipo_corpus']].copy()
    for metric in metrics:
        cols = [MODEL_METRIC_COLS[m][metric] for m in models]
        cons_df[f'avg_{metric}'] = df[cols].mean(axis=1)
    MODEL_METRIC_COLS['average'] = {'perplexity': 'avg_perplexity', 'attention_entropy': 'avg_attention_entropy', 'mad': 'avg_mad'}
    return cons_df

def procrustes_vs_consensus(cons_scores: np.ndarray, model_dfs: dict, output_dir: Path) -> None:
    print(f"\n{'=' * 60}\nPROCRUSTES — each model vs consensus\n{'=' * 60}")
    rows = []
    for model, df in model_dfs.items():
        m_scores = df[['pc1', 'pc2']].values.astype(float)
        _, _, d = procrustes(cons_scores, m_scores)
        interp = 'negligible' if d < 0.05 else 'minor' if d < 0.15 else 'moderate' if d < 0.3 else 'severe'
        print(f'  {model:<22} vs consensus: disparity={d:.6f}  ({interp})')
        rows.append({'model': model, 'disparity_vs_consensus': round(d, 6), 'interpretation': interp})
    pd.DataFrame(rows).to_csv(output_dir / 'procrustes_disparity_v3.csv', index=False)

def main():
    print('=== AGNOSTIC ANALYSIS v3 (Multi-Model) ===')
    print(f'Input:  {INPUT_CSV}')
    print(f'Output: {OUTPUT_DIR}')
    df_raw = load_data(INPUT_CSV)
    print(f'Corpora loaded: {len(df_raw)}')
    models = _avail_models(df_raw)
    tipo_map = df_raw.set_index('nombre_corpus')['tipo_corpus'].to_dict() if 'tipo_corpus' in df_raw.columns else {}
    model_dfs = {}
    for model in models:
        print(f'\n--- Model: {model} ---')
        df, loadings, var_exp = run_pca(df_raw, OUTPUT_DIR, model=model)
        model_dfs[model] = df[['nombre_corpus', 'pc1', 'pc2']].copy()
        if model == 'salamandra':
            boot_ci = bootstrap_pca(df, OUTPUT_DIR)
            boot_ci.to_csv(OUTPUT_DIR / 'bootstrap_ci_v3.csv', index=False)
        spearman_df = spearman_matrix(df)
        score_cols = ['nombre_corpus', 'tipo_corpus', f'pc1_{model}', f'pc2_{model}', 'pc1', 'pc2'] + list(MODEL_METRIC_COLS[model].values())
        df[[c for c in score_cols if c in df.columns]].to_csv(OUTPUT_DIR / f'pca_scores_{model}_v3.csv', index=False)
        loadings_out = loadings.copy()
        loadings_out.loc['_var_explained'] = [round(v, 4) for v in var_exp]
        loadings_out.to_csv(OUTPUT_DIR / f'pca_loadings_{model}_v3.csv')
        if not spearman_df.empty:
            spearman_df.to_csv(OUTPUT_DIR / f'spearman_matrix_{model}_v3.csv', index=False)
    rank_df = multimodel_rank_stability(df_raw, OUTPUT_DIR)
    if not rank_df.empty:
        rank_df.to_csv(OUTPUT_DIR / 'multimodel_ranks_v3.csv', index=False)
    if len(model_dfs) >= 2:
        crossmodel_comparison(model_dfs, OUTPUT_DIR, tipo_map)
    print(f"\n{'=' * 60}\nCONSENSUS ANALYSIS (model average)\n{'=' * 60}")
    cons_df_raw = calculate_consensus(df_raw)
    cons_df, cons_load, cons_var = run_pca(cons_df_raw, OUTPUT_DIR, model='average')
    cons_df.to_csv(OUTPUT_DIR / 'pca_scores_average_v3.csv', index=False)
    cons_load_out = cons_load.copy()
    cons_load_out.loc['_var_explained'] = [round(v, 4) for v in cons_var]
    cons_load_out.to_csv(OUTPUT_DIR / 'pca_loadings_average_v3.csv')
    cons_scores = cons_df[['pc1', 'pc2']].values.astype(float)
    procrustes_vs_consensus(cons_scores, model_dfs, OUTPUT_DIR)
    print(f'\nAll outputs saved to {OUTPUT_DIR}')
    print('\nCSV:  ' + '  '.join((f.name for f in sorted(OUTPUT_DIR.glob('*.csv')))))
    print('PNG:  ' + '  '.join((f.name for f in sorted(OUTPUT_DIR.glob('*.png')))))
if __name__ == '__main__':
    main()
