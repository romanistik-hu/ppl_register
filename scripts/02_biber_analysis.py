import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
"\nPipeline Step 02b: Biber ↔ Agnostic Space Comparison\n\nPurpose:\n    Establishes the empirical dialogue between two heterogeneous register\n    analysis systems:\n\n        Biber (1988): Factor Analysis on 43 morphosyntactic features\n                      extracted via biberpy → Factor Scores (F1…F6)\n\n        Agnostic:     PCA on 5 distributional LLM-derived metrics\n                      → PC Scores (PC1, PC2)\n\n    The central claim tested here is convergent validity:\n    two methodologically independent approaches to register space produce\n    correlated orderings of the same corpora along the planned/spontaneous\n    continuum.\n\nAnalysis modules:\n    A. Biber Factor Analysis (canonical FA with varimax, as in Biber 1988)\n       — with corrected Jable pre-aggregation for consistency with pipeline v2\n    B. Biber PCA (parallel to agnostic PCA, for direct geometric comparison)\n    C. Spearman correlation: Biber-F1 ~ Agnostic-PC1\n    D. Procrustes analysis: Biber (F1,F2) ~ Agnostic (PC1,PC2)\n    E. Corpus displacement table: which corpora diverge most between systems\n\nTheoretical framing:\n    Biber F1 captures 'Involved vs Informational Production' via\n    morphosyntactic surface features (pronouns, contractions, nominalizations).\n    Agnostic PC1 captures the same gradient via token-level predictability\n    and attention structure, without any linguistic annotation.\n    Convergence between them (high ρ) supports the claim that LLM-derived\n    distributional metrics constitute a valid register-detection instrument.\n    Divergence for specific corpora (low Procrustes alignment locally) is\n    equally informative: it identifies register dimensions that morphosyntax\n    cannot capture but LLM metrics can (e.g., topic restriction in bot corpora).\n\nJable aggregation note:\n    Consistent with 01_calculate_metrics_v2.py, this script concatenates\n    the three Jable sub-corpora BEFORE feature extraction. This differs from\n    the original 02_biber_analysis.py which averaged post-hoc. For Biber\n    features (per-token rates), the difference is small but the pre-aggregation\n    approach is more principled and ensures n=13 in both pipelines.\n\nInput:\n    data/raw/textos_todos_corpus_new.json\n    data/processed/03_agnostic_pca/pca_scores_salamandra_v3.csv\n\nOutput:\n    data/processed/02_biber_mda/\n        biber_factor_scores.csv\n        biber_pca_scores.csv\n        biber_factor_loadings.csv\n        biber_fa_biplot.png\n        biber_pca_biplot.png\n        spearman_f1_pc1.csv\n        spearman_f1_pc1_scatter.png\n        procrustes_result.csv\n        procrustes_biplot_overlay.png\n        corpus_divergence.csv\n        corpus_divergence.png\n"
import argparse
import json
import logging
import re
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
BIBERPY_PATH = str(_paths.ROOT / 'third_party' / 'biberpy')
AGGREGATE_PATTERNS = [('^Jable', 'Jable', 'written-press')]
TYPE_COLORS = {'written': '#2166ac', 'written-press': '#4393c3', 'bot': '#d6604d', 'spoken-whatsapp': '#f4a582', 'spoken-presentation': '#74c476', 'spoken-semi-covert': '#31a354', 'spoken-interview': '#756bb1', 'spoken_interview': '#756bb1', 'spoken-covert': '#fdae6b'}
FLIP_BIBER_F1 = False

def load_and_aggregate(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raw = {}
    for entry in data['textos_completos']:
        stem = Path(entry['archivo']).stem
        raw[stem] = {'texto': entry['texto_completo'], 'tipo': entry['tipo_corpus']}
    aggregated = {}
    consumed = set()
    for pattern, agg_name, agg_tipo in AGGREGATE_PATTERNS:
        parts = {k: v for k, v in raw.items() if re.match(pattern, k)}
        if parts:
            combined = ' '.join((v['texto'] for v in parts.values()))
            aggregated[agg_name] = {'texto': combined, 'tipo': agg_tipo}
            consumed.update(parts.keys())
            logger.info(f"Aggregated {len(parts)} Jable sub-corpora → '{agg_name}' ({len(combined):,} chars)")
    for k, v in raw.items():
        if k not in consumed:
            aggregated[k] = v
    return aggregated

def extract_biber_features(corpora: dict) -> pd.DataFrame:
    sys.path.insert(0, BIBERPY_PATH)
    try:
        import biberpy
    except ImportError as e:
        raise ImportError(f'biberpy not found at {BIBERPY_PATH}. Check path configuration.\nOriginal error: {e}')
    biberpy.initwordlists(BIBERPY_PATH, 'es', 'ol')
    biberpy.verbosity = 0
    rows = []
    for name, info in corpora.items():
        logger.info(f'  Extracting Biber features: {name}')
        texto = info['texto']
        texto_clean = re.sub('[^\\w\\s.,!?]', ' ', texto.lower())
        texto_clean = re.sub('\\s+', ' ', texto_clean).strip()
        tokens = re.findall('[.?,]|\\w+', texto_clean)
        try:
            feats = biberpy.getbiberdims(tokens)
            if feats:
                row = {'nombre_corpus': name, 'tipo_corpus': info['tipo'], 'n_tokens': len(tokens), **feats}
                rows.append(row)
                logger.info(f'    {name}: {len(feats)} features extracted')
            else:
                logger.warning(f'    {name}: no features returned')
        except Exception as e:
            logger.error(f'    {name}: extraction failed — {e}')
    return pd.DataFrame(rows)

def stable_factor_scores(X: pd.DataFrame, loadings: np.ndarray) -> np.ndarray:
    """Regression factor scores with the Moore-Penrose pseudo-inverse of the
    feature correlation matrix. With more features than corpora that matrix
    is singular (rank n - 1), and the linear solve inside
    FactorAnalyzer.transform then returns machine-dependent values. The
    pseudo-inverse gives the unique minimum-norm solution."""
    Z = (X.values - X.values.mean(axis=0)) / X.values.std(axis=0)
    R = np.corrcoef(X.values.T)
    return Z @ (np.linalg.pinv(R, rcond=1e-10) @ loadings)

def biber_factor_analysis(biber_df: pd.DataFrame, feat_cols: list, output_dir: Path, n_factors: int=6) -> pd.DataFrame:
    try:
        from factor_analyzer import FactorAnalyzer
        from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    except ImportError:
        logger.error('factor_analyzer not installed. Run: pip install factor_analyzer --break-system-packages')
        return pd.DataFrame()
    print('\n[A] Biber Factor Analysis (varimax)')
    X = biber_df[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_cols = [c for c in X.columns if not X[c].isnull().all() and X[c].var() > 0]
    X = X[valid_cols]
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    logger.info(f'  Valid features for FA: {len(valid_cols)}/{len(feat_cols)}')
    try:
        chi2, p_bartlett = calculate_bartlett_sphericity(X)
        _, kmo = calculate_kmo(X)
        print(f'  Bartlett p={p_bartlett:.4f} | KMO={kmo:.3f}')
        if p_bartlett >= 0.05:
            print('  ⚠ Bartlett p ≥ 0.05 — FA may be inappropriate')
        if kmo < 0.6:
            print('  ⚠ KMO < 0.6 — FA may be inappropriate')
        print('  Note: with n=13 these tests have very low power. Report values but do not use as acceptance/rejection criterion.')
    except Exception as e:
        logger.warning(f'  Adequacy tests failed: {e}')
    fa_full = FactorAnalyzer(n_factors=min(len(valid_cols) - 1, n_factors), rotation=None)
    fa_full.fit(X)
    ev, _ = fa_full.get_eigenvalues()
    n_use = min(max(int(np.sum(ev > 1.0)), 2), n_factors)
    print(f'  Factors with eigenvalue > 1: {n_use}')
    fa = FactorAnalyzer(n_factors=n_use, rotation='varimax')
    fa.fit(X)
    scores = stable_factor_scores(X, fa.loadings_)
    scores_df = pd.DataFrame(scores, columns=[f'F{i + 1}' for i in range(n_use)])
    scores_df.insert(0, 'nombre_corpus', biber_df['nombre_corpus'].values)
    scores_df.insert(1, 'tipo_corpus', biber_df['tipo_corpus'].values)
    loadings_df = pd.DataFrame(fa.loadings_, columns=[f'F{i + 1}' for i in range(n_use)], index=valid_cols)
    if 'E14' in loadings_df.index and loadings_df.loc['E14', 'F1'] > 0:
        scores_df['F1'] *= -1
        loadings_df['F1'] *= -1
        print('  F1 orientation: flipped so E14 (nominalizations) → negative = planned')
    variance = fa.get_factor_variance()
    var_exp = variance[1]
    print(f"  Variance explained per factor: {', '.join((f'F{i + 1}={v:.1%}' for i, v in enumerate(var_exp)))}")
    scores_df.to_csv(output_dir / 'biber_factor_scores.csv', index=False)
    loadings_df.to_csv(output_dir / 'biber_factor_loadings.csv')
    _biplot(scores_df, 'F1', 'F2', x_label=f'F1 ({var_exp[0]:.1%} var. explicada)', y_label=f'F2 ({var_exp[1]:.1%} var. explicada)', title='Biber Factor Analysis — F1 vs F2\n(Biber 1988, varimax, español)', output_path=output_dir / 'biber_fa_biplot.png')
    return (scores_df, loadings_df)

def biber_pca(biber_df: pd.DataFrame, feat_cols: list, output_dir: Path) -> pd.DataFrame:
    print('\n[B] Biber PCA (parallel to agnostic PCA)')
    X = biber_df[feat_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    valid = [c for c in X.columns if X[c].std() > 0]
    X = X[valid]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    scores = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    e14_idx = list(X.columns).index('E14') if 'E14' in X.columns else 0
    if pca.components_[0, e14_idx] < 0:
        pca.components_[0] *= -1
        scores[:, 0] *= -1
    scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2'])
    scores_df.insert(0, 'nombre_corpus', biber_df['nombre_corpus'].values)
    scores_df.insert(1, 'tipo_corpus', biber_df['tipo_corpus'].values)
    loadings_df = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=valid)
    scores_df.to_csv(output_dir / 'biber_pca_scores.csv', index=False)
    loadings_df.to_csv(output_dir / 'biber_pca_loadings.csv')
    _biplot(scores_df, 'PC1', 'PC2', x_label=f'PC1 ({var_exp[0]:.1%} var. explicada)', y_label=f'PC2 ({var_exp[1]:.1%} var. explicada)', title='Biber PCA — PC1 vs PC2\n(features morfosintácticos, comparación paralela)', output_path=output_dir / 'biber_pca_biplot.png', loadings=loadings_df, top_loadings=8)
    print(f'  PC1: {var_exp[0]:.1%}  |  PC2: {var_exp[1]:.1%}')
    return scores_df

def spearman_f1_pc1(biber_fa_scores: pd.DataFrame, agnostic_scores: pd.DataFrame, output_dir: Path) -> None:
    print('\n[C] Spearman: Biber-F1 ~ Agnostic-PC1')
    fa_col = 'F1' if 'F1' in biber_fa_scores.columns else biber_fa_scores.columns[2]
    ag_col = 'PC1' if 'PC1' in agnostic_scores.columns else agnostic_scores.columns[2]
    merged = biber_fa_scores[['nombre_corpus', 'tipo_corpus', fa_col]].merge(agnostic_scores[['nombre_corpus', ag_col]], on='nombre_corpus')
    missing_biber = set(biber_fa_scores['nombre_corpus']) - set(agnostic_scores['nombre_corpus'])
    missing_agn = set(agnostic_scores['nombre_corpus']) - set(biber_fa_scores['nombre_corpus'])
    if missing_biber:
        print(f'  ⚠ En Biber pero no en Agnostico: {missing_biber}')
    if missing_agn:
        print(f'  ⚠ En Agnostico pero no en Biber: {missing_agn}')
    if len(merged) < 5:
        print('  Insuficientes corpora en común para Spearman.')
        return
    f1 = merged[fa_col].values
    if FLIP_BIBER_F1:
        f1 = -f1
    pc1 = merged[ag_col].values
    rho, p_two = spearmanr(f1, pc1)
    p_one = p_two / 2
    n = len(merged)
    print(f'\n  n = {n}')
    print(f'  ρ = {rho:.3f}')
    print(f'  p (dos colas) = {p_two:.4f}')
    print(f'  p (una cola)  = {p_one:.4f}')
    if rho > 0.45 and p_one < 0.05:
        verdict = '✓ Convergencia significativa'
    elif rho > 0.45:
        verdict = '~ Tendencia convergente, p marginal (n pequeño)'
    elif rho > 0.0:
        verdict = '~ Correlación débil, no significativa'
    else:
        verdict = '✗ Sin convergencia'
    print(f'  {verdict}')
    pd.DataFrame([{'n': n, 'rho': round(rho, 3), 'p_two_tailed': round(p_two, 4), 'p_one_tailed': round(p_one, 4), 'verdict': verdict, 'biber_f1_flipped': FLIP_BIBER_F1}]).to_csv(output_dir / 'spearman_f1_pc1.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, row in merged.iterrows():
        color = TYPE_COLORS.get(row['tipo_corpus'], '#888888')
        ax.scatter(-row[fa_col] if FLIP_BIBER_F1 else row[fa_col], row[ag_col], color=color, s=160, edgecolors='k', linewidth=0.6, zorder=3)
        ax.annotate(row['nombre_corpus'], (-row[fa_col] if FLIP_BIBER_F1 else row[fa_col], row[ag_col]), xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
    m, b = np.polyfit(f1, pc1, 1)
    x_line = np.linspace(f1.min(), f1.max(), 100)
    ax.plot(x_line, m * x_line + b, '--', color='#444444', lw=1.4, alpha=0.7, label=f'ρ={rho:.3f}, p={p_one:.3f} (1-tail)')
    ax.set_xlabel(f"Biber F1 {('(invertido)' if FLIP_BIBER_F1 else '')} ← oral | escrito →", fontsize=10)
    ax.set_ylabel('PC1 agnostico ← oral | escrito →', fontsize=10)
    ax.set_title('Convergencia: Biber F1 ~ PC1 Agnostico\n(Spearman — validez convergente entre métodos)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    legend_handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items() if t in merged['tipo_corpus'].values]
    ax.legend(handles=legend_handles + [plt.Line2D([0], [0], ls='--', color='#444444', label=f'ρ={rho:.3f}, p={p_one:.3f}')], fontsize=8, loc='best')
    plt.tight_layout()
    fig.savefig(output_dir / 'spearman_f1_pc1_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def procrustes_comparison(biber_fa_scores: pd.DataFrame, agnostic_scores: pd.DataFrame, output_dir: Path) -> None:
    print('\n[D] Procrustes: Biber (F1,F2) ~ Agnostic (PC1,PC2)')
    shared = set(biber_fa_scores['nombre_corpus']) & set(agnostic_scores['nombre_corpus'])
    if len(shared) < 4:
        print('  Insuficientes corpora compartidos.')
        return
    b = biber_fa_scores[biber_fa_scores['nombre_corpus'].isin(shared)].sort_values('nombre_corpus').reset_index(drop=True)
    a = agnostic_scores[agnostic_scores['nombre_corpus'].isin(shared)].sort_values('nombre_corpus').reset_index(drop=True)
    fa_cols = [c for c in ['F1', 'F2'] if c in b.columns]
    ag_cols = [c for c in ['PC1', 'PC2'] if c in a.columns]
    if len(fa_cols) < 2 or len(ag_cols) < 2:
        print('  Faltan columnas para Procrustes 2D.')
        return
    B = b[fa_cols].values.astype(float)
    A = a[ag_cols].values.astype(float)
    if FLIP_BIBER_F1:
        B[:, 0] *= -1
    mtx1, mtx2, disparity = procrustes(A, B)
    print(f'  Disparity de Procrustes: {disparity:.4f}')
    if disparity < 0.1:
        interp = 'Excelente correspondencia — los espacios son casi idénticos'
    elif disparity < 0.2:
        interp = 'Buena correspondencia — divergencia menor'
    elif disparity < 0.35:
        interp = 'Correspondencia moderada — algunos corpora divergen entre métodos'
    else:
        interp = 'Correspondencia débil — los espacios capturan dimensiones distintas'
    print(f'  Interpretación: {interp}')
    pd.DataFrame([{'n_corpora': len(shared), 'disparity': round(disparity, 4), 'interpretation': interp}]).to_csv(output_dir / 'procrustes_result.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, name in enumerate(b['nombre_corpus']):
        tipo = b['tipo_corpus'].iloc[i]
        color = TYPE_COLORS.get(tipo, '#888888')
        ax.scatter(mtx2[i, 0], mtx2[i, 1], marker='s', color=color, s=140, edgecolors='k', linewidths=0.8, zorder=3, alpha=0.85)
        ax.scatter(mtx1[i, 0], mtx1[i, 1], marker='o', color=color, s=140, edgecolors='k', linewidths=0.8, zorder=3, alpha=0.85)
        ax.annotate('', xy=(mtx2[i, 0], mtx2[i, 1]), xytext=(mtx1[i, 0], mtx1[i, 1]), arrowprops=dict(arrowstyle='->', color=color, lw=1.3, alpha=0.6, connectionstyle='arc3,rad=0.1'))
        mx = (mtx1[i, 0] + mtx2[i, 0]) / 2
        my = (mtx1[i, 1] + mtx2[i, 1]) / 2
        ax.annotate(name, (mx, my), xytext=(4, 4), textcoords='offset points', fontsize=8, fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.75, edgecolor='none'))
    type_handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items() if t in b['tipo_corpus'].values]
    method_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#555', markersize=9, label='Agnostico (referencia)'), plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#555', markersize=9, label='Biber FA (alineado)')]
    ax.legend(handles=type_handles + method_handles, fontsize=8, loc='lower right', framealpha=0.85)
    ax.axhline(0, color='#cccccc', lw=0.6, ls='--')
    ax.axvline(0, color='#cccccc', lw=0.6, ls='--')
    ax.set_xlabel('Dimensión 1 (Procrustes-aligned)', fontsize=10)
    ax.set_ylabel('Dimensión 2 (Procrustes-aligned)', fontsize=10)
    ax.set_title(f'Superimposición Procrustes: Biber FA vs. Agnostico\nDisparity = {disparity:.4f} — {interp}', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_dir / 'procrustes_biplot_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()

def corpus_divergence(biber_fa_scores: pd.DataFrame, agnostic_scores: pd.DataFrame, output_dir: Path) -> None:
    print('\n[E] Tabla de divergencia por corpus')
    shared = set(biber_fa_scores['nombre_corpus']) & set(agnostic_scores['nombre_corpus'])
    b = biber_fa_scores[biber_fa_scores['nombre_corpus'].isin(shared)].sort_values('nombre_corpus').reset_index(drop=True)
    a = agnostic_scores[agnostic_scores['nombre_corpus'].isin(shared)].sort_values('nombre_corpus').reset_index(drop=True)
    fa_cols = [c for c in ['F1', 'F2'] if c in b.columns]
    ag_cols = [c for c in ['PC1', 'PC2'] if c in a.columns]
    if len(fa_cols) < 2 or len(ag_cols) < 2:
        return
    B = b[fa_cols].values.astype(float)
    A = a[ag_cols].values.astype(float)
    if FLIP_BIBER_F1:
        B[:, 0] *= -1
    _, mtx2, _ = procrustes(A, B)
    mtx1 = A / (np.sqrt((A ** 2).sum()) + 1e-12)
    displacements = np.sqrt(((mtx1 - mtx2) ** 2).sum(axis=1))
    div_df = pd.DataFrame({'nombre_corpus': b['nombre_corpus'], 'tipo_corpus': b['tipo_corpus'], 'divergence': displacements.round(4), 'f1_biber': B[:, 0].round(3), 'f2_biber': B[:, 1].round(3), 'pc1_agnostic': A[:, 0].round(3), 'pc2_agnostic': A[:, 1].round(3)}).sort_values('divergence', ascending=False)
    print(div_df.to_string(index=False))
    div_df.to_csv(output_dir / 'corpus_divergence.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [TYPE_COLORS.get(t, '#888888') for t in div_df['tipo_corpus']]
    ax.barh(div_df['nombre_corpus'], div_df['divergence'], color=colors, edgecolor='k', linewidth=0.5, alpha=0.85)
    ax.axvline(0.1, color='#d6604d', lw=1.2, ls='--', label='Umbral orientativo (0.10)')
    ax.set_xlabel('Divergencia (distancia Procrustes-aligned)', fontsize=10)
    ax.set_title('Divergencia entre métodos por corpus\nMayor divergencia = el corpus se posiciona de forma distinta en Biber vs. Agnostico', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / 'corpus_divergence.png', dpi=300, bbox_inches='tight')
    plt.close()

def _biplot(scores_df: pd.DataFrame, xc: str, yc: str, x_label: str, y_label: str, title: str, output_path: Path, loadings: pd.DataFrame=None, top_loadings: int=0):
    fig, ax = plt.subplots(figsize=(12, 8))
    for _, row in scores_df.iterrows():
        color = TYPE_COLORS.get(row['tipo_corpus'], '#888888')
        ax.scatter(row[xc], row[yc], color=color, s=160, edgecolors='k', linewidths=0.6, zorder=3)
        ax.annotate(row['nombre_corpus'], (row[xc], row[yc]), xytext=(5, 6), textcoords='offset points', fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, edgecolor='#cccccc'))
    if loadings is not None and top_loadings > 0:
        top = loadings[xc].abs().nlargest(top_loadings).index
        scale = 0.4 * max(scores_df[xc].abs().max(), scores_df[yc].abs().max()) / max(loadings.loc[top, [xc, yc]].abs().values.max(), 1e-06)
        for feat in top:
            vx = loadings.loc[feat, xc] * scale
            vy = loadings.loc[feat, yc] * scale
            ax.annotate('', xy=(vx, vy), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#666', lw=1.2, alpha=0.65))
            ax.text(vx * 1.15, vy * 1.15, feat, color='#8b0000', fontsize=8, fontweight='bold', ha='center', va='center')
    ax.axhline(0, color='#cccccc', lw=0.6, ls='--')
    ax.axvline(0, color='#cccccc', lw=0.6, ls='--')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.grid(alpha=0.2)
    legend_handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items() if t in scores_df['tipo_corpus'].values]
    ax.legend(handles=legend_handles, fontsize=8, loc='lower right', framealpha=0.85)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Biber ↔ Agnostic Comparison — Pipeline Step 02b')
    parser.add_argument('--input_json', type=str, default=str(_paths.CORPUS_JSON))
    parser.add_argument('--input_agnostic', type=str, default=str(_paths.DATA_DIR / '03_agnostic_pca' / 'pca_scores_salamandra_v3.csv'), help='Agnostic PCA scores from 03_agnostic_analysis_v2.py')
    parser.add_argument('--output_dir', type=str, default=str(_paths.DATA_DIR / '02_biber_mda'))
    parser.add_argument('--n_factors', type=int, default=6, help='Maximum number of factors for Biber FA (eigenvalue>1 rule applied)')
    parser.add_argument('--skip_extraction', action='store_true', help='Skip biberpy extraction; load from --biber_features_csv instead')
    parser.add_argument('--biber_features_csv', type=str, default=str(_paths.DATA_DIR / '02_biber_mda' / 'biber_features_reaggregated.csv'), help='Pre-computed Biber features (used with --skip_extraction)')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_extraction:
        logger.info(f'Loading pre-computed Biber features: {args.biber_features_csv}')
        p = Path(args.biber_features_csv)
        if not p.exists():
            logger.error(f'File not found: {p}')
            return
        if p.suffix == '.xlsx':
            biber_df = pd.read_excel(p)
        else:
            biber_df = pd.read_csv(p)
        logger.info(f'Loaded {len(biber_df)} corpora, {biber_df.shape[1]} columns')
    else:
        logger.info('Extracting Biber features from corpus JSON...')
        corpora = load_and_aggregate(args.input_json)
        biber_df = extract_biber_features(corpora)
        if biber_df.empty:
            logger.error('Biber feature extraction failed.')
            return
        biber_df.to_csv(output_dir / 'biber_features_reaggregated.csv', index=False)
    meta_cols = {'nombre_corpus', 'tipo_corpus', 'archivo_original', 'longitud_texto', 'n_tokens'}
    feat_cols = [c for c in biber_df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(biber_df[c])]
    logger.info(f'Biber feature columns: {len(feat_cols)}')
    agn_path = Path(args.input_agnostic)
    if not agn_path.exists():
        logger.error(f'Agnostic scores not found: {agn_path}\nRun 03_agnostic_analysis_v2.py first.')
        return
    agnostic_scores = pd.read_csv(agn_path)
    if 'PC1' not in agnostic_scores.columns and 'pc1_salamandra' in agnostic_scores.columns:
        # orientation of the paper: negative PC1 is the planned pole
        agnostic_scores['PC1'] = -agnostic_scores['pc1_salamandra']
        agnostic_scores['PC2'] = -agnostic_scores['pc2_salamandra']
    logger.info(f'Agnostic scores: {len(agnostic_scores)} corpora')
    fa_result = biber_factor_analysis(biber_df, feat_cols, output_dir, n_factors=args.n_factors)
    if isinstance(fa_result, tuple):
        fa_scores, fa_loadings = fa_result
    else:
        logger.error('Factor Analysis failed. Aborting.')
        return
    biber_pca_scores = biber_pca(biber_df, feat_cols, output_dir)
    conv_dir = output_dir.parent / '05_biber_convergence'
    conv_dir.mkdir(parents=True, exist_ok=True)
    spearman_f1_pc1(fa_scores, agnostic_scores, conv_dir)
    procrustes_comparison(fa_scores, agnostic_scores, output_dir)
    corpus_divergence(fa_scores, agnostic_scores, conv_dir)
    print(f'\n✅ Comparación completada. Resultados en: {output_dir}')
if __name__ == '__main__':
    main()
