import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
"\nPipeline Step 03b: PCA Comparison — Full vs. Length-Controlled\n\nPurpose:\n    Tests whether the PCA structure obtained from full-length corpora\n    (03_agnostic_analysis_v3.py) is preserved when all corpora are\n    subsampled to a common length ceiling (01b_sensitivity.py output).\n\n    This is the visual and quantitative counterpart to the Spearman\n    ranking comparisons in 01b. Where 01b tests metric-by-metric rank\n    stability, this script tests the stability of the JOINT structure\n    (the position of each corpus in the 2D PCA space).\n\nAnalysis modules:\n    A. PCA on sensitivity metrics (same 5 features as 03_v3)\n    B. Side-by-side biplot: full vs. length-controlled\n    C. Procrustes analysis: quantifies geometric distortion between the\n       two PCA spaces after optimal rotation/scaling alignment\n    D. Per-corpus displacement table: which corpora shift most in the\n       PC space when length is controlled\n\nInterpretation:\n    - Low Procrustes disparity (< 0.15) → PCA structure robust to length\n    - High displacement for a specific corpus → that corpus's position\n      is length-confounded; flag in paper\n    - If bot corpora show large displacement → the perplexity result\n      may be partially driven by their shorter text length\n\nInput:\n    compiled_metrics_v3.csv          (full-length, from 01_v3)\n    compiled_metrics_sensitivity.csv  (length-controlled, from 01b)\n\nOutput:\n    pca_sensitivity_biplot.png\n    pca_comparison_side_by_side.png\n    procrustes_result.csv\n    corpus_displacement.csv\n"
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes
warnings.filterwarnings('ignore')
CORE_FEATURES = ['perplexity', 'attention_entropy', 'mad']
TYPE_COLORS = {'written': '#2166ac', 'written-press': '#4393c3', 'bot': '#d6604d', 'spoken-whatsapp': '#f4a582', 'spoken-presentation': '#74c476', 'spoken-semi-covert': '#31a354', 'spoken-interview': '#756bb1', 'spoken_interview': '#756bb1', 'spoken-covert': '#fdae6b'}

def fit_pca(df: pd.DataFrame, features: list) -> tuple:
    available = [f for f in features if f in df.columns]
    X = df[available].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    scores = pca.fit_transform(X_scaled)
    ppl_idx = available.index('perplexity') if 'perplexity' in available else 0
    if pca.components_[0, ppl_idx] < 0:
        pca.components_[0] *= -1
        scores[:, 0] *= -1
    if 'attention_entropy' in available:
        ae_idx = available.index('attention_entropy')
        if pca.components_[1, ae_idx] < 0:
            pca.components_[1] *= -1
            scores[:, 1] *= -1
    return (scores, pca, scaler, available)

def make_biplot(ax, scores: np.ndarray, loadings: np.ndarray, feature_names: list, labels: list, types: list, var_exp: list, title: str):
    for i, (name, tipo) in enumerate(zip(labels, types)):
        color = TYPE_COLORS.get(tipo, '#888888')
        ax.scatter(scores[i, 0], scores[i, 1], color=color, s=160, edgecolors='k', linewidths=0.6, zorder=3)
        ax.annotate(name, (scores[i, 0], scores[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=7.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, edgecolor='#cccccc'))
    score_range = max(abs(scores[:, 0]).max(), abs(scores[:, 1]).max())
    loading_max = max(abs(loadings).max(), 1e-06)
    scale = 0.42 * score_range / loading_max
    for j, feat in enumerate(feature_names):
        vx, vy = (loadings[j, 0] * scale, loadings[j, 1] * scale)
        ax.annotate('', xy=(vx, vy), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#555555', lw=1.3, alpha=0.7))
        ax.text(vx * 1.18, vy * 1.18, feat, color='#8b0000', fontsize=8, fontweight='bold', ha='center', va='center')
    ax.axhline(0, color='#cccccc', lw=0.6, ls='--')
    ax.axvline(0, color='#cccccc', lw=0.6, ls='--')
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1%})', fontsize=9)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1%})', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.grid(alpha=0.2)

def run_sensitivity_pca(sens_df: pd.DataFrame, features: list, output_dir: Path) -> tuple:
    print('\n[A] PCA sobre métricas de sensibilidad (textos recortados)')
    scores, pca, scaler, available = fit_pca(sens_df, features)
    var_exp = pca.explained_variance_ratio_
    loadings = pca.components_.T
    fig, ax = plt.subplots(figsize=(11, 8))
    make_biplot(ax, scores, loadings, available, labels=sens_df['nombre_corpus'].tolist(), types=sens_df['tipo_corpus'].tolist(), var_exp=var_exp, title='PCA Agnostico — textos recortados (longitud controlada)')
    legend_handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items() if t in sens_df['tipo_corpus'].values]
    ax.legend(handles=legend_handles, fontsize=8, loc='lower right', framealpha=0.85)
    plt.tight_layout()
    fig.savefig(output_dir / 'pca_sensitivity_biplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2'])
    scores_df['nombre_corpus'] = sens_df['nombre_corpus'].values
    scores_df['tipo_corpus'] = sens_df['tipo_corpus'].values
    scores_df.to_csv(output_dir / 'pca_sensitivity_scores_v3.csv', index=False)
    loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=available)
    loadings_df.to_csv(output_dir / 'pca_sensitivity_loadings_v3.csv')
    print(f'  PC1: {var_exp[0]:.1%}  |  PC2: {var_exp[1]:.1%}')
    print(f"  Loadings PC1:\n{loadings_df['PC1'].sort_values().round(3)}")
    return (scores, pca, available)

def side_by_side_biplot(full_df: pd.DataFrame, sens_df: pd.DataFrame, features: list, output_dir: Path) -> tuple:
    print('\n[B] Biplot comparativo (full vs. length-controlled)')
    shared = set(full_df['nombre_corpus']) & set(sens_df['nombre_corpus'])
    if len(shared) < 4:
        print(f'  Solo {len(shared)} corpora compartidos — omitiendo comparación.')
        return (None, None)
    full_sub = full_df[full_df['nombre_corpus'].isin(shared)].reset_index(drop=True)
    sens_sub = sens_df[sens_df['nombre_corpus'].isin(shared)].reset_index(drop=True)
    full_sub = full_sub.sort_values('nombre_corpus').reset_index(drop=True)
    sens_sub = sens_sub.sort_values('nombre_corpus').reset_index(drop=True)
    scores_full, pca_full, _, avail_full = fit_pca(full_sub, features)
    scores_sens, pca_sens, _, avail_sens = fit_pca(sens_sub, features)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"PCA Agnostico: textos completos vs. longitud controlada\n(ceiling: corpus más corto | métricas: {', '.join(avail_full)})", fontsize=12, fontweight='bold', y=1.01)
    make_biplot(axes[0], scores_full, pca_full.components_.T, avail_full, labels=full_sub['nombre_corpus'].tolist(), types=full_sub['tipo_corpus'].tolist(), var_exp=pca_full.explained_variance_ratio_, title='Textos completos')
    make_biplot(axes[1], scores_sens, pca_sens.components_.T, avail_sens, labels=sens_sub['nombre_corpus'].tolist(), types=sens_sub['tipo_corpus'].tolist(), var_exp=pca_sens.explained_variance_ratio_, title='Longitud controlada')
    legend_handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items() if t in full_sub['tipo_corpus'].values]
    fig.legend(handles=legend_handles, fontsize=8, loc='lower center', ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.04), framealpha=0.85)
    plt.tight_layout()
    fig.savefig(output_dir / 'pca_comparison_side_by_side.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  Guardado: pca_comparison_side_by_side.png')
    return (scores_full, scores_sens)

def procrustes_analysis(full_df: pd.DataFrame, sens_df: pd.DataFrame, features: list, output_dir: Path) -> None:
    print('\n[C] Análisis de Procrustes (distorsión geométrica)')
    shared = set(full_df['nombre_corpus']) & set(sens_df['nombre_corpus'])
    if len(shared) < 4:
        print('  Insuficientes corpora compartidos para Procrustes.')
        return
    full_sub = full_df[full_df['nombre_corpus'].isin(shared)].sort_values('nombre_corpus').reset_index(drop=True)
    sens_sub = sens_df[sens_df['nombre_corpus'].isin(shared)].sort_values('nombre_corpus').reset_index(drop=True)
    scores_full, _, _, _ = fit_pca(full_sub, features)
    scores_sens, _, _, _ = fit_pca(sens_sub, features)
    _, _, disparity = procrustes(scores_full, scores_sens)
    print(f'  Disparity de Procrustes: {disparity:.4f}')
    if disparity < 0.05:
        interpretation = 'Negligible — estructura PCA idéntica bajo control de longitud'
    elif disparity < 0.15:
        interpretation = 'Menor — estructura preservada, longitud no es confound severo'
    elif disparity < 0.3:
        interpretation = 'Moderada — algunos corpora desplazados; revisar tabla de desplazamientos'
    else:
        interpretation = 'Severa — longitud confunde la estructura PCA; subsampling obligatorio'
    print(f'  Interpretación: {interpretation}')
    result = pd.DataFrame([{'n_corpora': len(shared), 'disparity': round(disparity, 4), 'interpretation': interpretation}])
    result.to_csv(output_dir / 'procrustes_results_sensitivity_v3.csv', index=False)

def corpus_displacement(full_df: pd.DataFrame, sens_df: pd.DataFrame, features: list, output_dir: Path) -> None:
    print('\n[D] Desplazamiento por corpus (Procrustes-aligned)')
    shared = set(full_df['nombre_corpus']) & set(sens_df['nombre_corpus'])
    if len(shared) < 4:
        print('  Insuficientes corpora.')
        return
    full_sub = full_df[full_df['nombre_corpus'].isin(shared)].sort_values('nombre_corpus').reset_index(drop=True)
    sens_sub = sens_df[sens_df['nombre_corpus'].isin(shared)].sort_values('nombre_corpus').reset_index(drop=True)
    scores_full, _, _, _ = fit_pca(full_sub, features)
    scores_sens, _, _, _ = fit_pca(sens_sub, features)
    mtx1, mtx2, _ = procrustes(scores_full, scores_sens)
    displacements = np.sqrt(((mtx1 - mtx2) ** 2).sum(axis=1))
    disp_df = pd.DataFrame({'nombre_corpus': full_sub['nombre_corpus'], 'tipo_corpus': full_sub['tipo_corpus'], 'displacement': displacements.round(4), 'pc1_full': mtx1[:, 0].round(4), 'pc2_full': mtx1[:, 1].round(4), 'pc1_controlled': mtx2[:, 0].round(4), 'pc2_controlled': mtx2[:, 1].round(4)}).sort_values('displacement', ascending=False)
    print(disp_df.to_string(index=False))
    disp_df.to_csv(output_dir / 'corpus_displacement_v3.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [TYPE_COLORS.get(t, '#888888') for t in disp_df['tipo_corpus']]
    ax.barh(disp_df['nombre_corpus'], disp_df['displacement'], color=colors, edgecolor='k', linewidth=0.5, alpha=0.85)
    ax.axvline(0.1, color='#d6604d', lw=1.2, ls='--', label='Umbral orientativo (0.10)')
    ax.set_xlabel('Desplazamiento Euclidiano (espacio Procrustes)', fontsize=10)
    ax.set_title('Desplazamiento por corpus: textos completos → longitud controlada\nMayor desplazamiento = posición más afectada por la longitud', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / 'corpus_displacement.png', dpi=300, bbox_inches='tight')
    plt.close()
    flagged = disp_df[disp_df['displacement'] > 0.1]['nombre_corpus'].tolist()
    if flagged:
        print(f'\n  ⚠ Corpora con desplazamiento notable (>0.10): {flagged}')
        print('    Interpretar su posición en el PCA completo con cautela.')
    else:
        print('\n  ✓ Ningún corpus muestra desplazamiento notable.')

def main():
    parser = argparse.ArgumentParser(description='PCA Sensitivity Comparison — Pipeline Step 03b')
    parser.add_argument('--input_full', type=str, default=str(_paths.DATA_DIR / '01_metrics' / 'compiled_metrics_v3.csv'))
    parser.add_argument('--input_sensitivity', type=str, default=str(_paths.DATA_DIR / '01_metrics' / 'compiled_metrics_sensitivity_v3.csv'))
    parser.add_argument('--output_dir', type=str, default=str(_paths.DATA_DIR / '07_length_sensitivity'))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = Path(args.input_full)
    sens_path = Path(args.input_sensitivity)
    if not full_path.exists():
        print(f'Error: {full_path} no encontrado. Ejecuta 01_calculate_metrics_v3.py')
        return
    if not sens_path.exists():
        print(f'Error: {sens_path} no encontrado. Ejecuta 01b_sensitivity.py')
        return
    full_df = pd.read_csv(full_path)
    sens_df = pd.read_csv(sens_path)
    print(f'Full metrics:        {len(full_df)} corpora')
    print(f'Sensitivity metrics: {len(sens_df)} corpora')
    available = [f for f in CORE_FEATURES if f in sens_df.columns]
    missing = [f for f in CORE_FEATURES if f not in sens_df.columns]
    if missing:
        print(f'Métricas ausentes en sensitivity CSV: {missing}')
    if len(available) < 2:
        print('Insuficientes métricas. Revisa el output de 01b_sensitivity.py')
        return
    run_sensitivity_pca(sens_df, available, output_dir)
    side_by_side_biplot(full_df, sens_df, available, output_dir)
    procrustes_analysis(full_df, sens_df, available, output_dir)
    corpus_displacement(full_df, sens_df, available, output_dir)
    print(f'\n✅ Análisis completado. Resultados en: {output_dir}')
if __name__ == '__main__':
    main()
