import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as _paths
B = _paths.DATA_DIR
OUT = _paths.FIGURES_DIR
ALIAS = {'PARCAN_declaraciones': 'CoParCan-D', 'PARCAN_discursos': 'CoParCan-S', 'Jable': 'Jable', 'Bot_individual': 'CoACan-Indiv', 'Bot_pareja': 'CoACan-Pair', 'Carto_expo': 'CoACan-Pres', 'Carto_grupo': 'CoACan-Cart', 'COSER': 'COSER', 'PRESEEA_alto': 'PRESEEA-H', 'PRESEEA_medio': 'PRESEEA-M', 'PRESEEA_bajo': 'PRESEEA-L', 'WhCan': 'CoCanWA', 'AMERESCO': 'AMERESCO'}
MODE = {'PARCAN_declaraciones': 'written', 'PARCAN_discursos': 'written', 'Jable': 'written', 'Bot_individual': 'oral, asynchronous', 'Bot_pareja': 'oral, asynchronous', 'WhCan': 'oral, asynchronous'}
STYLE = {'written': ('#2a78d6', 's'), 'oral, asynchronous': ('#eb6834', '^'), 'oral': ('#3d8f5f', 'o')}
INK, MUTED = ('#222222', '#777777')
plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})

def mode(c):
    return MODE.get(c, 'oral')

def scatter_modes(ax, x, y, size=18):
    for md, (col, mk) in STYLE.items():
        sel = [c for c in x.index if mode(c) == md]
        ax.scatter(x[sel], y[sel], s=size, c=col, marker=mk, edgecolors='white', linewidths=0.5, label=md, zorder=3)

def label_points(ax, x, y, offsets, size=5.5):
    for c in x.index:
        dx, dy, ha = offsets.get(c, (4, 3, 'left'))
        ax.annotate(ALIAS[c], (x[c], y[c]), xytext=(dx, dy), textcoords='offset points', fontsize=size, color=INK, ha=ha, va='center')

def fig_convergence():
    f1 = pd.read_csv(B / '02_biber_mda/biber_factor_scores.csv').set_index('nombre_corpus')['F1']
    pc = -pd.read_csv(B / '03_agnostic_pca/pca_scores_salamandra_v3.csv').set_index('nombre_corpus')['pc1_salamandra']
    f1 = f1.loc[pc.index]
    fig, a = plt.subplots(figsize=(3.3, 2.9))
    scatter_modes(a, f1, pc, size=22)
    off = {'PARCAN_declaraciones': (5, 1, 'left'), 'PARCAN_discursos': (5, -4, 'left'), 'Jable': (5, 2, 'left'), 'Bot_individual': (6, -6, 'left'), 'Bot_pareja': (6, 5, 'left'), 'Carto_expo': (5, 0, 'left'), 'Carto_grupo': (5, -4, 'left'), 'WhCan': (5, -3, 'left'), 'COSER': (-1, -7, 'center'), 'PRESEEA_alto': (-5, -3, 'right'), 'PRESEEA_medio': (-5, 4, 'right'), 'PRESEEA_bajo': (5, 4, 'left'), 'AMERESCO': (-5, 0, 'right')}
    label_points(a, f1, pc, off, size=6)
    a.axhline(0, color=MUTED, lw=0.4, zorder=1)
    a.axvline(0, color=MUTED, lw=0.4, zorder=1)
    a.set_xlim(-2.4, 2.6)
    a.set_ylim(-3.1, 3.0)
    a.set_xlabel('MD Factor 1 (informational to involved)')
    a.set_ylabel('Agnostic PC1 (planned to spontaneous)')
    a.legend(frameon=False, fontsize=6, loc='lower center', ncol=3, handletextpad=0.1, columnspacing=0.8, bbox_to_anchor=(0.5, 1.0), borderpad=0)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_convergence.png', dpi=300)
    plt.close(fig)

def fig_biplot():
    sc = pd.read_csv(B / '03_agnostic_pca/pca_scores_salamandra_v3.csv').set_index('nombre_corpus')
    ld = pd.read_csv(B / '03_agnostic_pca/pca_loadings_salamandra_v3.csv', index_col=0)
    x, y = (-sc['pc1_salamandra'], sc['pc2_salamandra'])
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    scatter_modes(ax, x, y)
    off = {'PARCAN_declaraciones': (5, 0, 'left'), 'PARCAN_discursos': (5, 4, 'left'), 'Jable': (5, -4, 'left'), 'Bot_individual': (5, -3, 'left'), 'Bot_pareja': (-5, 2, 'right'), 'Carto_expo': (-5, 3, 'right'), 'Carto_grupo': (5, -4, 'left'), 'WhCan': (5, 4, 'left'), 'COSER': (-5, 3, 'right'), 'PRESEEA_alto': (-5, -1, 'right'), 'PRESEEA_medio': (0, -7, 'center'), 'PRESEEA_bajo': (5, 3, 'left'), 'AMERESCO': (-2, 7, 'center')}
    label_points(ax, x, y, off)
    names = {'perplexity': 'PPL', 'attention_entropy': 'H', 'mad': 'MAD'}
    for k, lab in names.items():
        lx, ly = (-ld.loc[k, 'PC1'], ld.loc[k, 'PC2'])
        ax.annotate('', xy=(lx, ly), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.8, shrinkA=0, shrinkB=0))
        ax.text(lx * 1.12, ly * 1.12, lab, fontsize=6.5, color=INK, ha='center', va='center', fontweight='bold')
    ax.axhline(0, color=MUTED, lw=0.4, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.4, zorder=1)
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-1.0, 1.1)
    ax.set_xlabel('PC1, 91.3% of variance (planned to spontaneous)')
    ax.set_ylabel('PC2, 7.8% of variance')
    ax.legend(frameon=False, fontsize=6, loc='lower center', ncol=3, handletextpad=0.1, columnspacing=0.8, bbox_to_anchor=(0.5, 1.0), borderpad=0)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_pca_biplot.png', dpi=300)
    plt.close(fig)

def fig_ranks():
    m = pd.read_csv(B / '01_metrics/compiled_metrics_v3.csv').set_index('nombre_corpus').sort_values('perplexity')
    models = [('perplexity', 'Salamandra-7B', '#2a78d6'), ('perplexity_llama3', 'LLaMA3-8B', '#eb6834'), ('perplexity_mistral', 'Mistral-7B-v0.3', '#3d8f5f')]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    w = 0.26
    for k, (col, lab, colour) in enumerate(models):
        ax.bar([i + (k - 1) * w for i in range(len(m))], m[col], width=w - 0.02, color=colour, label=lab, zorder=3)
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels([ALIAS[c] for c in m.index], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Perplexity')
    ax.grid(axis='y', color='#dddddd', lw=0.4, zorder=0)
    ax.tick_params(axis='x', length=0)
    ax.legend(frameon=False, fontsize=6, loc='upper left', handlelength=1.0, borderpad=0)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_ranks.png', dpi=300)
    plt.close(fig)
if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_ranks()
    fig_convergence()
    fig_biplot()
    print('figures written to', OUT)
