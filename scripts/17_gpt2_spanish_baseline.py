import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CORPUS_JSON = str(_paths.CORPUS_JSON)
BASE = _paths.DATA_DIR
OUT = BASE / '17_gpt2_baseline'
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = 'datificate/gpt2-small-spanish'
STRIDE, MAX_LEN = (256, 512)

def load_and_aggregate_corpora(json_path):
    with open(json_path) as f:
        data = json.load(f)
    raw = {Path(e['archivo']).stem: e['texto_completo'] for e in data['textos_completos']}
    aggregated, consumed = ({}, set())
    parts = {k: v for k, v in raw.items() if re.match('^Jable', k)}
    if parts:
        aggregated['Jable'] = ' '.join(parts.values())
        consumed.update(parts.keys())
    for k, v in raw.items():
        if k not in consumed:
            aggregated[k] = v
    return aggregated

def corpus_ppl(token_ids, model, device):
    import torch
    loss_numer, loss_denom = (0.0, 0)
    for i in range(0, len(token_ids), STRIDE):
        chunk = token_ids[i:i + MAX_LEN]
        if len(chunk) < 10:
            continue
        new_tokens = len(chunk) if i == 0 else min(STRIDE, len(chunk))
        input_ids = torch.tensor([chunk]).to(device)
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
            loss_numer += out.loss.item() * new_tokens
            loss_denom += new_tokens
    return float(np.exp(loss_numer / loss_denom)) if loss_denom > 0 else float('nan')

def main():
    if not Path(CORPUS_JSON).exists():
        print('Corpus texts not found: reusing the stored metrics per corpus.')
        df = pd.read_csv(OUT / 'gpt2_spanish_ppl.csv').set_index('nombre_corpus')
        n_params = float(pd.read_csv(OUT / 'gpt2_baseline_spearman.csv')['n_params_M'].iloc[0]) * 1000000.0
        return correlations(df, n_params)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device).eval()
    n_params = sum((p.numel() for p in model.parameters()))
    print(f'Loaded {MODEL_ID}: {n_params / 1000000.0:.1f}M params, device={device}')
    corpora = load_and_aggregate_corpora(CORPUS_JSON)
    rows = []
    for name, text in sorted(corpora.items()):
        ids = tok.encode(text, add_special_tokens=False)
        ppl = corpus_ppl(ids, model, device)
        rows.append({'nombre_corpus': name, 'gpt2_es_ppl': ppl})
        print(f'  {name:<24} PPL={ppl:.2f}')
    df = pd.DataFrame(rows).set_index('nombre_corpus')
    df.to_csv(OUT / 'gpt2_spanish_ppl.csv')
    print(f"\nSaved: {OUT / 'gpt2_spanish_ppl.csv'}")
    correlations(df, n_params)

def correlations(df, n_params):
    f1 = pd.read_csv(BASE / '02_biber_mda/biber_factor_scores.csv').set_index('nombre_corpus')['F1']
    ALIAS = {'PARCAN_declaraciones': 'CoParCan-D', 'PARCAN_discursos': 'CoParCan-S', 'Jable': 'Jable', 'Bot_individual': 'CoACan-Indiv', 'Bot_pareja': 'CoACan-Pair', 'Carto_expo': 'CoACan-Pres', 'Carto_grupo': 'CoACan-Cart', 'COSER': 'COSER', 'PRESEEA_alto': 'PRESEEA-H', 'PRESEEA_medio': 'PRESEEA-M', 'PRESEEA_bajo': 'PRESEEA-L', 'WhCan': 'CoCanWA', 'AMERESCO': 'AMERESCO'}
    ko_new = pd.read_csv(BASE / '06_ko_analysis/ko_coding_matrix_final.csv', index_col=0)['ko_sum_final']
    ppl_sal = pd.read_csv(BASE / '01_metrics/compiled_metrics_v3.csv').set_index('nombre_corpus')['perplexity'].rename(index=ALIAS)
    df_alias = df.rename(index=ALIAS)
    common = sorted(set(df_alias.index) & set(f1.rename(index=ALIAS).index) & set(ko_new.index))
    f1_a = f1.rename(index=ALIAS)
    r_f1, p_f1 = spearmanr(df_alias.loc[common, 'gpt2_es_ppl'], f1_a.loc[common])
    r_ko, p_ko = spearmanr(df_alias.loc[common, 'gpt2_es_ppl'], ko_new.loc[common])
    r_sal, p_sal = spearmanr(df_alias.loc[common, 'gpt2_es_ppl'], ppl_sal.loc[common])
    print(f'\nrho(GPT2-es PPL, MD F1)  = {r_f1:+.3f} (p={p_f1:.4f})')
    print(f'rho(GPT2-es PPL, K&O)    = {r_ko:+.3f} (p={p_ko:.4f})')
    print(f'rho(GPT2-es PPL, Salamandra PPL) = {r_sal:+.3f} (p={p_sal:.4f})')
    pd.DataFrame([{'baseline': 'gpt2_small_spanish_124M', 'n_params_M': round(n_params / 1000000.0, 1), 'rho_MD_F1': round(r_f1, 3), 'p_MD_F1': round(p_f1, 5), 'rho_KO': round(r_ko, 3), 'p_KO': round(p_ko, 5), 'rho_vs_salamandra': round(r_sal, 3), 'p_vs_salamandra': round(p_sal, 5), 'n': len(common)}]).to_csv(OUT / 'gpt2_baseline_spearman.csv', index=False)
    print(f"Saved: {OUT / 'gpt2_baseline_spearman.csv'}")
if __name__ == '__main__':
    main()
