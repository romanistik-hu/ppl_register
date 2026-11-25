import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
'Small-model baseline for the BNC contamination probe: does a small,\nEnglish-native GPT-2 (124M, openai-community/gpt2) show the same\nwritten/spoken gradient on BNC as the three much larger target LLMs?\nParallel to the char-5-gram / GPT-2-Spanish comparison run on the main\n13-corpus study -- same "does scale matter, independent of contamination"\nquestion, applied to the contamination-check corpus itself.\n\nOutput: data/processed/20_gpt2_bnc/gpt2_bnc_ppl.csv\n'
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
BNC_JSON = str(_paths.ROOT / 'data' / 'raw' / 'bnc_corpus.json')
OUT = _paths.DATA_DIR / '20_gpt2_bnc'
OUT.mkdir(parents=True, exist_ok=True)
STRIDE, MAX_LEN = (256, 512)
CHAR_CAP = 300000
MODEL_ID = 'openai-community/gpt2'

def load_bnc_by_genre():
    with open(BNC_JSON) as f:
        data = json.load(f)
    genres = {}
    for e in data['textos_completos']:
        key = e['tipo_corpus']
        genres.setdefault(key, {'medium': e['medium'], 'text': ''})
        if len(genres[key]['text']) < CHAR_CAP:
            genres[key]['text'] += e['texto_completo']
    for k in genres:
        genres[k]['text'] = genres[k]['text'][:CHAR_CAP]
    return genres

def corpus_ppl(token_ids, model, device):
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    genres = load_bnc_by_genre()
    print(f'BNC genres: {len(genres)}, device={device}')
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device).eval()
    n_params = sum((p.numel() for p in model.parameters()))
    print(f'Loaded {MODEL_ID}: {n_params / 1000000.0:.1f}M params')
    rows = []
    for i, (genre, info) in enumerate(sorted(genres.items())):
        ids = tok.encode(info['text'])
        ppl = corpus_ppl(ids, model, device)
        rows.append({'tipo_corpus': genre, 'medium': info['medium'], 'ppl': ppl, 'n_chars': len(info['text'])})
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(genres)} genres done')
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'gpt2_bnc_ppl.csv', index=False)
    print(f"Saved: {OUT / 'gpt2_bnc_ppl.csv'}")
    summ = df.groupby('medium')['ppl'].agg(['mean', 'std', 'count'])
    print(f'\n{summ}')
    ratio = summ.loc['spoken', 'mean'] / summ.loc['written', 'mean']
    print(f'spoken/written PPL ratio: {ratio:.2f}')
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(df[df.medium == 'spoken']['ppl'], df[df.medium == 'written']['ppl'], alternative='greater')
    print(f'Mann-Whitney U={u:.0f}, p={p:.4f}')
if __name__ == '__main__':
    main()
