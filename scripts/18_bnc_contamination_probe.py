import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
'Contamination-probe supplementary check (user-requested extension): BNC\n(British National Corpus) is one of the most widely-redistributed English\ncorpora in NLP and is highly plausible to appear in web-crawl-based\npretraining data for all three target LLMs -- unlike the private Canarian\noral corpora, this is a corpus we can be reasonably confident IS\n"contaminated". If perplexity mainly reflects training-data familiarity\nrather than genuine register/production-condition sensitivity, a\nnear-certainly-contaminated corpus like BNC should show flat/compressed\nperplexity across registers (memorization ceiling), not a graded\nwritten/spoken structure. Tests all THREE target models (Salamandra,\nLLaMA3, Mistral) -- not just the two English-dominant ones -- since\nSalamandra\'s own pretraining (per its model card) downsamples English only\nto 50%, not zero, so English text is still a substantial share.\n\nRegisters aggregated by BNC\'s raw genre code (tipo_corpus, ~70 categories);\neach genre\'s concatenated text capped at 300k characters to bound compute.\n\nOutput: paper_final/data/18_bnc_probe/bnc_perplexity_by_genre.csv\n'
import gc
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
BNC_JSON = str(_paths.ROOT / 'data' / 'raw' / 'bnc_corpus.json')
BASE = _paths.DATA_DIR
OUT = BASE / '18_bnc_probe'
OUT.mkdir(parents=True, exist_ok=True)
STRIDE, MAX_LEN = (256, 512)
CHAR_CAP = 300000
MODELS = {'salamandra': 'BSC-LT/salamandra-7b', 'llama3': 'meta-llama/Meta-Llama-3-8B', 'mistral': 'mistralai/Mistral-7B-v0.3'}

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
    print(f'BNC genres: {len(genres)}')
    hf_token = os.environ.get('HF_TOKEN')
    for model_key, model_id in MODELS.items():
        out_csv = OUT / f'bnc_ppl_{model_key}.csv'
        if out_csv.exists():
            print(f'{model_key}: already done, skipping')
            continue
        print(f'--- Loading {model_id} ---')
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
        if model_key == 'salamandra':
            model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token).to(device).eval().half()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, torch_dtype=torch.float16, device_map='auto').eval()
        print(f'  loaded in {time.time() - t0:.1f}s')
        rows = []
        t1 = time.time()
        for i, (genre, info) in enumerate(sorted(genres.items())):
            ids = tok.encode(info['text'], add_special_tokens=False)
            ppl = corpus_ppl(ids, model, device)
            rows.append({'tipo_corpus': genre, 'medium': info['medium'], 'ppl': ppl, 'n_chars': len(info['text'])})
            if (i + 1) % 10 == 0:
                print(f'  [{model_key}] {i + 1}/{len(genres)} genres, {(time.time() - t1) / 60:.1f} min elapsed')
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f'  Saved: {out_csv} ({(time.time() - t0) / 60:.1f} min total)')
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
    print('\n--- Summary (written vs spoken mean PPL per model) ---')
    for model_key in MODELS:
        p = OUT / f'bnc_ppl_{model_key}.csv'
        if p.exists():
            df = pd.read_csv(p)
            summ = df.groupby('medium')['ppl'].agg(['mean', 'std', 'count'])
            print(f'\n{model_key}:\n{summ}')
            ratio = summ.loc['spoken', 'mean'] / summ.loc['written', 'mean']
            print(f'  spoken/written PPL ratio: {ratio:.2f}')
if __name__ == '__main__':
    main()
