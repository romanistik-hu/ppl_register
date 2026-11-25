import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
'\nPipeline Step 01 (v3): Agnostic Metrics — Multi-Model Validation\n=================================================================\n\nSection 3 updated: replaces Ollama REST backend with native HuggingFace\ntransformers loading for llama3 and mistral. Same sliding-window\nstride-weighted NLL as Salamandra — no API timeouts, no chunking approximation.\n\nHuggingFace token required for gated models:\n    huggingface-cli login\n    or: export HF_TOKEN=your_token\n    or: --hf_token argument\n\nModels (HF IDs):\n    meta-llama/Meta-Llama-3-8B\n    mistralai/Mistral-7B-v0.3\n\nMemory: loaded and unloaded sequentially. Peak VRAM = one 7-8B model at a time\n(~16 GB fp16). A100 80GB handles this comfortably alongside Salamandra.\n\nUsage:\n    python 01_calculate_metrics_v3.py\n    python 01_calculate_metrics_v3.py --skip_validation   # Salamandra only\n    python 01_calculate_metrics_v3.py --hf_token hf_xxx\n'
import os
import re
import gc
import json
import logging
import warnings
import numpy as np
import pandas as pd
import torch
from collections import Counter
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import entropy as scipy_entropy
import nltk
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

class LexicalMetrics:

    def clean_and_tokenize(self, text: str) -> list:
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub('\\d+', '', text)
        text = re.sub('[^\\w\\s]', ' ', text)
        return [t for t in re.sub('\\s+', ' ', text).strip().split() if len(t) >= 2]

    def shannon_entropy(self, tokens: list) -> float:
        if not tokens:
            return 0.0
        total = len(tokens)
        freqs = Counter(tokens)
        return -sum((f / total * np.log2(f / total) for f in freqs.values()))

    def mtld(self, tokens: list, ttr_threshold: float=0.72) -> float:
        if not tokens or len(tokens) < 2:
            return 0.0
        if len(tokens) > 50000:
            step = len(tokens) // 50000
            tokens = tokens[::step]
        return self._mtld_one_pass(tokens, ttr_threshold)

    def _mtld_one_pass(self, tokens: list, thr: float) -> float:
        segments = []
        start = 0
        while start < len(tokens):
            counts, unique, best_end = ({}, 0, start + 1)
            end = start + 1
            while end <= min(start + 1000, len(tokens)):
                tok = tokens[end - 1]
                if tok not in counts:
                    unique += 1
                counts[tok] = counts.get(tok, 0) + 1
                if end - start >= 2 and unique / (end - start) >= thr:
                    best_end = end
                end += 1
            segments.append(best_end - start)
            start = best_end if best_end > start + 1 else start + 2
        return np.mean(segments) if segments else 0.0
PUNCTUATION_SEPARATORS = ['.', '?', '!']

def build_sep_ids(tokenizer) -> set:
    sep_ids = set()
    for sep in PUNCTUATION_SEPARATORS:
        sep_ids.update(tokenizer.encode(sep, add_special_tokens=False))
        sep_ids.update(tokenizer.encode(' ' + sep, add_special_tokens=False))
    return sep_ids

def calculate_ipi(token_ids: list, sep_ids: set) -> float:
    distances, last_pos = ([], -1)
    for i, tid in enumerate(token_ids):
        if tid in sep_ids:
            if i - last_pos > 0:
                distances.append(i - last_pos)
            last_pos = i
    if last_pos < len(token_ids) - 1:
        distances.append(len(token_ids) - 1 - last_pos)
    return float(np.mean(distances)) if distances else float(len(token_ids))

def calculate_mad(attention_matrix: torch.Tensor) -> float:
    num_heads, seq_len, _ = attention_matrix.shape
    idx = torch.arange(seq_len, device=attention_matrix.device).float()
    distances = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    distances = distances.unsqueeze(0).expand(num_heads, -1, -1)
    return ((attention_matrix * distances).sum() / (seq_len * num_heads)).item()

def calculate_attention_entropy(attention_matrix: torch.Tensor) -> float:
    arr = attention_matrix.cpu().numpy()
    return float(np.mean(scipy_entropy(arr, axis=-1)))

def salamandra_metrics(token_ids: list, model, device: str, stride: int=256, max_len: int=512, n_layers_attention: int=4) -> dict:
    mads, att_ents = ([], [])
    loss_numer, loss_denom = (0.0, 0)
    for i in range(0, len(token_ids), stride):
        chunk = token_ids[i:i + max_len]
        if len(chunk) < 10:
            continue
        new_tokens = len(chunk) if i == 0 else min(stride, len(chunk))
        input_ids = torch.tensor([chunk]).to(device)
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
            loss_numer += out.loss.item() * new_tokens
            loss_denom += new_tokens
            last_layers = torch.stack(out.attentions[-n_layers_attention:])
            mean_attn = last_layers.mean(dim=0)[0]
            mads.append(calculate_mad(mean_attn))
            att_ents.append(calculate_attention_entropy(mean_attn))
    return {'perplexity': float(np.exp(loss_numer / loss_denom)) if loss_denom > 0 else 0.0, 'mad': float(np.mean(mads)) if mads else 0.0, 'attention_entropy': float(np.mean(att_ents)) if att_ents else 0.0}

def hf_full_metrics(token_ids: list, model, device: str, stride: int=256, max_len: int=512, n_layers_attention: int=4) -> dict:
    mads, att_ents = ([], [])
    loss_numer, loss_denom = (0.0, 0)
    for i in range(0, len(token_ids), stride):
        chunk = token_ids[i:i + max_len]
        if len(chunk) < 10:
            continue
        new_tokens = len(chunk) if i == 0 else min(stride, len(chunk))
        input_ids = torch.tensor([chunk]).to(device)
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
            loss_numer += out.loss.item() * new_tokens
            loss_denom += new_tokens
            last_layers = torch.stack(out.attentions[-n_layers_attention:])
            mean_attn = last_layers.mean(dim=0)[0]
            mads.append(calculate_mad(mean_attn))
            att_ents.append(calculate_attention_entropy(mean_attn))
    return {'perplexity': float(np.exp(loss_numer / loss_denom)) if loss_denom > 0 else float('nan'), 'attention_entropy': float(np.mean(att_ents)) if att_ents else float('nan'), 'mad': float(np.mean(mads)) if mads else float('nan')}

def load_hf_model(model_id: str, device: str):
    hf_token = os.environ.get('HF_TOKEN', None)
    logger.info(f'Loading {model_id} ...')
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True, output_attentions=True).eval()
    logger.info(f'  Loaded: {model_id}')
    return (tokenizer, model)

def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info('  Model unloaded, VRAM freed')

def load_and_aggregate_corpora(json_path: str) -> dict:
    AGGREGATE_PATTERNS = [('^Jable', 'Jable', 'written-press')]
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raw = {Path(e['archivo']).stem: {'texto': e['texto_completo'], 'tipo': e['tipo_corpus']} for e in data['textos_completos']}
    aggregated, consumed = ({}, set())
    for pattern, agg_name, agg_tipo in AGGREGATE_PATTERNS:
        parts = {k: v for k, v in raw.items() if re.match(pattern, k)}
        if parts:
            combined = ' '.join((v['texto'] for v in parts.values()))
            aggregated[agg_name] = {'texto': combined, 'tipo': agg_tipo}
            consumed.update(parts.keys())
            logger.info(f"Aggregated {len(parts)} sub-corpora → '{agg_name}' ({len(combined):,} chars)")
    for k, v in raw.items():
        if k not in consumed:
            aggregated[k] = v
    return aggregated
HF_VALIDATION_MODELS = {'llama3': 'meta-llama/Meta-Llama-3-8B', 'mistral': 'mistralai/Mistral-7B-v0.3'}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Agnostic Metrics Pipeline v3 — Native HF Multi-model')
    parser.add_argument('--input', type=str, default=str(_paths.CORPUS_JSON))
    parser.add_argument('--output_dir', type=str, default=str(_paths.DATA_DIR / '01_metrics'))
    parser.add_argument('--stride', type=int, default=256, help='Sliding window stride in tokens (default 256)')
    parser.add_argument('--max_len', type=int, default=512, help='Max window length in tokens (default 512)')
    parser.add_argument('--skip_validation', action='store_true', help='Skip llama3/mistral — run Salamandra only')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token (alternative to HF_TOKEN env var)')
    args = parser.parse_args()
    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'=== PIPELINE v3 — device: {device.upper()} ===')
    corpora = load_and_aggregate_corpora(args.input)
    logger.info(f'Corpora: {list(corpora.keys())}')
    results = {name: {'nombre_corpus': name, 'tipo_corpus': info['tipo']} for name, info in corpora.items()}
    logger.info('--- Lexical metrics ---')
    lm = LexicalMetrics()
    for name, info in corpora.items():
        tokens = lm.clean_and_tokenize(info['texto'])
        results[name]['entropy'] = lm.shannon_entropy(tokens)
        results[name]['mtld'] = lm.mtld(tokens)
        logger.info(f"  {name}: entropy={results[name]['entropy']:.3f}  mtld={results[name]['mtld']:.2f}")
    logger.info('--- Salamandra-7B (PPL + MAD + attention_entropy) ---')
    try:
        sal_tok = AutoTokenizer.from_pretrained('BSC-LT/salamandra-7b')
        sal_mdl = AutoModelForCausalLM.from_pretrained('BSC-LT/salamandra-7b', output_attentions=True).to(device).eval().half()
        sep_ids = build_sep_ids(sal_tok)
        for name, info in corpora.items():
            logger.info(f'  Salamandra → {name}')
            token_ids = sal_tok.encode(info['texto'], add_special_tokens=False)
            results[name]['ipi'] = calculate_ipi(token_ids, sep_ids)
            sal = salamandra_metrics(token_ids, sal_mdl, device, stride=args.stride, max_len=args.max_len)
            results[name].update(sal)
            logger.info(f"    ppl={results[name]['perplexity']:.2f}  mad={results[name]['mad']:.2f}  att_ent={results[name]['attention_entropy']:.4f}")
        unload_model(sal_mdl, sal_tok)
    except Exception as e:
        logger.error(f'Salamandra failed: {e}')
        raise
    if args.skip_validation:
        logger.info('--- Validation models: skipped ---')
    else:
        for col_suffix, model_id in HF_VALIDATION_MODELS.items():
            logger.info(f'--- {model_id} (PPL + attention_entropy + MAD) ---')
            try:
                tok, mdl = load_hf_model(model_id, device)
            except Exception as e:
                logger.error(f'  Failed to load {model_id}: {e}')
                logger.error(f'  Set HF_TOKEN env var and ensure you have model access.')
                for name in corpora:
                    for m in ('perplexity', 'attention_entropy', 'mad'):
                        results[name][f'{m}_{col_suffix}'] = float('nan')
                continue
            for name, info in corpora.items():
                logger.info(f'  {col_suffix} → {name}')
                try:
                    token_ids = tok.encode(info['texto'], add_special_tokens=False)
                    metrics = hf_full_metrics(token_ids, mdl, device, stride=args.stride, max_len=args.max_len)
                    results[name][f'perplexity_{col_suffix}'] = metrics['perplexity']
                    results[name][f'attention_entropy_{col_suffix}'] = metrics['attention_entropy']
                    results[name][f'mad_{col_suffix}'] = metrics['mad']
                    logger.info(f"    ppl={metrics['perplexity']:.2f}  att_ent={metrics['attention_entropy']:.4f}  mad={metrics['mad']:.4f}")
                except Exception as e:
                    logger.error(f'    Error on {name}: {e}')
                    for m in ('perplexity', 'attention_entropy', 'mad'):
                        results[name][f'{m}_{col_suffix}'] = float('nan')
            unload_model(mdl, tok)
    df = pd.DataFrame(list(results.values()))
    col_order = ['nombre_corpus', 'tipo_corpus', 'entropy', 'mtld', 'perplexity', 'attention_entropy', 'mad', 'ipi', 'perplexity_llama3', 'attention_entropy_llama3', 'mad_llama3', 'perplexity_mistral', 'attention_entropy_mistral', 'mad_mistral']
    df = df[[c for c in col_order if c in df.columns]]
    out_path = Path(args.output_dir) / 'compiled_metrics_v3.csv'
    df.to_csv(out_path, index=False)
    logger.info(f'Saved: {out_path}')
    ppl_cols = [c for c in ['perplexity', 'perplexity_llama3', 'perplexity_mistral'] if c in df.columns and df[c].notna().any()]
    if len(ppl_cols) >= 2:
        from scipy.stats import spearmanr
        print('\nRANK STABILITY BY PERPLEXITY')
        rank_df = pd.DataFrame({'corpus': df['nombre_corpus']})
        for col in ppl_cols:
            lbl = col.replace('perplexity_', '').replace('perplexity', 'salamandra')
            rank_df[lbl] = df[col].rank(ascending=True).astype('Int64')
        print(rank_df.sort_values('salamandra').to_string(index=False))
        print('\nSpearman rho between model PPL rankings:')
        for i, c1 in enumerate(ppl_cols):
            for c2 in ppl_cols[i + 1:]:
                valid = df[[c1, c2]].dropna()
                if len(valid) >= 5:
                    rho, p = spearmanr(valid[c1], valid[c2])
                    l1 = c1.replace('perplexity_', '').replace('perplexity', 'salamandra')
                    l2 = c2.replace('perplexity_', '').replace('perplexity', 'salamandra')
                    print(f'  {l1} ~ {l2}: rho={rho:.4f}  p={p:.5f}')
        for metric in ('attention_entropy', 'mad'):
            metric_cols = [c for c in df.columns if c == metric or c.startswith(f'{metric}_')]
            if len(metric_cols) >= 2:
                print(f'\nSpearman rho between model {metric} rankings:')
                for i, c1 in enumerate(metric_cols):
                    for c2 in metric_cols[i + 1:]:
                        valid = df[[c1, c2]].dropna()
                        if len(valid) >= 5:
                            rho, p = spearmanr(valid[c1], valid[c2])
                            l1 = c1.replace(f'{metric}_', '').replace(metric, 'salamandra')
                            l2 = c2.replace(f'{metric}_', '').replace(metric, 'salamandra')
                            print(f'  {l1} ~ {l2}: rho={rho:.4f}  p={p:.5f}')
    return df
if __name__ == '__main__':
    main()
