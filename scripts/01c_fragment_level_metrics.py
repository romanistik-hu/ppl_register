import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
'\nPipeline Step 01c: Fragment-Level Metrics (fixes the method/code mismatch in\n01_calculate_metrics.py, which computes ONE continuous stride-window pass per\nwhole concatenated corpus rather than segmenting into ~500-word fragments and\naveraging per-fragment perplexity, as the paper\'s Method section describes).\n\nFor each of the 3 models (Salamandra-7B, LLaMA3-8B, Mistral-7B-v0.3), for\nevery ~500-word fragment of every corpus, computes in a single forward pass:\n  - perplexity (identical stride-weighted NLL as salamandra_metrics/\n    hf_full_metrics in 01_calculate_metrics.py, just scoped to one fragment\n    instead of the whole corpus)\n  - attention_entropy, mad (identical logic, reused)\n  - mean-pooled last-layer hidden state (embedding baseline — answers the\n    "extracts attention but ignores hidden states" criticism)\n  - Min-20%-Prob surprisal (mean NLL of the 20% least-likely "new" tokens per\n    fragment — a membership-inference-style contamination probe input)\n\nOutputs (written incrementally per corpus so partial progress is never lost):\n  paper_final/data/08_fragment_level/fragment_level_ppl_v4.csv   (long format)\n  paper_final/data/08_fragment_level/compiled_metrics_v4.csv     (corpus-level\n      aggregate, same column schema as compiled_metrics_v3.csv — drop-in\n      replacement for 03_agnostic_analysis.py / compute_bpc.py /\n      bootstrap_ci_loo.py)\n  paper_final/data/08_fragment_level/hidden_states_<model>.npy   (n_fragments\n      x hidden_dim, row order matches fragment_level_ppl_v4.csv for that model)\n\nUsage:\n    HF_TOKEN=hf_xxx python3 scripts/01c_fragment_level_metrics.py\n'
import gc
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
CORPUS_JSON = str(_paths.CORPUS_JSON)
OUT_DIR = _paths.DATA_DIR / '08_fragment_level'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FRAGMENT_WORDS = 500
MIN_TAIL_WORDS = 100
STRIDE = 256
MAX_LEN = 512
N_LAYERS_ATTENTION = 4
MINK_FRACTION = 0.2
MODELS = {'salamandra': 'BSC-LT/salamandra-7b', 'llama3': 'meta-llama/Meta-Llama-3-8B', 'mistral': 'mistralai/Mistral-7B-v0.3'}

def load_and_aggregate_corpora(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raw = {Path(e['archivo']).stem: {'texto': e['texto_completo'], 'tipo': e['tipo_corpus']} for e in data['textos_completos']}
    aggregated, consumed = ({}, set())
    parts = {k: v for k, v in raw.items() if re.match('^Jable', k)}
    if parts:
        combined = ' '.join((v['texto'] for v in parts.values()))
        aggregated['Jable'] = {'texto': combined, 'tipo': 'written-press'}
        consumed.update(parts.keys())
    for k, v in raw.items():
        if k not in consumed:
            aggregated[k] = v
    return aggregated

def segment_fragments(text: str, n_words: int=FRAGMENT_WORDS, min_tail: int=MIN_TAIL_WORDS):
    words = text.split()
    fragments = []
    for i in range(0, len(words), n_words):
        fragments.append(' '.join(words[i:i + n_words]))
    if len(fragments) >= 2 and len(fragments[-1].split()) < min_tail:
        fragments[-2] = fragments[-2] + ' ' + fragments[-1]
        fragments.pop()
    return fragments

def calculate_mad(attention_matrix: torch.Tensor, normalize: bool=False) -> float:
    num_heads, seq_len, _ = attention_matrix.shape
    idx = torch.arange(seq_len, device=attention_matrix.device).float()
    distances = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    distances = distances.unsqueeze(0).expand(num_heads, -1, -1)
    raw = ((attention_matrix * distances).sum() / (seq_len * num_heads)).item()
    if normalize:
        return raw / max(seq_len - 1, 1)
    return raw

def calculate_attention_entropy(attention_matrix: torch.Tensor) -> float:
    arr = attention_matrix.cpu().numpy()
    return float(np.mean(scipy_entropy(arr, axis=-1)))

def fragment_metrics(token_ids: list, model, device: str):
    mads, mads_norm, att_ents = ([], [], [])
    loss_numer, loss_denom = (0.0, 0)
    new_token_nlls = []
    hidden_sum, hidden_weight = (None, 0)
    for i in range(0, len(token_ids), STRIDE):
        chunk = token_ids[i:i + MAX_LEN]
        if len(chunk) < 10:
            continue
        new_tokens = len(chunk) if i == 0 else min(STRIDE, len(chunk))
        input_ids = torch.tensor([chunk]).to(device)
        with torch.no_grad():
            out = model(input_ids, output_attentions=True, output_hidden_states=True)
            logits = out.logits[0, :-1, :]
            labels = input_ids[0, 1:]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_nll = -log_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
            window_loss = token_nll.mean().item()
            loss_numer += window_loss * new_tokens
            loss_denom += new_tokens
            tail_n = min(new_tokens, token_nll.shape[0])
            new_token_nlls.append(token_nll[-tail_n:].detach().cpu().numpy())
            last_layers = torch.stack(out.attentions[-N_LAYERS_ATTENTION:])
            mean_attn = last_layers.mean(dim=0)[0]
            mads.append(calculate_mad(mean_attn, normalize=False))
            mads_norm.append(calculate_mad(mean_attn, normalize=True))
            att_ents.append(calculate_attention_entropy(mean_attn))
            last_hidden = out.hidden_states[-1][0]
            tail_h = last_hidden[-tail_n:].float().mean(dim=0).cpu().numpy()
            if hidden_sum is None:
                hidden_sum = tail_h * new_tokens
            else:
                hidden_sum = hidden_sum + tail_h * new_tokens
            hidden_weight += new_tokens
        del out, logits, log_probs, token_nll
    torch.cuda.empty_cache()
    all_nlls = np.concatenate(new_token_nlls) if new_token_nlls else np.array([])
    if all_nlls.size > 0:
        k = max(1, int(len(all_nlls) * MINK_FRACTION))
        mink_nll = float(np.mean(np.sort(all_nlls)[-k:]))
    else:
        mink_nll = float('nan')
    hidden_mean = hidden_sum / hidden_weight if hidden_weight > 0 else None
    return {'perplexity': float(np.exp(loss_numer / loss_denom)) if loss_denom > 0 else float('nan'), 'attention_entropy': float(np.mean(att_ents)) if att_ents else float('nan'), 'mad': float(np.mean(mads)) if mads else float('nan'), 'mad_norm': float(np.mean(mads_norm)) if mads_norm else float('nan'), 'mink20_nll': mink_nll, 'hidden': hidden_mean}

def load_model(model_key: str, model_id: str, device: str):
    hf_token = os.environ.get('HF_TOKEN')
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(token=hf_token, trust_remote_code=True, output_attentions=True, output_hidden_states=True)
    if model_key == 'salamandra':
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device).eval().half()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto', **kwargs).eval()
    return (tokenizer, model)

def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'=== Fragment-level metrics — device: {device.upper()} ===')
    corpora = load_and_aggregate_corpora(CORPUS_JSON)
    fragments_by_corpus = {name: segment_fragments(info['texto']) for name, info in corpora.items()}
    total_fragments = sum((len(v) for v in fragments_by_corpus.values()))
    logger.info(f'Corpora: {len(corpora)}, total fragments: {total_fragments}')
    for name, frags in fragments_by_corpus.items():
        logger.info(f'  {name}: {len(frags)} fragments')
    for model_key, model_id in MODELS.items():
        row_csv = OUT_DIR / f'fragment_level_ppl_v4_{model_key}.csv'
        hidden_npy = OUT_DIR / f'hidden_states_{model_key}.npy'
        if row_csv.exists():
            logger.info(f'{model_key}: output already exists, skipping ({row_csv})')
            continue
        logger.info(f'--- Loading {model_id} ---')
        t0 = time.time()
        try:
            tok, mdl = load_model(model_key, model_id, device)
        except Exception as e:
            logger.error(f'Failed to load {model_id}: {e}')
            continue
        logger.info(f'  Loaded in {time.time() - t0:.1f}s')
        rows = []
        hidden_vecs = []
        n_done = 0
        t_start = time.time()
        for corpus_name, info in corpora.items():
            frags = fragments_by_corpus[corpus_name]
            for frag_idx, frag_text in enumerate(frags):
                token_ids = tok.encode(frag_text, add_special_tokens=False)
                try:
                    m = fragment_metrics(token_ids, mdl, device)
                except Exception as e:
                    logger.error(f'  error on {corpus_name}#{frag_idx}: {e}')
                    m = {'perplexity': float('nan'), 'attention_entropy': float('nan'), 'mad': float('nan'), 'mad_norm': float('nan'), 'mink20_nll': float('nan'), 'hidden': None}
                rows.append({'fragment_id': f'{corpus_name}_{frag_idx:05d}', 'corpus': corpus_name, 'tipo_corpus': info['tipo'], 'word_count': len(frag_text.split()), 'perplexity': m['perplexity'], 'attention_entropy': m['attention_entropy'], 'mad': m['mad'], 'mad_norm': m['mad_norm'], 'mink20_nll': m['mink20_nll']})
                hidden_dim = m['hidden'].shape[0] if m['hidden'] is not None else 0
                hidden_vecs.append(m['hidden'] if m['hidden'] is not None else np.full((hidden_dim or 4096,), np.nan, dtype=np.float32))
                n_done += 1
                if n_done % 500 == 0:
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed
                    eta_min = (total_fragments - n_done) / rate / 60 if rate > 0 else float('nan')
                    logger.info(f'  [{model_key}] {n_done}/{total_fragments} fragments ({rate:.2f}/s, ETA {eta_min:.0f} min)')
            pd.DataFrame(rows).to_csv(row_csv, index=False)
            np.save(hidden_npy, np.vstack(hidden_vecs).astype(np.float32))
            logger.info(f"  Checkpoint saved after corpus '{corpus_name}' ({len(rows)} fragments so far)")
        unload_model(mdl, tok)
        logger.info(f'--- {model_key} done in {(time.time() - t0) / 60:.1f} min ---')
    logger.info('--- Compiling corpus-level v4 aggregate ---')
    per_model_agg = {}
    for model_key in MODELS:
        row_csv = OUT_DIR / f'fragment_level_ppl_v4_{model_key}.csv'
        if not row_csv.exists():
            logger.warning(f'Missing {row_csv}, skipping in aggregate')
            continue
        df = pd.read_csv(row_csv)
        agg = df.groupby(['corpus', 'tipo_corpus']).agg(perplexity=('perplexity', 'mean'), attention_entropy=('attention_entropy', 'mean'), mad=('mad', 'mean'), mad_norm=('mad_norm', 'mean'), mink20_nll=('mink20_nll', 'mean'), n_fragments=('perplexity', 'size')).reset_index()
        per_model_agg[model_key] = agg
    if 'salamandra' in per_model_agg:
        compiled = per_model_agg['salamandra'].rename(columns={'corpus': 'nombre_corpus'})
        for model_key in ('llama3', 'mistral'):
            if model_key in per_model_agg:
                sub = per_model_agg[model_key][['corpus', 'perplexity', 'attention_entropy', 'mad', 'mad_norm']]
                sub = sub.rename(columns={'corpus': 'nombre_corpus', 'perplexity': f'perplexity_{model_key}', 'attention_entropy': f'attention_entropy_{model_key}', 'mad': f'mad_{model_key}', 'mad_norm': f'mad_norm_{model_key}'})
                compiled = compiled.merge(sub, on='nombre_corpus', how='left')
        compiled.to_csv(OUT_DIR / 'compiled_metrics_v4.csv', index=False)
        logger.info(f"Saved: {OUT_DIR / 'compiled_metrics_v4.csv'}")
        print(compiled)
    else:
        logger.warning('Salamandra results missing — cannot build compiled_metrics_v4.csv yet')
if __name__ == '__main__':
    main()
