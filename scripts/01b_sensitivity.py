import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
'\nPipeline Step 01b: Length Sensitivity Analysis\n\nPurpose:\n    Tests whether the perplexity and attention-based rankings obtained in\n    01_calculate_metrics_v3.py are confounded by corpus length differences.\n\nMethod:\n    1. Identifies the shortest corpus in the collection (length ceiling).\n    2. Subsamples every longer corpus to that ceiling using a fixed contiguous\n       window (reproducible; window start selected via fixed random seed).\n    3. Re-computes Salamandra-7B metrics (perplexity, MAD, attention_entropy)\n       and IPI on the subsampled texts.\n    4. Exports compiled_metrics_sensitivity.csv with the same schema as\n       compiled_metrics_v3.csv.\n    5. Compares the two rankings (full vs. subsampled) via Spearman correlation\n       for each metric. High ρ → metric is robust to length differences.\n\nWhy a contiguous window (not random token sampling):\n    Random token sampling breaks syntactic and discourse structure, making\n    perplexity estimates unreliable (the model sees decontextualised fragments).\n    A contiguous window preserves local coherence while controlling length.\n\nLimitations (to report in paper):\n    - A single window per corpus introduces position bias. To reduce it, the\n      script averages over --n_windows independent contiguous windows per corpus\n      (default: 5, each starting at a different equidistant position).\n    - Entropy and MTLD are recomputed on the subsampled token sequences for\n      completeness, but length-dependence of MTLD is already controlled by\n      its algorithm design (McCarthy & Jarvis 2010).\n\nInput:\n    data/raw/textos_todos_corpus_new.json\n    data/processed/01_metrics/compiled_metrics_v3.csv  (full metrics)\n\nOutput:\n    data/processed/01_metrics/compiled_metrics_sensitivity.csv\n    data/processed/01_metrics/sensitivity_spearman.csv\n    data/processed/01_metrics/sensitivity_ranking_comparison.csv\n'
import argparse
import gc
import json
import logging
import re
import warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy as scipy_entropy
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
MODEL_NAME = 'BSC-LT/salamandra-7b'
PUNCTUATION_SEPARATORS = ['.', '?', '!']
STRIDE = 256
MAX_LEN = 512
N_LAYERS_ATTENTION = 4
METRICS_TO_COMPARE = ['perplexity', 'attention_entropy', 'mad', 'ipi', 'entropy', 'mtld']
AGGREGATE_PATTERNS = [('^Jable', 'Jable', 'written-press')]

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
            logger.info(f"Aggregated {len(parts)} sub-corpora → '{agg_name}' ({len(combined):,} chars)")
    for k, v in raw.items():
        if k not in consumed:
            aggregated[k] = v
    return aggregated

def build_sep_ids(tokenizer) -> set:
    sep_ids = set()
    for sep in PUNCTUATION_SEPARATORS:
        sep_ids.update(tokenizer.encode(sep, add_special_tokens=False))
        sep_ids.update(tokenizer.encode(' ' + sep, add_special_tokens=False))
    return sep_ids

def calculate_ipi(token_ids: list, sep_ids: set) -> float:
    distances = []
    last_pos = -1
    for i, tid in enumerate(token_ids):
        if tid in sep_ids:
            if i - last_pos > 0:
                distances.append(i - last_pos)
            last_pos = i
    if last_pos < len(token_ids) - 1:
        distances.append(len(token_ids) - 1 - last_pos)
    return float(np.mean(distances)) if distances else float(len(token_ids))

def calculate_mad(attn: torch.Tensor) -> float:
    num_heads, seq_len, _ = attn.shape
    idx = torch.arange(seq_len, device=attn.device).float()
    dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    dist = dist.unsqueeze(0).expand(num_heads, -1, -1)
    return ((attn * dist).sum() / (seq_len * num_heads)).item()

def calculate_attention_entropy(attn: torch.Tensor) -> float:
    arr = attn.cpu().numpy()
    return float(np.mean(scipy_entropy(arr, axis=-1)))

def salamandra_metrics_on_tokens(token_ids: list, model, device: str, sep_ids: set) -> dict:
    mads, att_ents = ([], [])
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
            last_layers = torch.stack(out.attentions[-N_LAYERS_ATTENTION:])
            mean_attn = last_layers.mean(dim=0)[0]
            mads.append(calculate_mad(mean_attn))
            att_ents.append(calculate_attention_entropy(mean_attn))
    return {'perplexity': float(np.exp(loss_numer / loss_denom)) if loss_denom > 0 else 0.0, 'mad': float(np.mean(mads)) if mads else 0.0, 'attention_entropy': float(np.mean(att_ents)) if att_ents else 0.0, 'ipi': calculate_ipi(token_ids, sep_ids)}

def lexical_metrics(tokens: list) -> dict:
    if not tokens:
        return {'entropy': 0.0, 'mtld': 0.0}
    total = len(tokens)
    freqs = Counter(tokens)
    ent = -sum((f / total * np.log2(f / total) for f in freqs.values()))
    thr = 0.72
    segments = []
    start = 0
    while start < len(tokens):
        window = min(start + 1000, len(tokens))
        counts_w = {}
        unique = 0
        best_end = start + 1
        end = start + 1
        while end <= window:
            tok = tokens[end - 1]
            if tok not in counts_w:
                counts_w[tok] = 0
                unique += 1
            counts_w[tok] += 1
            span = end - start
            if span >= 2 and unique / span >= thr:
                best_end = end
            end += 1
        if best_end > start + 1:
            segments.append(best_end - start)
            start = best_end
        else:
            segments.append(2)
            start += 2
    mtld = float(np.mean(segments)) if segments else 0.0
    return {'entropy': float(ent), 'mtld': mtld}

def word_tokenize_simple(text: str) -> list:
    text = text.lower()
    text = re.sub('\\d+', '', text)
    text = re.sub('[^\\w\\s]', ' ', text)
    text = re.sub('\\s+', ' ', text)
    return [t for t in text.strip().split() if len(t) >= 2]

def subsample_tokens(token_ids: list, ceiling: int, n_windows: int, seed: int) -> list:
    if len(token_ids) <= ceiling:
        return [token_ids]
    rng = np.random.default_rng(seed)
    spacing = (len(token_ids) - ceiling) / max(n_windows - 1, 1)
    windows = []
    for w in range(n_windows):
        base_start = int(w * spacing)
        jitter = int(rng.integers(-int(0.05 * spacing + 1), int(0.05 * spacing + 1) + 1))
        start = max(0, min(base_start + jitter, len(token_ids) - ceiling))
        windows.append(token_ids[start:start + ceiling])
    return windows

def main():
    parser = argparse.ArgumentParser(description='Length Sensitivity Analysis — Pipeline Step 01b')
    parser.add_argument('--input_json', type=str, default=str(_paths.CORPUS_JSON))
    parser.add_argument('--input_full_metrics', type=str, default=str(_paths.DATA_DIR / '01_metrics' / 'compiled_metrics_v3.csv'), help='Full-length metrics CSV from 01_calculate_metrics_v3.py')
    parser.add_argument('--output_dir', type=str, default=str(_paths.DATA_DIR / '01_metrics'))
    parser.add_argument('--n_windows', type=int, default=5, help='Number of equidistant windows per corpus for averaging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for window jitter (reproducibility)')
    parser.add_argument('--ceiling_corpus', type=str, default=None, help='Force a specific corpus as length ceiling (default: auto = shortest)')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('=== PIPELINE 01b: LENGTH SENSITIVITY ===')
    corpora = load_and_aggregate(args.input_json)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Cargando tokenizador: {MODEL_NAME}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sep_ids = build_sep_ids(tokenizer)
    logger.info('Tokenizando todos los corpora...')
    tokenized = {}
    word_tokens = {}
    for name, info in corpora.items():
        tokenized[name] = tokenizer.encode(info['texto'], add_special_tokens=False)
        word_tokens[name] = word_tokenize_simple(info['texto'])
        logger.info(f"  {name}: {len(tokenized[name]):,} tokens (chars: {len(info['texto']):,})")
    lengths = {name: len(toks) for name, toks in tokenized.items()}
    if args.ceiling_corpus:
        if args.ceiling_corpus not in lengths:
            logger.error(f"--ceiling_corpus '{args.ceiling_corpus}' not found. Available: {list(lengths.keys())}")
            return
        ceiling_name = args.ceiling_corpus
    else:
        ceiling_name = min(lengths, key=lengths.get)
    ceiling = lengths[ceiling_name]
    logger.info(f"\nCeiling corpus: '{ceiling_name}' → {ceiling:,} tokens\nCorpora that will be subsampled: {[n for n, l in lengths.items() if l > ceiling]}")
    logger.info(f'\nCargando modelo: {MODEL_NAME}')
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_attentions=True).to(device).eval().half()
    logger.info(f'Modelo en {device.upper()}')
    sensitivity_results = []
    for name, info in corpora.items():
        logger.info(f'\nProcesando: {name} ({lengths[name]:,} tokens)')
        windows = subsample_tokens(tokenized[name], ceiling, args.n_windows, args.seed)
        logger.info(f'  Ventanas: {len(windows)} × {len(windows[0]):,} tokens')
        window_metrics = []
        for w_idx, window_toks in enumerate(windows):
            logger.info(f'  Ventana {w_idx + 1}/{len(windows)}...')
            sal = salamandra_metrics_on_tokens(window_toks, model, device, sep_ids)
            ratio = len(window_toks) / max(len(tokenized[name]), 1)
            n_word_toks = max(int(ratio * len(word_tokens[name])), 1)
            word_start = int((len(word_tokens[name]) - n_word_toks) * (w_idx / max(len(windows) - 1, 1)))
            word_start = max(0, min(word_start, len(word_tokens[name]) - n_word_toks))
            lex = lexical_metrics(word_tokens[name][word_start:word_start + n_word_toks])
            window_metrics.append({**sal, **lex})
        avg = {metric: float(np.mean([wm[metric] for wm in window_metrics])) for metric in window_metrics[0]}
        row = {'nombre_corpus': name, 'tipo_corpus': info['tipo'], 'n_tokens_full': lengths[name], 'n_tokens_subsampled': ceiling, 'n_windows': len(windows), **avg}
        sensitivity_results.append(row)
        logger.info(f"  ppl={avg['perplexity']:.2f}  mad={avg['mad']:.2f}  att_ent={avg['attention_entropy']:.4f}  ipi={avg['ipi']:.1f}  entropy={avg['entropy']:.3f}  mtld={avg['mtld']:.2f}")
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    sens_df = pd.DataFrame(sensitivity_results)
    sens_path = output_dir / 'compiled_metrics_sensitivity_v3.csv'
    sens_df.to_csv(sens_path, index=False)
    logger.info(f'\nSensitivity metrics saved: {sens_path}')
    full_path = Path(args.input_full_metrics)
    if not full_path.exists():
        logger.warning(f'Full metrics CSV not found at {full_path}. Skipping ranking comparison.')
        return
    full_df = pd.read_csv(full_path)
    merged = full_df.merge(sens_df[['nombre_corpus'] + METRICS_TO_COMPARE], on='nombre_corpus', suffixes=('_full', '_sub'))
    if len(merged) < 5:
        logger.warning('Fewer than 5 corpora matched between full and sensitivity CSVs. Check corpus name consistency.')
        return
    logger.info('\n=== RANKING COMPARISON: full vs. length-controlled ===')
    comparison_records = []
    for metric in METRICS_TO_COMPARE:
        col_full = f'{metric}_full'
        col_sub = f'{metric}_sub'
        if col_full not in merged.columns or col_sub not in merged.columns:
            continue
        rho, p = spearmanr(merged[col_full], merged[col_sub])
        comparison_records.append({'metric': metric, 'rho': round(rho, 3), 'p_value': round(p, 4), 'robust': rho > 0.7 and p < 0.05})
        logger.info(f"  {metric:<20} ρ={rho:.3f}  p={p:.4f}  {('✓ robust' if rho > 0.7 else '⚠ sensitive to length')}")
    comp_df = pd.DataFrame(comparison_records)
    comp_path = output_dir / 'sensitivity_spearman_v3.csv'
    comp_df.to_csv(comp_path, index=False)
    shift_records = []
    for metric in METRICS_TO_COMPARE:
        col_full = f'{metric}_full'
        col_sub = f'{metric}_sub'
        if col_full not in merged.columns:
            continue
        merged_m = merged[['nombre_corpus', col_full, col_sub]].copy()
        merged_m['rank_full'] = merged_m[col_full].rank()
        merged_m['rank_sub'] = merged_m[col_sub].rank()
        merged_m['rank_shift'] = (merged_m['rank_sub'] - merged_m['rank_full']).abs()
        merged_m['metric'] = metric
        shift_records.append(merged_m[['metric', 'nombre_corpus', 'rank_full', 'rank_sub', 'rank_shift']])
    if shift_records:
        shift_df = pd.concat(shift_records, ignore_index=True)
        shift_path = output_dir / 'sensitivity_ranking_comparison_v3.csv'
        shift_df.to_csv(shift_path, index=False)
        logger.info(f'\nRanking shifts saved: {shift_path}')
        ppl_shifts = shift_df[shift_df['metric'] == 'perplexity'].sort_values('rank_shift', ascending=False)
        logger.info('\nPerplexity rank shifts (full → length-controlled):')
        logger.info(ppl_shifts.to_string(index=False))
    logger.info('\n✅ Análisis de sensibilidad completado.')
    logger.info(f'   Outputs en: {output_dir}')
if __name__ == '__main__':
    main()
