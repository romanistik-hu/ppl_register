import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
'Novel-sentence contamination probe (yEQA "Contamination Risk"): 20 sentences\n(10 planned/formal, 10 spontaneous/oral Canarian), written for this check and\nnever previously published, so they cannot be memorized by any model\'s\npretraining data by construction. If the LLM ordering (spontaneous > planned\nperplexity) holds on guaranteed-novel text, that is direct evidence the\nsignal is not driven by training-data familiarity with our specific corpora.\n\nComplements, rather than replaces, the corpus-level BNC probe\n(18_bnc_contamination_probe.py): BNC bounds contamination risk on real,\npossibly-seen text; this bounds it on text that is provably unseen.\n\nOutput: data/processed/19_probe_sentences/probe_sentence_ppl.csv\n'
import gc
import os
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from transformers import AutoModelForCausalLM, AutoTokenizer
OUT = _paths.DATA_DIR / '19_probe_sentences'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {'salamandra': 'BSC-LT/salamandra-7b', 'llama3': 'meta-llama/Meta-Llama-3-8B', 'mistral': 'mistralai/Mistral-7B-v0.3'}
PLANNED = ['La Consejería de Turismo ha aprobado hoy una nueva partida presupuestaria destinada a la mejora de las infraestructuras portuarias del archipiélago.', 'En sesión plenaria celebrada el pasado martes, la Cámara acordó por unanimidad remitir el expediente a la comisión correspondiente para su estudio detallado.', 'El presente informe recoge las conclusiones alcanzadas tras el análisis pormenorizado de los datos aportados por las administraciones competentes.', 'Según fuentes oficiales, el proyecto de rehabilitación del casco histórico contempla una inversión superior a los tres millones de euros.', 'La corporación municipal ha manifestado su compromiso con la sostenibilidad ambiental a través de la firma de un nuevo convenio de colaboración.', 'Se hace constar que la presente resolución podrá ser recurrida en el plazo de un mes ante el órgano administrativo correspondiente.', 'El pleno del cabildo insular procedió a la votación de las enmiendas presentadas por los distintos grupos políticos.', 'La memoria anual detalla los avances registrados en materia de empleo juvenil durante el último ejercicio.', 'Los servicios técnicos han elaborado un informe favorable sobre la viabilidad del proyecto presentado por la entidad promotora.', 'Con motivo de la efeméride, las autoridades insulares han convocado un acto institucional en la sede del organismo.']
SPONTANEOUS = ["Mira, yo ayer fui pa' casa de mi agüela y ella tenía la guagua esperándome allí en la parada, pero llegué tarde y ya se había ido.", 'Oye, y tú ¿qué hiciste al final con lo del trabajo ese que te dije, lo dejaste o qué fisco pasó ahí?', 'No sé, chacho, es que a mí esas cosas me ponen de un nervioso que no veas, en serio te lo digo.', 'Bueno pues nada, allí estábamos los tres esperando y va y se pone a llover, imagínate el show que se armó.', 'Y yo le digo, oye pero eso no es así, y ella ni caso, siguió a lo suyo como si nada.', "Fíjate que el otro día fui al mercadillo y había una cola que no veas, tuve que esperar como media hora pa' comprar cuatro papas.", 'La verdad es que no tengo ni idea, eso pregúntaselo a mi hermano que él sabe más de eso que yo.', 'Ay mija, no me hables de eso que todavía me da hasta rabia de pensarlo, fue un lío tremendo.', 'Total que al final ni fuimos a la playa ni nada, nos quedamos ahí en casa viendo la tele todo el día.', 'Eso sí, cuando llegue le voy a decir cuatro cosas, porque así no se puede seguir, oye.']

def sentence_ppl(text, tok, model, device):
    ids = tok.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([ids]).to(device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return float(np.exp(out.loss.item()))

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hf_token = os.environ.get('HF_TOKEN')
    all_rows = []
    for model_key, model_id in MODELS.items():
        out_csv = OUT / f'probe_ppl_{model_key}.csv'
        if out_csv.exists():
            print(f'{model_key}: already done, skipping')
            all_rows.append(pd.read_csv(out_csv))
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
        for pole, sentences in (('planned', PLANNED), ('spontaneous', SPONTANEOUS)):
            for i, s in enumerate(sentences):
                ppl = sentence_ppl(s, tok, model, device)
                rows.append({'model': model_key, 'pole': pole, 'sentence_id': i, 'ppl': ppl, 'text': s})
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        all_rows.append(df)
        print(f'  Saved: {out_csv} ({(time.time() - t0) / 60:.1f} min)')
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT / 'probe_sentence_ppl.csv', index=False)
    print('\n--- Summary (planned vs spontaneous PPL per model) ---')
    summary_rows = []
    for model_key in MODELS:
        sub = combined[combined['model'] == model_key]
        planned = sub[sub['pole'] == 'planned']['ppl'].values
        spontaneous = sub[sub['pole'] == 'spontaneous']['ppl'].values
        u, p = mannwhitneyu(spontaneous, planned, alternative='greater')
        ratio = spontaneous.mean() / planned.mean()
        print(f'{model_key}: planned mean={planned.mean():.2f}  spontaneous mean={spontaneous.mean():.2f}  ratio={ratio:.2f}  Mann-Whitney U={u:.0f}  p={p:.4f}')
        summary_rows.append({'model': model_key, 'planned_mean': planned.mean(), 'spontaneous_mean': spontaneous.mean(), 'ratio': ratio, 'mannwhitney_u': u, 'p_value': p})
    pd.DataFrame(summary_rows).to_csv(OUT / 'probe_sentence_summary.csv', index=False)
if __name__ == '__main__':
    main()
