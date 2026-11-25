# Perplexity as a Register Classifier

Code and derived data for:

Johnatan E. Bonilla. *Perplexity as a Register Classifier: Neural Language Models and the Planned–Spontaneous Continuum in Spanish.* EACL 2027.

## Contents

- `data/processed/`: metrics per corpus and per fragment, reference measures and results.
- `data/annotations/`: Koch and Oesterreicher coding of the thirteen corpora.
- `scripts/`: analysis scripts. `check_paper_numbers.py` recomputes the numbers reported in the paper.
- `results/figures/`: figures of the paper.

The corpus texts are not included. The corpora are available from their holders:

- CoACan: Bonilla, Bastias Grünwald and Bouzouita (2025), <https://doi.org/10.5281/zenodo.17413533>
- CoParCan: Bonilla and Bouzouita (2025), <https://doi.org/10.5281/zenodo.17413688>
- CorpusCanarioWA: Betti (2026), <https://doi.org/10.5281/zenodo.20290207>
- Jable: Universidad de Las Palmas de Gran Canaria (2008), <https://jable.ulpgc.es/>
- COSER, Corpus Oral y Sonoro del Español Rural: Fernández-Ordóñez (2005), <https://corpusrural.es/>
- PRESEEA: PRESEEA (2014), <https://preseea.uah.es>
- AMERESCO: Albelda Marco and Estellés Arguedas (2011), <http://esvaratenuacion.es/ameresco/>

## Factor scores

The factor analysis has 43 features and 13 corpora, so the correlation matrix of the features is singular. `02_biber_analysis.py` computes the factor scores with its pseudo-inverse, which makes them reproducible across machines.

## Running the scripts without the corpus texts

The scripts that read the corpus texts (`09`, `12`, `17`, `21`) and the hidden states (`15`) reuse the stored metrics per corpus when those inputs are absent, so the correlation tables can be recomputed from this repository alone. `02_biber_analysis.py --skip_extraction` does the same for the factor analysis.

## Acknowledgments

This research was carried out within the Collaborative Research Centre SFB/CRC 1412 *Register: Language Users' Knowledge of Situational-Functional Variation*, funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project Number 416591334, specifically within its subproject A09, *On the interplay between register and socio-geographic variation in Canarian Spanish* (PI: Miriam Bouzouita).

## Licence

MIT (see `LICENSE`).
