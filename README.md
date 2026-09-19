# Prenatal PubCaseFinder
Vietnamese doctor-reviewed prenatal phenotype workspace.

## Use
1. Open the GitHub Pages site.
2. Choose **Kết nối**, enter the access code supplied by the administrator.
3. Search HPO terms or extract phrases from an ultrasound paragraph.
4. Review each HPO and its Có / Nghi ngờ / Không status.
5. Confirm review and compare disease profiles.
6. Select diseases for the lab handoff sheet and print/save PDF.

The UI does not persist case text or the access code. Only the API address is saved.
Similarity is IC-weighted phenotype similarity, not disease probability.
Unrecorded features include postnatal findings and require clinician interpretation.
Gene associations are displayed as associations, not proof of causality.
This is a research/clinical-review tool, not a validated diagnostic device.

## Layout
- `docs/`: dependency-free HTML/CSS/JavaScript published by GitHub Pages.
- `backend/`: Python HTTP boundary, existing matching engine, assertion rules and strict Model 1 runner.
- Clinical cases, HPO data files, model weights, credentials and training runs are not published here.

## Backend
Use the existing Python environment on the Ubuntu GPU host.
Supply licensed/local files in `backend/data/`: `hp.obo`, `phenotype.hpoa`,
`genes_to_disease.txt`, `hpo_catalog_vi.json`; the clinical phrase dictionary is optional.
Set `MODEL1_ADAPTER` to the selected LoRA directory and `MODEL1_WORKER_DIR`
to the original evaluated v3.8 package (worker.py, prepare.py, core.py, tokenizer).
The serving wrapper calls that worker without changing training files.

Run `python serve_web.py` from the backend directory.
Set `MODEL1_PRELOAD=1` to preload on service startup.
Set `WEB_ACCESS_TOKEN` and `WEB_ALLOWED_ORIGINS` before enabling public HTTPS.
The service binds loopback; HTTPS is provided by the Ubuntu reverse proxy.
`server.py` is retained only as a compatibility base for data loading. Use `serve_web.py` as the entrypoint.

## Training and updates
The web service uses a fixed adapter path. Training completion never automatically promotes a checkpoint.
If training needs all GPU memory, stop `prenatal-web` first, then restart after training.
Validate a new adapter separately before changing `MODEL1_ADAPTER` in the private environment file.
Restart and smoke-test extraction and ranking; revert the path if validation fails.
No training is started by this web application.

## Checks
`python test_web.py` checks search, alias normalization, input validation, ranking metadata,
review requirements, and rejection of ambiguous span alignment.
GPU/API/UI deployment checks are documented separately; these tests do not establish clinical accuracy.

