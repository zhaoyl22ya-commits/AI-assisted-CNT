# DeepCNT 3.0

**Physics-Structured Literature Learning Enables Transferable Ranking and Domain-Aware Prioritization of Carbon Nanotube Growth Conditions**

DeepCNT 3.0 is a literature-to-design framework for carbon nanotube (CNT) chemical vapor deposition (CVD). The project converts heterogeneous CNT growth literature into structured synthesis records, represents experimental conditions through physics-organized latent descriptors, trains predictive models, evaluates generalization and robustness, ranks literature-supported candidate routes, and provides WorldMap-based exploration and application interfaces.

The repository is designed as a **reproducible code and model-analysis package**. The complete historical literature corpus is intentionally **not included**.

---

## 1. Project overview

The system is organized into four major blocks:

```text
DeepCNT 3.0
│
├── BLOCK 1 — PIPELINE
│   Literature → structured records → physics representation
│
├── BLOCK 2 — MODELS
│   120D representation → predictive models → scientific validation
│
├── BLOCK 3 — RECOMMENDATION
│   Catalyst + route → physics-consistent candidates → prioritization
│
└── BLOCK 4 — APPLICATION / WORLDMAP
    OneClick + Fusion + Global Atlas + Evaluation + 3D + interpretation
```

The current codebase contains approximately **50 formal Python scripts**:

| Block | Main role | Scripts |
|---|---|---:|
| Pipeline | Literature processing and representation | 19 |
| Models | Training, validation, robustness and explanation | 14 |
| Recommendation | Candidate construction and ranking | 8 |
| Application / WorldMap | Interactive analysis and visualization | 9 |
| **Total** | | **50** |

---

## 2. Scientific data flow

The central data flow is:

```text
CNT CVD literature
        │
        ▼
PDF / DOCX
        │
        ▼
01_markdown
        │
        ▼
Growth-mode classification (A–I)
        │
        ▼
Structured extraction
        │
        ▼
Canonical cleaning / qualification
        │
        ▼
Image-supported information supplementation
        │
        ▼
Deterministic finalization
        │
        ▼
Physics representation
        │
        ├── N: Nucleation / catalyst state
        │      N1–N18
        │
        ├── C: Carbon-supply state
        │      C1–C12
        │
        ├── F: Flow / transport state
        │      F1–F18
        │
        └── P: Process boundary state
               P1–P12
        │
        ▼
60 physical descriptors
        +
60 missingness indicators
        │
        ▼
120D formal model input
        │
        ▼
Prediction / validation / ranking
        │
        ▼
Candidate prioritization
        │
        ▼
WorldMap / application layer
```

The 60 physical descriptors are organized as:

```text
18 N descriptors
12 C descriptors
18 F descriptors
12 P descriptors
----------------
60 physical descriptors
```

The corresponding missingness indicators are retained explicitly. Therefore, the downstream formal model representation contains **120 dimensions**: 60 physical descriptors plus 60 missingness indicators.

Metadata such as paper identifiers, year and sample identifiers are retained for provenance, splitting and auditing, but are not treated as physical model features.

---

# 3. Repository structure

The recommended repository layout is:

```text
20260721-CNTcodes/
│
├── app/
│   ├── app_oneclick_tabs.py
│   ├── app_oneclick_worldmap.py
│   ├── evaluate_worldmap.py
│   ├── mapmodel_analysis.py
│   ├── oneclick_core.py
│   ├── oneclick_fusion.py
│   ├── phase_diagram_interactive_3d.py
│   ├── phase_diagram_worldmap.py
│   ├── run_worldmap_app.py
│   └── support_distance_robustness.py
│
├── data/
│   └── 11_flattened/
│       ├── ml_dataset_3p0_full60D.csv
│       ├── feature_schema.json
│       ├── dataset_manifest.csv
│       └── build_report.csv
│
├── models/
│   ├── evaluation/
│   │   ├── external_validation/
│   │   ├── ablation.py
│   │   ├── catalyst_family_holdout.py
│   │   ├── composite_robustness.py
│   │   ├── composite_sensitivity.py
│   │   ├── representation_baseline.py
│   │   ├── retrospective_ranking.py
│   │   ├── uncertainty.py
│   │   └── validate_generalization.py
│   │
│   ├── explain/
│   │   └── explain_l1_shap.py
│   │
│   └── training/
│       ├── train_l1_composite.py
│       ├── train_l1_targets.py
│       └── train_l12_quantile.py
│
├── pipeline/
│   ├── 01_pdf_to_markdown.py
│   ├── 02_extract_and_classify.py
│   ├── 03_run_extract_forD.py
│   ├── 04_split_by_growth_mode.py
│   ├── 05_assign_L123.py
│   ├── 06a_clean_A.py
│   ├── 06b_clean_B.py
│   ├── 06c_clean_C.py
│   ├── 06d_clean_D.py
│   ├── 06e_clean_E.py
│   ├── 07_qwen_pdf_image_api.py
│   ├── 07b_merge_image_patches.py
│   ├── 08_finalize_cnt_json.py
│   ├── 09a_extract_znuc.py
│   ├── 09b_extract_zc.py
│   ├── 09c_extract_zflow.py
│   ├── 09d_extract_zproc.py
│   ├── 10_merge_latent.py
│   └── 11_flatten_dataset.py
│
├── prompts/
│   ├── A_prompt_extract_cnt.txt
│   ├── B_prompt_extract_cnt.txt
│   ├── B_prompt_calc_cnt.txt
│   ├── C_prompt_extract_cnt.txt
│   ├── C_prompt_calc_cnt.txt
│   ├── D_prompt_extract_cnt.txt
│   ├── D_prompt_calc_cnt.txt
│   ├── E_prompt_extract_cnt.txt
│   ├── E_prompt_calc_cnt.txt
│   ├── F_prompt_extract_cnt.txt
│   ├── G_prompt_extract_cnt.txt
│   ├── H_prompt_extract_cnt.txt
│   ├── I_prompt_extract_cnt.txt
│   ├── gpt_auto_prompt.txt
│   ├── prompt_classify_mode.txt
│   └── prompt_deepseek_top3_analysis.txt
│
├── promptsV2/
│   ├── A_prompt_zC.txt
│   ├── A_prompt_zflow.txt
│   ├── A_prompt_znuc.txt
│   ├── A_prompt_zproc.txt
│   ├── CNT_latent_space_explanation.txt
│   ├── elements_properties.txt
│   ├── elements_properties.xlsx
│   ├── gas_properteis.txt
│   ├── gas_properteis.xlsx
│   └── catalyst_input_template.csv
│
├── recommendation/
│   ├── feature_builders/
│   │   ├── build_candidate_priors.py
│   │   ├── build_route_templates.py
│   │   ├── build_zc.py
│   │   └── build_znuc.py
│   ├── candidate_stability.py
│   ├── optimize_physical_state.py
│   ├── rank_candidate_routes.py
│   └── rank_catalysts.py
│
├── outputs/
│   └── generated results are written here
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

Some generated directories from the full literature-processing workflow are intentionally omitted from the public repository.

---

# 4. What is included

This repository is intended to contain the materials required to understand and reproduce the **modeling, recommendation and application layers** without distributing the full historical literature corpus.

### Included

- All project Python source code.
- `prompts/` and its literature-understanding / extraction prompts.
- `promptsV2/` and its physics-representation prompts and supporting property tables.
- The flattened machine-learning dataset:
  - `data/11_flattened/ml_dataset_3p0_full60D.csv`
- The feature definition and audit files when included:
  - `feature_schema.json`
  - `dataset_manifest.csv`
  - `build_report.csv`
- WorldMap visualization assets used by the application, including:
  - `worldmap001.png`
  - `worldmap002.png`
  - `worldmap003.png`
- Configuration templates such as `.env.example`.
- Dependency specification such as `requirements.txt`.

### Not included

The following are deliberately excluded from the public repository:

- The complete historical PDF corpus.
- Raw downloaded literature files.
- Intermediate Markdown files generated from the literature corpus.
- Large intermediate JSON collections.
- API logs containing sensitive or unnecessary information.
- API keys, passwords, tokens or other secrets.
- Large temporary image collections.
- Private experimental or unpublished datasets.

The repository therefore should **not** be described as containing the complete raw literature database. Instead, it provides the code and the final flattened dataset needed for downstream modeling and application use.

---

# 5. Important distinction: literature retrieval labels vs scientific growth modes

The original literature retrieval routes should not be treated as scientific labels.

The authoritative growth-mode classification is performed later from the article content and produces the scientific categories:

```text
A / B / C / D / E / F / G / H / I
```

In other words:

```text
WoS retrieval route
        ≠
scientific growth-mode label
```

This distinction is important for reproducibility and for avoiding leakage from the literature retrieval process into downstream modeling.

---

# 6. Pipeline block

The Pipeline converts literature into the formal model interface.

### Main stages

```text
01_pdf_to_markdown.py
        ↓
02_extract_and_classify.py
        ↓
03_run_extract_forD.py
        ↓
04_split_by_growth_mode.py
        ↓
05_assign_L123.py
        ↓
06a–06e_clean_*.py
        ↓
07_qwen_pdf_image_api.py
        ↓
07b_merge_image_patches.py
        ↓
08_finalize_cnt_json.py
        ↓
09a–09d_extract_*.py
        ↓
10_merge_latent.py
        ↓
11_flatten_dataset.py
```

The formal output of the complete pipeline is:

```text
data/11_flattened/
├── ml_dataset_3p0_full60D.csv
├── feature_schema.json
├── dataset_manifest.csv
└── build_report.csv
```

The public repository does not require the raw corpus to be redistributed in order to expose this final modeling interface.

### Document ingestion

`01_pdf_to_markdown.py` converts literature documents into machine-readable text. PDF input is the primary supported literature format, with DOCX also supported by the document conversion stage.

### Extraction and classification

`02_extract_and_classify.py` performs the main LLM-based scientific understanding stage, including growth-mode classification and mode-specific structured extraction. B/C/E branches can use dedicated physics-calculation prompts before final extraction, while D has a dedicated intermediate processing route.

### Canonical cleaning

`05_assign_L123.py` and `06a–06e_clean_*.py` qualify and normalize the extracted records before the physical representation stage.

### Physics representation

`09a–09d_extract_*.py` map finalized synthesis records into four organized representation blocks:

```text
N — Nucleation / catalyst
C — Carbon supply
F — Flow / transport
P — Process boundary
```

These are not simply an arbitrary collection of raw experimental parameters; they provide the structured physical representation used by the downstream modeling and recommendation layers.

---

# 7. Models block

The Models block operates on the flattened representation rather than directly on raw papers.

It contains code for:

- target and composite model training;
- quantile modeling;
- catalyst-family holdout;
- representation baselines;
- ablation analysis;
- sensitivity and robustness analysis;
- uncertainty analysis;
- retrospective ranking;
- external validation;
- SHAP-based interpretation.

The principal dataset interface is:

```text
data/11_flattened/ml_dataset_3p0_full60D.csv
```

The formal model representation uses:

```text
60 physical descriptors
+
60 missingness indicators
=
120D model input
```

The dataset filename retains `full60D` because the underlying physical representation itself consists of 60 descriptors; the explicit missingness indicators are an additional modeling layer.

---

# 8. Recommendation block

The Recommendation block converts model outputs and physical representations into prioritized candidate routes.

The main components include:

```text
feature_builders/
├── build_candidate_priors.py
├── build_route_templates.py
├── build_zc.py
└── build_znuc.py

candidate_stability.py
optimize_physical_state.py
rank_candidate_routes.py
rank_catalysts.py
```

The recommendation workflow is conceptually:

```text
Catalyst / route definition
        ↓
Physical-state construction
        ↓
Physical feasibility / consistency checks
        ↓
Model prediction
        ↓
Literature-support assessment
        ↓
Candidate ranking
        ↓
Prioritized candidates
```

A candidate supported by the literature should not be described as experimentally validated. Literature support and prospective experimental validation are distinct evidence levels.

---

# 9. Application / WorldMap block

The application layer provides interfaces for exploring the learned CNT growth landscape.

The current application block includes:

- OneClick interfaces.
- Fusion analysis.
- Global WorldMap.
- Phase-diagram visualization.
- Interactive 3D exploration.
- WorldMap evaluation.
- Support-distance robustness analysis.
- DeepSeek-assisted interpretation.

The application layer is designed to consume the already constructed modeling outputs and candidate tables rather than rebuilding the literature-processing pipeline.

The WorldMap candidate construction explicitly maintains route-dependent physical consistency between catalyst/carbon descriptors and process-route descriptors.

---

# 10. Environment setup

Create a Python environment and install the dependencies listed in `requirements.txt`.

Example using PowerShell:

```powershell
cd E:\pycharm_files\20260721-CNTcodes

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt
```

If PowerShell blocks activation because of the local execution policy, the environment can also be used directly through its Python executable:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

# 11. Environment variables

Do **not** commit real API keys to GitHub.

Create a local `.env` file from the provided template:

```text
.env.example
    ↓ copy
.env
```

The `.env` file should contain only local secrets and machine-specific configuration.

Typical variables may include API credentials required by the current scripts, for example:

```text
DEEPSEEK_API_KEY=your_key_here
QWEN_API_KEY=your_key_here
```

Use the exact variable names expected by the current source code and `.env.example`. Do not place real keys in:

- Python source files.
- Prompt files.
- README.
- YAML configuration committed to GitHub.
- CSV files.
- Git history.

If a service is not used by a particular execution path, its corresponding key is not required for that path.

---

# 12. Running the downstream system from the public package

Because the raw literature corpus is not included, the most direct public-repository workflow begins from the flattened dataset and the already prepared project assets.

### Step 1 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 2 — Configure local environment variables

```text
.env.example → .env
```

Add only the credentials required by the scripts you intend to run.

### Step 3 — Verify the dataset

Confirm that:

```text
data/11_flattened/ml_dataset_3p0_full60D.csv
```

is present.

If available, also verify:

```text
data/11_flattened/feature_schema.json
data/11_flattened/dataset_manifest.csv
data/11_flattened/build_report.csv
```

### Step 4 — Train or validate models

Use the scripts under:

```text
models/training/
models/evaluation/
models/explain/
```

The exact script and execution order should follow the input/output paths defined in the current source code.

### Step 5 — Run recommendation workflows

Use:

```text
recommendation/
```

for catalyst and route construction, physical-state optimization and candidate ranking.

### Step 6 — Run the application layer

Use the scripts under:

```text
app/
```

for WorldMap, phase-diagram, 3D exploration and application interfaces.

---

# 13. Reconstructing the complete literature pipeline

The full upstream pipeline can be reconstructed if a user has an authorized copy of the historical literature corpus.

The required conceptual sequence is:

```text
raw papers
    ↓
Markdown conversion
    ↓
LLM classification / extraction
    ↓
growth-mode routing
    ↓
L1/L2/L3 assignment
    ↓
canonical cleaning
    ↓
image supplementation
    ↓
finalization
    ↓
N/C/F/P representation
    ↓
latent merge
    ↓
flattened dataset
```

The public repository intentionally stops short of distributing the raw historical corpus.

Users who want to rebuild the dataset from scratch must supply their own authorized literature files and configure the corresponding external API services.

---

# 14. Reproducibility principles

The repository follows several principles:

### 14.1 No raw literature redistribution

The complete historical literature corpus is not committed to GitHub.

### 14.2 No secrets in source control

API keys and authentication credentials must remain local.

### 14.3 Explicit representation

The model input is explicitly organized around:

```text
N / C / F / P
```

with explicit missingness indicators.

### 14.4 Provenance separation

Paper metadata and retrieval provenance are retained for audit and splitting purposes but are not treated as physical model features.

### 14.5 Scientific label separation

Literature retrieval routes are not used as substitutes for content-based scientific growth-mode classification.

### 14.6 Evidence-level separation

Literature-supported candidates are not equivalent to experimentally validated candidates. Prospective experimental validation is a separate step.

---

# 15. Expected outputs

Depending on the workflow, generated outputs may include:

- trained model artifacts;
- validation metrics;
- external-validation results;
- robustness and sensitivity analyses;
- uncertainty estimates;
- SHAP explanations;
- candidate ranking tables;
- physical-state optimization results;
- WorldMap candidate tables;
- phase diagrams;
- interactive application outputs.

Generated outputs should normally remain outside version control unless they are intentionally selected as lightweight, reproducible example results.

---

# 16. Data and evidence boundaries

This repository represents a **literature-derived computational framework**.

The final flattened dataset is derived from retrospective literature information. Therefore:

```text
literature support
        ≠
experimental validation
```

Likewise:

```text
model prediction
        ≠
guaranteed experimental outcome
```

Candidate rankings should be interpreted as computational prioritization within the domain represented by the literature and model, rather than as unconditional claims of experimental success.

Prospective experiments on a small set of prioritized candidates would provide an independent level of validation and practical confirmation.

---

# 17. Citation and scientific use

If this repository is used in academic work, please cite the associated DeepCNT 3.0 publication and the original literature sources underlying the dataset.

The repository is intended to make the computational workflow, representation design, modeling procedures and downstream candidate prioritization transparent and reproducible without redistributing the complete source corpus.

---

# 18. Current repository scope

The public package is best understood as:

```text
                 PUBLIC REPOSITORY
                         │
        ┌────────────────┼────────────────┐
        │                │                │
      CODE             PROMPTS         FINAL DATA
        │                │                │
   pipeline/         prompts/       ml_dataset...
   models/           promptsV2/     schema / audit
   recommendation/
   app/
        │
        └───────────────┬────────────────┘
                        ▼
                REPRODUCIBLE
             DOWNSTREAM WORKFLOW
                        │
                        ▼
              MODEL / RANKING /
             WORLDMAP / APP
```

The omitted raw corpus remains an external input rather than a repository asset.

---

## 19. License

Add the project license appropriate to the final publication and institutional requirements before making the repository public.

Until a license is explicitly added, users should not assume that the source code or associated materials are released under a particular open-source license.

---

## 20. Contact

For scientific questions, reproducibility issues, or collaboration requests, please use the contact information associated with the project or the corresponding publication.
