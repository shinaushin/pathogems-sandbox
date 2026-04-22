# Stage 2 — Data Acquisition

Two scripts live here, by design:

- **`fetch_cbioportal_brca.py` (Stage-2-lite).** Fetches TCGA-BRCA RNA-seq +
  clinical from cBioPortal. Used by the omics-only baseline. One zip,
  ~100 MB, ~30 seconds. See ADR 0003 for the rationale.
- **`agent.py` (full Stage 2).** The brief's agent that queries GDC for all
  four modalities and emits download scripts for WSIs. Not yet implemented;
  will be added once the omics baseline is stable and we're ready to move
  to multimodal (ADR 0001 lists this as a follow-up).

## Running Stage-2-lite

```bash
python stage2_data/fetch_cbioportal_brca.py
```

> **Note:** cBioPortal previously distributed study bundles from a public S3 bucket
> (`cbioportal-datahub.s3.amazonaws.com`). That bucket now returns HTTP 403.
> The script has been updated to use cBioPortal's own download endpoint instead.
> If the script still fails, download the study manually — see below.

After the script completes you should see:

```
stage2_data/raw/brca_tcga_pan_can_atlas_2018/
├── data_mrna_seq_v2_rsem.txt         # gene × sample RSEM matrix
├── data_clinical_patient.txt         # patient-level clinical (OS_STATUS, OS_MONTHS, ...)
├── data_clinical_sample.txt          # sample-level clinical (sample → patient mapping)
└── manifest.json                     # provenance: source URL + SHA256 + fetched_at
```

`stage2_data/raw/` is gitignored; the script is idempotent, so recovering
from a lost volume is `python stage2_data/fetch_cbioportal_brca.py` and
nothing else.

**Manual download fallback** (if all URLs fail):

1. Go directly to the study page:
   <https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018>
2. Click the **Download** button in the top-right corner of the page
3. Unzip and place these three files into
   `stage2_data/raw/brca_tcga_pan_can_atlas_2018/`:
   - `data_mrna_seq_v2_rsem.txt`
   - `data_clinical_patient.txt`
   - `data_clinical_sample.txt`

## Clinical fields we rely on

Stage 3's data module reads exactly two clinical fields from
`data_clinical_patient.txt`:

| Field        | Values                          | Meaning                              |
|--------------|----------------------------------|--------------------------------------|
| `OS_STATUS`  | `"0:LIVING"`, `"1:DECEASED"`     | Event indicator. `1` = death observed. |
| `OS_MONTHS`  | float, months                    | Time to event or last follow-up.     |

If a row is missing either value it is dropped before training.
cBioPortal's clinical files have a four-line header of metadata before the
real header; Stage 3's parser is aware of this.
