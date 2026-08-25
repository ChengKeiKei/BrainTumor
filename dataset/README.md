# Dataset code

Only preprocessing source code and notebooks are included. All notebook
outputs were cleared before packaging. No patient data, processed CSV, split
file, MRI, spreadsheet, or chart is included.

## Expected local layout

```text
dataset/
├── first/
│   ├── Raw/
│   │   ├── MU-Glioma-Post_ClinicalData-July2025.xlsx
│   │   ├── segmentation_volumes.xlsx
│   │   └── MRI/
│   ├── Processed/        # generated; ignored by Git
│   └── splits/           # generated; ignored by Git
└── second/
    ├── Raw/
    │   ├── MU-Glioma-Post_ClinicalData-July2025.xlsx
    │   ├── segmentation_volumes.xlsx
    │   └── MRI/
    ├── Processed/        # generated; ignored by Git
    ├── splits/           # generated; ignored by Git
    └── splits_cv/        # generated; ignored by Git
```

Run notebooks from their own directory so `Path.cwd()` resolves correctly.

First recurrence:

```bash
cd dataset/first
jupyter lab preprocessing.ipynb
```

Second recurrence, in order:

```bash
cd dataset/second
jupyter lab Clinical_preprocessing.ipynb
jupyter lab Image_preprocessing.ipynb
jupyter lab Radiomic_preprocessing.ipynb
```

`_landmark.py` is the shared source of second-recurrence cohort, label, and
landmark logic. `audit_followup_censoring.py` builds the fixed-landmark
sensitivity cohort used to avoid prospective claims from the original
outcome-dependent landmark.
