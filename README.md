# UK Pension Risk Transfer Opportunity Analysis

This repository contains the Python code used for the empirical analysis in a
dissertation on observable characteristics of UK Pension Risk Transfer (PRT)
opportunities and the probability that an opportunity was Won by the focal
insurer.

The workflow covers sample preparation, descriptive statistics, unadjusted
tests, the principal binary logistic regression, model diagnostics, a
quadratic specification, a Won-versus-Lost sensitivity model, an alternative
IA/DA specification, and a Premium by Product Group interaction model.

## Confidentiality and data availability

The source workbook is not included because it contains confidential
commercial information. No project names, client or scheme identifiers,
insurer identifiers, or opportunity-level records are included in this
repository.

Authorised users may supply an appropriately structured workbook locally. The
required variables and derived measures are described in
`documentation/data_dictionary.md`.

## Repository contents

- `analysis/prt_analysis.py`: complete analysis workflow.
- `data/README.md`: instructions for supplying an authorised local workbook.
- `documentation/data_dictionary.md`: input and derived-variable definitions.
- `documentation/analysis_mapping.md`: mapping between code and dissertation sections.
- `documentation/dissertation_appendix_text.md`: suggested repository statement for the dissertation appendix.
- `outputs/public/`: non-confidential figure already reported in the dissertation.
- `requirements.txt`: Python package versions used to verify the repository.

## Software environment

The repository was verified with Python 3.12.13 and the package versions listed
in `requirements.txt`.

Create a virtual environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running the analysis

Place the authorised workbook at:

```text
data/prt_project_data.xlsx
```

The workbook must contain a worksheet named `PRT Data`. Run:

```powershell
python analysis/prt_analysis.py
```

An alternative input file and output directory can be supplied as positional
arguments:

```powershell
python analysis/prt_analysis.py "C:\path\authorised_data.xlsx" "outputs\generated"
```

Generated files are written to `outputs/generated/` by default. This directory
is excluded from Git because some generated files contain opportunity-level
diagnostics. Review all outputs before sharing them.

## Reproducibility boundary

The repository permits review of the analytical procedures, but the numerical
results cannot be reproduced externally without authorised access to a
workbook with the required structure. The published figure in `outputs/public/`
is included only because it presents aggregate results already reported in the
dissertation.

## Before publishing

1. Confirm that the data-providing organisation permits publication of the code.
2. Check the complete Git history for confidential files and local paths.
3. Add the author's details to `CITATION.cff.example` and rename it to `CITATION.cff` if desired.
4. Create a fixed release, such as `v1.0.0`, and record its commit hash in the dissertation appendix.
5. Check whether anonymous marking restricts use of a personally identifiable GitHub account.

Unless a separate licence is added, no permission to reuse or redistribute the
code is granted beyond viewing the public repository.
