# Data directory

The confidential source workbook is deliberately excluded from this
repository.

Authorised users should place the workbook at:

```text
data/prt_project_data.xlsx
```

The workbook must contain a worksheet named `PRT Data` and the required fields
listed in `../documentation/data_dictionary.md`.

Do not commit the workbook, extracts, cleaned samples, project identifiers, or
opportunity-level analytical outputs. The repository `.gitignore` excludes
common spreadsheet and data formats as an additional safeguard.
