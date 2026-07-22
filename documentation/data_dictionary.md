# Data dictionary

## Required input fields

| Field | Type | Use in the analysis |
|---|---|---|
| `Project Status` | Categorical | Records Won, Lost, Cold, Declined or Active. |
| `Product` | Categorical | Used to construct the four Product Groups. |
| `Estimated Premium` | Positive numeric | Available proxy for the expected monetary scale of the opportunity. |
| `IAs` | Non-negative numeric count | Number of immediate annuities already in payment. |
| `DAs` | Non-negative numeric count | Number of deferred annuities expected to enter payment in the future. |

The worksheet containing these fields must be named `PRT Data`.

## Optional fields

| Field | Use |
|---|---|
| `Target Signing Date` | Used only to report the recorded target-date range in the generated summary. |
| `Source Row` | Used only as a local diagnostic reference. If absent, it is generated from the worksheet row number. |

Project names, client identifiers and scheme identifiers are not required by
the analytical model and should not be included in any public data extract.

## Derived variables

| Variable | Definition |
|---|---|
| `success` | 1 for Won; 0 for Lost, Cold or Declined. Active is excluded. |
| `product_group` | IA & DA / Buyout, IA & DA / Buy-In, IA / Buy-In, or Other product types. |
| `total_annuity_count` | IAs + DAs. |
| `da_share` | DAs / (IAs + DAs), calculated where the total is positive. |
| `log_premium` | Natural logarithm of Estimated Premium. |
| `log_total_annuity_count` | Natural logarithm of Total Annuity Count. |
| `log_ias_plus_1` | Natural logarithm of 1 + IAs. |
| `log_das_plus_1` | Natural logarithm of 1 + DAs. |
| `premium_size_band` | Sample quartiles labelled Small, Medium, Large and Very Large. |

Total Annuity Count and DA Share are count-based proxies. They do not represent
the monetary value, duration, age profile or detailed benefit structure of the
liabilities.
