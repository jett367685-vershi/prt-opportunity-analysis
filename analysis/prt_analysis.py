"""
Reproducible analysis code for the PRT dissertation dataset.

The script prepares opportunity-level data, creates descriptive tables,
performs unadjusted tests, estimates the principal logistic regression,
and runs diagnostics and supplementary model specifications.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "data" / "prt_project_data.xlsx"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "generated"


def load_figure_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a common font while retaining a cross-platform fallback."""
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _regularized_gamma_q(a: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(a, x), used for chi-square p-values."""
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0

    eps = 1e-12
    max_iter = 1000
    gln = math.lgamma(a)

    if x < a + 1:
        ap = a
        delta = 1 / a
        total = delta
        for _ in range(max_iter):
            ap += 1
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * eps:
                break
        p = total * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))

    b = x + 1 - a
    c = 1 / 1e-300
    d = 1 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, q))


def chi_square_test(table: pd.DataFrame) -> dict[str, float]:
    observed = table.to_numpy(dtype=float)
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_totals @ col_totals / total
    mask = expected > 0
    statistic = float(((observed[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    df = int((observed.shape[0] - 1) * (observed.shape[1] - 1))
    p_value = _regularized_gamma_q(df / 2, statistic / 2) if df > 0 else float("nan")
    n = float(total)
    min_dimension = min(observed.shape[0] - 1, observed.shape[1] - 1)
    cramers_v = math.sqrt(statistic / (n * min_dimension)) if min_dimension > 0 else float("nan")
    return {
        "chi_square": statistic,
        "df": df,
        "p_value": p_value,
        "cramers_v": cramers_v,
        "min_expected_count": float(expected.min()),
        "cells_with_expected_below_5": int((expected < 5).sum()),
    }


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def mann_whitney_u_test(group_a: pd.Series, group_b: pd.Series) -> dict[str, float]:
    a = group_a.dropna().astype(float)
    b = group_b.dropna().astype(float)
    combined = pd.concat([a, b], ignore_index=True)
    ranks = combined.rank(method="average")
    n1 = len(a)
    n2 = len(b)
    rank_sum_a = float(ranks.iloc[:n1].sum())
    u1 = rank_sum_a - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    tie_counts = combined.value_counts().to_numpy()
    tie_correction = 1 - ((tie_counts**3 - tie_counts).sum() / ((n1 + n2) ** 3 - (n1 + n2)))
    mean_u = n1 * n2 / 2
    sd_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) * tie_correction / 12)
    z = (u - mean_u) / sd_u if sd_u else float("nan")
    p_value = float(2 * (1 - normal_cdf(np.array([abs(z)]))[0])) if not math.isnan(z) else float("nan")
    effect_size_r = abs(z) / math.sqrt(n1 + n2) if not math.isnan(z) else float("nan")

    return {
        "n_successful": int(n1),
        "n_unsuccessful": int(n2),
        "median_successful": float(a.median()),
        "median_unsuccessful": float(b.median()),
        "u_statistic": float(u),
        "z_value": float(z),
        "p_value": p_value,
        "effect_size_r": float(effect_size_r),
    }


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_logit(
    x: pd.DataFrame,
    y: pd.Series,
    max_iter: int = 100,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    x_matrix = x.to_numpy(dtype=float)
    y_vector = y.to_numpy(dtype=float)
    beta = np.zeros(x_matrix.shape[1])
    converged = False
    iterations = max_iter

    for iteration in range(1, max_iter + 1):
        eta = np.clip(x_matrix @ beta, -35, 35)
        p = 1 / (1 + np.exp(-eta))
        weights = np.clip(p * (1 - p), 1e-8, None)
        gradient = x_matrix.T @ (y_vector - p)
        hessian = -(x_matrix.T * weights) @ x_matrix
        step = np.linalg.pinv(hessian) @ gradient
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            converged = True
            iterations = iteration
            break
        beta = beta_new

    eta = np.clip(x_matrix @ beta, -35, 35)
    p = 1 / (1 + np.exp(-eta))
    weights = np.clip(p * (1 - p), 1e-8, None)
    information = (x_matrix.T * weights) @ x_matrix
    covariance = np.linalg.pinv(information)
    se = np.sqrt(np.diag(covariance))
    z = beta / se
    p_values = 2 * (1 - normal_cdf(np.abs(z)))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    eps = 1e-12
    log_likelihood = float(np.sum(y_vector * np.log(p + eps) + (1 - y_vector) * np.log(1 - p + eps)))
    null_prob = float(y_vector.mean())
    null_log_likelihood = float(
        np.sum(y_vector * np.log(null_prob + eps) + (1 - y_vector) * np.log(1 - null_prob + eps))
    )
    predicted_class = (p >= 0.5).astype(int)
    metrics = {
        "n_observations": int(len(y_vector)),
        "converged": bool(converged),
        "iterations": int(iterations),
        "log_likelihood": log_likelihood,
        "null_log_likelihood": null_log_likelihood,
        "lr_chi_square": 2 * (log_likelihood - null_log_likelihood),
        "lr_df": int(x_matrix.shape[1] - 1),
        "lr_p_value": _regularized_gamma_q((x_matrix.shape[1] - 1) / 2, (2 * (log_likelihood - null_log_likelihood)) / 2),
        "mcfadden_pseudo_r2": 1 - log_likelihood / null_log_likelihood,
        "accuracy_at_0_5_cutoff": float((predicted_class == y_vector).mean()),
        "majority_class_benchmark_accuracy": float(max(y_vector.mean(), 1 - y_vector.mean())),
        "auc": float(auc_score(y_vector, p)),
    }

    results = pd.DataFrame(
        {
            "variable": x.columns,
            "coefficient": beta,
            "std_error": se,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "z_value": z,
            "p_value": p_values,
            "odds_ratio": np.exp(beta),
            "odds_ratio_ci_95_low": np.exp(ci_low),
            "odds_ratio_ci_95_high": np.exp(ci_high),
        }
    )

    leverage = weights * np.einsum("ij,jk,ik->i", x_matrix, covariance, x_matrix)
    leverage = np.clip(leverage, 0, 1 - 1e-10)
    pearson_residual = (y_vector - p) / np.sqrt(weights)
    standardized_pearson = pearson_residual / np.sqrt(1 - leverage)
    parameter_count = x_matrix.shape[1]
    cooks_distance = (
        standardized_pearson**2 * leverage / (parameter_count * np.clip(1 - leverage, 1e-10, None))
    )
    diagnostics = pd.DataFrame(
        {
            "observed": y_vector.astype(int),
            "predicted_probability": p,
            "predicted_class": predicted_class,
            "pearson_residual": pearson_residual,
            "standardized_pearson_residual": standardized_pearson,
            "leverage": leverage,
            "cooks_distance": cooks_distance,
        },
        index=x.index,
    )
    return results, metrics, diagnostics


def average_marginal_premium_predictions(
    model_df: pd.DataFrame,
    design_matrix: pd.DataFrame,
    model_results: pd.DataFrame,
    premium_center: float,
    grid_size: int = 120,
) -> pd.DataFrame:
    """Average predicted probabilities and delta-method confidence intervals."""
    beta = model_results.set_index("variable").loc[design_matrix.columns, "coefficient"].to_numpy()
    fitted_probability = 1 / (1 + np.exp(-np.clip(design_matrix.to_numpy() @ beta, -35, 35)))
    weights = np.clip(fitted_probability * (1 - fitted_probability), 1e-8, None)
    information = (design_matrix.to_numpy().T * weights) @ design_matrix.to_numpy()
    covariance = np.linalg.pinv(information)

    premium_low, premium_high = model_df["Estimated Premium"].quantile([0.05, 0.95])
    premium_grid = np.geomspace(premium_low, premium_high, grid_size)
    rows = []
    for premium in premium_grid:
        prediction_matrix = design_matrix.copy()
        log_premium = math.log(premium)
        prediction_matrix["log_premium"] = log_premium
        prediction_matrix["log_premium_centered_squared"] = (log_premium - premium_center) ** 2
        prediction_values = prediction_matrix.to_numpy(dtype=float)
        linear_predictor = np.clip(prediction_values @ beta, -35, 35)
        probability = 1 / (1 + np.exp(-linear_predictor))
        average_probability = float(probability.mean())
        gradient = np.mean((probability * (1 - probability))[:, None] * prediction_values, axis=0)
        standard_error = math.sqrt(max(0.0, float(gradient @ covariance @ gradient)))
        rows.append(
            {
                "estimated_premium_millions": float(premium / 1_000_000),
                "predicted_probability": average_probability,
                "ci_95_low": max(0.0, average_probability - 1.96 * standard_error),
                "ci_95_high": min(1.0, average_probability + 1.96 * standard_error),
            }
        )
    return pd.DataFrame(rows)


def save_premium_prediction_figure(predictions: pd.DataFrame, output_dir: Path) -> None:
    """Save a publication-ready predicted-probability figure."""
    width, height = 2160, 1380
    left, top, right, bottom = 330, 120, 2050, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_figure_font(40)
    small_font = load_figure_font(34)

    premium = predictions["estimated_premium_millions"].to_numpy(dtype=float)
    log_premium = np.log10(premium)
    log_min, log_max = float(log_premium.min()), float(log_premium.max())

    def x_position(value: float) -> float:
        return left + (math.log10(value) - log_min) / (log_max - log_min) * (right - left)

    def y_position(value: float) -> float:
        return bottom - value * (bottom - top)

    for probability in np.arange(0, 1.01, 0.2):
        y_coord = y_position(float(probability))
        draw.line([(left, y_coord), (right, y_coord)], fill="#D9D9D9", width=2)
        label = f"{probability:.0%}"
        label_box = draw.textbbox((0, 0), label, font=small_font)
        draw.text(
            (left - 28 - (label_box[2] - label_box[0]), y_coord - 18),
            label,
            fill="#333333",
            font=small_font,
        )

    candidate_ticks = [1, 3, 10, 30, 100, 300, 1000, 3000]
    x_ticks = [value for value in candidate_ticks if premium.min() <= value <= premium.max()]
    for value in x_ticks:
        x_coord = x_position(value)
        draw.line([(x_coord, bottom), (x_coord, bottom + 12)], fill="#333333", width=2)
        label = f"\u00a3{value:,.0f}m"
        label_box = draw.textbbox((0, 0), label, font=small_font)
        draw.text(
            (x_coord - (label_box[2] - label_box[0]) / 2, bottom + 24),
            label,
            fill="#333333",
            font=small_font,
        )

    lower_points = [
        (x_position(row.estimated_premium_millions), y_position(row.ci_95_low))
        for row in predictions.itertuples()
    ]
    upper_points = [
        (x_position(row.estimated_premium_millions), y_position(row.ci_95_high))
        for row in predictions.iloc[::-1].itertuples()
    ]
    draw.polygon(lower_points + upper_points, fill=(155, 183, 212, 95))
    line_points = [
        (x_position(row.estimated_premium_millions), y_position(row.predicted_probability))
        for row in predictions.itertuples()
    ]
    draw.line(line_points, fill="#1F4E79", width=8, joint="curve")

    draw.line([(left, top), (left, bottom)], fill="#333333", width=3)
    draw.line([(left, bottom), (right, bottom)], fill="#333333", width=3)

    x_label = "Estimated Premium (\u00a3 million, logarithmic scale)"
    x_box = draw.textbbox((0, 0), x_label, font=font)
    draw.text(
        ((left + right - (x_box[2] - x_box[0])) / 2, height - 115),
        x_label,
        fill="#222222",
        font=font,
    )

    y_label = "Predicted probability of transaction success"
    y_box = draw.textbbox((0, 0), y_label, font=font)
    y_image = Image.new("RGBA", (y_box[2] - y_box[0] + 30, y_box[3] - y_box[1] + 30), (255, 255, 255, 0))
    ImageDraw.Draw(y_image).text((15, 8), y_label, fill="#222222", font=font)
    y_image = y_image.rotate(90, expand=True)
    image.paste(y_image, (55, int((top + bottom - y_image.height) / 2)), y_image)

    legend_x, legend_y = right - 640, top + 35
    draw.rectangle((legend_x, legend_y + 6, legend_x + 90, legend_y + 36), fill=(155, 183, 212, 95))
    draw.text((legend_x + 110, legend_y), "95% confidence interval", fill="#333333", font=small_font)
    draw.line(
        [(legend_x, legend_y + 68), (legend_x + 90, legend_y + 68)],
        fill="#1F4E79",
        width=8,
    )
    draw.text((legend_x + 110, legend_y + 48), "Average predicted probability", fill="#333333", font=small_font)

    image.save(
        output_dir / "figure_4_1_predicted_success_probability.png",
        format="PNG",
        dpi=(300, 300),
    )


PRODUCT_CATEGORIES = [
    "IA & DA / Buy-In",
    "IA & DA / Buyout",
    "IA / Buy-In",
    "Other product types",
]

REQUIRED_COLUMNS = [
    "Product",
    "Project Status",
    "Estimated Premium",
    "IAs",
    "DAs",
]


def product_dummy_variables(data: pd.DataFrame) -> pd.DataFrame:
    product = pd.Categorical(data["product_group"], categories=PRODUCT_CATEGORIES)
    return pd.get_dummies(product, prefix="product", drop_first=True, dtype=float).set_axis(data.index)


def principal_design_matrix(data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data[["log_premium", "log_total_annuity_count", "da_share"]].astype(float),
            product_dummy_variables(data),
        ],
        axis=1,
    )


def alternative_annuity_design_matrix(data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data[["log_premium", "log_ias_plus_1", "log_das_plus_1"]].astype(float),
            product_dummy_variables(data),
        ],
        axis=1,
    )


def variance_inflation_factors(x: pd.DataFrame) -> pd.DataFrame:
    predictors = x.drop(columns=["intercept"], errors="ignore").astype(float)
    rows = []
    for variable in predictors.columns:
        y = predictors[variable].to_numpy()
        other = predictors.drop(columns=[variable])
        design = np.column_stack([np.ones(len(other)), other.to_numpy(dtype=float)])
        coefficients = np.linalg.pinv(design) @ y
        residuals = y - design @ coefficients
        total_sum_squares = float(((y - y.mean()) ** 2).sum())
        residual_sum_squares = float((residuals**2).sum())
        r_squared = 1 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else float("nan")
        vif = 1 / (1 - r_squared) if not math.isnan(r_squared) and r_squared < 1 else float("inf")
        rows.append({"variable": variable, "r_squared": r_squared, "vif": vif})
    return pd.DataFrame(rows)


def likelihood_ratio_comparison(
    reduced_metrics: dict[str, float],
    full_metrics: dict[str, float],
    added_parameters: int,
    comparison: str,
) -> dict[str, float | str]:
    statistic = 2 * (full_metrics["log_likelihood"] - reduced_metrics["log_likelihood"])
    statistic = max(0.0, float(statistic))
    return {
        "comparison": comparison,
        "lr_chi_square": statistic,
        "df": int(added_parameters),
        "p_value": _regularized_gamma_q(added_parameters / 2, statistic / 2),
    }


def logit_linearity_checks(
    x: pd.DataFrame,
    y: pd.Series,
    principal_metrics: dict[str, float],
) -> pd.DataFrame:
    checks = []
    for variable in ["log_premium", "log_total_annuity_count", "da_share"]:
        centered = x[variable] - x[variable].mean()
        extended = x.copy()
        extended[f"{variable}_squared"] = centered**2
        _, extended_metrics, _ = fit_logit(extended, y)
        comparison = likelihood_ratio_comparison(
            principal_metrics,
            extended_metrics,
            added_parameters=1,
            comparison=f"Add squared term for {variable}",
        )
        comparison["variable"] = variable
        checks.append(comparison)
    return pd.DataFrame(checks)[["variable", "lr_chi_square", "df", "p_value", "comparison"]]


def prepare_data(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(input_path, sheet_name="PRT Data")
    df = raw.copy()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input workbook is missing required columns: {missing_columns}")
    if "Source Row" not in df.columns:
        df["Source Row"] = df.index + 2

    for col in ["Product", "Project Status"]:
        df[col] = df[col].astype(str).str.strip()

    df["success"] = np.where(df["Project Status"].eq("Won"), 1, 0)
    df["outcome_final"] = ~df["Project Status"].eq("Active")
    df["total_annuity_count"] = df["IAs"] + df["DAs"]
    df["da_share"] = np.where(df["total_annuity_count"] > 0, df["DAs"] / df["total_annuity_count"], np.nan)
    df["log_premium"] = np.log(df["Estimated Premium"])
    df["log_total_annuity_count"] = np.nan
    positive_total = df["total_annuity_count"].gt(0)
    df.loc[positive_total, "log_total_annuity_count"] = np.log(
        df.loc[positive_total, "total_annuity_count"]
    )
    df["log_ias_plus_1"] = np.log1p(df["IAs"])
    df["log_das_plus_1"] = np.log1p(df["DAs"])

    main_products = ["IA & DA / Buyout", "IA & DA / Buy-In", "IA / Buy-In"]
    df["product_group"] = np.where(
        df["Product"].isin(main_products),
        df["Product"],
        "Other product types",
    )

    outcome_sample = df[df["outcome_final"]].copy()
    full_variable_sample = outcome_sample.dropna(
        subset=[
            "success",
            "Product",
            "product_group",
            "Estimated Premium",
            "IAs",
            "DAs",
            "total_annuity_count",
            "da_share",
        ]
    ).copy()
    full_variable_sample = full_variable_sample[
        full_variable_sample["Estimated Premium"].gt(0)
        & full_variable_sample["total_annuity_count"].gt(0)
    ].copy()
    full_variable_sample["premium_size_band"] = pd.qcut(
        full_variable_sample["Estimated Premium"],
        q=4,
        labels=["Small", "Medium", "Large", "Very Large"],
        duplicates="drop",
    )
    df["premium_size_band"] = pd.NA
    df.loc[full_variable_sample.index, "premium_size_band"] = full_variable_sample["premium_size_band"].astype(str)
    won_lost_sample = full_variable_sample[full_variable_sample["Project Status"].isin(["Won", "Lost"])].copy()
    return df, full_variable_sample, won_lost_sample


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    if not input_path.exists():
        raise FileNotFoundError(
            "Input workbook not found. Supply an authorised workbook as the first "
            "command-line argument or place it at data/prt_project_data.xlsx."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, sample, won_lost = prepare_data(input_path)
    final_outcome = raw[raw["Project Status"].ne("Active")]

    sample.to_csv(output_dir / "clean_prt_analysis_sample.csv", index=False, encoding="utf-8-sig")
    won_lost.to_csv(output_dir / "clean_prt_won_lost_sample.csv", index=False, encoding="utf-8-sig")

    target_signing_date = (
        pd.to_datetime(raw["Target Signing Date"], errors="coerce")
        if "Target Signing Date" in raw.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    summary = {
        "input_file": input_path.name,
        "initial_rows": int(len(raw)),
        "outcome_sample_rows_excluding_active": int(len(final_outcome)),
        "full_variable_sample_rows": int(len(sample)),
        "won_lost_sensitivity_rows": int(len(won_lost)),
        "target_signing_date_min": (
            str(target_signing_date.min().date()) if target_signing_date.notna().any() else None
        ),
        "target_signing_date_max": (
            str(target_signing_date.max().date()) if target_signing_date.notna().any() else None
        ),
        "status_counts": raw["Project Status"].value_counts(dropna=False).to_dict(),
        "product_counts": raw["Product"].value_counts(dropna=False).to_dict(),
        "missing_values": raw.isna().sum().to_dict(),
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    descriptive_variables = [
        "Estimated Premium",
        "IAs",
        "DAs",
        "total_annuity_count",
        "da_share",
    ]
    descriptive = sample[descriptive_variables].describe(percentiles=[0.25, 0.5, 0.75]).T
    descriptive["skewness"] = sample[descriptive_variables].skew()
    descriptive.index.name = "variable"
    descriptive.to_csv(output_dir / "descriptive_statistics.csv", encoding="utf-8-sig")

    status_by_product = pd.crosstab(sample["product_group"], sample["success"], margins=True)
    status_by_size = pd.crosstab(sample["premium_size_band"], sample["success"], margins=True)
    status_by_product.to_csv(output_dir / "success_by_product.csv", encoding="utf-8-sig")
    status_by_size.to_csv(output_dir / "success_by_premium_size_band.csv", encoding="utf-8-sig")

    rates = (
        sample.groupby("product_group", observed=False)["success"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "opportunities", "sum": "successful", "mean": "success_rate"})
        .sort_values("opportunities", ascending=False)
    )
    rates.to_csv(output_dir / "product_success_rates.csv", encoding="utf-8-sig")

    chi_tests = pd.DataFrame(
        [
            {
                "test": "Product Group x Success",
                **chi_square_test(pd.crosstab(sample["product_group"], sample["success"])),
            },
            {
                "test": "Premium Size Band x Success",
                **chi_square_test(pd.crosstab(sample["premium_size_band"], sample["success"])),
            },
        ]
    )
    chi_tests.to_csv(output_dir / "chi_square_tests.csv", index=False, encoding="utf-8-sig")

    successful = sample[sample["success"].eq(1)]
    unsuccessful = sample[sample["success"].eq(0)]
    continuous_tests = pd.DataFrame(
        [
            {
                "variable": "Estimated Premium",
                **mann_whitney_u_test(successful["Estimated Premium"], unsuccessful["Estimated Premium"]),
            },
            {"variable": "IAs", **mann_whitney_u_test(successful["IAs"], unsuccessful["IAs"])},
            {"variable": "DAs", **mann_whitney_u_test(successful["DAs"], unsuccessful["DAs"])},
            {
                "variable": "Total Annuity Count",
                **mann_whitney_u_test(successful["total_annuity_count"], unsuccessful["total_annuity_count"]),
            },
            {"variable": "DA Share", **mann_whitney_u_test(successful["da_share"], unsuccessful["da_share"])},
        ]
    )
    continuous_tests.to_csv(output_dir / "mann_whitney_tests.csv", index=False, encoding="utf-8-sig")

    status_descriptive = (
        sample.groupby("Project Status", observed=False)
        .agg(
            opportunities=("success", "size"),
            median_premium=("Estimated Premium", "median"),
            median_ias=("IAs", "median"),
            median_das=("DAs", "median"),
            median_total_annuity_count=("total_annuity_count", "median"),
            median_da_share=("da_share", "median"),
        )
        .reset_index()
    )
    status_descriptive.to_csv(
        output_dir / "descriptive_statistics_by_original_status.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.crosstab(sample["Project Status"], sample["product_group"]).to_csv(
        output_dir / "product_group_by_original_status.csv",
        encoding="utf-8-sig",
    )

    premium_band_ranges = (
        sample.groupby("premium_size_band", observed=False)["Estimated Premium"]
        .agg(["count", "min", "max"])
        .reset_index()
        .rename(columns={"premium_size_band": "project_size_band"})
    )
    premium_band_ranges.to_csv(
        output_dir / "premium_size_band_cutpoints.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_df = sample.copy()
    x = principal_design_matrix(model_df)
    model_results, model_metrics, model_diagnostics = fit_logit(x, model_df["success"])
    model_results.to_csv(output_dir / "logistic_regression_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([model_metrics]).to_csv(
        output_dir / "logistic_regression_model_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    x.drop(columns=["intercept"]).corr().to_csv(
        output_dir / "logistic_regression_predictor_correlations.csv",
        encoding="utf-8-sig",
    )
    model_df[["log_premium", "log_total_annuity_count", "da_share"]].corr().to_csv(
        output_dir / "logistic_regression_continuous_predictor_correlations.csv",
        encoding="utf-8-sig",
    )
    variance_inflation_factors(x).to_csv(
        output_dir / "logistic_regression_vif.csv",
        index=False,
        encoding="utf-8-sig",
    )
    logit_linearity_checks(x, model_df["success"], model_metrics).to_csv(
        output_dir / "logistic_regression_linearity_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    influence = model_df[["Source Row", "Project Status", "success"]].join(model_diagnostics)
    influence.to_csv(
        output_dir / "logistic_regression_influence_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    n_observations = len(model_df)
    n_parameters = x.shape[1]
    influence_summary = {
        "n_observations": n_observations,
        "n_parameters": n_parameters,
        "leverage_threshold_2p_over_n": 2 * n_parameters / n_observations,
        "observations_above_leverage_threshold": int(
            (model_diagnostics["leverage"] > 2 * n_parameters / n_observations).sum()
        ),
        "cooks_distance_threshold_4_over_n": 4 / n_observations,
        "observations_above_cooks_threshold": int(
            (model_diagnostics["cooks_distance"] > 4 / n_observations).sum()
        ),
        "observations_with_abs_standardized_pearson_above_3": int(
            (model_diagnostics["standardized_pearson_residual"].abs() > 3).sum()
        ),
        "maximum_leverage": float(model_diagnostics["leverage"].max()),
        "maximum_cooks_distance": float(model_diagnostics["cooks_distance"].max()),
        "maximum_abs_standardized_pearson_residual": float(
            model_diagnostics["standardized_pearson_residual"].abs().max()
        ),
    }
    pd.DataFrame([influence_summary]).to_csv(
        output_dir / "logistic_regression_influence_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    won_lost_x = principal_design_matrix(won_lost)
    won_lost_results, won_lost_metrics, _ = fit_logit(won_lost_x, won_lost["success"])
    won_lost_results.to_csv(
        output_dir / "sensitivity_won_vs_lost_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([won_lost_metrics]).to_csv(
        output_dir / "sensitivity_won_vs_lost_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    alternative_x = alternative_annuity_design_matrix(model_df)
    alternative_results, alternative_metrics, _ = fit_logit(alternative_x, model_df["success"])
    alternative_results.to_csv(
        output_dir / "sensitivity_alternative_ia_da_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([alternative_metrics]).to_csv(
        output_dir / "sensitivity_alternative_ia_da_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_df[["log_premium", "log_ias_plus_1", "log_das_plus_1"]].corr().to_csv(
        output_dir / "sensitivity_alternative_ia_da_correlations.csv",
        encoding="utf-8-sig",
    )
    variance_inflation_factors(alternative_x).to_csv(
        output_dir / "sensitivity_alternative_ia_da_vif.csv",
        index=False,
        encoding="utf-8-sig",
    )

    interaction_x = x.copy()
    centered_log_premium = model_df["log_premium"] - model_df["log_premium"].mean()
    product_columns = [column for column in x.columns if column.startswith("product_")]
    for column in product_columns:
        interaction_x[f"log_premium_x_{column}"] = centered_log_premium * x[column]
    interaction_results, interaction_metrics, _ = fit_logit(interaction_x, model_df["success"])
    interaction_results.to_csv(
        output_dir / "supplementary_premium_product_interaction_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    interaction_comparison = likelihood_ratio_comparison(
        model_metrics,
        interaction_metrics,
        added_parameters=len(product_columns),
        comparison="Principal model vs Log Premium x Product Group interaction model",
    )
    pd.DataFrame([interaction_comparison]).to_csv(
        output_dir / "supplementary_premium_product_interaction_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    nonlinear_x = x.copy()
    nonlinear_variables = ["log_premium", "log_total_annuity_count", "da_share"]
    for variable in nonlinear_variables:
        centered = model_df[variable] - model_df[variable].mean()
        nonlinear_x[f"{variable}_centered_squared"] = centered**2
    nonlinear_results, nonlinear_metrics, _ = fit_logit(nonlinear_x, model_df["success"])
    nonlinear_results.to_csv(
        output_dir / "supplementary_quadratic_specification_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nonlinear_comparison = likelihood_ratio_comparison(
        model_metrics,
        nonlinear_metrics,
        added_parameters=len(nonlinear_variables),
        comparison="Principal model vs quadratic continuous-predictor specification",
    )
    pd.DataFrame([nonlinear_comparison]).to_csv(
        output_dir / "supplementary_quadratic_specification_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    premium_predictions = average_marginal_premium_predictions(
        model_df,
        nonlinear_x,
        nonlinear_results,
        premium_center=float(model_df["log_premium"].mean()),
    )
    premium_predictions.to_csv(
        output_dir / "figure_4_1_predicted_success_probability_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_premium_prediction_figure(premium_predictions, output_dir)

    model_comparison_rows = []
    for model_name, metrics in [
        ("Principal binary model", model_metrics),
        ("Won-versus-Lost sensitivity model", won_lost_metrics),
        ("Alternative IA and DA model", alternative_metrics),
        ("Premium x Product Group interaction model", interaction_metrics),
        ("Quadratic continuous-predictor model", nonlinear_metrics),
    ]:
        model_comparison_rows.append({"model": model_name, **metrics})
    pd.DataFrame(model_comparison_rows).to_csv(
        output_dir / "logistic_regression_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nAnalysis outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
