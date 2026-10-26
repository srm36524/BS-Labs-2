import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Business Statistics Lab",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Business Statistics Lab")
st.caption("Step-by-step calculations for Business Statistics")


# ============================================================
# COMMON HELPER FUNCTIONS
# ============================================================

def display_table(data):
    """Display a dataframe using Streamlit."""
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )


def parse_numbers(text):
    """Convert comma-separated values into a numeric array."""
    values = np.array(
        [float(x.strip()) for x in text.split(",") if x.strip()],
        dtype=float
    )

    if len(values) == 0:
        raise ValueError("Enter at least one observation.")

    if not np.all(np.isfinite(values)):
        raise ValueError("All observations must be finite numbers.")

    return values


def show_t_test_result(t_stat, df, alternative):
    """Display t-statistic, degrees of freedom and p-value."""

    if alternative == "Two-tailed":
        p_value = 2 * stats.t.sf(abs(t_stat), df)
    elif alternative == "Right-tailed":
        p_value = stats.t.sf(t_stat, df)
    else:
        p_value = stats.t.cdf(t_stat, df)

    confidence = 0.95
    alpha = 1 - confidence

    st.subheader("Test results")

    display_table(
        pd.DataFrame({
            "Measure": [
                "t-statistic",
                "Degrees of freedom",
                "p-value",
                "Significance level",
                "Critical t-value"
            ],
            "Value": [
                t_stat,
                df,
                p_value,
                alpha,
                (
                    stats.t.ppf(1 - alpha / 2, df)
                    if alternative == "Two-tailed"
                    else stats.t.ppf(
                        1 - alpha if alternative == "Right-tailed"
                        else alpha,
                        df
                    )
                )
            ]
        }).round(6)
    )

    st.write(f"**Alternative hypothesis:** {alternative}")

    if p_value < alpha:
        st.success(
            "Reject the null hypothesis at the 5% significance level."
        )
    else:
        st.info(
            "Fail to reject the null hypothesis at the 5% significance level."
        )


# ============================================================
# MULTIPLE LINEAR REGRESSION COEFFICIENT CALCULATION
# ============================================================

def calculate_mlr_coefficients(data, y_col, x_cols):
    """
    Calculate multiple regression coefficients using centered sums.

    For two predictors, use the direct coefficient formulas.
    For three predictors, use determinants (Cramer's rule).
    """

    y = data[y_col].to_numpy(dtype=float)
    X = data[x_cols].to_numpy(dtype=float)

    n = len(y)
    k = len(x_cols)

    y_mean = np.mean(y)
    x_means = np.mean(X, axis=0)

    # Deviations from means
    Xc = X - x_means
    yc = y - y_mean

    # Centered sums of squares and cross-products
    Sxx = Xc.T @ Xc
    Sxy = Xc.T @ yc

    if np.linalg.matrix_rank(Sxx) < k:
        raise ValueError(
            "The independent variables are linearly dependent. "
            "Use a different dataset or different predictors."
        )

    coefficient_steps = []

    if k == 2:
        S11 = Sxx[0, 0]
        S22 = Sxx[1, 1]
        S12 = Sxx[0, 1]
        S1y = Sxy[0]
        S2y = Sxy[1]

        denominator = S11 * S22 - S12**2

        if abs(denominator) < 1e-12:
            raise ValueError(
                "The coefficient denominator is zero."
            )

        b1 = (
            S1y * S22 - S2y * S12
        ) / denominator

        b2 = (
            S2y * S11 - S1y * S12
        ) / denominator

        coefficients_x = np.array([b1, b2])

        coefficient_steps.append(
            f"S11 = {S11:.4f}, S22 = {S22:.4f}, "
            f"S12 = {S12:.4f}"
        )

        coefficient_steps.append(
            f"S1y = {S1y:.4f}, S2y = {S2y:.4f}"
        )

        coefficient_steps.append(
            f"Denominator = S11 × S22 − S12² "
            f"= {S11:.4f} × {S22:.4f} − "
            f"({S12:.4f})² = {denominator:.4f}"
        )

        coefficient_steps.append(
            f"b1 = (S1y × S22 − S2y × S12) / Denominator "
            f"= ({S1y:.4f} × {S22:.4f} − "
            f"{S2y:.4f} × {S12:.4f}) / "
            f"{denominator:.4f} = {b1:.4f}"
        )

        coefficient_steps.append(
            f"b2 = (S2y × S11 − S1y × S12) / Denominator "
            f"= ({S2y:.4f} × {S11:.4f} − "
            f"{S1y:.4f} × {S12:.4f}) / "
            f"{denominator:.4f} = {b2:.4f}"
        )

    else:
        # Three independent variables: Cramer's rule
        denominator = np.linalg.det(Sxx)

        if abs(denominator) < 1e-12:
            raise ValueError(
                "The coefficient denominator is zero."
            )

        coefficients_x = np.zeros(k)

        for j in range(k):
            replaced = Sxx.copy()
            replaced[:, j] = Sxy

            numerator = np.linalg.det(replaced)
            coefficients_x[j] = numerator / denominator

            coefficient_steps.append(
                f"b{j + 1} = determinant of the matrix with "
                f"column {j + 1} replaced by Sxy / determinant(Sxx) "
                f"= {numerator:.4f} / {denominator:.4f} "
                f"= {coefficients_x[j]:.4f}"
            )

    # Intercept
    intercept = y_mean - np.dot(coefficients_x, x_means)

    coefficients = np.concatenate(
        ([intercept], coefficients_x)
    )

    return (
        coefficients,
        y_mean,
        x_means,
        Sxx,
        Sxy,
        coefficient_steps
    )


# ============================================================
# MULTIPLE LINEAR REGRESSION
# ============================================================

def multiple_regression_section():

    st.header("Multiple Linear Regression")

    st.write(
        "Estimate one dependent variable using two or three "
        "independent variables."
    )

    st.latex(
        r"\hat{Y}=a+b_1X_1+b_2X_2+\cdots+b_kX_k"
    )

    st.write(
        "The coefficients are calculated using centered sums "
        "and regression coefficient formulas."
    )

    default_data = pd.DataFrame({
        "Sales": [10, 12, 13, 15, 15, 18],
        "Advertising": [1, 2, 1, 2, 3, 3],
        "Salespersons": [1, 1, 2, 2, 1, 2],
        "Price": [8, 7, 8, 6, 7, 5]
    })

    uploaded = st.file_uploader(
        "Upload a CSV file (optional)",
        type=["csv"],
        key="mlr_upload"
    )

    if uploaded is not None:
        try:
            source = pd.read_csv(uploaded)
        except Exception as error:
            st.error(f"Unable to read the CSV file: {error}")
            return
    else:
        source = default_data.copy()

    numeric_columns = list(
        source.select_dtypes(include=np.number).columns
    )

    if len(numeric_columns) < 3:
        st.error(
            "The dataset must contain at least three numeric columns."
        )
        return

    y_col = st.selectbox(
        "Select dependent variable (Y)",
        numeric_columns,
        key="mlr_y"
    )

    possible_x = [
        col for col in numeric_columns if col != y_col
    ]

    default_x = possible_x[:2]

    x_cols = st.multiselect(
        "Select two or three independent variables",
        possible_x,
        default=default_x,
        key="mlr_x"
    )

    st.write("Edit the sample data or the uploaded data below.")

    edited = st.data_editor(
        source,
        num_rows="dynamic",
        use_container_width=True,
        key="mlr_data"
    )

    if st.button(
        "Calculate MLR step by step",
        type="primary",
        key="mlr_calculate"
    ):

        if len(x_cols) not in (2, 3):
            st.error(
                "Select exactly two or three independent variables."
            )
            return

        try:
            required = [y_col] + x_cols

            data = edited[required].apply(
                pd.to_numeric,
                errors="coerce"
            )

            data = data.replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            n = len(data)
            k = len(x_cols)

            if n <= k + 1:
                st.error(
                    f"Enter at least {k + 2} complete observations."
                )
                return

            (
                coefficients,
                y_mean,
                x_means,
                Sxx,
                Sxy,
                steps
            ) = calculate_mlr_coefficients(
                data,
                y_col,
                x_cols
            )

            intercept = coefficients[0]
            slopes = coefficients[1:]

            y = data[y_col].to_numpy(dtype=float)
            X = data[x_cols].to_numpy(dtype=float)

            predicted = intercept + X @ slopes
            residuals = y - predicted

            SSE = np.sum(residuals**2)
            SST = np.sum((y - y_mean)**2)
            SSR = SST - SSE

            R2 = (
                1 - SSE / SST
                if SST > 0
                else np.nan
            )

            adjusted_R2 = (
                1 - (1 - R2) * (n - 1) / (n - k - 1)
                if np.isfinite(R2) and n > k + 1
                else np.nan
            )

            # STEP 1: DATA
            st.subheader("Step 1: Original data")

            display_table(
                data.reset_index(drop=True).round(4)
            )

            st.write(f"Observations: n = {n}")
            st.write(f"Independent variables: k = {k}")

            # STEP 2: MEANS
            st.subheader("Step 2: Calculate the means")

            mean_table = pd.DataFrame({
                "Variable": [y_col] + x_cols,
                "Mean": [y_mean] + list(x_means)
            })

            display_table(mean_table.round(4))

            # STEP 3: DEVIATIONS AND CROSS-PRODUCTS
            st.subheader(
                "Step 3: Calculate deviations and cross-products"
            )

            centered = pd.DataFrame(index=data.index)

            centered[f"{y_col} deviation"] = (
                data[y_col] - y_mean
            )

            for i, col in enumerate(x_cols):
                centered[f"{col} deviation"] = (
                    data[col] - x_means[i]
                )

            for i, col in enumerate(x_cols):
                centered[f"{col} deviation squared"] = (
                    data[col] - x_means[i]
                ) ** 2

                centered[f"{col} deviation × Y deviation"] = (
                    (data[col] - x_means[i])
                    * (data[y_col] - y_mean)
                )

            for i in range(k):
                for j in range(i + 1, k):
                    centered[
                        f"{x_cols[i]} deviation × "
                        f"{x_cols[j]} deviation"
                    ] = (
                        (data[x_cols[i]] - x_means[i])
                        * (data[x_cols[j]] - x_means[j])
                    )

            display_table(centered.reset_index(drop=True).round(4))

            # STEP 4: CENTERED SUMS
            st.subheader("Step 4: Calculate the centered sums")

            st.write(
                "Sxx contains the sums of squared deviations "
                "and cross-products among the X variables."
            )

            st.latex(
                r"S_{ij}=\sum(X_i-\bar{X}_i)(X_j-\bar{X}_j)"
            )

            display_table(
                pd.DataFrame(
                    Sxx,
                    index=x_cols,
                    columns=x_cols
                ).reset_index(names="Variable").round(4)
            )

            st.write(
                "Sxy contains the sums of products between "
                "each X variable and Y."
            )

            display_table(
                pd.DataFrame({
                    "Independent variable": x_cols,
                    "Sxy": Sxy
                }).round(4)
            )

            # STEP 5: COEFFICIENTS
            st.subheader(
                "Step 5: Calculate the regression coefficients"
            )

            if k == 2:

                st.write(
                    "For two independent variables, use these formulas:"
                )

                st.latex(
                    r"b_1=\frac{S_{1y}S_{22}-S_{2y}S_{12}}"
                    r"{S_{11}S_{22}-S_{12}^{2}}"
                )

                st.latex(
                    r"b_2=\frac{S_{2y}S_{11}-S_{1y}S_{12}}"
                    r"{S_{11}S_{22}-S_{12}^{2}}"
                )

            else:

                st.write(
                    "For three independent variables, use Cramer's "
                    "rule on the centered normal equations."
                )

                st.latex(
                    r"\mathbf{S}_{xx}\mathbf{b}=\mathbf{S}_{xy}"
                )

                st.latex(
                    r"b_j=\frac{\det(\mathbf{S}_{xx}^{(j)})}"
                    r"{\det(\mathbf{S}_{xx})}"
                )

            for step in steps:
                st.write(step)

            # INTERCEPT
            st.write("Calculate the intercept:")

            st.latex(
                r"a=\bar{Y}-b_1\bar{X}_1-\cdots-b_k\bar{X}_k"
            )

            terms = [
                f"{slopes[i]:.4f} × {x_means[i]:.4f}"
                for i in range(k)
            ]

            st.write(
                f"a = {y_mean:.4f} − "
                + " − ".join(terms)
                + f" = {intercept:.4f}"
            )

            # STEP 6: FINAL EQUATION
            st.subheader("Step 6: Final regression equation")

            equation = f"Ŷ = {intercept:.4f}"

            for i, col in enumerate(x_cols):
                sign = "+" if slopes[i] >= 0 else "−"
                equation += (
                    f" {sign} {abs(slopes[i]):.4f}({col})"
                )

            st.success(equation)

            display_table(
                pd.DataFrame({
                    "Coefficient": (
                        ["a"] +
                        [f"b{i + 1}" for i in range(k)]
                    ),
                    "Variable": ["Intercept"] + x_cols,
                    "Value": coefficients
                }).round(6)
            )

            # STEP 7: PREDICTIONS
            st.subheader(
                "Step 7: Predicted values and residuals"
            )

            output = data.copy()
            output["Predicted Y"] = predicted
            output["Residual (Y − Ŷ)"] = residuals

            display_table(
                output.reset_index(drop=True).round(4)
            )

            # STEP 8: GOODNESS OF FIT
            st.subheader("Step 8: Goodness of fit")

            display_table(
                pd.DataFrame({
                    "Measure": [
                        "SSE — Error sum of squares",
                        "SST — Total sum of squares",
                        "SSR — Regression sum of squares",
                        "R²",
                        "Adjusted R²"
                    ],
                    "Value": [
                        SSE,
                        SST,
                        SSR,
                        R2,
                        adjusted_R2
                    ]
                }).round(6)
            )

            if np.isfinite(R2):
                st.write(
                    f"The model explains {R2 * 100:.2f}% "
                    f"of the variation in {y_col}."
                )

        except (
            ValueError,
            np.linalg.LinAlgError,
            FloatingPointError
        ) as error:
            st.error(f"Calculation failed: {error}")


# ============================================================
# ONE-SAMPLE T-TEST
# ============================================================

def one_sample_t_section():

    st.header("One-Sample t-Test")

    st.write(
        "Tests whether a sample mean differs from a specified "
        "population mean."
    )

    sample_text = st.text_area(
        "Enter sample values separated by commas",
        value="12, 14, 15, 13, 16, 14, 15",
        key="one_t_data"
    )

    population_mean = st.number_input(
        "Hypothesized population mean",
        value=12.0,
        key="one_t_mean"
    )

    alternative = st.selectbox(
        "Alternative hypothesis",
        ["Two-tailed", "Right-tailed", "Left-tailed"],
        key="one_t_alternative"
    )

    if st.button("Calculate one-sample t-test"):

        try:
            sample = parse_numbers(sample_text)

            n = len(sample)

            if n < 2:
                st.error("At least two observations are required.")
                return

            mean = np.mean(sample)
            sd = np.std(sample, ddof=1)
            se = sd / np.sqrt(n)
            df = n - 1

            if se == 0:
                if mean == population_mean:
                    st.error(
                        "The sample has zero variance and its mean "
                        "equals the hypothesized mean; the t-statistic "
                        "is undefined."
                    )
                else:
                    st.error(
                        "The sample has zero variance but its mean "
                        "differs from the hypothesized mean; the "
                        "ordinary t-statistic is undefined."
                    )
                return

            t_stat = (mean - population_mean) / se

            st.subheader("Step-by-step calculation")

            display_table(
                pd.DataFrame({
                    "Measure": [
                        "Sample size (n)",
                        "Sample mean",
                        "Sample standard deviation",
                        "Standard error",
                        "Hypothesized mean",
                        "Degrees of freedom"
                    ],
                    "Value": [
                        n,
                        mean,
                        sd,
                        se,
                        population_mean,
                        df
                    ]
                }).round(6)
            )

            st.latex(
                r"t=\frac{\bar{x}-\mu_0}{s/\sqrt{n}}"
            )

            st.write(
                f"t = ({mean:.4f} − {population_mean:.4f}) "
                f"/ {se:.4f} = {t_stat:.4f}"
            )

            show_t_test_result(t_stat, df, alternative)

        except ValueError as error:
            st.error(str(error))


# ============================================================
# INDEPENDENT TWO-SAMPLE T-TEST
# ============================================================

def independent_t_section():

    st.header("Independent Two-Sample t-Test")

    st.write(
        "Compare the means of two independent groups using "
        "Welch's test or the pooled-variance t-test."
    )

    group1_text = st.text_area(
        "Group 1 values, separated by commas",
        value="12, 14, 15, 16, 13",
        key="ind_t_group1"
    )

    group2_text = st.text_area(
        "Group 2 values, separated by commas",
        value="17, 18, 16, 19, 20",
        key="ind_t_group2"
    )

    method = st.radio(
        "Choose the test",
        [
            "Welch t-test (unequal variances)",
            "Pooled t-test (equal variances)"
        ],
        key="ind_t_method"
    )

    alternative = st.selectbox(
        "Alternative hypothesis",
        ["Two-tailed", "Right-tailed", "Left-tailed"],
        key="ind_t_alternative"
    )

    if st.button("Calculate independent t-test"):

        try:
            x1 = parse_numbers(group1_text)
            x2 = parse_numbers(group2_text)

            n1 = len(x1)
            n2 = len(x2)

            if n1 < 2 or n2 < 2:
                st.error(
                    "Each group must contain at least two observations."
                )
                return

            mean1 = np.mean(x1)
            mean2 = np.mean(x2)

            var1 = np.var(x1, ddof=1)
            var2 = np.var(x2, ddof=1)

            if method.startswith("Welch"):

                se_squared = var1 / n1 + var2 / n2

                if se_squared == 0:
                    st.error(
                        "Both groups have zero variance; "
                        "the t-statistic is undefined."
                    )
                    return

                se = np.sqrt(se_squared)

                t_stat = (mean1 - mean2) / se

                numerator = se_squared**2

                denominator = (
                    (var1 / n1)**2 / (n1 - 1)
                    + (var2 / n2)**2 / (n2 - 1)
                )

                if denominator == 0:
                    st.error("Degrees of freedom cannot be calculated.")
                    return

                df = numerator / denominator

                st.subheader("Welch t-test calculation")

                st.latex(
                    r"t=\frac{\bar{x}_1-\bar{x}_2}"
                    r"\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}"
                )

                st.write(
                    f"SE = √({var1:.4f}/{n1} + "
                    f"{var2:.4f}/{n2}) = {se:.4f}"
                )

                st.write(
                    f"t = ({mean1:.4f} − {mean2:.4f}) "
                    f"/ {se:.4f} = {t_stat:.4f}"
                )

                st.write(
                    "Welch-Satterthwaite degrees of freedom:"
                )

                st.latex(
                    r"df=\frac{(s_1^2/n_1+s_2^2/n_2)^2}"
                    r"{\frac{(s_1^2/n_1)^2}{n_1-1}"
                    r"+\frac{(s_2^2/n_2)^2}{n_2-1}}"
                )

            else:

                df = n1 + n2 - 2

                pooled_variance = (
                    ((n1 - 1) * var1 + (n2 - 1) * var2)
                    / df
                )

                se_squared = pooled_variance * (
                    1 / n1 + 1 / n2
                )

                if se_squared == 0:
                    st.error(
                        "The pooled standard error is zero; "
                        "the t-statistic is undefined."
                    )
                    return

                se = np.sqrt(se_squared)

                t_stat = (mean1 - mean2) / se

                st.subheader("Pooled-variance t-test calculation")

                st.latex(
                    r"s_p^2=\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}"
                    r"{n_1+n_2-2}"
                )

                st.write(
                    f"Pooled variance = {pooled_variance:.4f}"
                )

                st.latex(
                    r"t=\frac{\bar{x}_1-\bar{x}_2}"
                    r"{s_p\sqrt{1/n_1+1/n_2}}"
                )

                st.write(
                    f"SE = √[{pooled_variance:.4f} "
                    f"× (1/{n1} + 1/{n2})] = {se:.4f}"
                )

                st.write(
                    f"t = ({mean1:.4f} − {mean2:.4f}) "
                    f"/ {se:.4f} = {t_stat:.4f}"
                )

            st.subheader("Sample summary")

            display_table(
                pd.DataFrame({
                    "Measure": [
                        "Group 1 sample size",
                        "Group 1 mean",
                        "Group 1 variance",
                        "Group 2 sample size",
                        "Group 2 mean",
                        "Group 2 variance",
                        "Standard error",
                        "Degrees of freedom"
                    ],
                    "Value": [
                        n1,
                        mean1,
                        var1,
                        n2,
                        mean2,
                        var2,
                        se,
                        df
                    ]
                }).round(6)
            )

            show_t_test_result(t_stat, df, alternative)

        except ValueError as error:
            st.error(str(error))


# ============================================================
# PAIRED-SAMPLES T-TEST
# ============================================================

def paired_t_section():

    st.header("Paired-Samples t-Test")

    st.write(
        "Compare two measurements taken from the same subjects, "
        "such as before and after training."
    )

    before_text = st.text_area(
        "Before values, separated by commas",
        value="10, 12, 14, 13, 15",
        key="paired_before"
    )

    after_text = st.text_area(
        "After values, separated by commas",
        value="12, 13, 16, 15, 18",
        key="paired_after"
    )

    direction = st.radio(
        "Difference direction",
        ["After − Before", "Before − After"],
        key="paired_direction"
    )

    alternative = st.selectbox(
        "Alternative hypothesis",
        ["Two-tailed", "Right-tailed", "Left-tailed"],
        key="paired_alternative"
    )

    if st.button("Calculate paired t-test"):

        try:
            before = parse_numbers(before_text)
            after = parse_numbers(after_text)

            if len(before) != len(after):
                st.error(
                    "Before and after must have the same number "
                    "of observations."
                )
                return

            n = len(before)

            if n < 2:
                st.error(
                    "At least two paired observations are required."
                )
                return

            if direction == "After − Before":
                differences = after - before
            else:
                differences = before - after

            mean_d = np.mean(differences)
            sd_d = np.std(differences, ddof=1)
            se = sd_d / np.sqrt(n)
            df = n - 1

            if se == 0:
                st.error(
                    "The differences have zero variance; "
                    "the t-statistic is undefined."
                )
                return

            t_stat = mean_d / se

            st.subheader("Step-by-step calculation")

            display_table(
                pd.DataFrame({
                    "Before": before,
                    "After": after,
                    "Difference": differences
                }).round(4)
            )

            st.write(f"Mean difference = {mean_d:.4f}")
            st.write(f"Standard deviation of differences = {sd_d:.4f}")
            st.write(f"Standard error = {se:.4f}")
            st.write(f"Degrees of freedom = {df}")

            st.latex(
                r"t=\frac{\bar{d}}{s_d/\sqrt{n}}"
            )

            st.write(
                f"t = {mean_d:.4f} / {se:.4f} "
                f"= {t_stat:.4f}"
            )

            show_t_test_result(t_stat, df, alternative)

        except ValueError as error:
            st.error(str(error))


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Choose a module",
    [
        "Multiple Linear Regression",
        "One-Sample t-Test",
        "Independent Two-Sample t-Test",
        "Paired-Samples t-Test"
    ]
)

if page == "Multiple Linear Regression":
    multiple_regression_section()

elif page == "One-Sample t-Test":
    one_sample_t_section()

elif page == "Independent Two-Sample t-Test":
    independent_t_section()

elif page == "Paired-Samples t-Test":
    paired_t_section()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Business Statistics Lab | Educational use. "
    "Check assumptions and interpret statistical results "
    "in the context of the problem."
)
