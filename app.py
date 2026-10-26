import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

st.set_page_config(
    page_title="Business Statistics Lab",
    page_icon="📊",
    layout="wide"
)

st.title("Business Statistics Lab")
st.caption("Multiple Regression and t-Tests with Step-by-Step Calculations")


# ============================================================
# COMMON FUNCTIONS
# ============================================================

def display_table(data):
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )


def parse_values(text):
    """Convert comma-separated values into a NumPy array."""
    values = [
        float(value.strip())
        for value in text.split(",")
        if value.strip()
    ]

    return np.array(values, dtype=float)


def show_t_test_result(
    t_stat,
    df,
    alternative,
    alpha,
    h0,
    h1
):
    """Display critical value, p-value and decision."""

    if alternative == "Two-tailed":
        p_value = 2 * stats.t.sf(abs(t_stat), df)
        critical = stats.t.ppf(1 - alpha / 2, df)

        reject = abs(t_stat) > critical

        rule = (
            f"Reject H₀ if |t| > {critical:.4f}, "
            f"or if p-value < {alpha}."
        )

    elif alternative == "Left-tailed":
        p_value = stats.t.cdf(t_stat, df)
        critical = stats.t.ppf(alpha, df)

        reject = t_stat < critical

        rule = (
            f"Reject H₀ if t < {critical:.4f}, "
            f"or if p-value < {alpha}."
        )

    else:
        p_value = stats.t.sf(t_stat, df)
        critical = stats.t.ppf(1 - alpha, df)

        reject = t_stat > critical

        rule = (
            f"Reject H₀ if t > {critical:.4f}, "
            f"or if p-value < {alpha}."
        )

    st.subheader("Hypothesis Test Decision")

    st.write(f"Null hypothesis: {h0}")
    st.write(f"Alternative hypothesis: {h1}")
    st.write(f"Degrees of freedom = {df:.4f}")
    st.write(f"Calculated t-statistic = {t_stat:.4f}")
    st.write(f"Critical t-value = {critical:.4f}")
    st.write(f"p-value = {p_value:.6f}")

    st.write(rule)

    if reject:
        st.success(
            "Reject H₀ at the selected significance level."
        )
    else:
        st.info(
            "Fail to reject H₀ at the selected significance level."
        )


# ============================================================
# GAUSSIAN ELIMINATION
# ============================================================

def gaussian_elimination_steps(matrix, rhs):

    augmented = np.column_stack(
        [
            matrix.astype(float),
            rhs.astype(float)
        ]
    )

    n = len(rhs)
    notes = []

    for col in range(n):

        pivot = col + np.argmax(
            np.abs(augmented[col:, col])
        )

        if abs(augmented[pivot, col]) < 1e-12:
            raise ValueError(
                "The equations cannot be solved uniquely. "
                "Check whether the independent variables are "
                "perfectly related."
            )

        if pivot != col:
            augmented[[col, pivot]] = (
                augmented[[pivot, col]]
            )

            notes.append(
                f"Swap R{col + 1} and R{pivot + 1}"
            )

        pivot_value = augmented[col, col]

        augmented[col] = (
            augmented[col] / pivot_value
        )

        notes.append(
            f"Divide R{col + 1} by {pivot_value:.6f}"
        )

        for row in range(n):

            if row != col:

                factor = augmented[row, col]

                if abs(factor) > 1e-12:

                    augmented[row] = (
                        augmented[row]
                        - factor * augmented[col]
                    )

                    notes.append(
                        f"R{row + 1} - "
                        f"({factor:.6f})R{col + 1}"
                    )

    coefficients = augmented[:, -1]

    return coefficients, notes, augmented


# ============================================================
# MULTIPLE LINEAR REGRESSION
# ============================================================

def multiple_regression_section():

    st.header("Multiple Linear Regression")

    st.write(
        "Calculate a regression equation with two or three "
        "independent variables."
    )

    st.latex(
        r"\hat{Y}=a+b_1X_1+b_2X_2+\cdots+b_kX_k"
    )

    st.info(
        "The sample data can be edited, or you can upload "
        "your own CSV file."
    )

    default_data = pd.DataFrame({
        "Sales": [50, 55, 60, 65, 70, 75, 80],
        "Advertising": [2, 3, 4, 4, 5, 6, 7],
        "Price": [10, 9, 8, 9, 7, 6, 5],
        "Salespersons": [2, 2, 3, 3, 4, 4, 5]
    })

    uploaded_file = st.file_uploader(
        "Upload CSV file (optional)",
        type=["csv"],
        key="regression_upload"
    )

    if uploaded_file is not None:

        try:
            source_data = pd.read_csv(uploaded_file)

        except Exception as error:
            st.error(f"Could not read CSV: {error}")
            return

    else:
        source_data = default_data

    numeric_columns = list(
        source_data.select_dtypes(
            include=np.number
        ).columns
    )

    if len(numeric_columns) < 3:
        st.error(
            "The dataset must contain at least three "
            "numeric columns."
        )
        return

    dependent_variable = st.selectbox(
        "Select dependent variable (Y)",
        numeric_columns,
        key="regression_y"
    )

    available_predictors = [
        column
        for column in numeric_columns
        if column != dependent_variable
    ]

    default_predictors = available_predictors[
        :min(2, len(available_predictors))
    ]

    independent_variables = st.multiselect(
        "Select two or three independent variables",
        available_predictors,
        default=default_predictors,
        key="regression_x"
    )

    st.subheader("Enter or edit your data")

    edited_data = st.data_editor(
        source_data,
        num_rows="dynamic",
        use_container_width=True,
        key="regression_editor"
    )

    if st.button(
        "Calculate Multiple Regression",
        key="regression_calculate"
    ):

        if len(independent_variables) not in (2, 3):

            st.error(
                "Select exactly two or three independent variables."
            )
            return

        try:

            required_columns = [
                dependent_variable
            ] + independent_variables

            data = edited_data[
                required_columns
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )

            data = data.replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            if len(data) <= len(independent_variables) + 1:

                st.error(
                    "Enter more complete observations. "
                    "You need at least k + 2 observations, "
                    "where k is the number of predictors."
                )
                return

            y = data[
                dependent_variable
            ].to_numpy(dtype=float)

            x_values = data[
                independent_variables
            ].to_numpy(dtype=float)

            n, k = x_values.shape

            # Add a column of ones for the intercept.
            X = np.column_stack(
                [
                    np.ones(n),
                    x_values
                ]
            )

            coefficient_names = (
                ["Intercept"] + independent_variables
            )

            xtx = X.T @ X
            xty = X.T @ y

            if np.linalg.matrix_rank(X) < X.shape[1]:

                st.error(
                    "The predictors are perfectly or nearly "
                    "perfectly related. Use data with independent "
                    "variation in each predictor."
                )
                return

            # Solve normal equations.
            coefficients, operations, reduced_matrix = (
                gaussian_elimination_steps(xtx, xty)
            )

            predicted = X @ coefficients
            residuals = y - predicted

            sse = float(np.sum(residuals ** 2))

            sst = float(
                np.sum((y - y.mean()) ** 2)
            )

            ssr = sst - sse

            r_squared = (
                1 - sse / sst
                if sst > 0
                else np.nan
            )

            adjusted_r_squared = (
                1 - (1 - r_squared)
                * (n - 1) / (n - k - 1)
                if sst > 0 and n > k + 1
                else np.nan
            )

            # Step 1: Data
            st.subheader("Step 1: Data used")

            display_table(
                data.reset_index(drop=True).round(4)
            )

            st.write(f"Number of observations, n = {n}")
            st.write(f"Number of predictors, k = {k}")

            # Step 2: Calculation columns
            st.subheader("Step 2: Calculation table")

            calculation_table = data.copy()

            calculation_table["Y²"] = y ** 2

            for column in independent_variables:

                calculation_table[
                    f"{column}²"
                ] = data[column] ** 2

                calculation_table[
                    f"{column} × Y"
                ] = data[column] * data[
                    dependent_variable
                ]

            for i in range(k):

                for j in range(i + 1, k):

                    column_1 = independent_variables[i]
                    column_2 = independent_variables[j]

                    calculation_table[
                        f"{column_1} × {column_2}"
                    ] = (
                        data[column_1]
                        * data[column_2]
                    )

            display_table(
                calculation_table.round(4)
            )

            st.write("Column totals")

            totals = pd.DataFrame({
                "Column": calculation_table.columns,
                "Total": [
                    calculation_table[column].sum()
                    for column in calculation_table.columns
                ]
            })

            display_table(totals.round(4))

            # Step 3: Normal equations
            st.subheader("Step 3: Normal equations")

            st.write(
                "The coefficients are obtained from:"
            )

            st.latex(
                r"(X^TX)\beta=X^TY"
            )

            st.write("Matrix XᵀX")

            display_table(
                pd.DataFrame(
                    xtx,
                    index=coefficient_names,
                    columns=coefficient_names
                ).round(4)
            )

            st.write("Vector XᵀY")

            display_table(
                pd.DataFrame(
                    {"XᵀY": xty},
                    index=coefficient_names
                ).round(4)
            )

            st.write("Normal equations with numerical substitution")

            for i in range(k + 1):

                terms = []

                for j in range(k + 1):

                    terms.append(
                        f"({xtx[i, j]:.4f})"
                        f"({coefficient_names[j]})"
                    )

                equation = (
                    " + ".join(terms)
                    + f" = {xty[i]:.4f}"
                )

                st.write(equation)

            # Step 4: Gaussian elimination
            st.subheader(
                "Step 4: Solve using Gaussian elimination"
            )

            with st.expander("Show row operations and final matrix"):

                for number, operation in enumerate(
                    operations,
                    start=1
                ):

                    st.write(
                        f"{number}. {operation}"
                    )

                st.write("Final reduced augmented matrix")

                display_table(
                    pd.DataFrame(
                        reduced_matrix,
                        columns=coefficient_names + ["RHS"]
                    ).round(6)
                )

            # Step 5: Regression equation
            st.subheader("Step 5: Regression equation")

            equation = (
                f"Ŷ = {coefficients[0]:.4f}"
            )

            for i, column in enumerate(
                independent_variables,
                start=1
            ):

                sign = (
                    "+"
                    if coefficients[i] >= 0
                    else "−"
                )

                equation += (
                    f" {sign} "
                    f"{abs(coefficients[i]):.4f}"
                    f"({column})"
                )

            st.success(equation)

            display_table(
                pd.DataFrame({
                    "Coefficient": coefficient_names,
                    "Estimate": coefficients
                }).round(6)
            )

            # Step 6: Predictions
            st.subheader(
                "Step 6: Predicted values and residuals"
            )

            output = data.copy()

            output["Predicted Y"] = predicted
            output["Residual (Y − Ŷ)"] = residuals

            display_table(
                output.round(4)
            )

            # Step 7: Goodness of fit
            st.subheader("Step 7: Goodness of fit")

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
                        sse,
                        sst,
                        ssr,
                        r_squared,
                        adjusted_r_squared
                    ]
                }).round(6)
            )

            if np.isfinite(r_squared):

                st.write(
                    f"The model explains approximately "
                    f"{r_squared * 100:.2f}% of the variation "
                    f"in {dependent_variable} in this dataset."
                )

        except (
            ValueError,
            np.linalg.LinAlgError,
            FloatingPointError
        ) as error:

            st.error(
                f"Regression calculation failed: {error}"
            )


# ============================================================
# ONE-SAMPLE T-TEST
# ============================================================

def one_sample_t_section():

    st.header("One-Sample t-Test")

    st.write(
        "Tests whether a sample mean differs from a specified "
        "population mean when the population standard deviation "
        "is unknown."
    )

    values_text = st.text_input(
        "Sample values, separated by commas",
        "48, 51, 49, 52, 50, 47, 53",
        key="one_sample_values"
    )

    mu0 = st.number_input(
        "Hypothesized population mean (μ₀)",
        value=50.0,
        key="one_sample_mu"
    )

    alternative = st.selectbox(
        "Alternative hypothesis",
        ["Two-tailed", "Left-tailed", "Right-tailed"],
        key="one_sample_alternative"
    )

    alpha = st.selectbox(
        "Significance level",
        [0.10, 0.05, 0.01],
        index=1,
        key="one_sample_alpha"
    )

    if st.button(
        "Calculate One-Sample t-Test",
        key="one_sample_calculate"
    ):

        try:

            x = parse_values(values_text)

            if len(x) < 2:

                st.error(
                    "Enter at least two observations."
                )
                return

            if not np.isfinite(x).all():

                st.error(
                    "All observations must be finite numbers."
                )
                return

            n = len(x)
            xbar = x.mean()
            sd = x.std(ddof=1)
            se = sd / np.sqrt(n)
            df = n - 1

            st.subheader("Step 1: Sample data")

            display_table(
                pd.DataFrame({
                    "Observation": np.arange(1, n + 1),
                    "X": x
                }).round(4)
            )

            st.write(f"n = {n}")
            st.write(f"Σx = {x.sum():.4f}")
            st.write(f"Mean = {xbar:.4f}")
            st.write(f"Sample SD = {sd:.4f}")

            st.subheader("Step 2: Standard error")

            st.write(
                f"SE = s/√n = {sd:.4f}/√{n} = {se:.4f}"
            )

            st.subheader("Step 3: t-statistic")

            st.latex(
                r"t=\frac{\bar{x}-\mu_0}{s/\sqrt{n}}"
            )

            if se == 0:

                st.warning(
                    "The sample standard deviation is zero. "
                    "The usual t-statistic is undefined."
                )
                return

            t_stat = (xbar - mu0) / se

            st.write(
                f"t = ({xbar:.4f} − {mu0:.4f})/"
                f"{se:.4f} = {t_stat:.4f}"
            )

            h1 = (
                "μ ≠ μ₀"
                if alternative == "Two-tailed"
                else (
                    "μ < μ₀"
                    if alternative == "Left-tailed"
                    else "μ > μ₀"
                )
            )

            show_t_test_result(
                t_stat,
                df,
                alternative,
                alpha,
                "H₀: μ = μ₀",
                f"H₁: {h1}"
            )

        except ValueError:

            st.error(
                "Invalid input. Enter numerical values "
                "separated by commas."
            )


# ============================================================
# INDEPENDENT TWO-SAMPLE T-TEST
# ============================================================

def independent_t_section():

    st.header("Independent Two-Sample t-Test")

    st.write(
        "Compares the means of two independent groups."
    )

    col1, col2 = st.columns(2)

    with col1:

        group1_text = st.text_area(
            "Group 1 values, separated by commas",
            "12, 14, 13, 15, 11",
            key="group1_values"
        )

    with col2:

        group2_text = st.text_area(
            "Group 2 values, separated by commas",
            "10, 9, 12, 11, 8",
            key="group2_values"
        )

    method = st.radio(
        "Choose the method",
        [
            "Welch's t-test (unequal variances allowed)",
            "Pooled-variance t-test (equal variances assumed)"
        ],
        key="independent_method"
    )

    alternative = st.selectbox(
        "Alternative hypothesis",
        ["Two-tailed", "Left-tailed", "Right-tailed"],
        key="independent_alternative"
    )

    alpha = st.selectbox(
        "Significance level",
        [0.10, 0.05, 0.01],
        index=1,
        key="independent_alpha"
    )

    if st.button(
        "Calculate Independent t-Test",
        key="independent_calculate"
    ):

        try:

            x1 = parse_values(group1_text)
            x2 = parse_values(group2_text)

            if len(x1) < 2 or len(x2) < 2:

                st.error(
                    "Enter at least two observations in each group."
                )
                return

            if not (
                np.isfinite(x1).all()
                and np.isfinite(x2).all()
            ):

                st.error(
                    "All observations must be finite numbers."
                )
                return

            n1 = len(x1)
            n2 = len(x2)

            mean1 = x1.mean()
            mean2 = x2.mean()

            sd1 = x1.std(ddof=1)
            sd2 = x2.std(ddof=1)

            var1 = sd1 ** 2
            var2 = sd2 ** 2

            st.subheader("Step 1: Summary statistics")

            display_table(
                pd.DataFrame({
                    "Group": ["Group 1", "Group 2"],
                    "n": [n1, n2],
                    "Mean": [mean1, mean2],
                    "Sample SD": [sd1, sd2],
                    "Sample variance": [var1, var2]
                }).round(4)
            )

            st.subheader("Step 2: Calculate standard error")

            if method.startswith("Welch"):

                v1 = var1 / n1
                v2 = var2 / n2

                se = np.sqrt(v1 + v2)

                denominator = (
                    v1 ** 2 / (n1 - 1)
                    + v2 ** 2 / (n2 - 1)
                )

                if denominator == 0:

                    st.warning(
                        "Both groups have zero variance. "
                        "The usual t-statistic and Welch degrees "
                        "of freedom cannot be calculated."
                    )
                    return

                df = (v1 + v2) ** 2 / denominator

                st.write("Welch's t-test")

                st.latex(
                    r"t=\frac{\bar{x}_1-\bar{x}_2}"
                    r"{\sqrt{s_1^2/n_1+s_2^2/n_2}}"
                )

                st.write(
                    f"SE = √({var1:.4f}/{n1} "
                    f"+ {var2:.4f}/{n2}) "
                    f"= {se:.4f}"
                )

                st.write(
                    f"Welch degrees of freedom = {df:.4f}"
                )

            else:

                df = n1 + n2 - 2

                pooled_variance = (
                    (n1 - 1) * var1
                    + (n2 - 1) * var2
                ) / df

                se = np.sqrt(
                    pooled_variance * (1/n1 + 1/n2)
                )

                st.write("Pooled-variance t-test")

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

            if se == 0:

                difference = mean1 - mean2

                if abs(difference) < 1e-12:

                    st.warning(
                        "Both groups have zero standard error "
                        "and identical means. The usual t-statistic "
                        "is undefined."
                    )

                else:

                    st.warning(
                        "The standard error is zero while the "
                        "sample means differ. The usual t-statistic "
                        "is undefined; do not report an ordinary "
                        "finite t-statistic."
                    )

                return

            t_stat = (mean1 - mean2) / se

            st.subheader("Step 3: Calculate t-statistic")

            st.write(
                f"t = ({mean1:.4f} − {mean2:.4f})/"
                f"{se:.4f} = {t_stat:.4f}"
            )

            h1 = (
                "μ₁ ≠ μ₂"
                if alternative == "Two-tailed"
                else (
                    "μ₁ < μ₂"
                    if alternative == "Left-tailed"
                    else "μ₁ > μ₂"
                )
            )

            show_t_test_result(
                t_stat,
                df,
                alternative,
                alpha,
                "H₀: μ₁ = μ₂",
                f"H₁: {h1}"
            )

        except ValueError:

            st.error(
                "Invalid input. Enter numerical values "
                "separated by commas."
            )


# ============================================================
# PAIRED-SAMPLES T-TEST
# ============================================================

def paired_t_section():

    st.header("Paired-Samples t-Test")

    st.write(
        "Use this test when the same people or units are "
        "measured before and after an intervention."
    )

    col1, col2 = st.columns(2)

    with col1:

        before_text = st.text_area(
            "Before values, separated by commas",
            "10, 12, 9, 14, 11",
            key="paired_before"
        )

    with col2:

        after_text = st.text_area(
            "After values, separated by commas",
            "12, 13, 11, 15, 13",
            key="paired_after"
        )

    direction = st.selectbox(
        "Difference direction",
        ["After − Before", "Before − After"],
        key="paired_direction"
    )

    alternative = st.selectbox(
        "Alternative hypothesis",
        ["Two-tailed", "Left-tailed", "Right-tailed"],
        key="paired_alternative"
    )

    alpha = st.selectbox(
        "Significance level",
        [0.10, 0.05, 0.01],
        index=1,
        key="paired_alpha"
    )

    if st.button(
        "Calculate Paired t-Test",
        key="paired_calculate"
    ):

        try:

            before = parse_values(before_text)
            after = parse_values(after_text)

            if len(before) != len(after):

                st.error(
                    "Before and after must contain the same "
                    "number of observations."
                )

            elif len(before) < 2:

                st.error(
                    "Enter at least two complete pairs."
                )

            elif not (
                np.isfinite(before).all()
                and np.isfinite(after).all()
            ):

                st.error(
                    "All observations must be finite numbers."
                )

            else:

                differences = (
                    after - before
                    if direction == "After − Before"
                    else before - after
                )

                n = len(differences)
                mean_d = differences.mean()
                sd_d = differences.std(ddof=1)
                se = sd_d / np.sqrt(n)
                df = n - 1

                st.subheader(
                    "Step 1: Calculate paired differences"
                )

                display_table(
                    pd.DataFrame({
                        "Before": before,
                        "After": after,
                        "Difference": differences
                    }).round(4)
                )

                st.subheader(
                    "Step 2: Calculate mean difference"
                )

                st.write(
                    f"Σd = {differences.sum():.4f}"
                )

                st.write(
                    f"d̄ = Σd/n = "
                    f"{differences.sum():.4f}/{n} "
                    f"= {mean_d:.4f}"
                )

                st.subheader(
                    "Step 3: Calculate SD and standard error"
                )

                st.write(
                    f"Standard deviation of differences = "
                    f"{sd_d:.4f}"
                )

                st.write(
                    f"SE = s_d/√n = "
                    f"{sd_d:.4f}/√{n} = {se:.4f}"
                )

                st.subheader(
                    "Step 4: Calculate t-statistic"
                )

                st.latex(
                    r"t=\frac{\bar{d}}{s_d/\sqrt{n}}"
                )

                if se == 0:

                    if abs(mean_d) < 1e-12:

                        st.warning(
                            "All paired differences are zero. "
                            "The t-statistic is undefined (0/0); "
                            "there is no observed change."
                        )

                    else:

                        st.warning(
                            "All paired differences are identical "
                            "and non-zero. The sample standard "
                            "deviation is zero, so the usual "
                            "t-statistic is undefined."
                        )

                else:

                    t_stat = mean_d / se

                    st.write(
                        f"t = {mean_d:.4f}/{se:.4f} "
                        f"= {t_stat:.4f}"
                    )

                    h1 = (
                        "μd ≠ 0"
                        if alternative == "Two-tailed"
                        else (
                            "μd < 0"
                            if alternative == "Left-tailed"
                            else "μd > 0"
                        )
                    )

                    show_t_test_result(
                        t_stat,
                        df,
                        alternative,
                        alpha,
                        "H₀: μd = 0",
                        f"H₁: {h1}"
                    )

        except ValueError:

            st.error(
                "Invalid input. Enter valid numerical values "
                "separated by commas."
            )


# ============================================================
# APP NAVIGATION
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

else:
    paired_t_section()

st.divider()

st.caption(
    "Educational tool. Check statistical assumptions and "
    "interpret results in context."
)
```
