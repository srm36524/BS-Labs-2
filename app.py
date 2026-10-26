import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

st.set_page_config(
    page_title="Business Statistics Lab",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Business Statistics Lab")
st.caption("Step-by-step calculations for Regression and t-Tests")


# =========================================================
# COMMON FUNCTIONS
# =========================================================

def fmt(x, decimals=4):
    return f"{float(x):.{decimals}f}"


def clean_numeric(df, columns):
    result = df[columns].copy()

    for col in columns:
        result[col] = pd.to_numeric(
            result[col], errors="coerce"
        )

    return result.dropna().reset_index(drop=True)


def display_table(df):
    st.dataframe(df, use_container_width=True)


def gaussian_elimination(A, b, variable_names):
    """
    Solve Ax=b and show each row operation.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    augmented = np.hstack([A, b])
    n = len(b)

    st.subheader("Step 4: Solve the simultaneous equations")

    st.write(
        "Write the coefficients and constants as an "
        "augmented matrix."
    )

    labels = variable_names + ["Constant"]

    display_table(
        pd.DataFrame(
            augmented,
            columns=labels
        ).round(6)
    )

    for i in range(n):

        # Partial pivoting improves numerical stability.
        pivot_row = i + np.argmax(
            np.abs(augmented[i:, i])
        )

        if abs(augmented[pivot_row, i]) < 1e-12:
            raise ValueError(
                "The equations do not have a unique solution. "
                "Check whether predictors are redundant."
            )

        if pivot_row != i:
            augmented[[i, pivot_row]] = (
                augmented[[pivot_row, i]]
            )

            st.write(
                f"Swap Row {i + 1} with Row {pivot_row + 1} "
                "to obtain a usable pivot."
            )

        pivot = augmented[i, i]

        st.markdown(
            f"**Step 4.{i + 1}: Make the pivot in "
            f"Row {i + 1} equal to 1**"
        )

        augmented[i] = augmented[i] / pivot

        st.latex(
            rf"R_{{{i + 1}}} \rightarrow "
            rf"\frac{{R_{{{i + 1}}}}}{{{pivot:.6f}}}"
        )

        display_table(
            pd.DataFrame(
                augmented,
                columns=labels
            ).round(6)
        )

        # Eliminate the pivot-column values in other rows.
        for j in range(n):

            if j == i:
                continue

            factor = augmented[j, i]

            if abs(factor) > 1e-12:

                st.write(
                    f"Eliminate the entry in Row {j + 1}, "
                    f"Column {i + 1}:"
                )

                st.latex(
                    rf"R_{{{j + 1}}} \rightarrow "
                    rf"R_{{{j + 1}}} "
                    rf"- ({factor:.6f})R_{{{i + 1}}}"
                )

                augmented[j] = (
                    augmented[j] -
                    factor * augmented[i]
                )

                display_table(
                    pd.DataFrame(
                        augmented,
                        columns=labels
                    ).round(6)
                )

    solution = augmented[:, -1]

    st.success("The simultaneous equations are solved.")

    for name, value in zip(variable_names, solution):
        st.write(f"**{name} = {value:.6f}**")

    return solution


def show_t_test_result(
    t_stat,
    df,
    alternative,
    alpha,
    h0,
    h1
):
    if alternative == "Two-tailed":
        p_value = 2 * stats.t.sf(
            abs(t_stat), df
        )

        critical = stats.t.ppf(
            1 - alpha / 2, df
        )

        rule = (
            f"Reject H₀ if |t| > {critical:.4f} "
            f"or p < {alpha:.3f}."
        )

        critical_text = f"±{critical:.4f}"

    elif alternative == "Right-tailed":
        p_value = stats.t.sf(t_stat, df)

        critical = stats.t.ppf(
            1 - alpha, df
        )

        rule = (
            f"Reject H₀ if t > {critical:.4f} "
            f"or p < {alpha:.3f}."
        )

        critical_text = fmt(critical)

    else:
        p_value = stats.t.cdf(t_stat, df)

        critical = stats.t.ppf(
            alpha, df
        )

        rule = (
            f"Reject H₀ if t < {critical:.4f} "
            f"or p < {alpha:.3f}."
        )

        critical_text = fmt(critical)

    st.subheader("Hypothesis test decision")

    st.write(f"Null hypothesis: {h0}")
    st.write(f"Alternative hypothesis: {h1}")
    st.write(f"Significance level α = {alpha}")
    st.write(f"Calculated t-statistic = {t_stat:.4f}")
    st.write(f"Degrees of freedom = {df}")
    st.write(f"Critical value = {critical_text}")
    st.write(f"p-value = {p_value:.6f}")
    st.write(f"Decision rule: {rule}")

    if p_value < alpha:
        st.error(
            "Reject H₀. The result is statistically "
            "significant at the selected significance level."
        )
    else:
        st.success(
            "Fail to reject H₀. There is insufficient "
            "evidence to reject the null hypothesis."
        )


# =========================================================
# SIDEBAR
# =========================================================

module = st.sidebar.radio(
    "Select a statistical method",
    [
        "Multiple Linear Regression",
        "One-Sample t-Test",
        "Independent Two-Sample t-Test",
        "Paired t-Test"
    ]
)


# =========================================================
# MULTIPLE LINEAR REGRESSION
# =========================================================

if module == "Multiple Linear Regression":

    st.header("Multiple Linear Regression")

    st.write(
        "This module calculates the regression equation "
        "and explains the arithmetic at every stage."
    )

    number_of_x = st.radio(
        "Number of independent variables",
        [2, 3],
        horizontal=True
    )

    if number_of_x == 2:

        default_data = pd.DataFrame({
            "Y": [12, 15, 17, 20, 22, 25],
            "X1": [1, 2, 2, 3, 4, 5],
            "X2": [2, 2, 3, 3, 4, 5]
        })

        x_columns = ["X1", "X2"]

    else:

        default_data = pd.DataFrame({
            "Y": [15, 18, 20, 23, 25, 28, 31],
            "X1": [1, 2, 2, 3, 4, 5, 6],
            "X2": [2, 2, 3, 3, 4, 5, 6],
            "X3": [1, 1, 2, 2, 3, 4, 4]
        })

        x_columns = ["X1", "X2", "X3"]

    st.subheader("Step 1: Enter the observations")

    uploaded = st.file_uploader(
        "Optional: Upload a CSV file",
        type=["csv"],
        key="reg_csv"
    )

    if uploaded is not None:

        try:
            default_data = pd.read_csv(uploaded)

        except Exception as e:
            st.error(f"Unable to read CSV: {e}")

    st.info(
        "The dependent variable must be named Y. "
        "Independent variables must be named X1, X2 "
        "and, if applicable, X3."
    )

    edited_data = st.data_editor(
        default_data,
        num_rows="dynamic",
        use_container_width=True,
        key=f"reg_editor_{number_of_x}"
    )

    if st.button(
        "Calculate MLR step by step",
        type="primary"
    ):

        required = ["Y"] + x_columns

        if not all(c in edited_data.columns for c in required):

            st.error(
                f"The table must contain these columns: "
                f"{', '.join(required)}"
            )

        else:

            df = clean_numeric(
                edited_data,
                required
            )

            n = len(df)
            k = len(x_columns)

            if n <= k + 1:

                st.error(
                    f"Enter at least {k + 2} complete "
                    "observations. More observations are "
                    "recommended for meaningful regression."
                )

            else:

                # -------------------------------------------------
                # STEP 2: CALCULATE SUMS
                # -------------------------------------------------

                st.subheader(
                    "Step 2: Calculate the required columns"
                )

                calc = df.copy()

                for x in x_columns:
                    calc[f"{x}²"] = calc[x] ** 2
                    calc[f"{x}Y"] = calc[x] * calc["Y"]

                if k >= 2:
                    calc["X1X2"] = (
                        calc["X1"] * calc["X2"]
                    )

                if k == 3:
                    calc["X1X3"] = (
                        calc["X1"] * calc["X3"]
                    )

                    calc["X2X3"] = (
                        calc["X2"] * calc["X3"]
                    )

                calc["Y²"] = calc["Y"] ** 2

                display_table(calc.round(4))

                st.markdown("### Calculate column totals")

                totals = calc.sum(numeric_only=True)

                display_table(
                    pd.DataFrame({
                        "Column": totals.index,
                        "Total": totals.values
                    }).round(4)
                )

                st.write(f"Number of observations n = {n}")

                # -------------------------------------------------
                # STEP 3: FORM NORMAL EQUATIONS
                # -------------------------------------------------

                st.subheader(
                    "Step 3: Form the normal equations"
                )

                st.write(
                    "The regression model includes an intercept "
                    "and the selected independent variables."
                )

                if k == 2:

                    st.latex(
                        r"\sum Y=na+b_1\sum X_1+b_2\sum X_2"
                    )

                    st.latex(
                        r"\sum X_1Y=a\sum X_1"
                        r"+b_1\sum X_1^2"
                        r"+b_2\sum X_1X_2"
                    )

                    st.latex(
                        r"\sum X_2Y=a\sum X_2"
                        r"+b_1\sum X_1X_2"
                        r"+b_2\sum X_2^2"
                    )

                else:

                    st.latex(
                        r"\sum Y=na+b_1\sum X_1"
                        r"+b_2\sum X_2+b_3\sum X_3"
                    )

                    st.latex(
                        r"\sum X_1Y=a\sum X_1"
                        r"+b_1\sum X_1^2"
                        r"+b_2\sum X_1X_2"
                        r"+b_3\sum X_1X_3"
                    )

                    st.latex(
                        r"\sum X_2Y=a\sum X_2"
                        r"+b_1\sum X_1X_2"
                        r"+b_2\sum X_2^2"
                        r"+b_3\sum X_2X_3"
                    )

                    st.latex(
                        r"\sum X_3Y=a\sum X_3"
                        r"+b_1\sum X_1X_3"
                        r"+b_2\sum X_2X_3"
                        r"+b_3\sum X_3^2"
                    )

                # -------------------------------------------------
                # STEP 3A: SUBSTITUTE NUMERICAL VALUES
                # -------------------------------------------------

                st.subheader(
                    "Step 3A: Substitute the actual totals"
                )

                sums = {
                    "Y": df["Y"].sum(),
                    "n": n
                }

                for x in x_columns:
                    sums[x] = df[x].sum()
                    sums[x + "2"] = (
                        df[x] ** 2
                    ).sum()

                    sums[x + "Y"] = (
                        df[x] * df["Y"]
                    ).sum()

                sums["X1X2"] = (
                    df["X1"] * df["X2"]
                ).sum()

                if k == 3:

                    sums["X1X3"] = (
                        df["X1"] * df["X3"]
                    ).sum()

                    sums["X2X3"] = (
                        df["X2"] * df["X3"]
                    ).sum()

                st.write(
                    f"ΣY = {sums['Y']:.4f}"
                )

                for x in x_columns:

                    st.write(
                        f"Σ{x} = {sums[x]:.4f}"
                    )

                    st.write(
                        f"Σ{x}² = {sums[x + '2']:.4f}"
                    )

                    st.write(
                        f"Σ{x}Y = {sums[x + 'Y']:.4f}"
                    )

                st.write(
                    f"ΣX1X2 = {sums['X1X2']:.4f}"
                )

                if k == 3:

                    st.write(
                        f"ΣX1X3 = {sums['X1X3']:.4f}"
                    )

                    st.write(
                        f"ΣX2X3 = {sums['X2X3']:.4f}"
                    )

                # -------------------------------------------------
                # BUILD NORMAL EQUATION MATRIX
                # -------------------------------------------------

                X = np.column_stack(
                    [
                        np.ones(n)
                    ] + [
                        df[x].to_numpy(dtype=float)
                        for x in x_columns
                    ]
                )

                Y = df["Y"].to_numpy(dtype=float)

                A = X.T @ X
                b = X.T @ Y

                variable_names = ["a"] + [
                    f"b{i + 1}"
                    for i in range(k)
                ]

                st.markdown(
                    "### Numerical normal equations"
                )

                for i, name in enumerate(variable_names):

                    pieces = []

                    for j, other_name in enumerate(
                        variable_names
                    ):

                        pieces.append(
                            f"({A[i, j]:.4f}){other_name}"
                        )

                    st.latex(
                        " + ".join(pieces)
                        + f" = {b[i]:.4f}"
                    )

                # -------------------------------------------------
                # STEP 4: SOLVE EQUATIONS
                # -------------------------------------------------

                solution = gaussian_elimination(
                    A,
                    b,
                    variable_names
                )

                # -------------------------------------------------
                # STEP 5: FORM REGRESSION EQUATION
                # -------------------------------------------------

                st.subheader(
                    "Step 5: Form the regression equation"
                )

                equation = f"Ŷ = {solution[0]:.4f}"

                for i, x in enumerate(x_columns):

                    coefficient = solution[i + 1]

                    sign = (
                        "+"
                        if coefficient >= 0
                        else "-"
                    )

                    equation += (
                        f" {sign} "
                        f"{abs(coefficient):.4f}{x}"
                    )

                st.success(
                    f"Estimated regression equation: {equation}"
                )

                # -------------------------------------------------
                # STEP 6: PREDICTIONS AND RESIDUALS
                # -------------------------------------------------

                st.subheader(
                    "Step 6: Calculate predicted values"
                )

                predicted = X @ solution
                residuals = Y - predicted

                result = df.copy()

                result["Predicted Y"] = predicted
                result["Residual Y - Predicted Y"] = residuals
                result["Squared residual"] = residuals ** 2

                display_table(result.round(4))

                st.write(
                    "For each row, substitute its X values "
                    "into the estimated regression equation."
                )

                for i in range(min(n, 10)):

                    parts = [
                        f"{solution[0]:.4f}"
                    ]

                    for j, x in enumerate(x_columns):

                        parts.append(
                            f"({solution[j + 1]:.4f})"
                            f"({df.iloc[i][x]:.4f})"
                        )

                    st.write(
                        f"Observation {i + 1}: "
                        + " + ".join(parts)
                        + f" = {predicted[i]:.4f}"
                    )

                if n > 10:
                    st.caption(
                        "The table contains all observations; "
                        "the substitution examples above show "
                        "the first ten."
                    )

                # -------------------------------------------------
                # STEP 7: SSE, SST AND R SQUARED
                # -------------------------------------------------

                st.subheader(
                    "Step 7: Calculate goodness of fit"
                )

                mean_y = Y.mean()

                SSE = np.sum(residuals ** 2)
                SST = np.sum((Y - mean_y) ** 2)
                SSR = SST - SSE

                R2 = (
                    1 - SSE / SST
                    if SST > 0
                    else np.nan
                )

                df_error = n - k - 1

                adjusted_R2 = (
                    1 - (1 - R2) * (n - 1) / df_error
                    if df_error > 0 and np.isfinite(R2)
                    else np.nan
                )

                st.write(
                    f"Mean of Y = ΣY / n = "
                    f"{sums['Y']:.4f} / {n} "
                    f"= {mean_y:.4f}"
                )

                st.write(
                    f"SSE = Σ(Y − Ŷ)² = {SSE:.4f}"
                )

                st.write(
                    f"SST = Σ(Y − Ȳ)² = {SST:.4f}"
                )

                st.write(
                    f"SSR = SST − SSE = "
                    f"{SST:.4f} − {SSE:.4f} "
                    f"= {SSR:.4f}"
                )

                st.latex(
                    r"R^2=1-\frac{SSE}{SST}"
                )

                st.write(
                    f"R² = 1 − {SSE:.4f}/{SST:.4f} "
                    f"= {R2:.4f}"
                )

                st.write(
                    f"Adjusted R² = {adjusted_R2:.4f}"
                    if np.isfinite(adjusted_R2)
                    else "Adjusted R² is undefined."
                )

                st.info(
                    "R² measures the proportion of variation "
                    "in Y explained by the fitted model. "
                    "A high R² alone does not establish "
                    "causation or prove that the model is suitable."
                )


# =========================================================
# ONE-SAMPLE T-TEST
# =========================================================

elif module == "One-Sample t-Test":

    st.header("One-Sample t-Test")

    data = st.text_area(
        "Enter observations separated by commas",
        "48, 51, 49, 52, 47, 50, 53, 49"
    )

    mu0 = st.number_input(
        "Hypothesized population mean",
        value=50.0
    )

    alternative = st.selectbox(
        "Alternative hypothesis",
        [
            "Two-tailed",
            "Left-tailed",
            "Right-tailed"
        ]
    )

    alpha = st.selectbox(
        "Significance level α",
        [0.01, 0.05, 0.10],
        index=1
    )

    if st.button("Calculate one-sample t-test"):

        try:

            values = np.array(
                [
                    float(x.strip())
                    for x in data.split(",")
                    if x.strip()
                ]
            )

            n = len(values)

            if n < 2:
                st.error("Enter at least two observations.")

            else:

                mean = values.mean()
                sd = values.std(ddof=1)
                se = sd / np.sqrt(n)
                t_stat = (mean - mu0) / se
                df = n - 1

                st.subheader("Step 1: Calculate the mean")

                st.write(
                    f"ΣX = {values.sum():.4f}"
                )

                st.write(
                    f"x̄ = ΣX / n = "
                    f"{values.sum():.4f}/{n} "
                    f"= {mean:.4f}"
                )

                st.subheader("Step 2: Calculate the sample SD")

                st.write(
                    f"Sample standard deviation s = {sd:.4f}"
                )

                st.subheader("Step 3: Calculate standard error")

                st.latex(
                    r"SE=\frac{s}{\sqrt{n}}"
                )

                st.write(
                    f"SE = {sd:.4f}/√{n} = {se:.4f}"
                )

                st.subheader("Step 4: Calculate t")

                st.latex(
                    r"t=\frac{\bar{x}-\mu_0}{s/\sqrt{n}}"
                )

                st.write(
                    f"t = ({mean:.4f} − {mu0:.4f})/"
                    f"{se:.4f} = {t_stat:.4f}"
                )

                h1 = (
                    f"μ ≠ {mu0}"
                    if alternative == "Two-tailed"
                    else (
                        f"μ < {mu0}"
                        if alternative == "Left-tailed"
                        else f"μ > {mu0}"
                    )
                )

                show_t_test_result(
                    t_stat,
                    df,
                    alternative,
                    alpha,
                    f"H₀: μ = {mu0}",
                    f"H₁: {h1}"
                )

        except Exception as e:
            st.error(f"Check the entered data: {e}")


# =========================================================
# INDEPENDENT TWO-SAMPLE T-TEST
# =========================================================

elif module == "Independent Two-Sample t-Test":

    st.header("Independent Two-Sample t-Test")

    st.write(
        "Compare the means of two independent groups."
    )

    group1_text = st.text_area(
        "Group A observations",
        "18, 20, 17, 22, 19, 21"
    )

    group2_text = st.text_area(
        "Group B observations",
        "23, 25, 22, 24, 26, 23"
    )

    method = st.radio(
        "Method",
        [
            "Welch's t-test",
            "Pooled Student's t-test"
        ]
    )

    alternative = st.selectbox(
        "Alternative hypothesis",
        [
            "Two-tailed",
            "Left-tailed",
            "Right-tailed"
        ],
        key="ind_alt"
    )

    alpha = st.selectbox(
        "Significance level α",
        [0.01, 0.05, 0.10],
        index=1,
        key="ind_alpha"
    )

    if st.button("Calculate independent t-test"):

        try:

            a = np.array(
                [float(x.strip()) for x in group1_text.split(",")]
            )

            b = np.array(
                [float(x.strip()) for x in group2_text.split(",")]
            )

            n1, n2 = len(a), len(b)

            if n1 < 2 or n2 < 2:

                st.error(
                    "Enter at least two observations in each group."
                )

            else:

                m1, m2 = a.mean(), b.mean()
                s1, s2 = a.std(ddof=1), b.std(ddof=1)
                v1, v2 = s1**2, s2**2

                st.subheader("Step 1: Group summaries")

                display_table(
                    pd.DataFrame({
                        "Group A": [n1, m1, s1, v1],
                        "Group B": [n2, m2, s2, v2]
                    }, index=[
                        "Sample size",
                        "Mean",
                        "Sample SD",
                        "Sample variance"
                    ]).round(4)
                )

                difference = m1 - m2

                st.write(
                    f"Difference between means = "
                    f"{m1:.4f} − {m2:.4f} "
                    f"= {difference:.4f}"
                )

                if method == "Welch's t-test":

                    se = np.sqrt(
                        v1 / n1 + v2 / n2
                    )

                    df = (
                        (v1/n1 + v2/n2)**2
                        /
                        (
                            (v1/n1)**2/(n1-1)
                            +
                            (v2/n2)**2/(n2-1)
                        )
                    )

                    st.subheader(
                        "Step 2: Calculate the standard error"
                    )

                    st.latex(
                        r"SE=\sqrt{\frac{s_1^2}{n_1}"
                        r"+\frac{s_2^2}{n_2}}"
                    )

                    st.write(
                        f"SE = √({v1:.4f}/{n1} + "
                        f"{v2:.4f}/{n2}) = {se:.4f}"
                    )

                else:

                    df = n1 + n2 - 2

                    pooled_variance = (
                        (n1 - 1) * v1
                        +
                        (n2 - 1) * v2
                    ) / df

                    se = np.sqrt(
                        pooled_variance
                        *
                        (1/n1 + 1/n2)
                    )

                    st.subheader(
                        "Step 2: Calculate pooled variance"
                    )

                    st.latex(
                        r"s_p^2="
                        r"\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}"
                        r"{n_1+n_2-2}"
                    )

                    st.write(
                        f"Pooled variance = "
                        f"{pooled_variance:.4f}"
                    )

                    st.write(
                        f"SE = sₚ√(1/n₁ + 1/n₂) "
                        f"= {se:.4f}"
                    )

                st.subheader("Step 3: Calculate t-statistic")

                t_stat = difference / se

                st.latex(
                    r"t=\frac{\bar{x}_1-\bar{x}_2}{SE}"
                )

                st.write(
                    f"t = {difference:.4f}/{se:.4f} "
                    f"= {t_stat:.4f}"
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

        except Exception as e:
            st.error(f"Check the entered observations: {e}")


# =========================================================
# PAIRED T-TEST
# =========================================================

elif module == "Paired t-Test":

    st.header("Paired t-Test")

    st.write(
        "Use paired observations, such as marks before and "
        "after training for the same students."
    )

    before_text = st.text_area(
        "Before observations",
        "60, 65, 58, 72, 68, 63, 70"
    )

    after_text = st.text_area(
        "After observations",
        "66, 67, 61, 75, 73, 66, 74"
    )

    direction = st.radio(
        "Define the difference",
        [
            "After − Before",
            "Before − After"
        ]
    )

    alternative = st.selectbox(
        "Alternative hypothesis",
        [
            "Two-tailed",
            "Left-tailed",
            "Right-tailed"
        ],
        key="paired_alt"
    )

    alpha = st.selectbox(
        "Significance level α",
        [0.01, 0.05, 0.10],
        index=1,
        key="paired_alpha"
    )

    if st.button("Calculate paired t-test"):

        try:

            before = np.array(
                [
                    float(x.strip())
                    for x in before_text.split(",")
                ]
            )

            after = np.array(
                [
                    float(x.strip())
                    for x in after_text.split(",")
                ]
            )

            if len(before) != len(after):

                st.error(
                    "Before and after must contain the same "
                    "number of observations."
                )

            elif len(before) < 2:

                st.error(
                    "Enter at least two complete pairs."
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
                t_stat = mean_d / se
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
