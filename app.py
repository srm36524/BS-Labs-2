import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

st.set_page_config(page_title="Regression & t-Test Lab", page_icon="📊", layout="wide")
st.title("📊 Multiple Regression & t-Test Lab")
st.caption("Business Statistics • Step-by-step calculations")

with st.sidebar:
    st.header("Choose a module")
    module = st.radio("Analysis", [
        "Multiple Linear Regression",
        "One-Sample t-Test",
        "Independent Two-Sample t-Test",
        "Paired t-Test"
    ])
    st.divider()
    st.info("Enter your own observations or start with the example data. Results are calculated from the data currently shown.")

def data_editor_with_csv(key, default_df, help_text):
    st.write(help_text)
    uploaded = st.file_uploader("Optional: upload a CSV file", type=["csv"], key=f"{key}_upload")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("CSV loaded. Check column names and values below.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = default_df.copy()
    else:
        df = default_df.copy()
    return st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"{key}_editor")

def clean_numeric(df, cols):
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna().reset_index(drop=True)

def fmt(x, digits=4):
    try:
        if not np.isfinite(float(x)):
            return "Not defined"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

def tail_test(statistic, df, alternative):
    if alternative == "Two-tailed":
        p = 2 * stats.t.sf(abs(statistic), df)
        crit = stats.t.ppf(0.975, df)
        decision_rule = f"Reject H₀ if |t| > {fmt(crit)} or p < α."
        reject = p < alpha
        critical_text = f"±{fmt(crit)}"
    elif alternative == "Right-tailed":
        p = stats.t.sf(statistic, df)
        crit = stats.t.ppf(1-alpha, df)
        decision_rule = f"Reject H₀ if t > {fmt(crit)} or p < α."
        reject = p < alpha
        critical_text = fmt(crit)
    else:
        p = stats.t.cdf(statistic, df)
        crit = stats.t.ppf(alpha, df)
        decision_rule = f"Reject H₀ if t < {fmt(crit)} or p < α."
        reject = p < alpha
        critical_text = fmt(crit)
    return p, critical_text, decision_rule, reject

def show_test_result(t_stat, df, alternative, h0, h1):
    p, critical, rule, reject = tail_test(t_stat, df, alternative)
    st.subheader("Step-by-step hypothesis test")
    st.markdown(f"**Step 1 — Null hypothesis:** {h0}")
    st.markdown(f"**Step 2 — Alternative hypothesis ({alternative.lower()}):** {h1}")
    st.markdown(f"**Step 3 — Significance level:** α = {alpha:.3f}")
    st.markdown(f"**Step 4 — Test statistic:** t = {fmt(t_stat)}")
    st.markdown(f"**Step 5 — Degrees of freedom:** df = {fmt(df, 0)}")
    st.markdown(f"**Step 6 — Critical value:** {critical}")
    st.markdown(f"**Step 7 — p-value:** {fmt(p, 6)}")
    st.markdown(f"**Step 8 — Decision rule:** {rule}")
    if reject:
        st.error("Decision: Reject H₀. The result is statistically significant at the selected α level.")
    else:
        st.success("Decision: Fail to reject H₀. The evidence is not sufficient to reject the null hypothesis.")
    st.caption("A non-significant result does not prove that H₀ is true.")

if module == "Multiple Linear Regression":
    st.header("Multiple Linear Regression")
    st.write("Predict one dependent variable Y from either two or three independent variables.")
    n_predictors = st.radio("Number of independent variables", [2, 3], horizontal=True)
    if n_predictors == 2:
        default = pd.DataFrame({
            "Y_Sales": [12, 15, 17, 20, 22, 25, 27, 30],
            "X1_Advertising": [1, 2, 2, 3, 4, 4, 5, 6],
            "X2_Salespeople": [2, 2, 3, 3, 4, 5, 5, 6]
        })
        ycol, xcols = "Y_Sales", ["X1_Advertising", "X2_Salespeople"]
    else:
        default = pd.DataFrame({
            "Y_Sales": [15, 18, 20, 23, 25, 28, 31, 34, 36],
            "X1_Advertising": [1, 2, 2, 3, 4, 4, 5, 6, 7],
            "X2_Salespeople": [2, 2, 3, 3, 4, 5, 5, 6, 7],
            "X3_Discount": [1, 1, 2, 2, 3, 3, 4, 4, 5]
        })
        ycol, xcols = "Y_Sales", ["X1_Advertising", "X2_Salespeople", "X3_Discount"]
    edited = data_editor_with_csv("reg", default, "Edit the table directly. The first column is Y; the remaining columns are X variables. CSV uploads should use the same column names.")
    allcols = [ycol] + xcols
    if all(c in edited.columns for c in allcols):
        df = clean_numeric(edited, allcols)
    else:
        df = pd.DataFrame(columns=allcols)
    if st.button("Calculate regression", type="primary"):
        if len(df) <= len(xcols) + 1:
            st.error(f"Enter at least {len(xcols)+2} complete observations so there are positive residual degrees of freedom.")
        else:
            y = df[ycol].astype(float)
            X = df[xcols].astype(float)
            Xc = sm.add_constant(X, has_constant="add")
            try:
                model = sm.OLS(y, Xc).fit()
                n, k = len(df), len(xcols)
                st.subheader("Step 1 — Identify variables")
                st.write(f"Dependent variable: **{ycol}**")
                st.write(f"Independent variables: **{', '.join(xcols)}**")
                st.write(f"Number of observations n = {n}; predictors k = {k}.")
                st.subheader("Step 2 — Calculate means")
                means = df[allcols].mean().to_frame("Mean")
                st.dataframe(means.style.format("{:.4f}"), use_container_width=True)
                st.subheader("Step 3 — Set up the regression equation")
                equation = f"Ŷ = {model.params['const']:.4f}"
                for x in xcols:
                    coef = model.params[x]
                    equation += f" {'+' if coef >= 0 else '−'} {abs(coef):.4f}({x})"
                st.latex(equation.replace("Ŷ", r"\hat{Y}").replace("X1_Advertising", "X_1").replace("X2_Salespeople", "X_2").replace("X3_Discount", "X_3"))
                st.write("The coefficients are estimated by the least-squares method: β̂ = (X′X)⁻¹X′Y. The intercept column of 1s is included in X.")
                with st.expander("Show the design matrix X and response vector Y"):
                    st.write("Design matrix X (first column contains 1s for the intercept):")
                    st.dataframe(Xc, use_container_width=True)
                    st.write("Response vector Y:")
                    st.dataframe(y.to_frame(), use_container_width=True)
                    st.write("X′X:")
                    st.dataframe(pd.DataFrame(Xc.to_numpy().T @ Xc.to_numpy(), index=Xc.columns, columns=Xc.columns), use_container_width=True)
                    st.write("X′Y:")
                    st.dataframe(pd.DataFrame(Xc.to_numpy().T @ y.to_numpy(), index=Xc.columns, columns=["X′Y"]), use_container_width=True)
                st.subheader("Step 4 — Coefficients and interpretation")
                coef_table = pd.DataFrame({
                    "Coefficient": model.params,
                    "Standard error": model.bse,
                    "t statistic": model.tvalues,
                    "p-value": model.pvalues,
                    "95% CI lower": model.conf_int()[0],
                    "95% CI upper": model.conf_int()[1]
                })
                st.dataframe(coef_table.style.format("{:.5f}"), use_container_width=True)
                st.write("Interpretation: holding the other X variables constant, a one-unit increase in a predictor changes the predicted Y by its coefficient.")
                st.subheader("Step 5 — Calculate fitted values and residuals")
                result = df.copy()
                result["Predicted_Y"] = model.fittedvalues
                result["Residual (Y - Ŷ)"] = model.resid
                result["Squared residual"] = model.resid ** 2
                st.dataframe(result.style.format("{:.4f}"), use_container_width=True)
                st.write(f"Sum of squared errors (SSE) = Σ(Y − Ŷ)² = **{fmt(model.ssr)}**")
                st.write(f"Total sum of squares (SST) = Σ(Y − Ȳ)² = **{fmt(model.centered_tss)}**")
                st.write(f"Regression sum of squares (SSR) = SST − SSE = **{fmt(model.ess)}**")
                st.subheader("Step 6 — Goodness of fit")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R²", fmt(model.rsquared))
                c2.metric("Adjusted R²", fmt(model.rsquared_adj))
                c3.metric("Residual standard error", fmt(np.sqrt(model.mse_resid)))
                c4.metric("F statistic", fmt(model.fvalue))
                st.write(f"R² = 1 − SSE/SST = 1 − {fmt(model.ssr)}/{fmt(model.centered_tss)} = **{fmt(model.rsquared)}**.")
                st.write(f"Adjusted R² = **{fmt(model.rsquared_adj)}**; overall F-test p-value = **{fmt(model.f_pvalue, 6)}**.")
                st.subheader("Step 7 — ANOVA table for regression")
                anova = pd.DataFrame({
                    "Source": ["Regression", "Residual/Error", "Total"],
                    "Sum of squares": [model.ess, model.ssr, model.centered_tss],
                    "df": [k, model.df_resid, n-1],
                    "Mean square": [model.ess/k, model.mse_resid, np.nan],
                })
                anova["F"] = [model.fvalue, np.nan, np.nan]
                anova["p-value"] = [model.f_pvalue, np.nan, np.nan]
                st.dataframe(anova.style.format({"Sum of squares":"{:.4f}", "df":"{:.0f}", "Mean square":"{:.4f}", "F":"{:.4f}", "p-value":"{:.6f}"}), use_container_width=True)
                st.subheader("Step 8 — t-tests for individual regression coefficients")
                alpha_reg = st.select_slider("Significance level for coefficient tests", options=[0.01, 0.05, 0.10], value=0.05)
                st.write("For each predictor: H₀: βᵢ = 0; H₁: βᵢ ≠ 0. Test statistic t = estimated coefficient / standard error.")
                crit = stats.t.ppf(1-alpha_reg/2, model.df_resid)
                tests = []
                for name in xcols:
                    tests.append({
                        "Predictor": name,
                        "Coefficient": model.params[name],
                        "SE": model.bse[name],
                        "t = b/SE": model.tvalues[name],
                        "df": model.df_resid,
                        "Critical ±t": crit,
                        "p-value": model.pvalues[name],
                        "Decision": "Reject H₀" if model.pvalues[name] < alpha_reg else "Fail to reject H₀"
                    })
                st.dataframe(pd.DataFrame(tests).style.format({"Coefficient":"{:.4f}","SE":"{:.4f}","t = b/SE":"{:.4f}","df":"{:.0f}","Critical ±t":"{:.4f}","p-value":"{:.6f}"}), use_container_width=True)
                st.caption("Regression inference assumes a correctly specified linear model; conventional standard errors also rely on assumptions such as independent errors and constant error variance.")
            except Exception as e:
                st.error(f"Could not calculate regression. Check for constant/duplicate predictors or insufficient variation. Details: {e}")

elif module == "One-Sample t-Test":
    st.header("One-Sample t-Test")
    st.write("Tests whether the sample mean differs from a hypothesized population mean.")
    default = pd.DataFrame({"Observation": [48, 51, 49, 52, 47, 50, 53, 49]})
    df0 = data_editor_with_csv("one", default, "Enter one numeric observation per row.")
    values = pd.to_numeric(df0.get("Observation", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    mu0 = st.number_input("Hypothesized population mean (μ₀)", value=50.0)
    alternative = st.selectbox("Alternative hypothesis direction", ["Two-tailed", "Left-tailed", "Right-tailed"])
    alpha = st.select_slider("Significance level α", options=[0.01, 0.05, 0.10], value=0.05, key="one_alpha")
    if st.button("Calculate one-sample t-test", type="primary"):
        if len(values) < 2:
            st.error("Enter at least two valid observations.")
        else:
            n = len(values); mean = values.mean(); sd = values.std(ddof=1); se = sd / np.sqrt(n)
            t_stat = (mean - mu0) / se
            st.subheader("Step 1 — Descriptive statistics")
            st.dataframe(pd.DataFrame({"Value":[n, mean, sd, se]}, index=["n","Sample mean (x̄)","Sample SD (s)","Standard error (s/√n)"]), use_container_width=True)
            st.subheader("Step 2 — Formula and substitution")
            st.latex(r"t=\frac{\bar{x}-\mu_0}{s/\sqrt{n}}")
            st.latex(r"t=\frac{" + f"{mean:.4f}-{mu0:.4f}" + r"}{" + f"{sd:.4f}/\\sqrt{{{n}}}" + r"}=" + f"{t_stat:.4f}")
            h1 = "μ ≠ μ₀" if alternative == "Two-tailed" else ("μ < μ₀" if alternative == "Left-tailed" else "μ > μ₀")
            show_test_result(t_stat, n-1, alternative, f"H₀: μ = {mu0:g}", f"H₁: {h1.replace('μ₀', str(mu0))}")

elif module == "Independent Two-Sample t-Test":
    st.header("Independent Two-Sample t-Test")
    st.write("Compare the means of two independent groups.")
    default = pd.DataFrame({"Group_A": [18, 20, 17, 22, 19, 21], "Group_B": [23, 25, 22, 24, 26, 23]})
    df0 = data_editor_with_csv("ind", default, "Each column is a separate group. Leave missing values blank.")
    if "Group_A" in df0 and "Group_B" in df0:
        a = pd.to_numeric(df0["Group_A"], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(df0["Group_B"], errors="coerce").dropna().to_numpy(dtype=float)
    else:
        a, b = np.array([]), np.array([])
    method = st.radio("Test type", ["Welch's t-test (does not assume equal variances)", "Student's pooled t-test (assumes equal variances)"])
    alternative = st.selectbox("Alternative hypothesis direction", ["Two-tailed", "Left-tailed", "Right-tailed"])
    alpha = st.select_slider("Significance level α", options=[0.01, 0.05, 0.10], value=0.05, key="ind_alpha")
    if st.button("Calculate independent t-test", type="primary"):
        if len(a) < 2 or len(b) < 2:
            st.error("Each group needs at least two valid observations.")
        else:
            n1, n2 = len(a), len(b); m1, m2 = a.mean(), b.mean()
            s1, s2 = a.std(ddof=1), b.std(ddof=1)
            v1, v2 = s1**2, s2**2
            if method.startswith("Welch"):
                se = np.sqrt(v1/n1 + v2/n2)
                df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
                t_stat = (m1-m2)/se
                formula = "t=\\frac{\\bar{x}_1-\\bar{x}_2}{\\sqrt{s_1^2/n_1+s_2^2/n_2}}"
            else:
                df = n1+n2-2
                sp2 = ((n1-1)*v1 + (n2-1)*v2)/df
                se = np.sqrt(sp2*(1/n1+1/n2))
                t_stat = (m1-m2)/se
                st.write(f"Pooled variance: sₚ² = [({n1}−1)({v1:.4f}) + ({n2}−1)({v2:.4f})] / ({n1}+{n2}−2) = **{sp2:.4f}**.")
                formula = "t=\\frac{\\bar{x}_1-\\bar{x}_2}{s_p\\sqrt{1/n_1+1/n_2}}"
            st.subheader("Step 1 — Group summaries")
            st.dataframe(pd.DataFrame({"Group A":[n1,m1,s1,v1],"Group B":[n2,m2,s2,v2]}, index=["n","Mean","Sample SD","Sample variance"]), use_container_width=True)
            st.subheader("Step 2 — Standard error and t statistic")
            st.latex(formula)
            st.write(f"Difference in sample means = {m1:.4f} − {m2:.4f} = **{m1-m2:.4f}**.")
            st.write(f"Standard error = **{se:.4f}**; t = difference / SE = **{t_stat:.4f}**.")
            h1 = "μ₁ ≠ μ₂" if alternative == "Two-tailed" else ("μ₁ < μ₂" if alternative == "Left-tailed" else "μ₁ > μ₂")
            show_test_result(t_stat, df, alternative, "H₀: μ₁ = μ₂", f"H₁: {h1}")

elif module == "Paired t-Test":
    st.header("Paired t-Test")
    st.write("Use this when the same people/items are measured twice, such as before and after training.")
    default = pd.DataFrame({"Before": [60, 65, 58, 72, 68, 63, 70], "After": [66, 67, 61, 75, 73, 66, 74]})
    df0 = data_editor_with_csv("paired", default, "Each row must represent the same person/item measured twice.")
    if "Before" in df0 and "After" in df0:
        pairs = clean_numeric(df0, ["Before", "After"])
        before, after = pairs["Before"].to_numpy(dtype=float), pairs["After"].to_numpy(dtype=float)
    else:
        before, after = np.array([]), np.array([])
    alternative = st.selectbox("Alternative hypothesis direction", ["Two-tailed", "Left-tailed", "Right-tailed"], key="paired_alt")
    direction = st.radio("Define the difference as", ["After − Before", "Before − After"], horizontal=True)
    alpha = st.select_slider("Significance level α", options=[0.01, 0.05, 0.10], value=0.05, key="paired_alpha")
    if st.button("Calculate paired t-test", type="primary"):
        if len(before) < 2:
            st.error("Enter at least two complete pairs.")
        else:
            d = after-before if direction == "After − Before" else before-after
            n = len(d); md = d.mean(); sd = d.std(ddof=1); se = sd/np.sqrt(n); t_stat = md/se
            st.subheader("Step 1 — Calculate paired differences")
            pair_table = pairs.copy()
            pair_table["Difference"] = d
            st.dataframe(pair_table, use_container_width=True)
            st.subheader("Step 2 — Summarize differences")
            st.dataframe(pd.DataFrame({"Value":[n,md,sd,se]}, index=["Number of pairs (n)","Mean difference (d̄)","SD of differences (s_d)","Standard error (s_d/√n)"]), use_container_width=True)
            st.subheader("Step 3 — Substitute into formula")
            st.latex(r"t=\frac{\bar{d}-0}{s_d/\sqrt{n}}")
            st.write(f"t = ({md:.4f} − 0) / ({sd:.4f}/√{n}) = **{t_stat:.4f}**.")
            h1 = "μd ≠ 0" if alternative == "Two-tailed" else ("μd < 0" if alternative == "Left-tailed" else "μd > 0")
            show_test_result(t_stat, n-1, alternative, "H₀: μd = 0", f"H₁: {h1}")
