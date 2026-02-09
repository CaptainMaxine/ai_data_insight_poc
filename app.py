import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI Data Insight Assistant", layout="wide")

st.title("AI Data Insight Assistant (Enhanced Prototype)")
st.write("A concept demo showing how PMs could get low-friction, automated insights — including statistical profiling, risk detection, and AI-style summaries (mocked).")

# -----------------------
# 1. UPLOAD
# -----------------------
st.header("1. Upload your CSV")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("File uploaded successfully!")

    # -----------------------
    # 2. Summary
    # -----------------------
    st.header("2. Automatic Summary Statistics")
    st.write(df.describe(include="all"))

    # -----------------------
    # 3. Data Quality Check
    # -----------------------
    st.header("3. Data Quality Check")
    quality = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_rate": df.isnull().mean(),
        "unique_count": df.nunique()
    })
    st.write(quality)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    insights = []
    risks = []
    transformations = []
    correlations = []

    # -----------------------
    # 4. Statistical Insights
    # -----------------------
    st.header("4. Statistical Insight Engine (Rule-Based)")

    for col in numeric_cols:
        col_data = df[col].dropna()

        # a) Skew detection
        if col_data.mean() > col_data.median():
            insights.append(f"- **{col}** is right-skewed → consider log transform.")
            transformations.append(f"Log transform may stabilize {col} distribution.")

        # b) Outlier detection (IQR)
        q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        iqr = q3 - q1
        outliers = col_data[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]

        if len(outliers) > 0:
            risks.append(f"- **{col}** has {len(outliers)} potential outliers → may distort metrics.")

        # c) Zero / low variance
        if col_data.nunique() < 3:
            risks.append(f"- **{col}** has extremely low variance → limited signal strength.")

        # d) Peak detection (kurtosis)
        if col_data.kurtosis() > 3:
            insights.append(f"- **{col}** shows high kurtosis → heavy tails detected.")

    # -----------------------
    # Correlation Scan
    # -----------------------
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        pairs = high_corr[high_corr > 0.8].stack()

        if len(pairs) > 0:
            st.subheader("Strong Correlation Signals")
            for (c1, c2), val in pairs.items():
                correlations.append(f"- **{c1} ↔ {c2}** correlation = {val:.2f}")
        else:
            correlations.append("- No strong correlations detected.")

    # Display all sections
    st.subheader("Statistical Insights")
    st.write("\n".join(insights) if insights else "No significant skewness or patterns detected.")

    st.subheader("Potential Risks")
    st.write("\n".join(risks) if risks else "No major risks detected.")

    st.subheader("Suggested Transformations")
    st.write("\n".join(transformations) if transformations else "No transformations recommended.")

    st.subheader("Correlation Summary")
    st.write("\n".join(correlations))

    # -----------------------
    # 5. AI-STYLE SUMMARY (Mock)
    # -----------------------
    st.header("5. AI-Style Insight Summary (Mock Output)")
    mock_ai_summary = f"""
### Executive Summary
The uploaded dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.  
Initial automated analysis suggests several meaningful signals:

- Distribution patterns indicate potential skew in multiple fields.
- Outlier activity may distort trend interpretation in some numeric columns.
- Correlation structure remains stable with no strong dependencies, suggesting low multicollinearity.
- Data quality is high, with minimal missingness detected.

### Recommended PM Actions
- Validate outlier impact on downstream metrics.
- Apply recommended transformations where needed.
- Review distribution irregularities before modeling.
- Integrate insights into roadmap & launch readiness discussions.

*(This section simulates what an internal AI assistant could summarize.)*
"""
    st.markdown(mock_ai_summary)

    # -----------------------
    # 6. PM Summary Report
    # -----------------------
    st.header("6. PM Summary Report (Auto-Generated Prototype)")
    st.write("""
**Top Issues Identified**
- Potential skew in key numeric columns  
- Outliers requiring validation  
- Low variance features with limited predictive value  

**Recommended Fixes**
- Apply log transform to normalize heavy-tailed distributions  
- Investigate and clean outliers  
- Remove or bucket low-signal columns  

**Implications for PM**
- Cleaner data → more confident insights  
- Standardized analysis → faster alignment with engineering  
- Reduced manual prep → more time for decision-making  

**Feature Risk Summary**
- No major structural risks detected  
- Minor anomalies flagged for review  
    """)

else:
    st.info("Upload a sample CSV to begin.")
