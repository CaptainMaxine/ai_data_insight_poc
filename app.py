import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Data Insight Assistant", layout="wide")

# Header
st.title("AI Data Insight Assistant (Prototype)")
st.write("A fast demo showing how a PM could get low-friction data insights using automated processing.")

# Step 1: Upload CSV
st.header("1. Upload your CSV")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("File uploaded successfully!")

    # Step 2: Summary
    st.header("2. Automatic Summary Statistics")
    st.write(df.describe(include="all"))

    # Step 3: Data Quality Check
    st.header("3. Data Quality Check")
    quality = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_rate": df.isnull().mean(),
        "unique_count": df.nunique()
    })
    st.write(quality)

    # Step 4: Insight Generator (simple rule-based mock)
    st.header("4. Auto-Generated Insights (Prototype)")
    insights = []

    for col in df.columns:
        if df[col].dtype != "object":
            if df[col].isnull().mean() > 0.3:
                insights.append(f"- Column '{col}' has high missing rate → consider imputation or removal.")
            if df[col].nunique() < 5:
                insights.append(f"- Column '{col}' has low variance → may not contribute meaningful signal.")
            if df[col].mean() > df[col].median():
                insights.append(f"- Column '{col}' is right-skewed → consider log transform.")

    if not insights:
        insights.append("Data is clean and shows no immediate issues.")

    for line in insights:
        st.write(line)

    # Step 5: Recommended PM Actions
    st.header("5. Recommended PM Actions")
    st.write("""
    - Provide PMs with automated summaries to reduce manual exploration time.
    - Flag data-quality issues early to enable faster collaboration with engineering.
    - Auto-generate insights to support prioritization, roadmap conversations, and launch evaluations.
    """)

else:
    st.info("Upload a sample CSV to begin.")
