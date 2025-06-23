import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter, statistics
from textblob import TextBlob
import os
import matplotlib.pyplot as plt
import seaborn as sns

from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# 🔹 Title
st.title("📊 BioStatView:- Insights in Oncology")

# 🔹 File uploader
uploaded_file = st.file_uploader("📁 Upload your dataset (Structured or Unstructured)", type=['csv', 'xlsx', 'xls', 'txt', 'json'])

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()

    try:
        # 🔹 File Handling
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith('.json'):
            df = pd.read_json(uploaded_file)

        if 'df' in locals() and isinstance(df, pd.DataFrame) and df.shape[1] > 1:
            st.subheader("📊 Uploaded Structured Data Preview")
            st.dataframe(df.head())
            st.success("✅ Structured data detected. Choose analysis from sidebar.")

            # 🔹 Sidebar analysis selector
            analysis_type = st.sidebar.selectbox("📌 Select analysis", [
                "Descriptive Stats",
                "Kaplan-Meier + Log-Rank Test",
                "Cox Proportional Hazards"
            ])
            show_charts = st.sidebar.checkbox("📊 Show Charts for Numerical Data")

            # 🔸 Descriptive Stats
            if analysis_type == "Descriptive Stats":
                st.subheader("📌 Descriptive Statistics")

                # Categorical Summary
                cat_summary = []
                cat_cols = df.select_dtypes(include='object').columns
                for col in cat_cols:
                    total = df[col].shape[0]
                    missing = df[col].isnull().sum()
                    cat_summary.append({
                        'Variable': col,
                        'Unique Values': df[col].nunique(),
                        'Missing': missing,
                        'Total': total,
                        'Outliers': 'N/A'
                    })
                if cat_summary:
                    cat_df = pd.DataFrame(cat_summary)
                    st.markdown("### 📋 Categorical Summary Table")
                    st.dataframe(cat_df)

                # Numerical Summary
                num_summary = []
                num_cols = df.select_dtypes(include=np.number).columns
                if len(num_cols) == 0:
                    st.warning("⚠️ No numeric variables found in the dataset.")
                else:
                    for col in num_cols:
                        stats = df[col].describe()
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))].shape[0]
                        num_summary.append({
                            'Variable': col,
                            'Mean': round(stats['mean'], 2),
                            'Median': round(df[col].median(), 2),
                            'Std': round(stats['std'], 2),
                            'Min': round(stats['min'], 2),
                            'Max': round(stats['max'], 2),
                            'Outliers': outliers
                        })
                    num_df = pd.DataFrame(num_summary)
                    st.markdown("### 📈 Numerical Summary Table")
                    st.dataframe(num_df)

                    if show_charts:
                        st.markdown("### 🌐 Numeric Variable Distributions")
                        for col in num_cols:
                            fig, ax = plt.subplots()
                            sns.histplot(df[col].dropna(), kde=True, ax=ax)
                            ax.set_title(f"Distribution of {col}")
                            st.pyplot(fig)

            # 🔸 Kaplan-Meier + Log-Rank
            elif analysis_type == "Kaplan-Meier + Log-Rank Test":
                st.subheader("📈 Kaplan-Meier Survival Curve + Log-Rank Test")
                duration_col = st.selectbox("Select Duration Column", df.columns)
                event_col = st.selectbox("Select Event Column (1=event, 0=censored)", df.columns)
                group_col = st.selectbox("Select Grouping Column (optional)", ["None"] + list(df.columns))

                kmf = KaplanMeierFitter()

                if group_col == "None":
                    kmf.fit(df[duration_col], event_observed=df[event_col])
                    st.pyplot(kmf.plot_survival_function())
                else:
                    for group in df[group_col].dropna().unique():
                        kmf.fit(
                            df[df[group_col] == group][duration_col],
                            event_observed=df[df[group_col] == group][event_col],
                            label=str(group)
                        )
                        kmf.plot_survival_function()
                    st.pyplot()

                    unique_groups = df[group_col].dropna().unique()
                    if len(unique_groups) == 2:
                        group1 = df[df[group_col] == unique_groups[0]]
                        group2 = df[df[group_col] == unique_groups[1]]
                        result = statistics.logrank_test(
                            group1[duration_col], group2[duration_col],
                            event_observed_A=group1[event_col],
                            event_observed_B=group2[event_col]
                        )
                        st.write(f"**Log-rank p-value:** {result.p_value:.4f}")

            # 🔸 Cox Proportional Hazards
            elif analysis_type == "Cox Proportional Hazards":
                st.subheader("📌 Cox Proportional Hazards Model")
                duration_col = st.selectbox("Select Duration Column", df.columns)
                event_col = st.selectbox("Select Event Column", df.columns)
                covariates = st.multiselect("Select Covariates (Independent Variables)", [col for col in df.columns if col not in [duration_col, event_col]])

                if covariates:
                    cph = CoxPHFitter()
                    survival_df = df[[duration_col, event_col] + covariates].dropna()
                    cph.fit(survival_df, duration_col=duration_col, event_col=event_col)
                    st.write(cph.summary)
                    st.pyplot(cph.plot())

            # 🔸 AI Q&A
            st.subheader("💬 Ask AI a Question About Your Data")
            user_question = st.text_input("Type your question below:")

            if user_question:
                try:
                    os.environ["OPENAI_API_KEY"] = "sk-REPLACE-YOUR-KEY"
                    agent = create_pandas_dataframe_agent(
                        ChatOpenAI(temperature=0),
                        df,
                        verbose=False,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        handle_parsing_errors=True,
                        allow_dangerous_code=True
                    )
                    with st.spinner("Thinking..."):
                        response = agent.run(user_question)
                        st.success(response)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        elif file_name.endswith('.txt'):
            text_data = uploaded_file.read().decode("utf-8")
            st.subheader("📄 Uploaded Text Preview")
            st.text_area("Raw Text", text_data, height=300)
            st.success("✅ Unstructured text detected. Choose NLP task from sidebar.")

            text_option = st.sidebar.selectbox("🧠 Select NLP Task", ["Text Summarization", "Sentiment Analysis"])

            if text_option == "Text Summarization":
                st.subheader("📌 Summary")
                summary = ' '.join(text_data.split()[:100]) + "..."
                st.write(summary)

            elif text_option == "Sentiment Analysis":
                st.subheader("📌 Sentiment")
                blob = TextBlob(text_data)
                st.write(f"Polarity: {blob.sentiment.polarity:.2f}")
                st.write(f"Subjectivity: {blob.sentiment.subjectivity:.2f}")

        else:
            st.warning("⚠️ File format supported but not detected as structured or unstructured.")

    except Exception as e:
        st.error(f"❌ Error loading or processing file: {e}")

else:
    st.info("👈 Upload a dataset to begin.")
