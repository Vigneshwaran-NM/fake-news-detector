import streamlit as st
import pandas as pd
import altair as alt
import os

st.set_page_config(page_title="📊 Analytics Dashboard", layout="wide")
st.title("📈 Prediction History & Analytics")

history_path = "data/prediction_history.csv"

# Load history
if not os.path.exists(history_path):
    st.warning("No prediction history found. Make a prediction first.")
else:
    df = pd.read_csv(history_path)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sidebar filters
    st.sidebar.header("🔍 Filter")
    label_filter = st.sidebar.selectbox("Select Prediction Label", options=["All", "FAKE NEWS", "REAL NEWS"])
    if label_filter != "All":
        df = df[df['prediction'] == label_filter]

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Total Records:** {len(df)}")

    # Count chart
    st.subheader("🧮 Fake vs Real News Count")
    count_data = df['prediction'].value_counts().reset_index()
    count_data.columns = ['Prediction', 'Count']

    count_chart = alt.Chart(count_data).mark_bar().encode(
        x=alt.X('Prediction', sort=['FAKE NEWS', 'REAL NEWS']),
        y='Count',
        color='Prediction'
    ).properties(width=500)

    st.altair_chart(count_chart)

    # Confidence trend
    st.subheader("📊 Confidence Score Over Time")
    line_chart = alt.Chart(df).mark_line(point=True).encode(
        x='timestamp:T',
        y='confidence:Q',
        color='prediction:N',
        tooltip=['timestamp:T', 'prediction', 'confidence']
    ).interactive().properties(height=400)

    st.altair_chart(line_chart, use_container_width=True)

    # Raw data (optional)
    with st.expander("📄 Show Prediction Table"):
        st.dataframe(df[::-1], use_container_width=True)
