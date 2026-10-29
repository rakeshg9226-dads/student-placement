import pandas as pd
import plotly.express as px
import streamlit as st


def show_charts(df):
    """
    Display interactive charts for Student Placement Prediction.
    Expected target column: Placement Status
    """

    if df is None or df.empty:
        st.warning("No dataset available to display charts.")
        return

    # Find the placement status column
    target_col = next(
        (
            col for col in df.columns
            if col.strip().lower() == "placement status"
        ),
        None
    )

    if target_col is None:
        st.error(
            "The dataset must contain a 'Placement Status' column."
        )
        st.info(
          "check the column names in student_placement.csv."
        )
        return

    st.subheader("📊 Student Placement Analytics")

    # Clean placement status values
    chart_df = df.copy()
    chart_df[target_col] = (
        chart_df[target_col]
        .astype("string")
        .str.strip()
    )

    chart_df = chart_df.dropna(subset=[target_col])
    chart_df = chart_df[
        chart_df[target_col] != ""
    ]

    if chart_df.empty:
        st.warning("No valid placement status data found.")
        return

    # Calculate placement statistics
    status_counts = (
        chart_df[target_col]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "Placement Status",
        "Number of Students"
    ]

    total_students = len(chart_df)
    placed_students = chart_df[
        chart_df[target_col].str.lower().isin(
            ["placed", "yes", "1", "true"]
        )
    ].shape[0]

    placement_rate = (
        placed_students / total_students * 100
        if total_students > 0 else 0
    )

    # KPI cards
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Students",
        f"{total_students:,}"
    )

    col2.metric(
        "Placed Students",
        f"{placed_students:,}"
    )

    col3.metric(
        "Placement Rate",
        f"{placement_rate:.1f}%"
    )

    st.divider()

    # Placement status charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Placement Distribution")

        fig_pie = px.pie(
            status_counts,
            names="Placement Status",
            values="Number of Students",
            hole=0.45,
            title="Placed vs Not Placed"
        )

        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        fig_pie.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with col2:
        st.markdown("#### Student Count by Status")

        fig_bar = px.bar(
            status_counts,
            x="Placement Status",
            y="Number of Students",
            text="Number of Students",
            title="Placement Status Comparison"
        )

        fig_bar.update_traces(
            textposition="outside"
        )

        fig_bar.update_layout(
            height=400,
            xaxis_title="Placement Status",
            yaxis_title="Number of Students",
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    # CGPA analysis, if the column exists
    cgpa_col = next(
        (
            col for col in chart_df.columns
            if col.strip().lower() == "cgpa"
        ),
        None
    )

    if cgpa_col is not None:
        st.divider()
        st.markdown("#### CGPA Analysis by Placement Status")

        cgpa_df = chart_df.copy()

        cgpa_df[cgpa_col] = pd.to_numeric(
            cgpa_df[cgpa_col],
            errors="coerce"
        )

        cgpa_df = cgpa_df.dropna(
            subset=[cgpa_col]
        )

        if not cgpa_df.empty:
            fig_cgpa = px.box(
                cgpa_df,
                x=target_col,
                y=cgpa_col,
                color=target_col,
                title="CGPA Distribution by Placement Status",
                points="outliers"
            )

            fig_cgpa.update_layout(
                height=450,
                xaxis_title="Placement Status",
                yaxis_title="CGPA",
                showlegend=False
            )

            st.plotly_chart(
                fig_cgpa,
                use_container_width=True
            )

    # Show the summary table
    with st.expander("View Placement Summary Table"):
        st.dataframe(
            status_counts,
            use_container_width=True,
            hide_index=True
        )