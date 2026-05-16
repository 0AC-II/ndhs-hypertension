import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Page configuration
# Must be the first Streamlit command in the file
st.set_page_config(
    page_title="Nigeria Hypertension Risk",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the saved model
# @st.cache_resource means the model loads once and stays in memory
# Without this, it would reload on every user interaction
@st.cache_resource
def load_model():
    return joblib.load("app/model.pkl")

model = load_model()

# Zone and label mappings
# These match the DHS szone codes used during training
ZONE_MAP = {
    "North Central": 1,
    "North East":    2,
    "North West":    3,
    "South East":    4,
    "South South":   5,
    "South West":    6,
}

# Zonal findings from the analysis -- used in the dashboard page
ZONE_FINDINGS = pd.DataFrame({
    "Zone": [
        "North East", "North Central", "North West",
        "South South", "South West", "South East"
    ],
    "Screened Adults": [3611, 4659, 4317, 4160, 4087, 3499],
    "Diagnosed Rate (%)": [16.7, 14.8, 14.4, 14.2, 11.1, 10.4],
    "Unscreened Adults": [4570, 5436, 7887, 3114, 2602, 3064],
    "Est. High Risk (%)": [32.3, 15.7, 17.4, 15.8, 11.3, 8.0],
})

NATIONAL_AVG = 13.7  # national diagnosed hypertension rate from analysis

# Sidebar navigation
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/7/79/Flag_of_Nigeria.svg",
    width=80
)
st.sidebar.title("Nigeria Hypertension Risk")
st.sidebar.caption(
    "A machine learning tool built on Nigeria DHS 2024 data. "
    "Not a clinical diagnostic. For public health awareness only."
)

page = st.sidebar.radio(
    "Navigate",
    ["Risk Calculator", "Nigeria Dashboard"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built by** Anthony Okeibuno  \n"
    "Health Data Systems Strategist  \n"
    "[GitHub](https://github.com/0AC-II/ndhs-hypertension)"
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: RISK CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "Risk Calculator":

    st.title("🫀 Hypertension Risk Calculator")
    st.markdown(
        "This tool estimates your risk of having high blood pressure based on "
        "sociodemographic factors. It was trained on data from **39,050 women "
        "and 12,204 men** surveyed across Nigeria in 2024.  \n\n"
        "> ⚠️ This is not a medical diagnosis. If you are concerned about your "
        "blood pressure, please see a healthcare provider."
    )

    st.markdown("---")
    st.subheader("Enter your information")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider(
            "Age",
            min_value=15, max_value=59, value=35,
            help="Women 15-49 and men 15-59 were surveyed"
        )

        zone_label = st.selectbox(
            "Geopolitical Zone",
            options=list(ZONE_MAP.keys()),
            help="Nigeria's six geopolitical zones"
        )

        sex = st.radio(
            "Sex",
            options=["Female", "Male"],
            horizontal=True
        )

        urban_rural = st.radio(
            "Place of Residence",
            options=["Urban", "Rural"],
            horizontal=True
        )

    with col2:
        wealth = st.select_slider(
            "Household Wealth Level",
            options=["Poorest", "Poorer", "Middle", "Richer", "Richest"],
            value="Middle",
            help="Approximate relative wealth of your household"
        )

        education_level = st.selectbox(
            "Highest Education Level",
            options=["No education", "Primary", "Secondary", "Higher"],
        )

        years_education = st.slider(
            "Years of Education",
            min_value=0, max_value=20, value=9,
            help="From primary school"
        )

        children = st.slider(
            "Total Children Ever Born",
            min_value=0, max_value=15, value=2,
            help="For men, enter 0 if not applicable"
        )

        reads_newspaper = st.radio(
            "Do you read newspapers or magazines?",
            options=["Never", "Less than once a week", "At least once a week"],
            horizontal=False
        )

    # Encode inputs to match training feature format─
    zone_code     = ZONE_MAP[zone_label]
    sex_code      = 1 if sex == "Male" else 0
    urban_code    = 1 if urban_rural == "Urban" else 2  # DHS: 1=urban, 2=rural
    wealth_code   = ["Poorest","Poorer","Middle","Richer","Richest"].index(wealth) + 1
    edu_code      = ["No education","Primary","Secondary","Higher"].index(education_level)
    newspaper_code = ["Never","Less than once a week","At least once a week"].index(reads_newspaper)

    # Feature order must exactly match what the model was trained on:
    # v012, v025, v106, v190, v133, szone, v731, v157, v151, v201, sex
    input_features = np.array([[
        age,             # v012
        urban_code,      # v025
        edu_code,        # v106
        wealth_code,     # v190
        years_education, # v133
        zone_code,       # szone
        2,               # v731 -- worked in last 12 months (2=yes, most common)
        newspaper_code,  # v157
        1,               # v151 -- sex of household head (1=male, most common)
        children,        # v201
        sex_code,        # sex
    ]])

    # Predict
    st.markdown("---")

    if st.button("Calculate My Risk", type="primary", use_container_width=True):

        risk_score = model.predict_proba(input_features)[0][1]
        risk_pct   = risk_score * 100

        # Risk tier classification
        if risk_pct < 10:
            tier       = "Low Risk"
            color      = "#27ae60"
            emoji      = "✅"
            message    = (
                "Your profile suggests a lower likelihood of hypertension. "
                "Continue healthy habits and consider getting screened at your "
                "next health visit."
            )
        elif risk_pct < 20:
            tier       = "Moderate Risk"
            color      = "#f39c12"
            emoji      = "⚠️"
            message    = (
                "Your profile suggests a moderate likelihood of hypertension. "
                "We recommend getting your blood pressure checked at a nearby "
                "health facility."
            )
        else:
            tier       = "High Risk"
            color      = "#c0392b"
            emoji      = "🚨"
            message    = (
                "Your profile suggests a higher likelihood of hypertension. "
                "Please get your blood pressure checked as soon as possible. "
                "Early detection saves lives."
            )

        # Display result
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(
                f"""
                <div style='
                    background-color: {color}22;
                    border-left: 6px solid {color};
                    border-radius: 8px;
                    padding: 24px;
                    text-align: center;
                '>
                    <div style='font-size: 48px'>{emoji}</div>
                    <div style='font-size: 36px; font-weight: bold;
                                color: {color};'>{risk_pct:.1f}%</div>
                    <div style='font-size: 18px; font-weight: 600;
                                color: {color};'>{tier}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with res_col2:
            st.markdown(f"### {tier}")
            st.markdown(message)
            st.markdown(
                f"**National diagnosed hypertension rate:** {NATIONAL_AVG}%  \n"
                f"**Your zone ({zone_label}) diagnosed rate:** "
                f"{ZONE_FINDINGS.loc[ZONE_FINDINGS['Zone']==zone_label, 'Diagnosed Rate (%)'].values[0]}%"
            )
            st.info(
                "This estimate is based on sociodemographic patterns across "
                "51,000+ Nigerians surveyed in 2024. It is not a substitute "
                "for clinical measurement."
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: NIGERIA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Nigeria Dashboard":

    st.title("📊 Nigeria Hypertension Dashboard")
    st.markdown(
        "Findings from a machine learning analysis of the "
        "**Nigeria Demographic and Health Survey 2024** -- the most recent "
        "nationally representative health dataset for Nigeria."
    )
    st.markdown("---")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Adults Surveyed", "51,254")
    m2.metric("Screened for BP", "24,333")
    m3.metric("National Hypertension Rate", "13.7%")
    m4.metric("Never Screened", "26,673")

    st.markdown("---")

    # Chart 1: Diagnosed rate by zone─
    st.subheader("Diagnosed Hypertension Rate by Geopolitical Zone")
    st.caption("Among adults who have been screened for blood pressure")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    zones_sorted = ZONE_FINDINGS.sort_values("Diagnosed Rate (%)")
    colors = [
        "#c0392b" if r > NATIONAL_AVG else "#2980b9"
        for r in zones_sorted["Diagnosed Rate (%)"]
    ]

    ax1.barh(zones_sorted["Zone"], zones_sorted["Diagnosed Rate (%)"],
             color=colors, edgecolor="none")
    ax1.axvline(NATIONAL_AVG, color="black", linestyle="--",
                linewidth=1.2, label=f"National avg: {NATIONAL_AVG}%")

    for i, val in enumerate(zones_sorted["Diagnosed Rate (%)"]):
        ax1.text(val + 0.2, i, f"{val}%", va="center", fontsize=10)

    above = mpatches.Patch(color="#c0392b", label="Above national average")
    below = mpatches.Patch(color="#2980b9", label="Below national average")
    ax1.legend(handles=[above, below], loc="lower right", fontsize=8)
    ax1.set_xlabel("Diagnosed Hypertension Rate (%)")
    ax1.set_xlim(0, 22)
    ax1.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown("---")

    # Chart 2: Estimated high risk among unscreened─
    st.subheader("Estimated High Risk Among Unscreened Adults")
    st.caption(
        "Predicted hypertension risk applied to 26,673 adults who have "
        "never had their blood pressure checked. Threshold: >15% predicted risk."
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    unscreened_sorted = ZONE_FINDINGS.sort_values("Est. High Risk (%)")
    colors2 = [
        "#c0392b" if r > 15 else "#2980b9"
        for r in unscreened_sorted["Est. High Risk (%)"]
    ]

    ax2.barh(unscreened_sorted["Zone"], unscreened_sorted["Est. High Risk (%)"],
             color=colors2, edgecolor="none")
    ax2.axvline(15, color="black", linestyle="--",
                linewidth=1.2, label="High-risk threshold: 15%")

    for i, val in enumerate(unscreened_sorted["Est. High Risk (%)"]):
        ax2.text(val + 0.3, i, f"{val}%", va="center", fontsize=10)

    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_xlabel("Adults Estimated at High Risk (%)")
    ax2.set_xlim(0, 42)
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")

    # Feature importance chart─
    st.subheader("What Predicts Hypertension Risk?")
    st.caption(
        "Mean absolute SHAP values from the Gradient Boosting model -- "
        "higher value means stronger influence on predictions"
    )

    feature_importance = pd.DataFrame({
        "Feature": [
            "Age", "Geopolitical Zone", "Total Children Ever Born",
            "Wealth Index", "Years of Education", "Reads Newspaper",
            "Sex", "Urban/Rural Residence", "Education Level",
            "Worked in Last 12 Months", "Sex of Household Head"
        ],
        "Mean |SHAP|": [0.44, 0.23, 0.11, 0.08, 0.08,
                        0.03, 0.03, 0.025, 0.02, 0.015, 0.005]
    }).sort_values("Mean |SHAP|")

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    bar_colors = ["#c0392b" if f in ["Age", "Geopolitical Zone", "Total Children Ever Born"]
                  else "#7f8c8d" for f in feature_importance["Feature"]]
    ax3.barh(feature_importance["Feature"], feature_importance["Mean |SHAP|"],
             color=bar_colors, edgecolor="none")
    ax3.set_xlabel("Mean Absolute SHAP Value")
    ax3.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("---")

    # Data table
    st.subheader("Full Zonal Summary")
    st.dataframe(
        ZONE_FINDINGS.set_index("Zone"),
        use_container_width=True
    )

    # Footer─
    st.markdown("---")
    st.caption(
        "Data: Nigeria Demographic and Health Surveys (DHS) 2024. "
        "Model: Gradient Boosting Classifier (AUC = 0.669). "
        "Analysis and app: Anthony Okeibuno. "
        "GitHub: github.com/0AC-II/ndhs-hypertension"
    )