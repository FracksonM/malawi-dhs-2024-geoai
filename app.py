"""
Malawi Child Undernutrition Risk Prediction Tool
Based on: Survey-Weighted Machine Learning Analysis of the 2024 MDHS
Author: Frackson Makwangwala  
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# PAGE CONFIG  
st.set_page_config(
    page_title="Malawi Child Nutrition Risk Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PATHS  
PROJECT_ROOT   = Path("/Users/frack/Documents/PhD Data Science/malawi-dhs-2024-geoai")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS         = PROJECT_ROOT / "outputs" / "models"

# CONSTANTS  
FEATURES = [
    "imm_child_age_months", "enc_age_band", "enc_child_sex",
    "imm_birth_order", "imm_birth_interval", "imm_first_born",
    "imm_short_interval", "imm_high_birth_order",
    "enc_size_at_birth", "enc_had_diarrhea",
    "und_maternal_age", "und_maternal_edu_years", "enc_edu_level",
    "und_maternal_weight_kg", "und_maternal_height_cm",
    "und_maternal_bmi", "und_low_bmi", "und_maternal_stunted",
    "und_total_children", "und_household_size",
    "enc_wealth_index", "enc_residence", "enc_region",
    "enc_district", "enc_religion",
    "district_stunting_rate", "district_mean_age",
    "district_mean_wealth", "district_mean_mat_height"
]

FEATURE_LABELS = {
    "imm_child_age_months"  : "Child age (months)",
    "enc_age_band"          : "Age band",
    "enc_child_sex"         : "Child sex",
    "imm_birth_order"       : "Birth order",
    "imm_birth_interval"    : "Birth interval (months)",
    "imm_first_born"        : "First born",
    "imm_short_interval"    : "Short birth interval (<24m)",
    "imm_high_birth_order"  : "High birth order (4+)",
    "enc_size_at_birth"     : "Size at birth",
    "enc_had_diarrhea"      : "Diarrhea in past 2 weeks",
    "und_maternal_age"      : "Maternal age (years)",
    "und_maternal_edu_years": "Maternal education (years)",
    "enc_edu_level"         : "Maternal education level",
    "und_maternal_weight_kg": "Maternal weight (kg)",
    "und_maternal_height_cm": "Maternal height (cm)",
    "und_maternal_bmi"      : "Maternal BMI",
    "und_low_bmi"           : "Maternal low BMI (<18.5)",
    "und_maternal_stunted"  : "Maternal stunting (<145 cm)",
    "und_total_children"    : "Total children ever born",
    "und_household_size"    : "Household size",
    "enc_wealth_index"      : "Household wealth index",
    "enc_residence"         : "Residence type",
    "enc_region"            : "Region",
    "enc_district"          : "District",
    "enc_religion"          : "Religion",
    "district_stunting_rate": "District stunting rate",
    "district_mean_age"     : "District mean child age",
    "district_mean_wealth"  : "District mean wealth",
    "district_mean_mat_height": "District mean maternal height"
}

DISTRICTS = [
    "balaka", "blantyre", "blantyre city", "chikwawa", "chiradzulu",
    "chitipa", "dedza", "dowa", "karonga", "kasungu", "likoma",
    "lilongwe", "lilongwe city", "machinga", "mangochi", "mchinji",
    "mulanje", "mwanza", "mzimba", "mzuzu city", "neno", "nkhata bay",
    "nkhotakota", "nsanje", "ntcheu", "ntchisi", "phalombe", "rumphi",
    "salima", "thyolo", "zomba", "zomba city"
]
DISTRICT_ENC = {d: i for i, d in enumerate(DISTRICTS)}

REGIONS = {"Northern": 1, "Central": 0, "Southern": 2}
REGION_DISTRICTS = {
    "Northern": ["chitipa", "karonga", "likoma", "mzimba", "mzuzu city",
                 "nkhata bay", "rumphi"],
    "Central" : ["dedza", "dowa", "kasungu", "lilongwe", "lilongwe city",
                 "mchinji", "nkhotakota", "ntchisi", "salima"],
    "Southern": ["balaka", "blantyre", "blantyre city", "chikwawa",
                 "chiradzulu", "machinga", "mangochi", "mulanje", "mwanza",
                 "neno", "nsanje", "ntcheu", "phalombe", "thyolo",
                 "zomba", "zomba city"]
}
DISTRICT_REGION = {}
for region, dlist in REGION_DISTRICTS.items():
    for d in dlist:
        DISTRICT_REGION[d] = region

RELIGIONS   = {"Catholic": 0, "CCAP": 1, "Other Christian": 2,
               "Muslim": 3, "Other / None": 4}
RESIDENCES  = {"Rural": 0, "Urban": 1}
EDU_LEVELS  = {"No education": 0, "Primary": 1, "Secondary": 2, "Higher": 3}
BIRTH_SIZES = {"Very small": 1, "Small": 2, "Average": 3,
               "Large": 4, "Very large": 5}

# MODEL LOADING  
@st.cache_resource(show_spinner="Loading prediction models...")
def load_models():
    return {
        "rf_stunting"  : joblib.load(MODELS / "rf_stunting_v2.pkl"),
        "rf_wasting"   : joblib.load(MODELS / "rf_wasting_v1.pkl"),
        "rf_anemia"    : joblib.load(MODELS / "rf_anemia_v1.pkl"),
        "lgb_stunting" : joblib.load(MODELS / "lgb_stunting_v2.pkl"),
        "lgb_wasting"  : joblib.load(MODELS / "lgb_wasting_v1.pkl"),
        "lgb_anemia"   : joblib.load(MODELS / "lgb_anemia_v1.pkl"),
    }

@st.cache_data(show_spinner=False)
def load_district_stats():
    df = pd.read_parquet(DATA_PROCESSED / "model_ready_stunting_v2.parquet")
    stats = (df.groupby("str_district")
               .agg(
                   district_stunting_rate   = ("outcome_stunted",       "mean"),
                   district_mean_age        = ("imm_child_age_months",  "mean"),
                   district_mean_wealth     = ("enc_wealth_index",       "mean"),
                   district_mean_mat_height = ("und_maternal_height_cm", "mean")
               ).reset_index())
    return stats.set_index("str_district").to_dict(orient="index")

#  FEATURE ENGINEERING 
def build_feature_vector(inputs: dict, district_stats: dict) -> pd.DataFrame:
    age   = inputs["child_age"]
    bmi   = inputs["maternal_weight"] / ((inputs["maternal_height"] / 100) ** 2)
    bord  = inputs["birth_order"]
    bint  = inputs["birth_interval"]

    if   age <= 5:  age_band = 0
    elif age <= 11: age_band = 1
    elif age <= 23: age_band = 2
    elif age <= 35: age_band = 3
    elif age <= 47: age_band = 4
    else:           age_band = 5

    dist_key  = inputs["district"]
    dist_data = district_stats.get(dist_key, {
        "district_stunting_rate"   : 0.375,
        "district_mean_age"        : 29.0,
        "district_mean_wealth"     : 2.5,
        "district_mean_mat_height" : 156.0
    })

    row = {
        "imm_child_age_months"    : age,
        "enc_age_band"            : age_band,
        "enc_child_sex"           : inputs["child_sex"],
        "imm_birth_order"         : bord,
        "imm_birth_interval"      : bint,
        "imm_first_born"          : int(bord == 1),
        "imm_short_interval"      : int(bint < 24),
        "imm_high_birth_order"    : int(bord >= 4),
        "enc_size_at_birth"       : inputs["size_at_birth"],
        "enc_had_diarrhea"        : inputs["had_diarrhea"],
        "und_maternal_age"        : inputs["maternal_age"],
        "und_maternal_edu_years"  : inputs["maternal_edu_years"],
        "enc_edu_level"           : inputs["edu_level"],
        "und_maternal_weight_kg"  : inputs["maternal_weight"],
        "und_maternal_height_cm"  : inputs["maternal_height"],
        "und_maternal_bmi"        : round(bmi, 2),
        "und_low_bmi"             : int(bmi < 18.5),
        "und_maternal_stunted"    : int(inputs["maternal_height"] < 145),
        "und_total_children"      : inputs["total_children"],
        "und_household_size"      : inputs["household_size"],
        "enc_wealth_index"        : inputs["wealth_index"],
        "enc_residence"           : inputs["residence"],
        "enc_region"              : REGIONS[inputs["region"]],
        "enc_district"            : DISTRICT_ENC.get(dist_key, 11),
        "enc_religion"            : inputs["religion"],
        "district_stunting_rate"  : dist_data["district_stunting_rate"],
        "district_mean_age"       : dist_data["district_mean_age"],
        "district_mean_wealth"    : dist_data["district_mean_wealth"],
        "district_mean_mat_height": dist_data["district_mean_mat_height"],
    }
    return pd.DataFrame([row])[FEATURES]

# PREDICTION  
def predict_all(X: pd.DataFrame, models: dict) -> dict:
    return {
        "stunting_prob" : float(models["rf_stunting"].predict_proba(X)[0, 1]),
        "wasting_prob"  : float(models["rf_wasting"].predict_proba(X)[0, 1]),
        "anemia_prob"   : float(models["rf_anemia"].predict_proba(X)[0, 1]),
    }

def risk_level(prob: float) -> tuple:
    if prob < 0.30:
        return "Low",    "#2e7d32", "#e8f5e9"
    elif prob < 0.50:
        return "Moderate", "#f57f17", "#fff8e1"
    else:
        return "High",   "#c62828", "#ffebee"

# SHAP WATERFALL  
@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

def shap_waterfall(model, X: pd.DataFrame, feature_labels: dict, title: str):
    explainer = get_shap_explainer(model)
    sv        = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    vals   = sv[0]
    feats  = X.columns.tolist()
    labels = [feature_labels.get(f, f) for f in feats]
    base   = float(explainer.expected_value[1] if hasattr(
        explainer.expected_value, "__len__") else explainer.expected_value)

    pairs  = sorted(zip(vals, labels, X.values[0]), key=lambda x: abs(x[0]), reverse=True)
    top_n  = 12
    pairs  = pairs[:top_n]
    shap_vals, feat_names, feat_vals = zip(*pairs)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#c62828" if v > 0 else "#1565c0" for v in shap_vals]
    y_pos   = range(len(shap_vals))

    bars = ax.barh(list(y_pos), list(shap_vals), color=colors,
                   height=0.65, edgecolor="white", linewidth=0.5)

    tick_labels = [f"{n}  (={v:.2g})" for n, v in zip(feat_names, feat_vals)]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to predicted probability", fontsize=9)
    ax.set_title(f"{title}   (base rate = {base:.3f})", fontsize=10, fontweight="bold")

    for bar, val in zip(bars, shap_vals):
        ax.text(val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=8, color="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

#  GAUGE CHART  
def gauge_chart(prob: float, label: str, ax):
    level, color, _ = risk_level(prob)
    theta_start = np.pi
    theta_end   = 0.0
    theta_val   = theta_start + (theta_end - theta_start) * prob

    thetas = np.linspace(theta_start, theta_end, 300)
    ax.plot(np.cos(thetas), np.sin(thetas), color="#e0e0e0", linewidth=14,
            solid_capstyle="round")

    thetas_filled = np.linspace(theta_start, theta_val, 300)
    ax.plot(np.cos(thetas_filled), np.sin(thetas_filled),
            color=color, linewidth=14, solid_capstyle="round")

    ax.plot(0.78 * np.cos(theta_val), 0.78 * np.sin(theta_val),
            "o", color=color, markersize=12, zorder=5)

    ax.text(0, 0.2, f"{prob:.0%}", ha="center", va="center",
            fontsize=18, fontweight="bold", color=color)
    ax.text(0, -0.12, label, ha="center", va="center", fontsize=9, color="#555555")
    ax.text(0, -0.34, level, ha="center", va="center",
            fontsize=10, fontweight="bold", color=color)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.2)
    ax.axis("off")

#DISTRICT BAR CHART 
def district_bar_chart(district_stats_dict: dict, selected: str, pred_prob: float):
    items = sorted(district_stats_dict.items(),
                   key=lambda x: x[1]["district_stunting_rate"], reverse=True)
    dists = [k for k, _ in items]
    rates = [v["district_stunting_rate"] for _, v in items]

    colors = ["#ef9a9a" if d == selected else "#b0bec5" for d in dists]

    fig, ax = plt.subplots(figsize=(14, 4))
    bars = ax.bar(range(len(dists)), rates, color=colors, edgecolor="white", linewidth=0.4)

    sel_idx = dists.index(selected) if selected in dists else -1
    if sel_idx >= 0:
        ax.annotate(
            f"Selected: {selected.title()} ({rates[sel_idx]:.1%})",
            xy=(sel_idx, rates[sel_idx]),
            xytext=(sel_idx, rates[sel_idx] + 0.04),
            ha="center", fontsize=8, color="#c62828", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2)
        )

    ax.axhline(0.375, color="#1565c0", linewidth=1.2, linestyle="--",
               label="National average (37.5%)")
    ax.axhline(pred_prob, color="#e65100", linewidth=1.2, linestyle=":",
               label=f"This child ({pred_prob:.1%})")

    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels([d.title() for d in dists], rotation=45, ha="right",
                       fontsize=7.5)
    ax.set_ylabel("Stunting rate", fontsize=9)
    ax.set_title("Predicted stunting probability vs district burden, 2024 MDHS",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

#SIDEBAR 
def render_sidebar():
    with st.sidebar:
        st.markdown("## Malawi Child Nutrition Risk Tool")
        st.markdown("_2024 MDHS Survey-Weighted Random Forest_")
        st.markdown("---")
        role = st.selectbox(
            "Select user role",
            ["Health Surveillance Assistant (HSA)",
             "District Health Officer (DHO)",
             "Researcher / Data Analyst"],
            index=0
        )
        st.markdown("---")
        st.markdown(
            "**Model:** Random Forest (ROC-AUC = 0.686)\n\n"
            "**Data:** 2024 MDHS, n = 5,122 children\n\n"
            "**Outcomes:** Stunting, wasting, anemia\n\n"
            "**Weights:** DHS probability-of-selection weights applied during training"
        )
        st.markdown("---")
        st.caption(
            "Makwangwala F. (2025). Survey-Weighted Machine Learning for "
            "Nationally Representative Prediction of Child Stunting in Malawi. "
            "PhD Study, LUANAR / Pazel Conroy Consulting Ltd."
        )
    return role

#INPUT FORM  
def render_input_form(district_stats: dict):
    st.markdown("### Child and Household Information")
    st.markdown(
        "Enter the child and caregiver details below. "
        "District contextual features are automatically populated from the 2024 MDHS."
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Child characteristics**")
        child_age      = st.slider("Child age (months)", 0, 59, 18)
        child_sex      = st.radio("Child sex", ["Male", "Female"], horizontal=True)
        birth_order    = st.number_input("Birth order", 1, 15, 2)
        birth_interval = st.slider("Birth interval (months)",
                                   6, 120, 36,
                                   help="Months since previous birth. Set to 60 for first-born.")
        size_at_birth  = st.selectbox("Size at birth", list(BIRTH_SIZES.keys()), index=2)
        had_diarrhea   = st.radio("Diarrhea in past 2 weeks", ["No", "Yes"], horizontal=True)

    with col2:
        st.markdown("**Maternal characteristics**")
        maternal_age     = st.slider("Maternal age (years)", 15, 49, 26)
        maternal_height  = st.slider("Maternal height (cm)", 130.0, 190.0, 156.0, step=0.5)
        maternal_weight  = st.slider("Maternal weight (kg)", 30.0, 120.0, 57.0, step=0.5)
        maternal_edu_yrs = st.slider("Maternal education (years)", 0, 16, 6)
        edu_level        = st.selectbox("Education level", list(EDU_LEVELS.keys()), index=1)
        total_children   = st.number_input("Total children ever born", 1, 15, 2)

    with col3:
        st.markdown("**Household and location**")
        household_size = st.number_input("Household size", 1, 20, 5)
        wealth_index   = st.select_slider(
            "Wealth index", options=[1, 2, 3, 4, 5],
            value=2,
            format_func=lambda x: {1:"Poorest",2:"Poorer",3:"Middle",
                                   4:"Richer",5:"Richest"}[x]
        )
        residence      = st.radio("Residence", ["Rural", "Urban"], horizontal=True)
        religion       = st.selectbox("Religion", list(RELIGIONS.keys()), index=2)
        region         = st.selectbox("Region", list(REGIONS.keys()), index=1)
        valid_dists    = sorted(REGION_DISTRICTS.get(region, DISTRICTS))
        district       = st.selectbox("District", valid_dists, index=0)

    bmi_computed = maternal_weight / ((maternal_height / 100) ** 2)
    st.markdown(
        f"**Computed maternal BMI:** {bmi_computed:.1f}  "
        f"{'(Low BMI)' if bmi_computed < 18.5 else '(Normal)' if bmi_computed < 25 else '(Overweight/Obese)'}"
    )

    dist_data = district_stats.get(district, {})
    if dist_data:
        st.markdown(
            f"**District context (2024 MDHS):** "
            f"Stunting rate = {dist_data['district_stunting_rate']:.1%} | "
            f"Mean maternal height = {dist_data['district_mean_mat_height']:.1f} cm | "
            f"Mean wealth = {dist_data['district_mean_wealth']:.2f}"
        )

    inputs = {
        "child_age"       : child_age,
        "child_sex"       : 1 if child_sex == "Male" else 0,
        "birth_order"     : birth_order,
        "birth_interval"  : birth_interval,
        "size_at_birth"   : BIRTH_SIZES[size_at_birth],
        "had_diarrhea"    : 1 if had_diarrhea == "Yes" else 0,
        "maternal_age"    : maternal_age,
        "maternal_height" : maternal_height,
        "maternal_weight" : maternal_weight,
        "maternal_edu_years": maternal_edu_yrs,
        "edu_level"       : EDU_LEVELS[edu_level],
        "total_children"  : total_children,
        "household_size"  : household_size,
        "wealth_index"    : wealth_index,
        "residence"       : RESIDENCES[residence],
        "religion"        : RELIGIONS[religion],
        "region"          : region,
        "district"        : district,
    }
    return inputs

#  HSA VIEW  
def render_hsa_view(preds: dict, inputs: dict):
    st.markdown("---")
    st.markdown("## Nutrition Risk Assessment")

    stunting_level, stunting_color, stunting_bg = risk_level(preds["stunting_prob"])
    wasting_level,  wasting_color,  wasting_bg  = risk_level(preds["wasting_prob"])
    anemia_level,   anemia_color,   anemia_bg   = risk_level(preds["anemia_prob"])

    col1, col2, col3 = st.columns(3)
    for col, label, prob, level, color, bg in [
        (col1, "Stunting", preds["stunting_prob"], stunting_level, stunting_color, stunting_bg),
        (col2, "Wasting",  preds["wasting_prob"],  wasting_level,  wasting_color,  wasting_bg),
        (col3, "Anemia",   preds["anemia_prob"],   anemia_level,   anemia_color,   anemia_bg),
    ]:
        with col:
            st.markdown(
                f"""<div style="background:{bg}; border-left:5px solid {color};
                padding:16px 20px; border-radius:6px; margin-bottom:10px;">
                <span style="font-size:1.05em; font-weight:600; color:#333;">{label}</span><br>
                <span style="font-size:2.2em; font-weight:700;
                color:{color};">{prob:.0%}</span><br>
                <span style="font-size:0.95em; font-weight:600;
                color:{color};">{level} risk</span>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("### Recommended Action")

    overall_max = max(preds["stunting_prob"], preds["wasting_prob"], preds["anemia_prob"])
    if overall_max >= 0.50:
        action_color = "#c62828"
        action_bg    = "#ffebee"
        action_icon  = "HIGH RISK"
        action_text  = (
            "Refer this child to the nearest health facility today. "
            "Initiate nutrition counseling for the caregiver. "
            "Enroll in community management of acute malnutrition (CMAM) screening. "
            "Schedule a follow-up visit in 2 weeks."
        )
    elif overall_max >= 0.30:
        action_color = "#f57f17"
        action_bg    = "#fff8e1"
        action_icon  = "MODERATE RISK"
        action_text  = (
            "Refer for targeted nutrition assessment at the nearest health post. "
            "Provide caregiver counseling on dietary diversity and complementary feeding. "
            "Schedule growth monitoring follow-up in 4 weeks. "
            "Register in community nutrition programme if available."
        )
    else:
        action_color = "#2e7d32"
        action_bg    = "#e8f5e9"
        action_icon  = "LOW RISK"
        action_text  = (
            "Continue routine growth monitoring at scheduled visits. "
            "Reinforce positive feeding and care practices with caregiver. "
            "No immediate referral required. Reassess at next monthly visit."
        )

    child_age = inputs.get("child_age", 0)
    if child_age < 6:
        action_text += (
            " Note: this child is under 6 months. "
            "Promote exclusive breastfeeding and monitor closely."
        )
    elif 6 <= child_age <= 23:
        action_text += (
            " Note: this child is in the critical 6-23 month window. "
            "Ensure complementary feeding has begun alongside breastfeeding."
        )

    st.markdown(
        f"""<div style="background:{action_bg}; border-left:6px solid {action_color};
        padding:18px 22px; border-radius:6px;">
        <span style="font-size:0.85em; font-weight:700;
        color:{action_color}; letter-spacing:0.08em;">{action_icon}</span><br>
        <span style="font-size:1.0em; color:#333; line-height:1.7;">{action_text}</span>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#888;'>Predictions from survey-weighted Random Forest model "
        "trained on 2024 Malawi DHS (n = 5,122). Model ROC-AUC = 0.686. "
        "This tool supports but does not replace clinical judgment.</small>",
        unsafe_allow_html=True
    )

# DHO VIEW  
def render_dho_view(preds: dict, inputs: dict, district_stats: dict):
    st.markdown("---")
    st.markdown("## District Health Officer Dashboard")

    col_g1, col_g2, col_g3 = st.columns(3)
    for col, label, prob in [
        (col_g1, "Stunting",  preds["stunting_prob"]),
        (col_g2, "Wasting",   preds["wasting_prob"]),
        (col_g3, "Anemia",    preds["anemia_prob"]),
    ]:
        with col:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            gauge_chart(prob, label, ax)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("---")
    st.markdown("### Child stunting risk vs district burden")
    fig2 = district_bar_chart(
        district_stats, inputs["district"], preds["stunting_prob"]
    )
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown("---")
    st.markdown("### Prediction summary")

    df_summary = pd.DataFrame({
        "Outcome"            : ["Stunting", "Wasting", "Anemia"],
        "Predicted probability": [
            f"{preds['stunting_prob']:.1%}",
            f"{preds['wasting_prob']:.1%}",
            f"{preds['anemia_prob']:.1%}"
        ],
        "Risk level"         : [
            risk_level(preds["stunting_prob"])[0],
            risk_level(preds["wasting_prob"])[0],
            risk_level(preds["anemia_prob"])[0]
        ],
        "National threshold" : ["37.5%", "1.6%", "49.1%"],
    })
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    dist     = inputs["district"]
    dist_dat = district_stats.get(dist, {})
    if dist_dat:
        st.markdown(
            f"**District context ({dist.title()}):** "
            f"Stunting burden = {dist_dat['district_stunting_rate']:.1%} | "
            f"Mean maternal height = {dist_dat['district_mean_mat_height']:.1f} cm"
        )

# RESEARCHER VIEW  
def render_researcher_view(preds: dict, X: pd.DataFrame, models: dict):
    st.markdown("---")
    st.markdown("## Researcher Dashboard")

    col1, col2, col3 = st.columns(3)
    for col, label, key, color in [
        (col1, "Stunting (RF)", "stunting_prob", "#c62828"),
        (col2, "Wasting (RF)",  "wasting_prob",  "#e65100"),
        (col3, "Anemia (RF)",   "anemia_prob",   "#1565c0"),
    ]:
        with col:
            prob = preds[key]
            st.metric(label, f"{prob:.4f}", f"{prob:.1%}")

    st.markdown("---")
    st.markdown("### SHAP explanation: stunting prediction")
    st.markdown(
        "The waterfall below shows how each feature pushed the predicted "
        "stunting probability above or below the base rate."
    )
    with st.spinner("Computing SHAP values..."):
        fig_shap = shap_waterfall(
            models["rf_stunting"], X, FEATURE_LABELS,
            title="Stunting probability decomposition (Random Forest)"
        )
    st.pyplot(fig_shap, use_container_width=True)
    plt.close(fig_shap)

    st.markdown("---")
    st.markdown("### Full feature vector")
    X_display = X.T.copy()
    X_display.columns = ["Value"]
    X_display.index   = [FEATURE_LABELS.get(i, i) for i in X_display.index]
    st.dataframe(X_display, use_container_width=True)

    st.markdown("---")
    st.markdown("### Export prediction record")
    export = X.copy()
    export["pred_stunting"] = preds["stunting_prob"]
    export["pred_wasting"]  = preds["wasting_prob"]
    export["pred_anemia"]   = preds["anemia_prob"]
    export["risk_stunting"] = risk_level(preds["stunting_prob"])[0]

    st.download_button(
        label     = "Download prediction as CSV",
        data      = export.to_csv(index=False),
        file_name = "child_nutrition_prediction.csv",
        mime      = "text/csv"
    )

    st.markdown(
        "<small style='color:#888;'>Stunting model: Random Forest, "
        "ROC-AUC = 0.686 (95% CI: 0.666, 0.705), Brier Skill Score = 0.074. "
        "Wasting model: LightGBM (class-imbalanced, use with caution). "
        "Anemia model: Random Forest, ROC-AUC = 0.680. "
        "All models trained on 2024 Malawi DHS (n = 5,122) with "
        "probability-of-selection weights applied during training.</small>",
        unsafe_allow_html=True
    )

# MAIN 
def main():
    role           = render_sidebar()
    models         = load_models()
    district_stats = load_district_stats()

    st.markdown(
        "# Malawi Child Undernutrition Risk Prediction Tool  "
        "<small style='font-size:0.55em; color:#888; font-weight:400;'>"
        "2024 MDHS | Survey-Weighted Random Forest | v1.0</small>",
        unsafe_allow_html=True
    )
    st.markdown(
        "This tool predicts individual-level risk of stunting, wasting, and anemia "
        "for children under five years in Malawi using survey-weighted machine learning "
        "models trained on the 2024 Malawi Demographic and Health Survey."
    )

    inputs = render_input_form(district_stats)

    st.markdown("---")
    predict_btn = st.button("Run prediction", type="primary", use_container_width=True)

    if predict_btn:
        X     = build_feature_vector(inputs, district_stats)
        preds = predict_all(X, models)

        if "HSA" in role:
            render_hsa_view(preds, inputs)
        elif "DHO" in role:
            render_dho_view(preds, inputs, district_stats)
        else:
            render_researcher_view(preds, X, models)

if __name__ == "__main__":
    main()
