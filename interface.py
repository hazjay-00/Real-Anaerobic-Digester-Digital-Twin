from simulation_engine import run_plant_simulation
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import pickle

# Check if model exists; if not, train it automatically on deployment
if not os.path.exists("wastewater_brain_cod.pkl"):
    import ml_agent  # Running this script downloads Kaggle data and creates the .pkl file
    
st.set_page_config(page_title="Real Anaerobic Digester Digital Twin", layout="wide")

# --- EMERGENCY SYSTEM LOCK ---
if "kill_switch" not in st.session_state:
    st.session_state.kill_switch = False

if st.button("EMERGENCY TREATMENT PLANT BYPASS / LOCKOUT", use_container_width=True, type="primary"):
    st.session_state.kill_switch = not st.session_state.kill_switch

if st.session_state.kill_switch:
    st.error("PLANT ISOLATED: Aeration blowers stopped. Influent isolation gates closed to prevent environmental spill.")
    st.stop()

st.title("Real Anaerobic Digester Digital Twin")
st.caption("Melbourne Treatment Plant Framework | Empirical SCADA Machine Learning Interface")
st.markdown("---")

# Load real model artifacts safely
@st.cache_resource
def load_wastewater_brain():
    with open("wastewater_brain_cod.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts = load_wastewater_brain()
    ai_engine = artifacts["model"]
    model_accuracy = artifacts["r2_score"] # Explicitly defined here
except FileNotFoundError:
    st.error("Please run ml_agent.py first to parse the Kaggle repository data and train the model!")
    st.stop()

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("Live SCADA Actuator Controls")
slider_flow = st.sidebar.slider("Average Inflow Volumetric Rate (m³/day)", 15000, 85000, 45000, step=1000)
slider_cod_in = st.sidebar.slider("Incoming Organic Load (Influent COD mg/L)", 200, 1200, 580, step=20)
slider_ammonia = st.sidebar.slider("Incoming Ammonia Load (mg/L)", 10.0, 80.0, 35.0, step=0.5)
slider_temp = st.sidebar.slider("Influent Process Water Temperature (°C)", 5, 45, 18)

# --- RUN REAL PREDICTION PIPELINE ---
input_array = pd.DataFrame(
    [[slider_flow, slider_cod_in, slider_ammonia, slider_temp]], 
    columns=['avg_inflow', 'cod', 'am', 't']
)
predicted_energy_demand_wh = ai_engine.predict(input_array)[0]

# Convert Watt-hours (Wh) to Kilowatt-hours (kWh) for easier industrial display
predicted_energy_kwh = predicted_energy_demand_wh / 1000.0

# --- DYNAMIC ODE KINETICS & COMPLIANCE CALCULATIONS ---
# Calculate dilution rate D (Inflow rate / standard digester volume 50,000 m³)
dilution_rate = slider_flow / 50000.0

# Solve continuous ODE mass balance to retrieve dynamic kinetic effluent COD
s_effluent, x_biomass, methane_liters = run_plant_simulation(
    S_inflow=slider_cod_in, 
    D=dilution_rate, 
    Temperature=slider_temp
)

estimated_effluent_cod = max(0.0, s_effluent)
ENVIRONMENTAL_LIMIT_COD = 100.0

# Power utilities pricing: $0.14 per kWh of energy consumed by blower units
daily_electrical_cost = predicted_energy_kwh * 0.14

# Safe Discharge Estimation logic for water compliance tracking
estimated_effluent_cod = slider_cod_in * 0.15 # Assuming a baseline biological 85% reduction rate
ENVIRONMENTAL_LIMIT_COD = 100.0

if estimated_effluent_cod > ENVIRONMENTAL_LIMIT_COD:
    compliance_fine = 2500.00
    compliance_status = "CRITICAL BREACH DETECTED"
    compliance_color = "inverse"
else:
    compliance_fine = 0.00
    compliance_status = "STRICT COMPLIANCE TARGETS MET"
    compliance_color = "normal"

total_facility_overhead = daily_electrical_cost + compliance_fine
treatment_efficiency = ((slider_cod_in - estimated_effluent_cod) / slider_cod_in) * 100

# --- FINANCIAL DASHBOARD ROW ---
def status_badge(text, is_ok):
    color = "#2ecc71" if is_ok else "#e74c3c"
    return f"<span style='color:{color}; font-size: 0.85rem; font-weight: 500;'>{text}</span>"
    
st.subheader("Asset Operational Expense (OPEX) Monitoring")
fin_col1, fin_col2, fin_col3 = st.columns(3)

with fin_col1:
    ok_cost = daily_electrical_cost <= 50.0  # Budget limit $50.00/day
    st.metric(
        label="Blower Grid Energy Expenditures",
        value=f"${daily_electrical_cost:,.2f} / day"
    )
    st.markdown(
        status_badge("Budget Limit: <$50.00/day" if ok_cost else "Over Budget (>$50.00)", ok_cost),
        unsafe_allow_html=True
    )

with fin_col2:
    ok_fine = compliance_fine == 0.0
    st.metric(
        label="Regulatory Non-Compliance Penalty Fines",
        value=f"${compliance_fine:,.2f} / day"
    )
    st.markdown(
        status_badge("Target: $0.00/day (Compliant)" if ok_fine else "Penalty Breach Active", ok_fine),
        unsafe_allow_html=True
    )

with fin_col3:
    ok_total = total_facility_overhead <= 50.0  # Total OPEX target
    st.metric(
        label="Total Combined Overhead Burden",
        value=f"${total_facility_overhead:,.2f} / day"
    )
    st.markdown(
        status_badge("Target Max: $50.00/day" if ok_total else "Overhead Exceeded", ok_total),
        unsafe_allow_html=True
    )

st.markdown("---")

# --- TECHNICAL METRICS & KPIs ROW ---
st.subheader("Live Engineering Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    ok_flow = 15000 <= slider_flow <= 75000  # Nominal hydraulic capacity range
    st.metric(
        label="Measured Hydraulic Loading",
        value=f"{slider_flow:,.0f} m³/day"
    )
    st.markdown(
        status_badge("Nominal: 15k–75k m³/day" if ok_flow else "Flow Out of Range", ok_flow),
        unsafe_allow_html=True
    )

with col2:
    ok_kwh = predicted_energy_kwh <= 350.0  # Blower energy threshold
    st.metric(
        label="Predicted Infrastructure Power Demand",
        value=f"{predicted_energy_kwh:,.1f} kWh"
    )
    st.markdown(
        status_badge("Target: <350.0 kWh" if ok_kwh else "High Power Demand (>350 kWh)", ok_kwh),
        unsafe_allow_html=True
    )

with col3:
    ok_effluent = estimated_effluent_cod <= ENVIRONMENTAL_LIMIT_COD
    st.metric(
        label="Estimated Effluent Discharge",
        value=f"{estimated_effluent_cod:.1f} mg/L"
    )
    st.markdown(
        status_badge(
            f"EPA Limit: <{ENVIRONMENTAL_LIMIT_COD:.0f} mg/L" if ok_effluent else "EPA Limit Breached",
            ok_effluent
        ),
        unsafe_allow_html=True
    )

with col4:
    ok_acc = model_accuracy >= 80.0
    st.metric(
        label="ML Model R² Predictive Confidence",
        value=f"{model_accuracy:.2f} %"
    )
    st.markdown(
        status_badge("Benchmark: ≥80.0%" if ok_acc else "Low Validation Accuracy (<80%)", ok_acc),
        unsafe_allow_html=True
    )

st.markdown("---")

# --- MONITORING LAYOUT ---
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Plant Diagnostic Risk Alarms")
    
    if slider_ammonia > 55.0:
        st.error(f"AMMONIA TOXICITY WARNING ({slider_ammonia} mg/L): Nitrifying biomass bacteria stress thresholds exceeded. Risk of chemical process crash.")
    else:
        st.success("NITROGEN STOICHIOMETRIC BALANCES STABLE")
        
    if estimated_effluent_cod > ENVIRONMENTAL_LIMIT_COD:
        st.error(f"WATER QUALITY FAILURE: Estimated clear-well discharge breaches clean water statutory standards.")
        st.warning("MANIPULATION RECOVERY: Reduce inflow slider bounds to lengthen biological oxidation retention time.")
    else:
        st.success("DISCHARGE STREAM STABLE: Clarifier separation working within standard parameters.")

with col_right:
    st.subheader("Substrate Concentration Profile Vector")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Raw Wastewater Inflow', 'Estimated Outflow Discharge', 'EPA Boundary Limit'],
        y=[slider_cod_in, estimated_effluent_cod, ENVIRONMENTAL_LIMIT_COD],
        marker_color=['#E67E22', '#2ECC71', '#E74C3C']
    ))
    fig.update_layout(yaxis_title="Chemical Oxygen Demand (mg/L)", template="plotly_white", height=280)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- DOWNLOADABLE SHIFT REPORT ---
st.subheader("Operational Reporting Metrics")

report_data = pd.DataFrame({
    "Timestamp": [pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")],
    "Hydraulic Loading (m3/day)": [slider_flow],
    "Influent COD (mg/L)": [slider_cod_in],
    "Process Temp (C)": [slider_temp],
    "Ammonia Load (mg/L)": [slider_ammonia],
    "Predicted Energy (kWh)": [f"{predicted_energy_kwh:.2f}"],
    "Est. Purification KPI (%)": [f"{treatment_efficiency:.1f}%"],
    "Total Daily OPEX ($/day)": [f"${total_facility_overhead:.2f}"]
})

st.dataframe(report_data, hide_index=True, use_container_width=True)

st.download_button(label="Export Current Plant Metrics to Shift CSV Report", data=report_data.to_csv(index=False), file_name="ad_twin_melb_shift_report.csv", mime="text/csv", use_container_width=True)
