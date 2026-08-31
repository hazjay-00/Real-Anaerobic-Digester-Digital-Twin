# Real Anaerobic Digester Digital Twin (Melbourne Treatment Plant SCADA)

## Overview
An empirical SCADA Machine Learning dashboard and process control interface modeled after the Melbourne Treatment Plant framework. The application uses historical SCADA sensor records to predict infrastructure power demand, estimate effluent discharge compliance, and calculate real-time asset operational expenses (OPEX).

---

## Key Features

* **Live SCADA Actuator Controls:** Interactive sidebar sliders simulating volumetric inflow rates ($m^3/\text{day}$), influent COD, ammonia loads, and process water temperature.
* **Infrastructure Power & OPEX Forecasting:** Machine learning inference predicting blower grid energy expenditures ($kWh/\text{day}$) and financial overhead.
* **Regulatory Compliance & Penalty Tracking:** Real-time monitoring against EPA discharge thresholds ($<100\text{ mg/L}$ COD) with automated non-compliance fine calculations.
* **Diagnostic Risk Alarms:** Process safety warnings for ammonia toxicity ($>55\text{ mg/L}$) and biological clearance failures.
* **Emergency Treatment Plant Bypass:** One-click hardware isolation lockout to simulate stopping aeration blowers and closing isolation gates during spills.
* **Shift Log Export:** Facility metric reporting with direct CSV export capabilities for operational record-keeping.

---

## System Architecture & Data Pipeline

The project is structured into three execution modules:

1. **Empirical Data Pipeline & ML Engine (`ml_agent.py`)**
   * Downloads and parses full-scale municipal wastewater treatment logs via `kagglehub`.
   * Maps SCADA headers (`avg_inflow`, `cod`, `am`, `t`) to target grid power demand (`total_grid`).
   * Trains a `RandomForestRegressor` (80/20 train-test split) and exports verified model artifacts to `wastewater_brain_cod.pkl`.

2. **Kinetic Physics Engine (`simulation_engine.py`)**
   * Evaluates continuous mass balance ordinary differential equations (ODEs) using Monod growth kinetics and Arrhenius thermal factors.

3. **SCADA Dashboard Interface (`interface.py`)**
   * Streamlit control room dashboard displaying financial OPEX metrics, active risk notifications, and Plotly vector profiles.

---

## Tech Stack & Data Source

### Tech Stack
* **UI / Dashboard:** Streamlit
* **Data Ingestion & ML:** KaggleHub, Pandas, NumPy, Scikit-Learn
* **Visualization & Analytics:** Plotly Graph Objects
* **Numerical Methods:** SciPy (`odeint`)

### Data Source
* **Dataset:** Kaggle – Full-Scale Waste Water Treatment Plant Data (Melbourne SCADA Framework)
