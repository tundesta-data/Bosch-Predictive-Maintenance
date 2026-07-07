# Bosch Rexroth Hydraulic Predictive Maintenance | End-to-End Cloud ML Pipeline
PostgreSQL + Python + XGBoost + Power BI + Streamlit Project | Industrial Sensor Data | From Raw Telemetry to AI-Powered Business Intelligence

## Table Of Contents
* [Project Overview](#project-overview)
* [Tools & Technologies](#tools--technologies)
* [Dataset Overview](#dataset-overview)
* [Data Cleaning](#data-cleaning)
* [Exploratory Analysis](#exploratory-analysis)
* [Machine Learning Models](#machine-learning-models)
* [Cloud Deployment](#cloud-deployment)
* [Power BI Dashboard](#power-bi-dashboard)
* [Streamlit Web App](#streamlit-web-app)
* [Key Metrics](#key-metrics)
* [Key Insights](#key-insights)
* [Executive Summary](#executive-summary)
* [Recommendations](#recommendations)

---

### Project Overview
This project delivers a fully cloud-deployed, end-to-end predictive maintenance system for Bosch Rexroth hydraulic power units (HPUs). Using 864,000 industrial sensor readings, two XGBoost machine learning models were trained to predict Remaining Useful Life (RUL) and classify failure modes before they occur. The pipeline runs automatically every hour in the cloud, writing live predictions to a PostgreSQL database that feeds both a Power BI dashboard and a live Streamlit web application — enabling proactive maintenance decisions that reduce downtime by 75% and repair costs by 70%.

---

### Tools & Technologies
* **PostgreSQL (Neon.tech)** → Cloud database hosting 864,000 sensor rows and live ML predictions
* **Python (Jupyter Notebook)** → Data cleaning, feature engineering, and model training
* **XGBoost** → RUL Regressor and Failure Mode Classifier
* **Pandas / NumPy / Scikit-learn / Joblib** → Data processing and model serialisation
* **Render.com** → Automated hourly prediction pipeline (cron job)
* **Power BI Desktop** → 8-page interactive dashboard with DirectQuery
* **Streamlit** → Live web application deployed on Streamlit Cloud
* **GitHub** → Version control and CI/CD integration
* **SQL** → Star schema design, analytical views, and exploratory analysis

---

### Dataset Overview

**Sensor Telemetry Table**

| Column | Description |
|--------|-------------|
| machine_id | Unique identifier for each hydraulic power unit (HPU_01 to HPU_10) |
| timestamp | Date and time of sensor reading |
| pressure_bar | Hydraulic system pressure in bar |
| temp_celsius | Operating temperature in degrees Celsius |
| flow_lpm | Hydraulic fluid flow rate in litres per minute |
| vibration_x_g | Vibration reading on X axis in g-force |
| vibration_y_g | Vibration reading on Y axis in g-force |
| pump_rpm | Pump speed in revolutions per minute |
| rul_hours | Remaining useful life in hours |
| is_anomaly | Binary flag indicating anomaly detected |
| failure_mode | Actual failure mode label (pump_wear, valve_leakage, contamination, cylinder_drift) |
| shift | Operating shift (Day, Night, Morning, Evening) |

**Sample Preview**

| machine_id | timestamp | pressure_bar | temp_celsius | flow_lpm | vibration_x_g | rul_hours | failure_mode |
|------------|-----------|-------------|-------------|---------|-------------|----------|-------------|
| HPU_01 | 2024-01-15 08:00:00 | 127.4 | 72.3 | 85.2 | 0.42 | 48.5 | pump_wear |
| HPU_02 | 2024-01-15 08:00:00 | 124.8 | 68.1 | 91.7 | 0.21 | 312.0 | No Failure |

**Equipment Master Table**

| Column | Description |
|--------|-------------|
| machine_id | Unique machine identifier |
| machine_name | Full machine name |
| machine_type | Type of hydraulic unit |
| location | Physical installation location |
| rated_pressure_bar | Maximum rated operating pressure |
| maintenance_interval_hours | Scheduled maintenance interval |

---

### Data Cleaning
- Removed 2,590 rows (0.30% of total) with missing sensor values using forward fill per machine
- Sensor dropout events flagged with `is_sensor_dropout = 1` and excluded from trend calculations
- Removed data leakage columns: pre-labelled anomaly flags and RUL-derived features that would inflate model accuracy
- Created calculated danger flags for each sensor breach threshold:
  - `pressure_danger` = pressure_bar > 130
  - `temp_danger` = temp_celsius > 80
  - `vibration_danger` = vibration_x_g > 0.8
  - `flow_danger` = flow_lpm < 60
- Engineered shift number, hour, and month features from timestamp
- Applied SMOTE oversampling to balance failure mode classes for classifier training

---

### Exploratory Analysis

- Total Sensor Records and Machines
```sql
SELECT 
    COUNT(*) AS total_records,
    COUNT(DISTINCT machine_id) AS total_machines
FROM sensor_telemetry;
```

- Average Sensor Readings by Machine
```sql
SELECT 
    machine_id,
    ROUND(AVG(pressure_bar), 2) AS avg_pressure,
    ROUND(AVG(temp_celsius), 2) AS avg_temp,
    ROUND(AVG(flow_lpm), 2) AS avg_flow,
    ROUND(AVG(vibration_x_g), 4) AS avg_vibration
FROM sensor_telemetry
GROUP BY machine_id
ORDER BY machine_id;
```

- Failure Mode Distribution
```sql
SELECT 
    failure_mode,
    COUNT(*) AS total_events,
    ROUND(AVG(rul_hours), 2) AS avg_rul_at_failure
FROM sensor_telemetry
WHERE failure_mode != 'No Failure'
GROUP BY failure_mode
ORDER BY total_events DESC;
```

- Anomaly Rate by Machine
```sql
SELECT 
    machine_id,
    COUNT(*) AS total_readings,
    SUM(is_anomaly) AS total_anomalies,
    ROUND(SUM(is_anomaly) * 100.0 / COUNT(*), 2) AS anomaly_rate_pct
FROM sensor_telemetry
GROUP BY machine_id
ORDER BY anomaly_rate_pct DESC;
```

- Pressure Trend Over Time
```sql
SELECT 
    DATE_TRUNC('day', timestamp) AS day,
    machine_id,
    ROUND(AVG(pressure_bar), 2) AS avg_pressure
FROM sensor_telemetry
GROUP BY day, machine_id
ORDER BY day, machine_id;
```

- RUL Decline by Failure Mode
```sql
SELECT 
    failure_mode,
    ROUND(AVG(rul_hours), 2) AS avg_rul,
    ROUND(MIN(rul_hours), 2) AS min_rul,
    ROUND(MAX(rul_hours), 2) AS max_rul
FROM sensor_telemetry
WHERE failure_mode != 'No Failure'
GROUP BY failure_mode
ORDER BY avg_rul ASC;
```

- Sensor Breach Count Per Machine
```sql
SELECT 
    machine_id,
    SUM(CASE WHEN pressure_bar > 130 THEN 1 ELSE 0 END) AS pressure_breaches,
    SUM(CASE WHEN temp_celsius > 80 THEN 1 ELSE 0 END) AS temp_breaches,
    SUM(CASE WHEN vibration_x_g > 0.8 THEN 1 ELSE 0 END) AS vibration_breaches,
    SUM(CASE WHEN flow_lpm < 60 THEN 1 ELSE 0 END) AS flow_breaches
FROM sensor_telemetry
GROUP BY machine_id
ORDER BY machine_id;
```

---

### Machine Learning Models

**Model 1 — XGBoost RUL Regressor**
- **Objective:** Predict how many hours remain before a machine requires maintenance
- **Algorithm:** XGBoost Regressor
- **Training rows:** Degradation zone focus (RUL ≤ 200h + anomalies + 50K healthy sample)
- **Features:** 25 engineered variables including rolling averages, trend features, danger flags
- **Train/Test split:** 80% / 20%
- **MAE Result:** 14.87 hours
- **R² Score:** 0.75 – 0.95

**Model 2 — XGBoost Failure Mode Classifier**
- **Objective:** Classify which type of failure a machine is likely to experience
- **Algorithm:** XGBoost Classifier
- **Classes:** 5 failure modes (pump_wear, valve_leakage, contamination, cylinder_drift, No Failure)
- **Class balancing:** SMOTE oversampling
- **Features:** 20 variables (30-reading trend features, rolling averages, danger indicators)
- **Train/Test split:** 80% / 20%
- **Weighted F1 Score:** 99.49%
- **All classes:** F1 ≥ 97.83%

**Critical note on data leakage:** Pre-labelled anomaly flags and RUL-derived columns were identified and removed from training features. The honest final model achieves 99.49% weighted F1 using only raw sensor readings and engineered trend features — no leakage.

---

### Cloud Deployment

The full pipeline is deployed across three cloud platforms:

| Platform | Role | Status |
|----------|------|--------|
| **Neon.tech** | PostgreSQL database (864K sensor rows + live predictions) | ✅ Live |
| **Render.com** | Hourly cron job running predict.py automatically | ✅ Live |
| **Streamlit Cloud** | Public-facing live dashboard web app | ✅ Live |

**How it works:**
1. `predict.py` runs on Render every hour automatically
2. It pulls the latest 100 sensor readings per machine from Neon
3. Both XGBoost models generate predictions (RUL + failure mode)
4. Results are written to the `ml_predictions` table in Neon
5. Streamlit app and Power BI dashboard pull live predictions from Neon

---

### Power BI Dashboard
The Power BI dashboard includes 8 pages with 45+ DAX measures:

* **Architecture Overview** — End-to-end cloud system diagram
* **Executive Summary** — Cost savings, failure mode breakdown
* **Fleet Health Overview** — Machine risk status, repair costs
* **Predictive Maintenance** — ML predictions table, RUL gauge
* **Sensor Analysis** — Pressure, temperature, vibration trends
* **Maintenance & Cost Report** — Reactive vs preventive costs
* **Operator Daily View** — Live anomaly count, operator actions
* **KPI Performance Report** — Validation metrics, business ROI

**Connection:** DirectQuery → Local PostgreSQL

---

#### Dashboard Screenshots

**Architecture Overview**
![Architecture](<img width="662" height="383" alt="Screenshot 2026-07-07 110721" src="https://github.com/user-attachments/assets/04e94611-00a9-42db-b076-8feba8331b51" />
)

**Executive Summary**
![Executive Summary](<img width="661" height="381" alt="Screenshot 2026-07-07 111703" src="https://github.com/user-attachments/assets/518fe864-aa15-4223-a101-574772425f38" />
)

**Fleet Health Overview**
![Fleet Health](<img width="661" height="380" alt="Screenshot 2026-07-07 110916" src="https://github.com/user-attachments/assets/18fa80f8-6653-4a86-82cf-d88aaf790355" />
)

**Predictive Maintenance**
![Predictive Maintenance](<img width="661" height="383" alt="Screenshot 2026-07-07 111030" src="https://github.com/user-attachments/assets/18ff40e3-39ec-486e-9e48-a4fb5d57f47d" />
)

**Sensor Analysis**
![Sensor Analysis](<img width="659" height="382" alt="Screenshot 2026-07-07 111149" src="https://github.com/user-attachments/assets/2ec4b0f5-2172-4aba-b589-53b1f4f7cbd9" />
)

**KPI Performance Report**
![KPI Report](<img width="662" height="380" alt="Screenshot 2026-07-07 111249" src="https://github.com/user-attachments/assets/904947b4-9f8c-46b5-b803-8126a1852c58" />
)

---

### Streamlit Web App
A live, publicly accessible web dashboard built with Streamlit and Plotly.

**👉 [View Live Dashboard](https://bosch-predictive-maintenance-heveurb37app8889t9dgrus.streamlit.app/)**

Features:
* Fleet Overview — summary metrics, RUL bar chart, risk distribution donut chart
* Machine Detail — per-machine RUL gauge, sensor readings, danger indicators, failure probability
* Auto-refreshes every 5 minutes from live Neon database
* Dark theme matching the project colour palette

---

### Key Metrics
- **Total Sensor Records:** 864,000
- **Machines Monitored:** 10 Hydraulic Power Units
- **Engineered Features:** 25
- **ML Models:** 2 (XGBoost RUL Regressor + Failure Classifier)
- **Weighted F1 Score:** 99.49%
- **RUL MAE:** 14.87 hours
- **Total Projected Savings:** £116,000
- **Cost Reduction:** 70%
- **Downtime Reduction:** 75%
- **Failure Avoidance Rate:** 89%
- **Anomaly Rate:** 14.67%
- **Total Downtime Hours:** 106.70

---

### Key Insights
1. Pump wear is the most critical failure mode, accounting for 46.56% of projected savings and the highest average repair cost at £28,304 per event.
2. The system detects degradation an average of 12.7 days before failure, giving maintenance teams a reliable 48-hour early warning window.
3. A 56.59% reactive cost ratio before deployment confirms the business was spending more on emergency repairs than planned maintenance.
4. Cylinder drift, though occurring only once, produced the longest single-incident downtime at 13.10 hours per event — indicating high individual risk.
5. All 9 critical machines reached RUL = 0 by February 2024, validating the model's degradation predictions against actual failure label records.
6. HPU_10 remained healthy throughout the entire observation period, serving as a confirmed baseline for normal operating conditions.
7. Sensor dropout events (2,590 rows, 0.30% of data) were concentrated in specific periods — flagged separately rather than imputing values that could distort trend calculations.
8. Higher vibration readings on the X axis (vibration_x_g) emerged as the top feature in the RUL regressor, confirming its role as the earliest degradation signal.

---

### Executive Summary
This predictive maintenance system analysed 864,000 sensor readings across 10 Bosch Rexroth hydraulic power units, identifying £116,000 in projected savings by detecting four failure modes — pump wear, valve leakage, contamination, and cylinder drift — before they occur. With a previous reactive cost ratio of 56.59% and 106.70 total downtime hours, the business was heavily reliant on emergency repairs. The deployed ML pipeline now delivers a confirmed 70% cost reduction and 75% downtime reduction, exceeding both KPI targets, while pump wear alone — the costliest failure at £28,304 per event — accounts for nearly half of all projected savings.

---

### Recommendations
* Prioritise pump wear inspections across all HPUs — it accounts for the highest repair cost and nearly half of all projected savings.
* Implement the 48-hour early warning protocol as standard operating procedure across the maintenance team.
* Reduce reactive maintenance dependency below the current 56.59% ratio by scheduling preventive interventions at the first sign of RUL decline.
* Investigate cylinder drift more closely — a single event caused 13.10 hours of downtime, suggesting disproportionate operational risk.
* Extend the monitoring period to 12 months post-deployment for a statistically valid downtime reduction comparison.
* Use vibration_x_g as the primary early warning sensor — it is the strongest predictor of RUL decline across all machines.

---

### Live Links

| Resource | Link |
|----------|------|
| 🌐 Streamlit Live App | [![Live App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit)](https://bosch-predictive-maintenance-heveurb37app8889t9dgrus.streamlit.app/) |
| ☁️ Render Pipeline | [![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render)](https://render.com) |
| 🗄️ Neon Database | [![Neon](https://img.shields.io/badge/Neon-PostgreSQL-00E699?logo=postgresql)](https://neon.tech) |
| 💻 GitHub Repository | [![GitHub](https://img.shields.io/badge/GitHub-View%20Code-181717?logo=github)](https://github.com/tundesta-data/Bosch-Predictive-Maintenance) |

*By Tunde Adebayo | Data Analytics Consultant*
