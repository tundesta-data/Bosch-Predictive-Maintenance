# predict.py
# Runs ML predictions and writes to PostgreSQL
# Scheduled to run automatically in the cloud

import pandas as pd
import numpy as np
import joblib
import os
from sqlalchemy import create_engine

print("Starting prediction run...")

# ── Database connection ──────────────────────────────────────
# Uses environment variable for security
# Never put your real password in code
DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable not set"
    )

engine = create_engine(DATABASE_URL)
print("Connected to database")

# ── Load tables ──────────────────────────────────────────────
sensor = pd.read_sql(
    'SELECT * FROM sensor_telemetry', engine
)
equipment = pd.read_sql(
    'SELECT * FROM equipment_master', engine
)
print(f"Loaded {len(sensor):,} sensor rows")

# ── Clean data ───────────────────────────────────────────────
sensor['timestamp'] = pd.to_datetime(
    sensor['timestamp']
)
sensor['failure_mode'] = (
    sensor['failure_mode'].fillna('No Failure')
)

for col in [
    'pressure_bar','temp_celsius','flow_lpm',
    'vibration_x_g','vibration_y_g','pump_rpm',
    'rul_hours','is_anomaly','is_sensor_dropout',
    'day_of_week'
]:
    sensor[col] = pd.to_numeric(
        sensor[col], errors='coerce'
    )

sensor = sensor.dropna(
    subset=['rul_hours']
).sort_values(
    ['machine_id','timestamp']
).reset_index(drop=True)

# ── Build features ───────────────────────────────────────────
for col, name in [
    ('pressure_bar','pressure'),
    ('temp_celsius','temp'),
    ('vibration_x_g','vibration'),
    ('flow_lpm','flow')
]:
    sensor[f'{name}_avg_10'] = (
        sensor.groupby('machine_id')[col]
        .transform(
            lambda x: x.rolling(
                10, min_periods=1
            ).mean()
        )
    )
    sensor[f'{name}_std_10'] = (
        sensor.groupby('machine_id')[col]
        .transform(
            lambda x: x.rolling(
                10, min_periods=1
            ).std().fillna(0)
        )
    )

sensor['hour']      = sensor['timestamp'].dt.hour
sensor['month_num'] = sensor['timestamp'].dt.month
sensor['shift_num'] = sensor['shift'].map(
    {'Night':0,'Morning':1,'Day':2,'Evening':3}
).fillna(0).astype(int)

sensor['pressure_danger']  = (sensor['pressure_bar'] > 130).astype(int)
sensor['temp_danger']       = (sensor['temp_celsius'] > 80).astype(int)
sensor['vibration_danger']  = (sensor['vibration_x_g'] > 0.8).astype(int)
sensor['flow_danger']       = (sensor['flow_lpm'] < 60).astype(int)
sensor['total_dangers']     = (
    sensor['pressure_danger'] +
    sensor['temp_danger'] +
    sensor['vibration_danger'] +
    sensor['flow_danger']
)
sensor['rul_change'] = (
    sensor.groupby('machine_id')['rul_hours']
    .transform(lambda x: x.diff().fillna(0))
)

# ── Feature lists ────────────────────────────────────────────
input_features = [
    'pressure_bar','temp_celsius','flow_lpm',
    'vibration_x_g','vibration_y_g','pump_rpm',
    'is_anomaly','is_sensor_dropout',
    'pressure_avg_10','temp_avg_10',
    'vibration_avg_10','flow_avg_10',
    'pressure_std_10','temp_std_10',
    'vibration_std_10','flow_std_10',
    'hour','shift_num','total_dangers','rul_change'
]

improved_features = [
    'pressure_bar','temp_celsius','flow_lpm',
    'vibration_x_g','vibration_y_g','pump_rpm',
    'is_anomaly','is_sensor_dropout',
    'pressure_avg_10','temp_avg_10',
    'vibration_avg_10','flow_avg_10',
    'pressure_std_10','temp_std_10',
    'vibration_std_10','flow_std_10',
    'rul_avg_60','rul_drop_rate',
    'pressure_excess','danger_hours',
    'total_operating_hours',
    'hour','shift_num','total_dangers','month_num'
]

# ── Load models ──────────────────────────────────────────────
rul_model_v2 = joblib.load('rul_model.pkl')
clf_model    = joblib.load('failure_model.pkl')
le           = joblib.load('label_encoder.pkl')
print("Models loaded")

# ── Latest reading per machine ───────────────────────────────
latest = (
    sensor.sort_values('timestamp')
    .groupby('machine_id').last()
    .reset_index()
)

# ── Build improved features ──────────────────────────────────
rul_avg = (
    sensor.sort_values('timestamp')
    .groupby('machine_id')['rul_hours']
    .apply(
        lambda x: x.rolling(
            60, min_periods=1
        ).mean().iloc[-1]
    )
    .reset_index()
    .rename(columns={'rul_hours':'rul_avg_60'})
)

rul_drop = (
    sensor.sort_values('timestamp')
    .groupby('machine_id')['rul_hours']
    .apply(lambda x: x.diff().mean())
    .reset_index()
    .rename(columns={'rul_hours':'rul_drop_rate'})
)

latest['pressure_excess'] = (
    latest['pressure_bar'] - 100
).clip(lower=0)
latest['month_num']   = pd.to_datetime(
    latest['timestamp']
).dt.month
latest['rul_change']  = 0.0

danger = (
    sensor.groupby('machine_id')['rul_hours']
    .apply(lambda x: (x < 48).sum())
    .reset_index()
    .rename(columns={'rul_hours':'danger_hours'})
)

total_hrs = (
    sensor.groupby('machine_id').size()
    .reset_index()
    .rename(columns={0:'total_operating_hours'})
)

latest = latest.merge(rul_avg, on='machine_id', how='left')
latest = latest.merge(rul_drop, on='machine_id', how='left')
latest = latest.merge(danger, on='machine_id', how='left')
latest = latest.merge(total_hrs, on='machine_id', how='left')

all_cols = list(set(input_features + improved_features))
for col in all_cols:
    if col not in latest.columns:
        latest[col] = 0.0
latest[all_cols] = latest[all_cols].fillna(0)

# ── Predictions ──────────────────────────────────────────────
latest['predicted_rul_hours'] = (
    rul_model_v2.predict(latest[improved_features])
).round(2)

latest['predicted_failure_mode'] = le.inverse_transform(
    clf_model.predict(latest[input_features])
)

proba    = clf_model.predict_proba(latest[input_features])
proba_df = pd.DataFrame(
    proba, columns=le.classes_,
    index=latest['machine_id'].values
)
non_normal = [c for c in le.classes_ if c != 'No Failure']

latest['likely_failure_mode'] = (
    proba_df[non_normal].idxmax(axis=1).values
)
latest['failure_probability_pct'] = (
    proba_df[non_normal].max(axis=1).values * 100
).round(2)
latest['no_failure_probability_pct'] = (
    proba_df['No Failure'].values * 100
).round(2)

def classify_risk(rul):
    if rul <= 24:    return 'CRITICAL'
    elif rul <= 48:  return 'HIGH RISK'
    elif rul <= 72:  return 'ELEVATED'
    elif rul <= 168: return 'MODERATE'
    else:            return 'HEALTHY'

def get_action(rul):
    if rul <= 24:    return 'STOP MACHINE NOW'
    elif rul <= 48:  return 'CALL MAINTENANCE NOW'
    elif rul <= 72:  return 'SCHEDULE INSPECTION'
    elif rul <= 168: return 'MONITOR CLOSELY'
    else:            return 'NORMAL OPERATIONS'

def format_countdown(hours):
    if hours <= 0: return 'ALREADY SHUT DOWN'
    s = int(hours * 3600)
    d = s // 86400
    h = (s % 86400) // 3600
    m = (s % 3600) // 60
    return f"{d:02d}d {h:02d}h {m:02d}min"

def urgency_window(hours):
    if hours <= 0:    return 'ALREADY SHUT DOWN'
    elif hours <= 12: return '00:00 to 12:00'
    elif hours <= 24: return '12:00 to 24:00'
    elif hours <= 48: return '24:00 to 48:00'
    elif hours <= 72: return '48:00 to 72:00'
    else:             return '72:00 PLUS'

def shutdown_action(hours):
    if hours <= 0:    return 'INSPECT NOW'
    elif hours <= 12: return 'STOP MACHINE NOW'
    elif hours <= 24: return 'EMERGENCY MAINTENANCE TODAY'
    elif hours <= 48: return 'PLAN MAINTENANCE WITHIN 48H'
    elif hours <= 72: return 'QUEUE FOR NEXT WINDOW'
    else:             return 'MONITOR ONLY'

latest['risk_level']         = latest['predicted_rul_hours'].apply(classify_risk)
latest['recommended_action'] = latest['predicted_rul_hours'].apply(get_action)
latest['shutdown_countdown'] = latest['predicted_rul_hours'].apply(format_countdown)
latest['urgency_window']     = latest['predicted_rul_hours'].apply(urgency_window)
latest['shutdown_action']    = latest['predicted_rul_hours'].apply(shutdown_action)

# ── Write to PostgreSQL ──────────────────────────────────────
output = latest[[
    'machine_id','timestamp',
    'predicted_rul_hours',
    'shutdown_countdown',
    'urgency_window',
    'shutdown_action',
    'predicted_failure_mode',
    'likely_failure_mode',
    'failure_probability_pct',
    'no_failure_probability_pct',
    'risk_level',
    'recommended_action',
    'pressure_bar','temp_celsius',
    'vibration_x_g','flow_lpm',
    'pump_rpm','is_anomaly',
    'total_dangers'
]].copy()

output['prediction_generated_at'] = pd.Timestamp.now()
output['predicted_rul_hours'] = output['predicted_rul_hours'].round(2)

output.to_sql(
    'ml_predictions',
    engine,
    if_exists='replace',
    index=False
)

print(f"SUCCESS — {len(output)} predictions written")
print(f"Time: {pd.Timestamp.now()}")