# anomalies-detection-cold-storage-facility
Here we analyze 8 different sensor data for a cold storage facility using ML algorithm to detect anomalies in temperature to alert our monitoring system that results in less food loss. Detecting anomalies also helps to keep track of maintenance schedule if attention is needed. Also helps prepare if it's a fault or false positive.  

Cold Storage Sensor Dataset:
cold_storage_sensor_data.csv — 289 rows, 15-min intervals, 3 days (Jul 1–3, 2026).


**cold_storage_sensor_data_labeled.csv — same data plus `is_anomaly` (0/1) and `anomaly_type` columns.Use this only as an answer key to validate what your SQL/Python/ML approach catches — don't feed the labels into your model**

<img width="812" height="360" alt="image" src="https://github.com/user-attachments/assets/266460c8-1e17-4c33-a7a3-710c437700c4" />



Door left open (Jul 1, 19:00–20:30) — gradual temp rise, humidity/CO2/power drift up together (a correlated, multivariate anomaly).
Stuck/frozen sensor (Jul 2, 02:00–05:00) — temperature flatlines at exactly -18.0 (classic "sensor stuck" signature — zero variance).
Compressor vibration spike (Jul 2, 14:00–15:00) — short sharp mechanical fault burst.
Refrigerant leak (Jul 2 21:00 – Jul 3 03:00) — slow, steady pressure decline (a trend/drift anomaly, harder to catch with simple thresholds).
Power surges (Jul 1 11:00, Jul 3 09:15) — single-point spikes (~3x normal), easy point outliers.
CO2 ventilation failure (Jul 3, 06:00–09:00) — sustained ramp-up.
Airflow sensor dropout (Jul 1, 03:00–03:30) — missing values (NaN), a data-quality anomaly.
Humidity outlier glitch (Jul 2, 10:30) — single implausible point (sudden drop to 45%).
Gradual vibration drift (all of Jul 3) — slow upward creep simulating early bearing wear; overlaps with other anomalies to test multivariate detection.
Duplicate timestamp (row ~50) — a logging/ETL glitch, good for a SQL data-quality check rather than a statistical one.
This mix intentionally includes point outliers, contextual anomalies (only anomalous given time-of-day/door state), collective/trend anomalies, and data-quality issues (missing values, duplicates, flatlines) — the range an intermediate data scientist would expect to face.
Suggested workflow
SQL (exploration & data-quality checks)
`GROUP BY` hour/day to get baseline stats (AVG, STDDEV, MIN/MAX) per sensor.
Window functions (`LAG`/`LEAD`) to compute point-to-point deltas and flag jumps beyond N standard deviations.
Simple threshold rules (e.g., `WHERE refrigerant_pressure_psi < 24`) to catch out-of-range values.
`COUNT(*) ... HAVING COUNT(*) > 1 GROUP BY timestamp` to catch the duplicate-timestamp issue.
`WHERE column IS NULL` to catch the dropout.
Python (EDA + feature engineering)
`pandas` for loading, resampling, rolling mean/std, z-scores per sensor.
Rolling standard-deviation-based flags (e.g., z-score > 3) for point outliers.
Flatline detection: rolling `std() == 0` over a window catches the stuck sensor.
Correlate `door_status` with temperature/humidity/CO2/power to catch contextual anomalies.
Visualize with matplotlib/seaborn (line plots with shaded anomaly regions).
Machine learning (unsupervised, since real-world anomaly labels are rarely available)
`IsolationForest` or `LocalOutlierFactor` (scikit-learn) on the multivariate feature set (scaled) to catch anomalies that look normal on any single sensor but are abnormal jointly (e.g., the door-open event).
`STL` decomposition or a rolling z-score for detecting the slow refrigerant-pressure drift and vibration drift (trend anomalies isolation forest tends to miss).
Use the labeled file's `is_anomaly` column afterward to compute precision/recall and see which anomaly types your method caught vs. missed — a good intermediate-level exercise in comparing rule-based vs. statistical vs. ML approaches.
