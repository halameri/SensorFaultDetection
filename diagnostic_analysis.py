"""
DIAGNOSTIC ANALYSIS - Understanding Sensor Data Distribution
==============================================================
This script analyzes the raw data to understand:
1. What % of sensors are actually "Normal" (low error rate)
2. Whether AC/Humidity/Rain patterns exist
3. Error rate distributions
4. Why certain patterns aren't detected
"""

import pandas as pd
import numpy as np

print("="*80)
print("DIAGNOSTIC ANALYSIS OF SENSOR DATA")
print("="*80)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv("clean_joined_dataset.csv", encoding='utf-8-sig')
print(f"Total records: {len(df):,}")

# Calculate error columns (same as main script)
df["Signed_Error"] = df["Actual_Diff"] - df["Expected_Diff"]
df["Abs_Error"] = df["Signed_Error"].abs()

# Filter extreme outliers (same as main script)
extreme_outlier_mask = (df["Abs_Error"] > 100) | (df["Sensor_Temperature"].abs() > 100)
n_extreme = extreme_outlier_mask.sum()
print(f"Filtering {n_extreme:,} extreme outliers")
df = df[~extreme_outlier_mask].copy()
print(f"Cleaned records: {len(df):,}")

# Create error flags
df["Is_Error"] = df["Abs_Error"] >= 4
df["Is_High_Error"] = df["Signed_Error"] >= 4
df["Is_Low_Error"] = df["Signed_Error"] <= -4

print("\n" + "="*80)
print("1. OVERALL ERROR DISTRIBUTION")
print("="*80)
error_rate = df["Is_Error"].mean()
print(f"Overall Error Rate: {error_rate:.2%}")
print(f"High Error Rate: {df['Is_High_Error'].mean():.2%}")
print(f"Low Error Rate: {df['Is_Low_Error'].mean():.2%}")

print("\n" + "="*80)
print("2. SENSOR-LEVEL ANALYSIS (Potential 'Normal' Sensors)")
print("="*80)

sensor_stats = df.groupby('Sensor_Code').agg({
    'Is_Error': 'mean',
    'Is_High_Error': 'mean',
    'Is_Low_Error': 'mean',
    'Sensor_Temperature': 'count'
}).rename(columns={
    'Is_Error': 'Error_Rate',
    'Is_High_Error': 'High_Error_Rate',
    'Is_Low_Error': 'Low_Error_Rate',
    'Sensor_Temperature': 'Reading_Count'
})

# Classify sensors
sensor_stats['Classification'] = 'Unknown'
sensor_stats.loc[sensor_stats['Error_Rate'] < 0.01, 'Classification'] = 'Normal (<1% error)'
sensor_stats.loc[(sensor_stats['Error_Rate'] >= 0.01) & (sensor_stats['Error_Rate'] < 0.05), 'Classification'] = 'Low Error (1-5%)'
sensor_stats.loc[(sensor_stats['Error_Rate'] >= 0.05) & (sensor_stats['Error_Rate'] < 0.20), 'Classification'] = 'Moderate Error (5-20%)'
sensor_stats.loc[sensor_stats['Error_Rate'] >= 0.20, 'Classification'] = 'High Error (>20%)'

print("\nSensor Classification by Error Rate:")
print(sensor_stats['Classification'].value_counts().sort_index())
print(f"\nPercentage of 'Normal' sensors (<1% error): {(sensor_stats['Error_Rate'] < 0.01).sum() / len(sensor_stats) * 100:.1f}%")
print(f"Percentage of Low Error sensors (1-5%): {((sensor_stats['Error_Rate'] >= 0.01) & (sensor_stats['Error_Rate'] < 0.05)).sum() / len(sensor_stats) * 100:.1f}%")

# Show error rate distribution
print("\nError Rate Percentiles:")
print(f"  10th percentile: {sensor_stats['Error_Rate'].quantile(0.10):.2%}")
print(f"  25th percentile: {sensor_stats['Error_Rate'].quantile(0.25):.2%}")
print(f"  50th percentile (median): {sensor_stats['Error_Rate'].quantile(0.50):.2%}")
print(f"  75th percentile: {sensor_stats['Error_Rate'].quantile(0.75):.2%}")
print(f"  90th percentile: {sensor_stats['Error_Rate'].quantile(0.90):.2%}")

print("\n" + "="*80)
print("3. OPERATIONAL STATE ANALYSIS (FCB/AC)")
print("="*80)

# FCB analysis
fcb_on = df["FCB_On"].sum()
fcb_off = df["FCB_Off"].sum()
fcb_total = fcb_on + fcb_off
print(f"\nFCB State Distribution:")
print(f"  FCB On:  {fcb_on:,} readings ({fcb_on/fcb_total*100:.1f}%)")
print(f"  FCB Off: {fcb_off:,} readings ({fcb_off/fcb_total*100:.1f}%)")

fcb_on_err_rate = df[df["FCB_On"]]["Is_Error"].mean()
fcb_off_err_rate = df[df["FCB_Off"]]["Is_Error"].mean()
print(f"\nFCB Error Rates:")
print(f"  Error rate when FCB On:  {fcb_on_err_rate:.2%}")
print(f"  Error rate when FCB Off: {fcb_off_err_rate:.2%}")

# AC analysis
ac_on = df["AC_On"].sum()
ac_off = df["AC_Off"].sum()
ac_total = ac_on + ac_off
print(f"\nAC State Distribution:")
print(f"  AC On:  {ac_on:,} readings ({ac_on/ac_total*100:.1f}%)")
print(f"  AC Off: {ac_off:,} readings ({ac_off/ac_total*100:.1f}%)")

ac_on_err_rate = df[df["AC_On"]]["Is_Error"].mean()
ac_off_err_rate = df[df["AC_Off"]]["Is_Error"].mean()
print(f"\nAC Error Rates:")
print(f"  Error rate when AC On:  {ac_on_err_rate:.2%}")
print(f"  Error rate when AC Off: {ac_off_err_rate:.2%}")

# Check for AC-specific low error pattern
ac_on_low_err = df[df["AC_On"]]["Is_Low_Error"].mean()
ac_off_low_err = df[df["AC_Off"]]["Is_Low_Error"].mean()
print(f"\nAC Low Error Pattern Check:")
print(f"  Low error rate when AC On:  {ac_on_low_err:.2%}")
print(f"  Low error rate when AC Off: {ac_off_low_err:.2%}")
if ac_on_low_err > ac_off_low_err * 2:
    print("  ✓ AC-related low error pattern EXISTS in data!")
else:
    print("  ✗ No strong AC-related low error pattern")

print("\n" + "="*80)
print("4. ENVIRONMENTAL PATTERN ANALYSIS")
print("="*80)

# Humidity analysis
high_humidity = df["Met_NEM"] > 90
print(f"\nHigh Humidity (>90%) Distribution:")
print(f"  Readings with high humidity: {high_humidity.sum():,} ({high_humidity.mean()*100:.1f}%)")

if high_humidity.sum() > 0:
    high_hum_err_rate = df[high_humidity]["Is_Error"].mean()
    normal_hum_err_rate = df[~high_humidity]["Is_Error"].mean()
    print(f"  Error rate at high humidity: {high_hum_err_rate:.2%}")
    print(f"  Error rate at normal humidity: {normal_hum_err_rate:.2%}")
    if high_hum_err_rate > normal_hum_err_rate * 1.5:
        print("  ✓ Humidity-related error pattern EXISTS!")
    else:
        print("  ✗ No strong humidity-related pattern")

# Rain analysis
rain_mask = df["Weather_Simple"].isin(["Rain", "Snow"])
print(f"\nRainy Weather Distribution:")
print(f"  Readings during rain/snow: {rain_mask.sum():,} ({rain_mask.mean()*100:.1f}%)")

if rain_mask.sum() > 0:
    rain_err_rate = df[rain_mask]["Is_Error"].mean()
    clear_err_rate = df[~rain_mask]["Is_Error"].mean()
    print(f"  Error rate during rain: {rain_err_rate:.2%}")
    print(f"  Error rate during clear: {clear_err_rate:.2%}")
    if rain_err_rate > clear_err_rate * 1.5:
        print("  ✓ Rain-related error pattern EXISTS!")
    else:
        print("  ✗ No strong rain-related pattern")

# Daytime analysis
daytime = df["Is_Daytime"]
print(f"\nDaytime Distribution:")
print(f"  Daytime readings: {daytime.sum():,} ({daytime.mean()*100:.1f}%)")
day_high_err = df[daytime]["Is_High_Error"].mean()
night_high_err = df[~daytime]["Is_High_Error"].mean()
print(f"  High error rate during day: {day_high_err:.2%}")
print(f"  High error rate during night: {night_high_err:.2%}")
if day_high_err > night_high_err * 1.5:
    print("  ✓ Daytime high error pattern EXISTS!")
else:
    print("  ✗ No strong daytime pattern")

print("\n" + "="*80)
print("5. CONTINUOUSLY HIGH/LOW SENSOR DETECTION")
print("="*80)

# Group by sensor and check for continuously high/low patterns
errors_only = df[df["Is_Error"]].copy()
if len(errors_only) > 0:
    sensor_error_stats = errors_only.groupby('Sensor_Code').agg({
        'Is_High_Error': 'mean',
        'Is_Low_Error': 'mean',
        'Sensor_Code': 'count'
    }).rename(columns={'Sensor_Code': 'Error_Count'})

    # Sensors with >80% error rate AND >90% high errors
    continuously_high = sensor_stats[(sensor_stats['Error_Rate'] > 0.80) &
                                     (sensor_stats['High_Error_Rate'] / sensor_stats['Error_Rate'] > 0.90)]

    # Sensors with >80% error rate AND >90% low errors
    continuously_low = sensor_stats[(sensor_stats['Error_Rate'] > 0.80) &
                                    (sensor_stats['Low_Error_Rate'] / sensor_stats['Error_Rate'] > 0.90)]

    print(f"Sensors with 'Continuously High' pattern: {len(continuously_high)} ({len(continuously_high)/len(sensor_stats)*100:.1f}%)")
    print(f"Sensors with 'Continuously Low' pattern: {len(continuously_low)} ({len(continuously_low)/len(sensor_stats)*100:.1f}%)")

    if len(continuously_high) > 0:
        print(f"  ✓ Surekli_Yuksek pattern EXISTS in {len(continuously_high)} sensors")
    else:
        print(f"  ✗ No Surekli_Yuksek pattern found")

print("\n" + "="*80)
print("6. RECOMMENDATIONS")
print("="*80)

# Calculate what threshold would capture "normal" sensors
normal_threshold_1pct = (sensor_stats['Error_Rate'] < 0.01).sum()
normal_threshold_5pct = (sensor_stats['Error_Rate'] < 0.05).sum()
normal_threshold_10pct = (sensor_stats['Error_Rate'] < 0.10).sum()

print(f"\n💡 Normal Classification Thresholds:")
print(f"  < 1% error:  {normal_threshold_1pct} sensors ({normal_threshold_1pct/len(sensor_stats)*100:.1f}%)")
print(f"  < 5% error:  {normal_threshold_5pct} sensors ({normal_threshold_5pct/len(sensor_stats)*100:.1f}%)")
print(f"  < 10% error: {normal_threshold_10pct} sensors ({normal_threshold_10pct/len(sensor_stats)*100:.1f}%)")

print("\n💡 Suggested Actions:")
if normal_threshold_1pct / len(sensor_stats) > 0.10:
    print("  ✓ Increase Normal threshold from <1% to <5% to capture more normal sensors")
if normal_threshold_1pct / len(sensor_stats) < 0.05:
    print("  ⚠ Very few sensors have <1% error - consider <5% or <10% as 'Normal'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
