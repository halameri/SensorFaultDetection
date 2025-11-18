#!/usr/bin/env python3
"""
Diagnose why FCB_Off_Yuksek and AC_On_Dusuk patterns are not being detected
"""

import pandas as pd
import numpy as np
from clustering_approach_sensor_faults import extract_window_features

print("="*80)
print("DIAGNOSTIC: FCB and AC Pattern Analysis")
print("="*80)

# Load and extract features
print("\nLoading data and extracting features...")
df_raw = pd.read_csv('clean_joined_dataset.csv')
df_features = extract_window_features(df_raw)
print(f"Total windows: {len(df_features)}")

# Check FCB-related features
print("\n" + "="*80)
print("FCB ANALYSIS - Why FCB_Off_Yuksek might not be detected")
print("="*80)

# User's rule: >80% of errors when FCB off AND <20% error when FCB on
fcb_off_err_frac = df_features['FCB_Off_Error_Fraction']
fcb_on_err_rate = df_features['FCB_On_Error_Rate']
high_err_frac = df_features['High_Error_Fraction']
error_rate = df_features['Error_Rate']

print("\n1. FCB_Off_Error_Fraction distribution (need >0.80 for detection):")
print(f"   Min: {fcb_off_err_frac.min():.2%}")
print(f"   25%: {fcb_off_err_frac.quantile(0.25):.2%}")
print(f"   50%: {fcb_off_err_frac.quantile(0.50):.2%}")
print(f"   75%: {fcb_off_err_frac.quantile(0.75):.2%}")
print(f"   Max: {fcb_off_err_frac.max():.2%}")

n_above_80 = (fcb_off_err_frac >= 0.80).sum()
n_above_70 = (fcb_off_err_frac >= 0.70).sum()
n_above_60 = (fcb_off_err_frac >= 0.60).sum()
print(f"\n   Windows with FCB_Off_Error_Fraction >= 80%: {n_above_80} ({n_above_80/len(df_features)*100:.1f}%)")
print(f"   Windows with FCB_Off_Error_Fraction >= 70%: {n_above_70} ({n_above_70/len(df_features)*100:.1f}%)")
print(f"   Windows with FCB_Off_Error_Fraction >= 60%: {n_above_60} ({n_above_60/len(df_features)*100:.1f}%)")

print("\n2. FCB_On_Error_Rate distribution (need <0.20 for detection):")
print(f"   Min: {fcb_on_err_rate.min():.2%}")
print(f"   25%: {fcb_on_err_rate.quantile(0.25):.2%}")
print(f"   50%: {fcb_on_err_rate.quantile(0.50):.2%}")
print(f"   75%: {fcb_on_err_rate.quantile(0.75):.2%}")
print(f"   Max: {fcb_on_err_rate.max():.2%}")

n_below_20 = (fcb_on_err_rate < 0.20).sum()
n_below_30 = (fcb_on_err_rate < 0.30).sum()
n_below_40 = (fcb_on_err_rate < 0.40).sum()
print(f"\n   Windows with FCB_On_Error_Rate < 20%: {n_below_20} ({n_below_20/len(df_features)*100:.1f}%)")
print(f"   Windows with FCB_On_Error_Rate < 30%: {n_below_30} ({n_below_30/len(df_features)*100:.1f}%)")
print(f"   Windows with FCB_On_Error_Rate < 40%: {n_below_40} ({n_below_40/len(df_features)*100:.1f}%)")

# Combined check for FCB pattern
yuksek_hata_orani = high_err_frac * error_rate
fcb_on_count = df_features['FCB_On_Count']

potential_fcb = (
    (yuksek_hata_orani > 0.01) &
    (yuksek_hata_orani < 0.80) &
    (fcb_on_count > 20) &
    (fcb_off_err_frac >= 0.80) &
    (fcb_on_err_rate < 0.20)
)
print(f"\n3. Windows meeting STRICT FCB rule (>80% off, <20% on): {potential_fcb.sum()}")

potential_fcb_relaxed = (
    (yuksek_hata_orani > 0.01) &
    (yuksek_hata_orani < 0.80) &
    (fcb_on_count > 20) &
    (fcb_off_err_frac >= 0.60) &
    (fcb_on_err_rate < 0.40)
)
print(f"   Windows meeting RELAXED FCB rule (>60% off, <40% on): {potential_fcb_relaxed.sum()}")

# Show some examples
if potential_fcb_relaxed.sum() > 0:
    print("\n   Example windows that could be FCB_Off_Yuksek:")
    examples = df_features[potential_fcb_relaxed].head(5)
    for idx, row in examples.iterrows():
        print(f"   - Sensor {row['Sensor_Code']}: FCB_Off_Err={row['FCB_Off_Error_Fraction']:.0%}, FCB_On_Err={row['FCB_On_Error_Rate']:.0%}")

# Check AC-related features
print("\n" + "="*80)
print("AC ANALYSIS - Why AC_On_Dusuk might not be detected")
print("="*80)

# User's rule: >80% of errors when AC on AND <20% error when AC off
ac_on_err_frac = df_features['AC_On_Error_Fraction']
ac_off_err_rate = df_features['AC_Off_Error_Rate']
low_err_frac = df_features['Low_Error_Fraction']

print("\n1. AC_On_Error_Fraction distribution (need >0.80 for detection):")
print(f"   Min: {ac_on_err_frac.min():.2%}")
print(f"   25%: {ac_on_err_frac.quantile(0.25):.2%}")
print(f"   50%: {ac_on_err_frac.quantile(0.50):.2%}")
print(f"   75%: {ac_on_err_frac.quantile(0.75):.2%}")
print(f"   Max: {ac_on_err_frac.max():.2%}")

n_above_80 = (ac_on_err_frac >= 0.80).sum()
n_above_70 = (ac_on_err_frac >= 0.70).sum()
n_above_60 = (ac_on_err_frac >= 0.60).sum()
print(f"\n   Windows with AC_On_Error_Fraction >= 80%: {n_above_80} ({n_above_80/len(df_features)*100:.1f}%)")
print(f"   Windows with AC_On_Error_Fraction >= 70%: {n_above_70} ({n_above_70/len(df_features)*100:.1f}%)")
print(f"   Windows with AC_On_Error_Fraction >= 60%: {n_above_60} ({n_above_60/len(df_features)*100:.1f}%)")

print("\n2. AC_Off_Error_Rate distribution (need <0.20 for detection):")
print(f"   Min: {ac_off_err_rate.min():.2%}")
print(f"   25%: {ac_off_err_rate.quantile(0.25):.2%}")
print(f"   50%: {ac_off_err_rate.quantile(0.50):.2%}")
print(f"   75%: {ac_off_err_rate.quantile(0.75):.2%}")
print(f"   Max: {ac_off_err_rate.max():.2%}")

n_below_20 = (ac_off_err_rate < 0.20).sum()
n_below_30 = (ac_off_err_rate < 0.30).sum()
n_below_40 = (ac_off_err_rate < 0.40).sum()
print(f"\n   Windows with AC_Off_Error_Rate < 20%: {n_below_20} ({n_below_20/len(df_features)*100:.1f}%)")
print(f"   Windows with AC_Off_Error_Rate < 30%: {n_below_30} ({n_below_30/len(df_features)*100:.1f}%)")
print(f"   Windows with AC_Off_Error_Rate < 40%: {n_below_40} ({n_below_40/len(df_features)*100:.1f}%)")

# Combined check for AC pattern
dusuk_hata_orani = low_err_frac * error_rate
ac_off_count = df_features['AC_Off_Count']

potential_ac = (
    (dusuk_hata_orani > 0.01) &
    (dusuk_hata_orani < 0.80) &
    (ac_off_count > 20) &
    (ac_on_err_frac >= 0.80) &
    (ac_off_err_rate < 0.20)
)
print(f"\n3. Windows meeting STRICT AC rule (>80% on, <20% off): {potential_ac.sum()}")

potential_ac_relaxed = (
    (dusuk_hata_orani > 0.01) &
    (dusuk_hata_orani < 0.80) &
    (ac_off_count > 20) &
    (ac_on_err_frac >= 0.60) &
    (ac_off_err_rate < 0.40)
)
print(f"   Windows meeting RELAXED AC rule (>60% on, <40% off): {potential_ac_relaxed.sum()}")

# Show some examples
if potential_ac_relaxed.sum() > 0:
    print("\n   Example windows that could be AC_On_Dusuk:")
    examples = df_features[potential_ac_relaxed].head(5)
    for idx, row in examples.iterrows():
        print(f"   - Sensor {row['Sensor_Code']}: AC_On_Err={row['AC_On_Error_Fraction']:.0%}, AC_Off_Err={row['AC_Off_Error_Rate']:.0%}")

# Check FCB/AC count distributions
print("\n" + "="*80)
print("FCB/AC STATE COUNTS - Are there enough samples?")
print("="*80)

print("\n1. FCB_On_Count distribution (need >20 for detection):")
print(f"   Min: {fcb_on_count.min():.0f}")
print(f"   25%: {fcb_on_count.quantile(0.25):.0f}")
print(f"   50%: {fcb_on_count.quantile(0.50):.0f}")
print(f"   75%: {fcb_on_count.quantile(0.75):.0f}")
print(f"   Max: {fcb_on_count.max():.0f}")

n_fcb_enough = (fcb_on_count > 20).sum()
print(f"   Windows with FCB_On_Count > 20: {n_fcb_enough} ({n_fcb_enough/len(df_features)*100:.1f}%)")

print("\n2. AC_Off_Count distribution (need >20 for detection):")
print(f"   Min: {ac_off_count.min():.0f}")
print(f"   25%: {ac_off_count.quantile(0.25):.0f}")
print(f"   50%: {ac_off_count.quantile(0.50):.0f}")
print(f"   75%: {ac_off_count.quantile(0.75):.0f}")
print(f"   Max: {ac_off_count.max():.0f}")

n_ac_enough = (ac_off_count > 20).sum()
print(f"   Windows with AC_Off_Count > 20: {n_ac_enough} ({n_ac_enough/len(df_features)*100:.1f}%)")

# Summary
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

if potential_fcb.sum() == 0:
    if potential_fcb_relaxed.sum() > 0:
        print("\n⚠️  FCB_Off_Yuksek: Pattern EXISTS but doesn't meet 80% threshold")
        print("    → Consider lowering threshold to 60-70%")
    elif n_fcb_enough < len(df_features) * 0.5:
        print("\n⚠️  FCB_Off_Yuksek: Not enough FCB_On samples (FCB rarely runs)")
        print("    → Need more data when FCB is ON to verify pattern")
    else:
        print("\n❌ FCB_Off_Yuksek: Pattern likely doesn't exist in this dataset")
        print("    → Errors are evenly distributed between FCB states")

if potential_ac.sum() == 0:
    if potential_ac_relaxed.sum() > 0:
        print("\n⚠️  AC_On_Dusuk: Pattern EXISTS but doesn't meet 80% threshold")
        print("    → Consider lowering threshold to 60-70%")
    elif n_ac_enough < len(df_features) * 0.5:
        print("\n⚠️  AC_On_Dusuk: Not enough AC_Off samples (AC almost always on)")
        print("    → Need more data when AC is OFF to verify pattern")
    else:
        print("\n❌ AC_On_Dusuk: Pattern likely doesn't exist in this dataset")
        print("    → Errors are evenly distributed between AC states")

print("\n" + "="*80)
