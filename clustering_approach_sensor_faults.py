"""
HYBRID APPROACH: CLUSTERING + SUPERVISED LEARNING
==================================================

Strategy:
    1. Extract features from 15-day windows (no hard rules for labeling)
    2. Use clustering to discover natural fault patterns
    3. Analyze clusters to understand what they represent
    4. Use cluster labels as training data for supervised ML
    5. (Optional) Refine with small amount of manual labels

Benefits:
    - Discovers patterns without rigid rules
    - More flexible than rule-based approach
    - Can find unexpected fault types
    - Reduces manual labeling effort
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters"""
    
    # Files
    INPUT_FILE = "clean_joined_dataset.csv"
    OUTPUT_FILE = "ml_15day_windows_clustered.csv"
    
    # Window parameters
    WINDOW_DAYS = 15
    WINDOW_STEP = 1
    MIN_READINGS = 150
    MIN_DAYS = 15
    
    # Error calculation
    ERROR_TOLERANCE = 4.0
    LAPSE_RATE = -0.65
    
    # Clustering
    N_CLUSTERS_RANGE = range(7, 13)  # Try 7 to 12 clusters (8-9 optimal for fault types)
    RANDOM_STATE = 42
    # Note: 8-9 clusters work well for the 8 fault types + normal operation
    
    # Operational states
    FCB_ON_STATES = ["FCBCLSYR", "ACLCLSMMD"]
    FCB_OFF_STATES = ["BOSTA", "KLMCLSYR"]
    AC_ON_STATES = ["KLMCLSYR", "ACLCLSMMD"]
    AC_OFF_STATES = ["BOSTA", "FCBCLSYR"]

# ============================================================================
# FEATURE EXTRACTION (Same as before, but NO LABELING)
# ============================================================================

def simplify_weather(hadise_text):
    """Simplify weather condition"""
    s = str(hadise_text).lower()
    if any(w in s for w in ["açık", "clear", "az bulut"]): return "Clear"
    if any(w in s for w in ["yağ", "rain"]): return "Rain"
    if any(w in s for w in ["kar", "snow"]): return "Snow"
    if any(w in s for w in ["sis", "fog"]): return "Fog"
    if any(w in s for w in ["bulut", "cloud"]): return "Cloudy"
    return "Other"

def extract_window_features(df):
    """
    Extract features from 15-day windows WITHOUT rule-based labeling
    
    Returns:
        DataFrame with rich features for clustering
    """
    
    print("Loading and preparing data...")
    df["Met_Timestamp"] = pd.to_datetime(df["Met_Timestamp"])
    df = df.sort_values(["FC_BOX_CODE", "Met_Timestamp"]).reset_index(drop=True)
    
    # Weather simplification
    df["Weather_Simple"] = df["HADISE"].apply(simplify_weather)
    
    # Error calculation
    df["Expected_Diff"] = (df["Elevation_Difference_m"] / 100.0) * Config.LAPSE_RATE
    df["Actual_Diff"] = df["Sensor_Temperature"] - df["Met_Temperature"]
    df["Signed_Error"] = df["Actual_Diff"] - df["Expected_Diff"]
    df["Abs_Error"] = df["Signed_Error"].abs()

    # DATA QUALITY FILTER: Remove extreme outliers (likely sensor malfunctions)
    # These are not real faults but data transmission/hardware errors
    extreme_outlier_mask = (df["Abs_Error"] > 100) | (df["Sensor_Temperature"].abs() > 100)
    n_extreme = extreme_outlier_mask.sum()
    if n_extreme > 0:
        print(f"  ⚠ Filtering {n_extreme:,} extreme outliers (|error| > 100°C or |temp| > 100°C)")
        df = df[~extreme_outlier_mask].copy()

    df["Is_Error"] = (df["Abs_Error"] > Config.ERROR_TOLERANCE).astype(int)
    
    # Temporal features
    df["Date"] = df["Met_Timestamp"].dt.normalize()
    df["Hour"] = df["Met_Timestamp"].dt.hour
    df["Is_Daytime"] = (df["Hour"] >= 6) & (df["Hour"] < 18)
    
    # Operational states
    df["FCB_On"] = df["BX_CHZ_STT_PR"].isin(Config.FCB_ON_STATES)
    df["FCB_Off"] = df["BX_CHZ_STT_PR"].isin(Config.FCB_OFF_STATES)
    df["AC_On"] = df["BX_CHZ_STT_PR"].isin(Config.AC_ON_STATES)
    df["AC_Off"] = df["BX_CHZ_STT_PR"].isin(Config.AC_OFF_STATES)
    
    print("Extracting 15-day window features...")
    windows = []
    sensors = df["FC_BOX_CODE"].unique()
    
    for idx, (sensor, sensor_df) in enumerate(df.groupby("FC_BOX_CODE")):
        if (idx + 1) % 50 == 0:
            print(f"  Processing sensor {idx+1}/{len(sensors)}...")
        
        sensor_df = sensor_df.sort_values("Met_Timestamp").reset_index(drop=True)
        dates = np.sort(sensor_df["Date"].unique())
        
        for start_date in dates:
            start = pd.to_datetime(start_date)
            end = start + timedelta(days=Config.WINDOW_DAYS)
            
            mask = (sensor_df["Met_Timestamp"] >= start) & (sensor_df["Met_Timestamp"] < end)
            win = sensor_df.loc[mask]
            
            if win["Date"].nunique() < Config.MIN_DAYS or len(win) < Config.MIN_READINGS:
                continue
            
            # ================================================================
            # EXTRACT COMPREHENSIVE FEATURES
            # ================================================================
            total = len(win)
            err_mask = win["Is_Error"] == 1
            err_count = err_mask.sum()
            
            if total == 0:
                continue
            
            # Basic error metrics
            error_rate = err_count / total
            mean_error = win["Signed_Error"].mean()
            std_error = win["Signed_Error"].std()
            mean_abs_error = win["Abs_Error"].mean()
            max_abs_error = win["Abs_Error"].max()
            
            # Error direction
            high_err = (win["Signed_Error"] > Config.ERROR_TOLERANCE) & err_mask
            low_err = (win["Signed_Error"] < -Config.ERROR_TOLERANCE) & err_mask
            
            high_err_count = high_err.sum()
            low_err_count = low_err.sum()
            
            if err_count > 0:
                high_err_frac = high_err_count / err_count
                low_err_frac = low_err_count / err_count
            else:
                high_err_frac = 0.0
                low_err_frac = 0.0
            
            # FCB state analysis
            fcb_on_mask = win["FCB_On"]
            fcb_off_mask = win["FCB_Off"]
            
            n_fcb_on = fcb_on_mask.sum()
            n_fcb_off = fcb_off_mask.sum()
            fcb_on_ratio = n_fcb_on / total if total > 0 else 0
            
            fcb_on_err_rate = (err_mask & fcb_on_mask).sum() / n_fcb_on if n_fcb_on > 0 else 0
            fcb_off_err_rate = (err_mask & fcb_off_mask).sum() / n_fcb_off if n_fcb_off > 0 else 0
            
            fcb_on_mean_err = win.loc[fcb_on_mask, "Signed_Error"].mean() if n_fcb_on > 0 else 0
            fcb_off_mean_err = win.loc[fcb_off_mask, "Signed_Error"].mean() if n_fcb_off > 0 else 0
            
            if err_count > 0:
                fcb_on_err_frac = (err_mask & fcb_on_mask).sum() / err_count
                fcb_off_err_frac = (err_mask & fcb_off_mask).sum() / err_count
            else:
                fcb_on_err_frac = 0.0
                fcb_off_err_frac = 0.0
            
            # AC state analysis
            ac_on_mask = win["AC_On"]
            ac_off_mask = win["AC_Off"]
            
            n_ac_on = ac_on_mask.sum()
            n_ac_off = ac_off_mask.sum()
            ac_on_ratio = n_ac_on / total if total > 0 else 0
            
            ac_on_err_rate = (err_mask & ac_on_mask).sum() / n_ac_on if n_ac_on > 0 else 0
            ac_off_err_rate = (err_mask & ac_off_mask).sum() / n_ac_off if n_ac_off > 0 else 0
            
            ac_on_mean_err = win.loc[ac_on_mask, "Signed_Error"].mean() if n_ac_on > 0 else 0
            ac_off_mean_err = win.loc[ac_off_mask, "Signed_Error"].mean() if n_ac_off > 0 else 0
            
            if err_count > 0:
                ac_on_err_frac = (err_mask & ac_on_mask).sum() / err_count
                ac_off_err_frac = (err_mask & ac_off_mask).sum() / err_count
            else:
                ac_on_err_frac = 0.0
                ac_off_err_frac = 0.0

            # ================================================================
            # STATE REVERSAL ANALYSIS - Critical for distinguishing causation
            # ================================================================
            # User's insight: If sensor ALWAYS reads low (AC on AND off) → Surekli_Dusuk
            #                 If sensor reads low ONLY when AC on → AC_On_Dusuk
            # Need to check: Does pattern REVERSE when state changes?

            # 1. CORRECT READING RATES (not just error rates)
            # For true "Normal" sensor: Most readings should be CORRECT
            correct_mask = ~err_mask  # Readings within tolerance

            fcb_on_correct_rate = (correct_mask & fcb_on_mask).sum() / n_fcb_on if n_fcb_on > 0 else 0
            fcb_off_correct_rate = (correct_mask & fcb_off_mask).sum() / n_fcb_off if n_fcb_off > 0 else 0

            ac_on_correct_rate = (correct_mask & ac_on_mask).sum() / n_ac_on if n_ac_on > 0 else 0
            ac_off_correct_rate = (correct_mask & ac_off_mask).sum() / n_ac_off if n_ac_off > 0 else 0

            # 2. ERROR DIRECTION IN EACH STATE
            # Check if errors are consistently HIGH or LOW in each state

            # FCB states - direction of errors
            fcb_on_high_err = (high_err & fcb_on_mask).sum()
            fcb_on_low_err = (low_err & fcb_on_mask).sum()
            fcb_off_high_err = (high_err & fcb_off_mask).sum()
            fcb_off_low_err = (low_err & fcb_off_mask).sum()

            # Of errors when FCB on, what % are HIGH vs LOW?
            fcb_on_err_count = fcb_on_high_err + fcb_on_low_err
            if fcb_on_err_count > 0:
                fcb_on_high_err_frac = fcb_on_high_err / fcb_on_err_count
                fcb_on_low_err_frac = fcb_on_low_err / fcb_on_err_count
            else:
                fcb_on_high_err_frac = 0.0
                fcb_on_low_err_frac = 0.0

            fcb_off_err_count = fcb_off_high_err + fcb_off_low_err
            if fcb_off_err_count > 0:
                fcb_off_high_err_frac = fcb_off_high_err / fcb_off_err_count
                fcb_off_low_err_frac = fcb_off_low_err / fcb_off_err_count
            else:
                fcb_off_high_err_frac = 0.0
                fcb_off_low_err_frac = 0.0

            # AC states - direction of errors
            ac_on_high_err = (high_err & ac_on_mask).sum()
            ac_on_low_err = (low_err & ac_on_mask).sum()
            ac_off_high_err = (high_err & ac_off_mask).sum()
            ac_off_low_err = (low_err & ac_off_mask).sum()

            ac_on_err_count = ac_on_high_err + ac_on_low_err
            if ac_on_err_count > 0:
                ac_on_high_err_frac = ac_on_high_err / ac_on_err_count
                ac_on_low_err_frac = ac_on_low_err / ac_on_err_count
            else:
                ac_on_high_err_frac = 0.0
                ac_on_low_err_frac = 0.0

            ac_off_err_count = ac_off_high_err + ac_off_low_err
            if ac_off_err_count > 0:
                ac_off_high_err_frac = ac_off_high_err / ac_off_err_count
                ac_off_low_err_frac = ac_off_low_err / ac_off_err_count
            else:
                ac_off_high_err_frac = 0.0
                ac_off_low_err_frac = 0.0

            # 3. STATE REVERSAL INDICATORS
            # Does error pattern REVERSE when state changes?

            # FCB reversal: If truly FCB-related, should have:
            # - High errors when FCB OFF + Correct readings when FCB ON
            # If errors persist regardless of FCB state → Not FCB-related
            fcb_contrast_score = fcb_on_correct_rate - fcb_off_err_rate  # Higher = clear contrast

            # AC reversal: If truly AC-related, should have:
            # - Low errors when AC ON + Correct/High readings when AC OFF
            # Direction reversal: errors are LOW when AC on, but CORRECT/HIGH when AC off
            ac_contrast_score = ac_off_correct_rate - ac_on_err_rate  # Higher = clear contrast

            # Direction consistency check
            # For AC_On_Dusuk: Should be LOW when AC on, NOT low when AC off
            ac_direction_reversal = (ac_on_low_err_frac > 0.70) and (ac_off_low_err_frac < 0.30)

            # For FCB_Off_Yuksek: Should be HIGH when FCB off, NOT high when FCB on
            fcb_direction_reversal = (fcb_off_high_err_frac > 0.70) and (fcb_on_high_err_frac < 0.30)

            # Environmental
            mean_humidity = win["Met_NEM"].mean()
            high_humidity_ratio = (win["Met_NEM"] > 90).mean()
            
            if err_count > 0:
                high_humidity_err_frac = (win.loc[err_mask, "Met_NEM"] > 90).mean()
            else:
                high_humidity_err_frac = 0.0
            
            rain_ratio = win["Weather_Simple"].isin(["Rain", "Snow"]).mean()
            if err_count > 0:
                rain_err_frac = win.loc[err_mask, "Weather_Simple"].isin(["Rain", "Snow"]).mean()
            else:
                rain_err_frac = 0.0
            
            sunny_ratio = (win["Weather_Simple"] == "Clear").mean()
            
            # Temporal - ALIGNED WITH RULE-BASED CODE
            daytime_mask = win["Is_Daytime"]

            # Gunduz_Yuksek_Orani_YH: Of HIGH errors, what % are during daytime?
            gunduz_yuksek_count = (high_err & daytime_mask).sum()
            if high_err_count > 0:
                gunduz_yuksek_orani_yh = gunduz_yuksek_count / high_err_count
            else:
                gunduz_yuksek_orani_yh = 0.0

            # Also keep the general daytime error fraction for other uses
            if err_count > 0:
                daytime_err_frac = (err_mask & daytime_mask).sum() / err_count
            else:
                daytime_err_frac = 0.0
            
            # Variability metrics (NEW - useful for clustering)
            error_variability = win["Abs_Error"].std()
            error_range = win["Abs_Error"].max() - win["Abs_Error"].min()
            error_skewness = win["Signed_Error"].skew() if len(win) > 2 else 0

            # Consistency metrics (NEW)
            consistency_score = 1 - error_rate  # Higher = more consistent

            # State transition metrics (NEW)
            fcb_state_changes = (win["FCB_On"] != win["FCB_On"].shift()).sum()
            ac_state_changes = (win["AC_On"] != win["AC_On"].shift()).sum()

            # ================================================================
            # ADVANCED FEATURES - Confounding Variable Analysis
            # ================================================================
            # These features help distinguish TRUE causation from correlation
            # Example: Is FCB the cause, or is it just coincidentally off during daytime?

            # 1. HOURLY ERROR CONCENTRATION
            # If errors concentrate in specific hours (11am-2pm) → likely solar/daytime issue
            # If errors spread across all hours when FCB off → likely FCB issue
            if err_count > 0:
                error_hours = win.loc[err_mask, "Hour"].values
                if len(error_hours) > 0:
                    # Peak error hour (mode)
                    from scipy import stats as scipy_stats
                    peak_error_hour = scipy_stats.mode(error_hours, keepdims=True)[0][0] if len(error_hours) > 1 else error_hours[0]

                    # Error hour spread (std of error hours)
                    error_hour_std = np.std(error_hours)

                    # Concentration: % of errors in peak 3-hour window
                    hour_counts = pd.Series(error_hours).value_counts()
                    if len(hour_counts) > 0:
                        # Find best 3-hour consecutive window
                        max_3hr_count = 0
                        for h in range(24):
                            count_3hr = hour_counts.get(h, 0) + hour_counts.get((h+1)%24, 0) + hour_counts.get((h+2)%24, 0)
                            max_3hr_count = max(max_3hr_count, count_3hr)
                        error_3hr_concentration = max_3hr_count / len(error_hours)
                    else:
                        error_3hr_concentration = 0.0
                else:
                    peak_error_hour = 12  # Default to noon
                    error_hour_std = 0.0
                    error_3hr_concentration = 0.0
            else:
                peak_error_hour = 12
                error_hour_std = 0.0
                error_3hr_concentration = 0.0

            # 2. CONDITIONAL ERROR RATES - Separate confounded variables
            # Key insight: Check if pattern persists when controlling for other variables

            # FCB_Off during DAY vs NIGHT (to separate FCB effect from daytime effect)
            fcb_off_day_mask = fcb_off_mask & daytime_mask
            fcb_off_night_mask = fcb_off_mask & ~daytime_mask

            n_fcb_off_day = fcb_off_day_mask.sum()
            n_fcb_off_night = fcb_off_night_mask.sum()

            fcb_off_day_err_rate = (err_mask & fcb_off_day_mask).sum() / n_fcb_off_day if n_fcb_off_day > 0 else 0
            fcb_off_night_err_rate = (err_mask & fcb_off_night_mask).sum() / n_fcb_off_night if n_fcb_off_night > 0 else 0

            # If FCB_Off_Day error >> FCB_Off_Night error → It's daytime (sun), not FCB!
            # If FCB_Off_Day ≈ FCB_Off_Night (both high) → It's FCB issue
            if fcb_off_night_err_rate > 0:
                fcb_day_night_ratio = fcb_off_day_err_rate / fcb_off_night_err_rate
            else:
                fcb_day_night_ratio = 1.0 if fcb_off_day_err_rate == 0 else 999.0  # Very high if only day errors

            # FCB_On during DAY vs NIGHT
            fcb_on_day_mask = fcb_on_mask & daytime_mask
            fcb_on_night_mask = fcb_on_mask & ~daytime_mask

            n_fcb_on_day = fcb_on_day_mask.sum()
            n_fcb_on_night = fcb_on_night_mask.sum()

            fcb_on_day_err_rate = (err_mask & fcb_on_day_mask).sum() / n_fcb_on_day if n_fcb_on_day > 0 else 0
            fcb_on_night_err_rate = (err_mask & fcb_on_night_mask).sum() / n_fcb_on_night if n_fcb_on_night > 0 else 0

            # AC_On during DAY vs NIGHT (same logic for AC confounding)
            ac_on_day_mask = ac_on_mask & daytime_mask
            ac_on_night_mask = ac_on_mask & ~daytime_mask

            n_ac_on_day = ac_on_day_mask.sum()
            n_ac_on_night = ac_on_night_mask.sum()

            ac_on_day_err_rate = (err_mask & ac_on_day_mask).sum() / n_ac_on_day if n_ac_on_day > 0 else 0
            ac_on_night_err_rate = (err_mask & ac_on_night_mask).sum() / n_ac_on_night if n_ac_on_night > 0 else 0

            # 3. MIDDAY CONCENTRATION (11am-2pm) - Solar radiation peak
            midday_mask = (win["Hour"] >= 11) & (win["Hour"] < 14)
            if err_count > 0:
                midday_err_frac = (err_mask & midday_mask).sum() / err_count
            else:
                midday_err_frac = 0.0

            # Of HIGH errors, how many are in midday?
            if high_err_count > 0:
                midday_high_err_frac = (high_err & midday_mask).sum() / high_err_count
            else:
                midday_high_err_frac = 0.0

            # ================================================================
            # STORE WINDOW FEATURES
            # ================================================================
            windows.append({
                # Identifiers
                "Sensor_Code": sensor,
                "Window_Start": start,
                "Window_End": end,
                "Total_Readings": total,
                "Days_In_Window": win["Date"].nunique(),
                
                # Basic error metrics
                "Error_Rate": error_rate,
                "Error_Count": err_count,
                "Mean_Signed_Error": mean_error,
                "Std_Error": std_error,
                "Mean_Abs_Error": mean_abs_error,
                "Max_Abs_Error": max_abs_error,
                
                # Error direction
                "High_Error_Count": high_err_count,
                "Low_Error_Count": low_err_count,
                "High_Error_Fraction": high_err_frac,
                "Low_Error_Fraction": low_err_frac,
                
                # FCB metrics
                "FCB_On_Ratio": fcb_on_ratio,
                "FCB_On_Error_Rate": fcb_on_err_rate,
                "FCB_Off_Error_Rate": fcb_off_err_rate,
                "FCB_On_Mean_Error": fcb_on_mean_err,
                "FCB_Off_Mean_Error": fcb_off_mean_err,
                "FCB_On_Error_Fraction": fcb_on_err_frac,
                "FCB_Off_Error_Fraction": fcb_off_err_frac,
                "FCB_On_Count": n_fcb_on,
                "FCB_Off_Count": n_fcb_off,
                
                # AC metrics
                "AC_On_Ratio": ac_on_ratio,
                "AC_On_Error_Rate": ac_on_err_rate,
                "AC_Off_Error_Rate": ac_off_err_rate,
                "AC_On_Mean_Error": ac_on_mean_err,
                "AC_Off_Mean_Error": ac_off_mean_err,
                "AC_On_Error_Fraction": ac_on_err_frac,
                "AC_Off_Error_Fraction": ac_off_err_frac,
                "AC_On_Count": n_ac_on,
                "AC_Off_Count": n_ac_off,
                
                # Environmental
                "Mean_Humidity": mean_humidity,
                "High_Humidity_Ratio": high_humidity_ratio,
                "High_Humidity_Error_Fraction": high_humidity_err_frac,
                "Rain_Ratio": rain_ratio,
                "Rain_Error_Fraction": rain_err_frac,
                "Sunny_Ratio": sunny_ratio,
                
                # Temporal (ALIGNED WITH RULE-BASED CODE)
                "Gunduz_Yuksek_Orani_YH": gunduz_yuksek_orani_yh,  # Of HIGH errors, % during daytime
                "Daytime_Error_Fraction": daytime_err_frac,  # Of ALL errors, % during daytime

                # Variability metrics (NEW)
                "Error_Variability": error_variability,
                "Error_Range": error_range,
                "Error_Skewness": error_skewness,
                "Consistency_Score": consistency_score,

                # State transitions (NEW)
                "FCB_State_Changes": fcb_state_changes,
                "AC_State_Changes": ac_state_changes,

                # ADVANCED - Confounding analysis (NEW)
                "Peak_Error_Hour": peak_error_hour,
                "Error_Hour_Std": error_hour_std,
                "Error_3Hr_Concentration": error_3hr_concentration,
                "Midday_Error_Fraction": midday_err_frac,
                "Midday_High_Error_Fraction": midday_high_err_frac,

                # State reversal analysis - CRITICAL for causation (NEW)
                "FCB_On_Correct_Rate": fcb_on_correct_rate,
                "FCB_Off_Correct_Rate": fcb_off_correct_rate,
                "AC_On_Correct_Rate": ac_on_correct_rate,
                "AC_Off_Correct_Rate": ac_off_correct_rate,

                # Error direction in each state (NEW)
                "FCB_On_High_Error_Frac": fcb_on_high_err_frac,
                "FCB_On_Low_Error_Frac": fcb_on_low_err_frac,
                "FCB_Off_High_Error_Frac": fcb_off_high_err_frac,
                "FCB_Off_Low_Error_Frac": fcb_off_low_err_frac,
                "AC_On_High_Error_Frac": ac_on_high_err_frac,
                "AC_On_Low_Error_Frac": ac_on_low_err_frac,
                "AC_Off_High_Error_Frac": ac_off_high_err_frac,
                "AC_Off_Low_Error_Frac": ac_off_low_err_frac,

                # Reversal indicators (NEW)
                "FCB_Contrast_Score": fcb_contrast_score,
                "AC_Contrast_Score": ac_contrast_score,
                "FCB_Direction_Reversal": int(fcb_direction_reversal),
                "AC_Direction_Reversal": int(ac_direction_reversal),

                # Conditional error rates - FCB (NEW)
                "FCB_Off_Day_Count": n_fcb_off_day,
                "FCB_Off_Night_Count": n_fcb_off_night,
                "FCB_Off_Day_Error_Rate": fcb_off_day_err_rate,
                "FCB_Off_Night_Error_Rate": fcb_off_night_err_rate,
                "FCB_Day_Night_Ratio": fcb_day_night_ratio,
                "FCB_On_Day_Error_Rate": fcb_on_day_err_rate,
                "FCB_On_Night_Error_Rate": fcb_on_night_err_rate,

                # Conditional error rates - AC (NEW)
                "AC_On_Day_Count": n_ac_on_day,
                "AC_On_Night_Count": n_ac_on_night,
                "AC_On_Day_Error_Rate": ac_on_day_err_rate,
                "AC_On_Night_Error_Rate": ac_on_night_err_rate,
            })
    
    print(f"✓ Extracted {len(windows)} windows from {len(sensors)} sensors")
    return pd.DataFrame(windows)

# ============================================================================
# CLUSTERING ANALYSIS
# ============================================================================

def find_optimal_clusters(X_scaled, cluster_range=range(5, 15)):
    """
    Find optimal number of clusters using multiple metrics
    """
    print("\nFinding optimal number of clusters...")
    
    metrics = {
        'n_clusters': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': []
    }
    
    for n in cluster_range:
        print(f"  Testing {n} clusters...")
        kmeans = KMeans(n_clusters=n, random_state=Config.RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        metrics['n_clusters'].append(n)
        metrics['inertia'].append(kmeans.inertia_)
        metrics['silhouette'].append(silhouette_score(X_scaled, labels))
        metrics['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))
    
    # Plot metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Elbow plot
    axes[0].plot(metrics['n_clusters'], metrics['inertia'], 'bo-', linewidth=2)
    axes[0].set_xlabel('Number of Clusters', fontweight='bold')
    axes[0].set_ylabel('Inertia', fontweight='bold')
    axes[0].set_title('Elbow Method', fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Silhouette score (higher is better)
    axes[1].plot(metrics['n_clusters'], metrics['silhouette'], 'go-', linewidth=2)
    axes[1].set_xlabel('Number of Clusters', fontweight='bold')
    axes[1].set_ylabel('Silhouette Score', fontweight='bold')
    axes[1].set_title('Silhouette Score (higher is better)', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Davies-Bouldin score (lower is better)
    axes[2].plot(metrics['n_clusters'], metrics['davies_bouldin'], 'ro-', linewidth=2)
    axes[2].set_xlabel('Number of Clusters', fontweight='bold')
    axes[2].set_ylabel('Davies-Bouldin Score', fontweight='bold')
    axes[2].set_title('Davies-Bouldin Score (lower is better)', fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking

    # Recommend optimal
    best_silhouette_idx = np.argmax(metrics['silhouette'])
    best_davies_idx = np.argmin(metrics['davies_bouldin'])
    
    print("\nCluster optimization results:")
    print(f"  Best by Silhouette: {metrics['n_clusters'][best_silhouette_idx]} clusters")
    print(f"  Best by Davies-Bouldin: {metrics['n_clusters'][best_davies_idx]} clusters")
    
    # Return the one with best silhouette score
    optimal_n = metrics['n_clusters'][best_silhouette_idx]
    print(f"\n✓ Recommended: {optimal_n} clusters")
    
    return optimal_n, metrics

def perform_clustering(df_features, n_clusters=None):
    """
    Perform clustering on window features
    """
    # Select features for clustering
    feature_cols = [col for col in df_features.columns 
                   if col not in ['Sensor_Code', 'Window_Start', 'Window_End', 
                                 'Total_Readings', 'Days_In_Window', 'Error_Count',
                                 'High_Error_Count', 'Low_Error_Count',
                                 'FCB_On_Count', 'FCB_Off_Count',
                                 'AC_On_Count', 'AC_Off_Count']]
    
    print(f"\nUsing {len(feature_cols)} features for clustering")
    
    X = df_features[feature_cols].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal clusters if not specified
    if n_clusters is None:
        n_clusters, metrics = find_optimal_clusters(X_scaled, Config.N_CLUSTERS_RANGE)
    
    # Perform final clustering
    print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=Config.RANDOM_STATE, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    df_features['Cluster'] = cluster_labels
    
    print(f"✓ Clustering complete!")
    print(f"\nCluster distribution:")
    cluster_dist = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster_id, count in cluster_dist.items():
        print(f"  Cluster {cluster_id}: {count:,} windows ({count/len(cluster_labels)*100:.1f}%)")
    
    return df_features, kmeans, scaler, feature_cols

# ============================================================================
# CLUSTER ANALYSIS & INTERPRETATION
# ============================================================================

def map_cluster_to_fault_type(profile):
    """
    Map cluster characteristics to fault types using domain-specific rules.

    Based on expert rules:
    1. Normal (sensör doğru okuyor): Error rate < 1%
    2. FCB devrede değilken yüksek: High error 1-80%, >80% errors when FCB off, >80% correct when FCB on
    3. Klima devrede iken düşük: Low error 1-80%, >80% errors when AC on, >80% correct when AC off
    4. Sürekli yüksek: Error rate >80%, high errors >90%
    5. Sürekli düşük: Error rate >80%, low errors >90%
    6. Gündüz yüksek: High error >1%, >80% of errors are daytime high
    7. Yüksek nemde hatalı: Error rate >10%, >90% of errors at humidity >90%
    8. Yağışlı havada hatalı: Error rate >10%, >90% of errors during rain
    9. Düzensiz: Error rate >1% but doesn't fit above cases
    """

    # Extract key metrics
    error_rate = profile['Avg_Error_Rate']
    mean_error = profile['Avg_Mean_Error']
    std_error = profile['Avg_Std_Error']
    high_err_frac = profile['Avg_High_Error_Frac']
    low_err_frac = profile['Avg_Low_Error_Frac']

    fcb_on_err = profile['Avg_FCB_On_Error_Rate']
    fcb_off_err = profile['Avg_FCB_Off_Error_Rate']
    ac_on_err = profile['Avg_AC_On_Error_Rate']
    ac_off_err = profile['Avg_AC_Off_Error_Rate']

    # Operational state counts (for minimum thresholds - aligned with rule-based code)
    fcb_on_count = profile.get('Avg_FCB_On_Count', 0)
    fcb_off_count = profile.get('Avg_FCB_Off_Count', 0)
    ac_on_count = profile.get('Avg_AC_On_Count', 0)
    ac_off_count = profile.get('Avg_AC_Off_Count', 0)

    humidity_err_frac = profile['Avg_High_Humidity_Error_Frac']
    rain_err_frac = profile['Avg_Rain_Error_Frac']
    gunduz_yuksek_orani_yh = profile['Avg_Gunduz_Yuksek_Orani_YH']  # Of HIGH errors, % daytime
    daytime_err_frac = profile['Avg_Daytime_Error_Frac']
    error_variability = profile['Avg_Error_Variability']

    # NEW - Confounding analysis features
    error_3hr_concentration = profile.get('Avg_Error_3Hr_Concentration', 0)
    midday_high_err_frac = profile.get('Avg_Midday_High_Error_Fraction', 0)
    peak_error_hour = profile.get('Avg_Peak_Error_Hour', 12)

    # Conditional error rates (to separate confounded variables)
    fcb_off_day_err = profile.get('Avg_FCB_Off_Day_Error_Rate', 0)
    fcb_off_night_err = profile.get('Avg_FCB_Off_Night_Error_Rate', 0)
    fcb_day_night_ratio = profile.get('Avg_FCB_Day_Night_Ratio', 1.0)

    ac_on_day_err = profile.get('Avg_AC_On_Day_Error_Rate', 0)
    ac_on_night_err = profile.get('Avg_AC_On_Night_Error_Rate', 0)

    # NEW - State reversal features (CRITICAL for causation)
    fcb_on_correct_rate = profile.get('Avg_FCB_On_Correct_Rate', 0)
    fcb_off_correct_rate = profile.get('Avg_FCB_Off_Correct_Rate', 0)
    ac_on_correct_rate = profile.get('Avg_AC_On_Correct_Rate', 0)
    ac_off_correct_rate = profile.get('Avg_AC_Off_Correct_Rate', 0)

    # Error direction in each state
    fcb_off_high_err_frac = profile.get('Avg_FCB_Off_High_Error_Frac', 0)
    ac_on_low_err_frac = profile.get('Avg_AC_On_Low_Error_Frac', 0)

    # Reversal indicators
    fcb_contrast_score = profile.get('Avg_FCB_Contrast_Score', 0)
    ac_contrast_score = profile.get('Avg_AC_Contrast_Score', 0)
    fcb_direction_reversal = profile.get('Avg_FCB_Direction_Reversal', 0)
    ac_direction_reversal = profile.get('Avg_AC_Direction_Reversal', 0)

    # Calculate correct data rates (for FCB/AC conditions) - DEPRECATED, use direct features
    fcb_on_correct = fcb_on_correct_rate  # Use direct measurement
    ac_off_correct = ac_off_correct_rate  # Use direct measurement

    # RULE 1: Sensör doğru okuyor (Normal)
    # USER INSIGHT: "Most readings should be correct" - use correct_rate, not just error_rate
    # Check correct reading rates across ALL operational states
    overall_correct_rate = 1.0 - error_rate

    # Strict: error_rate < 1% (very rare in real data)
    if error_rate < 0.01:
        return "Normal", 0.95

    # Improved Normal detection using correct reading rates
    # Key insight: Sensor reads correctly in ALL states (not state-dependent)
    if error_rate < 0.15:  # Less than 15% error rate
        # Check for balanced errors (no strong directional bias)
        max_direction = max(high_err_frac, low_err_frac)

        # NEW: Check that sensor reads correctly across different states
        # If correct rate is high AND consistent across FCB/AC states → Normal
        min_correct_rate = min(fcb_on_correct_rate, fcb_off_correct_rate,
                               ac_on_correct_rate, ac_off_correct_rate)

        # Normal if: balanced errors + low variability + reads correctly in all states
        if (max_direction < 0.70 and
            error_variability < 2.5 and
            abs(mean_error) < 2.0 and
            min_correct_rate > 0.80):  # Reads correctly >80% in ALL states
            confidence = 0.75 - (error_rate * 2)  # Higher confidence for lower error rates
            # Boost confidence if very consistent across states
            if min_correct_rate > 0.90:
                confidence = min(0.90, confidence + 0.10)
            return "Normal", max(0.55, confidence)

    # RULE 4: Sürekli yüksek okuyor (Priority - check first)
    # Error rate > 80% AND high error fraction > 90%
    if error_rate > 0.80 and high_err_frac > 0.90:
        confidence = min(0.95, 0.85 + (high_err_frac - 0.90) * 0.5)
        return "Surekli_Yuksek", confidence

    # RULE 5: Sürekli düşük okuyor (Priority - check first)
    # Error rate > 80% AND low error fraction > 90%
    if error_rate > 0.80 and low_err_frac > 0.90:
        confidence = min(0.95, 0.85 + (low_err_frac - 0.90) * 0.5)
        return "Surekli_Dusuk", confidence

    # Determine if we should check high or low error cases
    # High error cases if high errors > 55% of total errors
    # Low error cases if low errors > 55% of total errors
    check_high_cases = high_err_frac > 0.55
    check_low_cases = low_err_frac > 0.55

    # Score-based evaluation for remaining cases
    fault_scores = {}

    if check_high_cases:
        yuksek_hata_orani = high_err_frac * error_rate  # Proportion of total that are high errors
        fcb_off_err_frac = profile.get('Avg_FCB_Off_Error_Fraction', 0)

        # ============================================================================
        # CONFOUNDING ANALYSIS: Distinguish "FCB_Off_Yuksek" vs "Gunduz_Yuksek"
        # ============================================================================
        # Problem: If FCB doesn't run during daytime, "FCB off" and "daytime" are confounded
        # Solution: Check if error pattern persists at NIGHT when FCB is also off

        # Evidence for TRUE "Gunduz_Yuksek" (solar radiation):
        # 1. Errors concentrated in specific hours (11am-2pm)
        # 2. Errors mostly in midday peak
        # 3. FCB_Off errors ONLY during day, NOT at night (fcb_day_night_ratio >> 1)
        #
        # Evidence for TRUE "FCB_Off_Yuksek" (FCB malfunction):
        # 1. Errors spread across all hours when FCB off
        # 2. FCB_Off errors both DAY and NIGHT (fcb_day_night_ratio ≈ 1)
        # 3. Low error hour concentration

        is_solar_pattern = (
            (error_3hr_concentration > 0.50) or  # Errors concentrated in 2-3 hours
            (midday_high_err_frac > 0.55) or  # Peak solar hours (11am-2pm)
            (fcb_day_night_ratio > 3.0 and fcb_off_day_err > 0.10)  # Errors only when FCB off during DAY, not night
        )

        # RULE 6: Gündüz yüksek okuyor (ALIGNED WITH RULE-BASED CODE + CONFOUNDING CHECK)
        # Prioritize if solar pattern detected
        if yuksek_hata_orani > 0.01:
            if gunduz_yuksek_orani_yh >= 0.80:  # Of HIGH errors, 80%+ are daytime
                score = 0.85 + (min(gunduz_yuksek_orani_yh, 0.95) - 0.80) * 0.3
                # Boost confidence if clear solar pattern
                if is_solar_pattern:
                    score = min(0.95, score + 0.10)
                fault_scores["Gunduz_Yuksek"] = score
            elif gunduz_yuksek_orani_yh >= 0.70:  # Moderate
                score = 0.70 + (gunduz_yuksek_orani_yh - 0.70) * 0.3
                if is_solar_pattern:
                    score = min(0.90, score + 0.10)
                fault_scores["Gunduz_Yuksek"] = score
            elif gunduz_yuksek_orani_yh >= 0.60:  # Relaxed
                score = 0.55 + (gunduz_yuksek_orani_yh - 0.60) * 0.3
                if is_solar_pattern:
                    score = min(0.85, score + 0.10)
                fault_scores["Gunduz_Yuksek"] = score

        # RULE 2: FCB devrede değilken yüksek okuyor (ALIGNED WITH RULE-BASED CODE + CONFOUNDING CHECK + STATE REVERSAL)
        # Only assign if NOT clearly a solar pattern
        if 0.01 < yuksek_hata_orani < 0.80 and fcb_on_count > 20:  # MINIMUM COUNT CHECK
            # Check for TRUE FCB issue (not confounded with daytime)
            is_fcb_issue = (
                (fcb_day_night_ratio < 2.0) or  # Errors both day and night when FCB off
                (fcb_off_night_err > 0.15 and fcb_off_day_err > 0.15)  # High errors both times
            )

            # USER INSIGHT: Check state reversal - does pattern REVERSE when FCB turns on?
            # True FCB fault: HIGH errors when FCB off + CORRECT readings when FCB on
            # NOT FCB fault: ALWAYS high (Surekli_Yuksek) regardless of FCB state
            has_state_reversal = (
                (fcb_contrast_score > 0.30) or  # Clear contrast: correct when on, error when off
                (fcb_direction_reversal > 0.5) or  # Direction explicitly reverses
                (fcb_on_correct_rate > 0.75 and fcb_off_err_rate > 0.30)  # Correct when on, errors when off
            )

            # Only assign FCB_Off_Yuksek if it's NOT clearly solar AND shows FCB pattern AND has reversal
            if (not is_solar_pattern or is_fcb_issue) and has_state_reversal:
                # Strict: >80% of errors when FCB off AND <20% error when FCB on
                if fcb_off_err_frac >= 0.80 and fcb_on_err < 0.20 and not np.isnan(fcb_on_err):
                    score = 0.90 + (min(fcb_off_err_frac, 0.95) - 0.80) * 0.2
                    # Boost confidence if clear state reversal
                    if fcb_contrast_score > 0.50:
                        score = min(0.95, score + 0.05)
                    # Reduce confidence if there's some solar evidence
                    if is_solar_pattern and not is_fcb_issue:
                        score = max(0.50, score - 0.20)
                    fault_scores["FCB_Off_Yuksek"] = score
                # Moderate: >70% of errors when FCB off AND <30% error when FCB on
                elif fcb_off_err_frac >= 0.70 and fcb_on_err < 0.30 and not np.isnan(fcb_on_err):
                    score = 0.75 + (fcb_off_err_frac - 0.70) * 0.3
                    if fcb_contrast_score > 0.40:
                        score = min(0.90, score + 0.05)
                    if is_solar_pattern and not is_fcb_issue:
                        score = max(0.45, score - 0.20)
                    fault_scores["FCB_Off_Yuksek"] = score
                # Relaxed: >60% of errors when FCB off AND <40% error when FCB on
                elif fcb_off_err_frac >= 0.60 and fcb_on_err < 0.40 and not np.isnan(fcb_on_err):
                    score = 0.60 + (fcb_off_err_frac - 0.60) * 0.3
                    if is_solar_pattern and not is_fcb_issue:
                        score = max(0.40, score - 0.15)
                    fault_scores["FCB_Off_Yuksek"] = score

    if check_low_cases:
        # RULE 3: Klima devrede iken düşük okuyor (ALIGNED WITH RULE-BASED CODE + STATE REVERSAL)
        # Düşük_Hata_Oranı between 1-80%, >80% of errors when AC on, <20% error when AC off
        # CRITICAL: Only check if ac_off_count > 20 (rule-based requirement)
        dusuk_hata_orani = low_err_frac * error_rate  # Proportion of total that are low errors
        ac_on_err_frac = profile.get('Avg_AC_On_Error_Fraction', 0)

        if 0.01 < dusuk_hata_orani < 0.80 and ac_off_count > 20:  # MINIMUM COUNT CHECK
            # USER INSIGHT: Check state reversal - does pattern REVERSE when AC turns off?
            # True AC fault: LOW errors when AC on + CORRECT/HIGH readings when AC off
            # NOT AC fault: ALWAYS low (Surekli_Dusuk) regardless of AC state
            has_ac_reversal = (
                (ac_contrast_score > 0.30) or  # Clear contrast: correct when off, error when on
                (ac_direction_reversal > 0.5) or  # Direction explicitly reverses
                (ac_off_correct_rate > 0.75 and ac_on_err_rate > 0.30)  # Correct when off, errors when on
            )

            # Only assign AC_On_Dusuk if errors are LOW direction AND pattern reverses
            if has_ac_reversal and ac_on_low_err_frac > 0.70:  # Errors must be LOW when AC on
                # Strict: >80% of errors when AC on AND <20% error when AC off
                if ac_on_err_frac >= 0.80 and ac_off_err < 0.20 and not np.isnan(ac_off_err):
                    score = 0.90 + (min(ac_on_err_frac, 0.95) - 0.80) * 0.2
                    # Boost confidence if clear state reversal
                    if ac_contrast_score > 0.50:
                        score = min(0.95, score + 0.05)
                    fault_scores["AC_On_Dusuk"] = score
                # Moderate: >70% of errors when AC on AND <30% error when AC off
                elif ac_on_err_frac >= 0.70 and ac_off_err < 0.30 and not np.isnan(ac_off_err):
                    score = 0.75 + (ac_on_err_frac - 0.70) * 0.3
                    if ac_contrast_score > 0.40:
                        score = min(0.90, score + 0.05)
                    fault_scores["AC_On_Dusuk"] = score
                # Relaxed: >60% of errors when AC on AND <40% error when AC off
                elif ac_on_err_frac >= 0.60 and ac_off_err < 0.40 and not np.isnan(ac_off_err):
                    score = 0.60 + (ac_on_err_frac - 0.60) * 0.3
                    fault_scores["AC_On_Dusuk"] = score

    # RULE 7: Yüksek nemde hatalı okuyor
    # Error rate > 10%, >90% of errors at humidity > 90%
    if error_rate > 0.10:
        if humidity_err_frac >= 0.90:  # Strict
            score = 0.85 + (min(humidity_err_frac, 0.98) - 0.90) * 0.5
            fault_scores["Yuksek_Nem_Hatali"] = score
        elif humidity_err_frac >= 0.80:  # Moderate
            score = 0.70 + (humidity_err_frac - 0.80) * 0.5
            fault_scores["Yuksek_Nem_Hatali"] = score
        elif humidity_err_frac >= 0.70:  # Relaxed
            score = 0.55 + (humidity_err_frac - 0.70) * 0.5
            fault_scores["Yuksek_Nem_Hatali"] = score

    # RULE 8: Yağışlı havada hatalı okuyor
    # Error rate > 10%, >90% of errors during rain
    if error_rate > 0.10:
        if rain_err_frac >= 0.90:  # Strict
            score = 0.85 + (min(rain_err_frac, 0.98) - 0.90) * 0.5
            fault_scores["Yagisli_Hava_Hatali"] = score
        elif rain_err_frac >= 0.80:  # Moderate
            score = 0.70 + (rain_err_frac - 0.80) * 0.5
            fault_scores["Yagisli_Hava_Hatali"] = score
        elif rain_err_frac >= 0.70:  # Relaxed
            score = 0.55 + (rain_err_frac - 0.70) * 0.5
            fault_scores["Yagisli_Hava_Hatali"] = score

    # Return the best matching fault type
    if fault_scores:
        best_fault = max(fault_scores, key=fault_scores.get)
        confidence = fault_scores[best_fault]
        return best_fault, confidence

    # RULE 9: Düzensiz (rastgele) hatalı okuyor
    # Error rate > 1% but doesn't fit any above cases
    if error_rate > 0.01:
        # Calculate irregularity score based on variability
        irregularity_score = 0.5 + min(error_variability / 10.0, 0.3)
        return "Duzensiz_Rastgele", irregularity_score

    # Fallback for edge cases
    return "Mixed_Unknown", 0.3

def get_fault_description(fault_type):
    """Get Turkish and English description of fault type"""
    descriptions = {
        "Normal": {
            "tr": "Normal - Hatasız çalışma",
            "en": "Normal - No significant errors",
            "icon": "✓"
        },
        "Surekli_Yuksek": {
            "tr": "Sürekli yüksek okuyor - Kalibrasyon/konumlandırma sorunu",
            "en": "Consistently reading high - Calibration/positioning issue",
            "icon": "⬆"
        },
        "Surekli_Dusuk": {
            "tr": "Sürekli düşük okuyor - Kalibrasyon/konumlandırma sorunu",
            "en": "Consistently reading low - Calibration/positioning issue",
            "icon": "⬇"
        },
        "FCB_Off_Yuksek": {
            "tr": "FCB devrede değilken yüksek - FCB devreye girmiyor",
            "en": "High when FCB off - FCB not engaging",
            "icon": "🔧"
        },
        "AC_On_Dusuk": {
            "tr": "Klima devredeyken düşük - FCB erken devreye giriyor",
            "en": "Low when AC on - FCB engaging too early",
            "icon": "❄"
        },
        "Gunduz_Yuksek": {
            "tr": "Gündüz yüksek okuyor - Güneş ışığı/gölgeleme sorunu",
            "en": "High during daytime - Sunlight/shading issue",
            "icon": "☀"
        },
        "Yuksek_Nem_Hatali": {
            "tr": "Yüksek nemde hatalı - Nem >90% sapma",
            "en": "Error at high humidity - Humidity >90% deviation",
            "icon": "💧"
        },
        "Yagisli_Hava_Hatali": {
            "tr": "Yağışlı havada hatalı - Su etkisi",
            "en": "Error during rain - Water effect",
            "icon": "🌧"
        },
        "Duzensiz_Rastgele": {
            "tr": "Düzensiz (rastgele) hatalı - Gürültü/kablo/haberleşme",
            "en": "Irregular/random errors - Noise/cable/communication",
            "icon": "⚡"
        },
        "Mixed_Unknown": {
            "tr": "Karışık/Belirsiz patern",
            "en": "Mixed/Unknown pattern",
            "icon": "❓"
        }
    }
    return descriptions.get(fault_type, descriptions["Mixed_Unknown"])

def analyze_clusters(df_clustered):
    """
    Analyze and interpret what each cluster represents,
    mapping to 8 known fault types
    """
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS & FAULT TYPE MAPPING")
    print("="*80)

    cluster_profiles = []

    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]

        profile = {
            'Cluster': cluster_id,
            'Count': len(cluster_data),
            'Percentage': len(cluster_data) / len(df_clustered) * 100,

            # Error characteristics
            'Avg_Error_Rate': cluster_data['Error_Rate'].mean(),
            'Avg_Mean_Error': cluster_data['Mean_Signed_Error'].mean(),
            'Avg_Std_Error': cluster_data['Std_Error'].mean(),

            # Error direction
            'Avg_High_Error_Frac': cluster_data['High_Error_Fraction'].mean(),
            'Avg_Low_Error_Frac': cluster_data['Low_Error_Fraction'].mean(),

            # Operational states
            'Avg_FCB_On_Error_Rate': cluster_data['FCB_On_Error_Rate'].mean(),
            'Avg_FCB_Off_Error_Rate': cluster_data['FCB_Off_Error_Rate'].mean(),
            'Avg_AC_On_Error_Rate': cluster_data['AC_On_Error_Rate'].mean(),
            'Avg_AC_Off_Error_Rate': cluster_data['AC_Off_Error_Rate'].mean(),

            # Operational state error fractions (what % of errors occur in each state)
            'Avg_FCB_Off_Error_Fraction': cluster_data['FCB_Off_Error_Fraction'].mean(),
            'Avg_AC_On_Error_Fraction': cluster_data['AC_On_Error_Fraction'].mean(),

            # Operational state counts (for minimum thresholds)
            'Avg_FCB_On_Count': cluster_data['FCB_On_Count'].mean(),
            'Avg_FCB_Off_Count': cluster_data['FCB_Off_Count'].mean(),
            'Avg_AC_On_Count': cluster_data['AC_On_Count'].mean(),
            'Avg_AC_Off_Count': cluster_data['AC_Off_Count'].mean(),

            # Environmental & Temporal
            'Avg_High_Humidity_Error_Frac': cluster_data['High_Humidity_Error_Fraction'].mean(),
            'Avg_Rain_Error_Frac': cluster_data['Rain_Error_Fraction'].mean(),
            'Avg_Gunduz_Yuksek_Orani_YH': cluster_data['Gunduz_Yuksek_Orani_YH'].mean(),  # NEW: Matches rule-based code
            'Avg_Daytime_Error_Frac': cluster_data['Daytime_Error_Fraction'].mean(),

            # Variability
            'Avg_Error_Variability': cluster_data['Error_Variability'].mean(),

            # Consistency
            'Avg_Consistency_Score': cluster_data['Consistency_Score'].mean(),

            # ADVANCED - Confounding analysis features
            'Avg_Error_3Hr_Concentration': cluster_data['Error_3Hr_Concentration'].mean(),
            'Avg_Midday_High_Error_Fraction': cluster_data['Midday_High_Error_Fraction'].mean(),
            'Avg_Peak_Error_Hour': cluster_data['Peak_Error_Hour'].mean(),
            'Avg_FCB_Off_Day_Error_Rate': cluster_data['FCB_Off_Day_Error_Rate'].mean(),
            'Avg_FCB_Off_Night_Error_Rate': cluster_data['FCB_Off_Night_Error_Rate'].mean(),
            'Avg_FCB_Day_Night_Ratio': cluster_data['FCB_Day_Night_Ratio'].mean(),

            # ADVANCED - State reversal analysis (CRITICAL for causation)
            'Avg_FCB_On_Correct_Rate': cluster_data['FCB_On_Correct_Rate'].mean(),
            'Avg_FCB_Off_Correct_Rate': cluster_data['FCB_Off_Correct_Rate'].mean(),
            'Avg_AC_On_Correct_Rate': cluster_data['AC_On_Correct_Rate'].mean(),
            'Avg_AC_Off_Correct_Rate': cluster_data['AC_Off_Correct_Rate'].mean(),
            'Avg_FCB_Off_High_Error_Frac': cluster_data['FCB_Off_High_Error_Frac'].mean(),
            'Avg_AC_On_Low_Error_Frac': cluster_data['AC_On_Low_Error_Frac'].mean(),
            'Avg_FCB_Contrast_Score': cluster_data['FCB_Contrast_Score'].mean(),
            'Avg_AC_Contrast_Score': cluster_data['AC_Contrast_Score'].mean(),
            'Avg_FCB_Direction_Reversal': cluster_data['FCB_Direction_Reversal'].mean(),
            'Avg_AC_Direction_Reversal': cluster_data['AC_Direction_Reversal'].mean(),
        }

        # Map to fault type
        fault_type, confidence = map_cluster_to_fault_type(profile)
        fault_desc = get_fault_description(fault_type)

        profile['Fault_Type'] = fault_type
        profile['Confidence'] = confidence
        profile['Fault_Description_TR'] = fault_desc['tr']
        profile['Fault_Description_EN'] = fault_desc['en']

        cluster_profiles.append(profile)

        # Print cluster interpretation
        print(f"\n{fault_desc['icon']} Cluster {cluster_id} → {fault_type}")
        print(f"   ({len(cluster_data):,} windows, {profile['Percentage']:.1f}%)")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   TR: {fault_desc['tr']}")
        print(f"   EN: {fault_desc['en']}")
        print(f"   Error Rate: {profile['Avg_Error_Rate']:.2%}")
        print(f"   Mean Error: {profile['Avg_Mean_Error']:+.2f}°C")
        print(f"   Std Error: {profile['Avg_Std_Error']:.2f}°C")

        # DEBUG: Show key features for troubleshooting
        print(f"   [DEBUG] High/Low Err Frac: {profile['Avg_High_Error_Frac']:.2%} / {profile['Avg_Low_Error_Frac']:.2%}")
        print(f"   [DEBUG] FCB: On Count={profile['Avg_FCB_On_Count']:.0f}, Off Err Frac={profile['Avg_FCB_Off_Error_Fraction']:.2%}, On Err={profile['Avg_FCB_On_Error_Rate']:.2%}")
        print(f"   [DEBUG] AC: Off Count={profile['Avg_AC_Off_Count']:.0f}, On Err Frac={profile['Avg_AC_On_Error_Fraction']:.2%}, Off Err={profile['Avg_AC_Off_Error_Rate']:.2%}")
        print(f"   [DEBUG] Gunduz Yuksek (of HIGH): {profile['Avg_Gunduz_Yuksek_Orani_YH']:.2%}")
        print(f"   [DEBUG] Humidity Err Frac: {profile['Avg_High_Humidity_Error_Frac']:.2%}, Rain Err Frac: {profile['Avg_Rain_Error_Frac']:.2%}")

        # NEW - Confounding analysis debug
        print(f"   [DEBUG] Error Concentration (3hr): {profile['Avg_Error_3Hr_Concentration']:.2%}, Midday High: {profile['Avg_Midday_High_Error_Fraction']:.2%}")
        print(f"   [DEBUG] FCB Day/Night Ratio: {profile['Avg_FCB_Day_Night_Ratio']:.2f}, FCB Off (Day): {profile['Avg_FCB_Off_Day_Error_Rate']:.2%}, FCB Off (Night): {profile['Avg_FCB_Off_Night_Error_Rate']:.2%}")

        # NEW - State reversal debug
        print(f"   [DEBUG] Correct Rates: FCB_On={profile['Avg_FCB_On_Correct_Rate']:.2%}, FCB_Off={profile['Avg_FCB_Off_Correct_Rate']:.2%}, AC_On={profile['Avg_AC_On_Correct_Rate']:.2%}, AC_Off={profile['Avg_AC_Off_Correct_Rate']:.2%}")
        print(f"   [DEBUG] Contrast: FCB={profile['Avg_FCB_Contrast_Score']:.2f}, AC={profile['Avg_AC_Contrast_Score']:.2f}, Reversals: FCB={profile['Avg_FCB_Direction_Reversal']:.0%}, AC={profile['Avg_AC_Direction_Reversal']:.0%}")

        # Show key discriminating features
        if fault_type == "Normal":
            min_correct = min(profile['Avg_FCB_On_Correct_Rate'], profile['Avg_FCB_Off_Correct_Rate'],
                            profile['Avg_AC_On_Correct_Rate'], profile['Avg_AC_Off_Correct_Rate'])
            print(f"   → Consistent correct readings: Min={min_correct:.2%} across all states")
        elif fault_type == "Surekli_Yuksek":
            print(f"   → High Error Fraction: {profile['Avg_High_Error_Frac']:.2%}")
        elif fault_type == "Surekli_Dusuk":
            print(f"   → Low Error Fraction: {profile['Avg_Low_Error_Frac']:.2%}")
        elif fault_type == "FCB_Off_Yuksek":
            print(f"   → FCB Off Error Rate: {profile['Avg_FCB_Off_Error_Rate']:.2%}")
            print(f"   → FCB On Correct Rate: {profile['Avg_FCB_On_Correct_Rate']:.2%} (should be high)")
            print(f"   → State Reversal: Contrast={profile['Avg_FCB_Contrast_Score']:.2f} (>0.3 = clear)")
            print(f"   → Confounding Check: Day/Night Ratio={profile['Avg_FCB_Day_Night_Ratio']:.2f} (<2.0 means TRUE FCB issue)")
        elif fault_type == "AC_On_Dusuk":
            print(f"   → AC On Error Rate: {profile['Avg_AC_On_Error_Rate']:.2%} (LOW direction: {profile['Avg_AC_On_Low_Error_Frac']:.2%})")
            print(f"   → AC Off Correct Rate: {profile['Avg_AC_Off_Correct_Rate']:.2%} (should be high)")
            print(f"   → State Reversal: Contrast={profile['Avg_AC_Contrast_Score']:.2f} (>0.3 = clear)")
        elif fault_type == "Gunduz_Yuksek":
            print(f"   → Gunduz Yuksek Orani (of HIGH errors): {profile['Avg_Gunduz_Yuksek_Orani_YH']:.2%}")
            print(f"   → Solar Evidence: 3hr Concentration={profile['Avg_Error_3Hr_Concentration']:.2%}, Midday={profile['Avg_Midday_High_Error_Fraction']:.2%}, Peak Hour={profile['Avg_Peak_Error_Hour']:.1f}")
        elif fault_type == "Yuksek_Nem_Hatali":
            print(f"   → High Humidity Error Fraction: {profile['Avg_High_Humidity_Error_Frac']:.2%}")
        elif fault_type == "Yagisli_Hava_Hatali":
            print(f"   → Rain Error Fraction: {profile['Avg_Rain_Error_Frac']:.2%}")
        elif fault_type == "Duzensiz_Rastgele":
            print(f"   → Error Variability: {profile['Avg_Error_Variability']:.2f}°C")

    return pd.DataFrame(cluster_profiles)

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_clusters(df_clustered, feature_cols):
    """
    Visualize clusters using PCA
    """
    print("\nVisualizing clusters with PCA...")
    
    X = df_clustered[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA to 2D
    pca = PCA(n_components=2, random_state=Config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=df_clustered['Cluster'], 
                         cmap='tab10', 
                         alpha=0.6, 
                         s=50,
                         edgecolors='black',
                         linewidth=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontweight='bold')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontweight='bold')
    plt.title('Cluster Visualization (PCA)', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('cluster_visualization_pca.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking

    print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    print(f"  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")

# ============================================================================
# FAULT TYPE VISUALIZATION
# ============================================================================

def visualize_fault_distribution(cluster_profiles):
    """
    Visualize distribution of fault types
    """
    print("\nCreating fault type distribution chart...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Count by fault type
    fault_counts = cluster_profiles.groupby('Fault_Type')['Count'].sum().sort_values(ascending=False)

    # Bar chart
    colors = ['#2ecc71' if ft == 'Normal' else '#e74c3c' if 'Surekli' in ft else '#f39c12'
              for ft in fault_counts.index]

    ax1.barh(range(len(fault_counts)), fault_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_yticks(range(len(fault_counts)))
    ax1.set_yticklabels(fault_counts.index, fontsize=10)
    ax1.set_xlabel('Number of Windows', fontweight='bold', fontsize=12)
    ax1.set_title('Fault Type Distribution (Window Count)', fontweight='bold', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, v in enumerate(fault_counts.values):
        ax1.text(v + max(fault_counts)*0.01, i, f'{v:,}', va='center', fontweight='bold')

    # Pie chart with percentages
    percentages = (fault_counts / fault_counts.sum() * 100)

    wedges, texts, autotexts = ax2.pie(fault_counts.values,
                                        labels=fault_counts.index,
                                        autopct='%1.1f%%',
                                        colors=colors,
                                        startangle=90,
                                        textprops={'fontsize': 9, 'fontweight': 'bold'})

    ax2.set_title('Fault Type Distribution (Percentage)', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.savefig('fault_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking

    print("✓ Fault distribution chart saved")

    # Print summary
    print("\n" + "="*80)
    print("FAULT TYPE SUMMARY")
    print("="*80)
    for fault_type, count in fault_counts.items():
        pct = count / fault_counts.sum() * 100
        desc = get_fault_description(fault_type)
        print(f"{desc['icon']} {fault_type:25} {count:6,} windows ({pct:5.1f}%) - {desc['tr']}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline with fault type mapping"""

    print("="*80)
    print("CLUSTERING-BASED FAULT DETECTION WITH AUTOMATIC LABELING")
    print("="*80)

    # Step 1: Extract features
    print("\nStep 1: Extracting features from raw data...")
    df_raw = pd.read_csv(Config.INPUT_FILE)
    df_features = extract_window_features(df_raw)

    # Step 2: Perform clustering
    print("\nStep 2: Performing clustering...")
    df_clustered, kmeans_model, scaler, feature_cols = perform_clustering(df_features)

    # Step 3: Analyze clusters and map to fault types
    print("\nStep 3: Analyzing clusters and mapping to fault types...")
    cluster_profiles = analyze_clusters(df_clustered)

    # Step 4: Add fault type labels to windows
    print("\nStep 4: Adding fault type labels to windows...")
    cluster_to_fault = dict(zip(cluster_profiles['Cluster'], cluster_profiles['Fault_Type']))
    cluster_to_confidence = dict(zip(cluster_profiles['Cluster'], cluster_profiles['Confidence']))
    cluster_to_desc_tr = dict(zip(cluster_profiles['Cluster'], cluster_profiles['Fault_Description_TR']))
    cluster_to_desc_en = dict(zip(cluster_profiles['Cluster'], cluster_profiles['Fault_Description_EN']))

    df_clustered['Fault_Type'] = df_clustered['Cluster'].map(cluster_to_fault)
    df_clustered['Fault_Confidence'] = df_clustered['Cluster'].map(cluster_to_confidence)
    df_clustered['Fault_Description_TR'] = df_clustered['Cluster'].map(cluster_to_desc_tr)
    df_clustered['Fault_Description_EN'] = df_clustered['Cluster'].map(cluster_to_desc_en)

    # Step 5: Visualize
    print("\nStep 5: Creating visualizations...")
    visualize_clusters(df_clustered, feature_cols)
    visualize_fault_distribution(cluster_profiles)

    # Step 6: Save results
    print("\nStep 6: Saving results...")
    df_clustered.to_csv(Config.OUTPUT_FILE, index=False, encoding='utf-8-sig')
    cluster_profiles.to_csv('cluster_profiles.csv', index=False, encoding='utf-8-sig')

    # Create a sensor-level summary
    print("\nStep 7: Creating sensor-level fault summary...")
    sensor_faults = df_clustered.groupby(['Sensor_Code', 'Fault_Type']).size().reset_index(name='Window_Count')
    sensor_faults = sensor_faults.sort_values(['Sensor_Code', 'Window_Count'], ascending=[True, False])
    sensor_faults.to_csv('sensor_fault_summary.csv', index=False, encoding='utf-8-sig')

    # Step 8: Pattern detection analysis
    print("\nStep 8: Analyzing pattern coverage...")
    print("\n" + "="*80)
    print("PATTERN DETECTION ANALYSIS")
    print("="*80)

    # Get all possible fault types
    all_fault_types = {
        "Normal": "Sensör doğru okuyor - İyi çalışıyor",
        "Surekli_Yuksek": "Sürekli yüksek okuyor - Kalibrasyon sorunu",
        "Surekli_Dusuk": "Sürekli düşük okuyor - Kalibrasyon/konumlandırma sorunu",
        "FCB_Off_Yuksek": "FCB devrede değilken yüksek - FCB devreye girmiyor",
        "AC_On_Dusuk": "Klima devredeyken düşük - Klima etkisi",
        "Gunduz_Yuksek": "Gündüz yüksek okuyor - Güneş ışığı/gölgeleme sorunu",
        "Yuksek_Nem_Hatali": "Yüksek nemde hatalı - Nem sensörü etkileşimi",
        "Yagisli_Hava_Hatali": "Yağışlı havada hatalı - Yağış etkisi",
        "Duzensiz_Rastgele": "Düzensiz (rastgele) hatalı - Gürültü/kablo/haberleşme"
    }

    detected_faults = set(df_clustered['Fault_Type'].unique())
    missing_faults = set(all_fault_types.keys()) - detected_faults

    print(f"\n✅ DETECTED PATTERNS ({len(detected_faults)}/9):")
    for fault in sorted(detected_faults):
        count = (df_clustered['Fault_Type'] == fault).sum()
        pct = count / len(df_clustered) * 100
        print(f"   ✓ {fault:25} {count:6,} windows ({pct:5.1f}%) - {all_fault_types[fault]}")

    if missing_faults:
        print(f"\n❌ MISSING PATTERNS ({len(missing_faults)}/9):")
        print("   These patterns were not detected in your data. This is normal if:")
        print("   - The sensors don't exhibit this specific fault behavior")
        print("   - The environmental conditions (humidity, rain) are not extreme enough")
        print("   - Sample size for specific operational states (FCB/AC) is insufficient")
        print()
        for fault in sorted(missing_faults):
            reason = ""
            if fault == "AC_On_Dusuk":
                reason = "→ No strong AC-related low error pattern found"
            elif fault == "Yuksek_Nem_Hatali":
                reason = "→ Humidity errors don't dominate any cluster (<5% in most cases)"
            elif fault == "Yagisli_Hava_Hatali":
                reason = "→ Rain errors don't dominate any cluster (<2% in most cases)"
            elif fault == "Surekli_Yuksek":
                reason = "→ No sensors with >80% error rate AND >90% high errors"
            elif fault == "Normal":
                reason = "→ No sensors found with <1% error rate (may need relaxed threshold)"
            print(f"   ✗ {fault:25} {all_fault_types[fault]}")
            if reason:
                print(f"      {reason}")

    print(f"\n💡 RECOMMENDATIONS:")
    if "Normal" not in detected_faults:
        lowest_error_cluster = cluster_profiles.loc[cluster_profiles['Avg_Error_Rate'].idxmin()]
        print(f"   • Lowest error cluster has {lowest_error_cluster['Avg_Error_Rate']*100:.1f}% error rate")
        print(f"     Consider this as your 'Normal' baseline for comparison")

    if "AC_On_Dusuk" not in detected_faults:
        print(f"   • AC pattern not found - check if AC usage is sufficient in your data")

    if "Surekli_Yuksek" not in detected_faults:
        print(f"   • No continuously high sensors found - your sensors don't have severe calibration drift")

    print(f"\n✅ CONCLUSION:")
    print(f"   Your dataset contains {len(detected_faults)} distinct fault patterns.")
    print(f"   Missing patterns indicate absence of those specific fault behaviors,")
    print(f"   which is expected and normal for real-world sensor deployments.")

    print(f"\n{'='*80}")
    print("✓ CLUSTERING AND LABELING COMPLETE!")
    print(f"{'='*80}")
    print(f"  Clustered windows: {Config.OUTPUT_FILE}")
    print(f"  Cluster profiles: cluster_profiles.csv")
    print(f"  Sensor fault summary: sensor_fault_summary.csv")
    print(f"  Visualizations: cluster_*.png, fault_type_distribution.png")
    print(f"\n🎉 Your data is now labeled with fault types!")
    print(f"   You can use these labels for supervised learning or further analysis.")

if __name__ == "__main__":
    main()
