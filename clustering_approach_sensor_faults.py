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
            
            # Temporal
            daytime_mask = win["Is_Daytime"]
            if err_count > 0:
                daytime_high_err_frac = (high_err & daytime_mask).sum() / err_count
            else:
                daytime_high_err_frac = 0.0
            
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
                
                # Temporal
                "Daytime_High_Error_Fraction": daytime_high_err_frac,
                
                # Variability metrics (NEW)
                "Error_Variability": error_variability,
                "Error_Range": error_range,
                "Error_Skewness": error_skewness,
                "Consistency_Score": consistency_score,
                
                # State transitions (NEW)
                "FCB_State_Changes": fcb_state_changes,
                "AC_State_Changes": ac_state_changes,
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
    plt.show()
    
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
    Map cluster characteristics to one of 8 fault types:

    1. Sürekli yüksek okuyor (Consistently reading high)
    2. Sürekli düşük okuyor (Consistently reading low)
    3. FCB devrede değilken yüksek okuyor (High when FCB off)
    4. Klima devredeyken düşük okuyor (Low when AC on)
    5. Gündüz yüksek okuyor (High during daytime)
    6. Yüksek nemde hatalı okuyor (Error at high humidity)
    7. Yağışlı havada hatalı okuyor (Error during rain)
    8. Düzensiz (rastgele) hatalı okuyor (Irregular/random)
    9. Normal (No significant errors)
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

    humidity_err_frac = profile['Avg_High_Humidity_Error_Frac']
    rain_err_frac = profile['Avg_Rain_Error_Frac']
    daytime_err_frac = profile['Avg_Daytime_High_Error_Frac']
    error_variability = profile['Avg_Error_Variability']

    # Decision tree for fault classification
    fault_scores = {}

    # Type 0: Normal (very low error rate)
    if error_rate < 0.05:
        return "Normal", 1.0

    # Type 1: Sürekli yüksek okuyor (Consistently high)
    if error_rate > 0.70 and high_err_frac > 0.85 and mean_error > 3.0 and std_error < 3.0:
        fault_scores["Surekli_Yuksek"] = 0.9 + (high_err_frac - 0.85) * 2

    # Type 2: Sürekli düşük okuyor (Consistently low)
    if error_rate > 0.70 and low_err_frac > 0.85 and mean_error < -3.0 and std_error < 3.0:
        fault_scores["Surekli_Dusuk"] = 0.9 + (low_err_frac - 0.85) * 2

    # Type 3: FCB devrede değilken yüksek (High when FCB off)
    if fcb_off_err > 0.40 and fcb_off_err > fcb_on_err * 2.5 and high_err_frac > 0.65:
        fault_scores["FCB_Off_Yuksek"] = 0.8 + (fcb_off_err / max(fcb_on_err, 0.01)) * 0.05

    # Type 4: Klima devredeyken düşük (Low when AC on)
    if ac_on_err > 0.40 and ac_on_err > ac_off_err * 2.5 and low_err_frac > 0.65:
        fault_scores["AC_On_Dusuk"] = 0.8 + (ac_on_err / max(ac_off_err, 0.01)) * 0.05

    # Type 5: Gündüz yüksek (High during daytime - sunlight)
    if daytime_err_frac > 0.65 and high_err_frac > 0.70 and error_rate > 0.30:
        fault_scores["Gunduz_Yuksek"] = 0.75 + daytime_err_frac * 0.2

    # Type 6: Yüksek nemde hatalı (Error at high humidity)
    if humidity_err_frac > 0.65 and error_rate > 0.30:
        fault_scores["Yuksek_Nem_Hatali"] = 0.75 + humidity_err_frac * 0.2

    # Type 7: Yağışlı havada hatalı (Error during rain)
    if rain_err_frac > 0.65 and error_rate > 0.30:
        fault_scores["Yagisli_Hava_Hatali"] = 0.75 + rain_err_frac * 0.2

    # Type 8: Düzensiz/rastgele (Irregular - high variability)
    if error_variability > 4.0 and std_error > 4.0 and error_rate > 0.25:
        fault_scores["Duzensiz_Rastgele"] = 0.7 + (error_variability / 10.0) * 0.2

    # Return the fault type with highest score
    if fault_scores:
        best_fault = max(fault_scores, key=fault_scores.get)
        confidence = min(fault_scores[best_fault], 1.0)
        return best_fault, confidence
    else:
        return "Mixed_Unknown", 0.5

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

            # Environmental
            'Avg_High_Humidity_Error_Frac': cluster_data['High_Humidity_Error_Fraction'].mean(),
            'Avg_Rain_Error_Frac': cluster_data['Rain_Error_Fraction'].mean(),
            'Avg_Daytime_High_Error_Frac': cluster_data['Daytime_High_Error_Fraction'].mean(),

            # Variability
            'Avg_Error_Variability': cluster_data['Error_Variability'].mean(),

            # Consistency
            'Avg_Consistency_Score': cluster_data['Consistency_Score'].mean(),
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

        # Show key discriminating features
        if fault_type == "Surekli_Yuksek":
            print(f"   → High Error Fraction: {profile['Avg_High_Error_Frac']:.2%}")
        elif fault_type == "Surekli_Dusuk":
            print(f"   → Low Error Fraction: {profile['Avg_Low_Error_Frac']:.2%}")
        elif fault_type == "FCB_Off_Yuksek":
            print(f"   → FCB Off Error Rate: {profile['Avg_FCB_Off_Error_Rate']:.2%}")
            print(f"   → FCB On Error Rate: {profile['Avg_FCB_On_Error_Rate']:.2%}")
        elif fault_type == "AC_On_Dusuk":
            print(f"   → AC On Error Rate: {profile['Avg_AC_On_Error_Rate']:.2%}")
            print(f"   → AC Off Error Rate: {profile['Avg_AC_Off_Error_Rate']:.2%}")
        elif fault_type == "Gunduz_Yuksek":
            print(f"   → Daytime High Error Fraction: {profile['Avg_Daytime_High_Error_Frac']:.2%}")
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
    plt.show()
    
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
    plt.show()

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
