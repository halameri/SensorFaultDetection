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
    N_CLUSTERS_RANGE = range(5, 15)  # Try 5 to 14 clusters
    RANDOM_STATE = 42
    
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

def analyze_clusters(df_clustered):
    """
    Analyze and interpret what each cluster represents
    """
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS & INTERPRETATION")
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
            
            # Consistency
            'Avg_Consistency_Score': cluster_data['Consistency_Score'].mean(),
        }
        
        cluster_profiles.append(profile)
        
        # Interpret cluster
        print(f"\nCluster {cluster_id} ({len(cluster_data):,} windows, {profile['Percentage']:.1f}%):")
        print(f"  Error Rate: {profile['Avg_Error_Rate']:.2%}")
        print(f"  Mean Error: {profile['Avg_Mean_Error']:+.2f}°C")
        
        # Identify dominant characteristics
        characteristics = []
        
        if profile['Avg_Error_Rate'] < 0.01:
            characteristics.append("✓ LOW ERROR RATE (Sensor working correctly)")
        elif profile['Avg_Error_Rate'] > 0.80:
            characteristics.append("⚠ VERY HIGH ERROR RATE (Continuous malfunction)")
        
        if profile['Avg_High_Error_Frac'] > 0.80:
            characteristics.append("↑ Predominantly HIGH errors")
        elif profile['Avg_Low_Error_Frac'] > 0.80:
            characteristics.append("↓ Predominantly LOW errors")
        
        if profile['Avg_FCB_Off_Error_Rate'] > profile['Avg_FCB_On_Error_Rate'] * 3:
            characteristics.append("🔧 Errors when FCB is OFF")
        
        if profile['Avg_AC_On_Error_Rate'] > profile['Avg_AC_Off_Error_Rate'] * 3:
            characteristics.append("❄ Errors when AC is ON")
        
        if profile['Avg_High_Humidity_Error_Frac'] > 0.70:
            characteristics.append("💧 High humidity correlation")
        
        if profile['Avg_Rain_Error_Frac'] > 0.70:
            characteristics.append("🌧 Precipitation correlation")
        
        if profile['Avg_Daytime_High_Error_Frac'] > 0.70:
            characteristics.append("☀ Daytime high errors")
        
        if characteristics:
            print("  Characteristics:")
            for char in characteristics:
                print(f"    {char}")
        else:
            print("  Characteristics: Mixed/irregular pattern")
    
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
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline"""
    
    print("="*80)
    print("CLUSTERING-BASED FAULT DETECTION")
    print("="*80)
    
    # Step 1: Extract features
    print("\nStep 1: Extracting features from raw data...")
    df_raw = pd.read_csv(Config.INPUT_FILE)
    df_features = extract_window_features(df_raw)
    
    # Step 2: Perform clustering
    print("\nStep 2: Performing clustering...")
    df_clustered, kmeans_model, scaler, feature_cols = perform_clustering(df_features)
    
    # Step 3: Analyze clusters
    print("\nStep 3: Analyzing cluster characteristics...")
    cluster_profiles = analyze_clusters(df_clustered)
    
    # Step 4: Visualize
    print("\nStep 4: Creating visualizations...")
    visualize_clusters(df_clustered, feature_cols)
    
    # Step 5: Save results
    print("\nStep 5: Saving results...")
    df_clustered.to_csv(Config.OUTPUT_FILE, index=False, encoding='utf-8-sig')
    cluster_profiles.to_csv('cluster_profiles.csv', index=False)
    
    print(f"\n✓ Clustering complete!")
    print(f"  Output: {Config.OUTPUT_FILE}")
    print(f"  Cluster profiles: cluster_profiles.csv")
    print(f"\n🎉 Ready for supervised learning using cluster labels!")

if __name__ == "__main__":
    main()
