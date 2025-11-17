#!/usr/bin/env python3
"""
Cluster Count Analysis - Visualize fault type distribution for K=4 to K=12
Shows how fault type distribution changes with different cluster counts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import the main clustering script functions
import sys
sys.path.insert(0, '/home/user/SensorFaultDetection')
from clustering_approach_sensor_faults import (
    extract_window_features,
    map_cluster_to_fault_type,
    get_fault_description
)

class Config:
    INPUT_FILE = 'clean_joined_dataset.csv'

def analyze_cluster_count(df_features, feature_cols, n_clusters):
    """
    Run clustering with specific K and return fault type distribution
    """
    print(f"\n{'='*60}")
    print(f"Analyzing K={n_clusters} clusters...")
    print(f"{'='*60}")

    # Prepare data
    X = df_features[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    df_features['Cluster'] = kmeans.fit_predict(X_scaled)

    # Analyze each cluster
    fault_distribution = {}

    for cluster_id in range(n_clusters):
        cluster_data = df_features[df_features['Cluster'] == cluster_id]
        n_windows = len(cluster_data)

        # Calculate cluster profile (averages)
        profile = {}
        for col in feature_cols:
            avg_col_name = f'Avg_{col}'
            profile[avg_col_name] = cluster_data[col].mean()

        # Add counts
        profile['Count'] = n_windows
        profile['Avg_Error_Rate'] = cluster_data['Error_Rate'].mean()
        profile['Avg_High_Error_Fraction'] = cluster_data['High_Error_Fraction'].mean()
        profile['Avg_Low_Error_Fraction'] = cluster_data['Low_Error_Fraction'].mean()

        # Map to fault type
        fault_type, confidence = map_cluster_to_fault_type(profile)
        fault_desc = get_fault_description(fault_type)

        if fault_type not in fault_distribution:
            fault_distribution[fault_type] = 0
        fault_distribution[fault_type] += n_windows

        print(f"  Cluster {cluster_id}: {fault_desc['icon']} {fault_type} ({n_windows} windows, {confidence:.0%} conf)")

    return fault_distribution

def visualize_cluster_analysis(all_results):
    """
    Create comprehensive visualization of cluster count analysis
    """
    # Prepare data for plotting
    k_values = sorted(all_results.keys())
    all_fault_types = set()
    for dist in all_results.values():
        all_fault_types.update(dist.keys())
    all_fault_types = sorted(all_fault_types)

    # Create data matrix
    data_matrix = np.zeros((len(all_fault_types), len(k_values)))
    for i, fault_type in enumerate(all_fault_types):
        for j, k in enumerate(k_values):
            data_matrix[i, j] = all_results[k].get(fault_type, 0)

    # Calculate percentages
    totals = data_matrix.sum(axis=0)
    data_pct = (data_matrix / totals) * 100

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Stacked Area Chart
    ax1 = plt.subplot(2, 2, 1)
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_fault_types)))
    ax1.stackplot(k_values, data_pct, labels=all_fault_types, colors=colors, alpha=0.8)
    ax1.set_xlabel('Number of Clusters (K)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Fault Type Distribution by Cluster Count\n(Stacked Area)',
                  fontweight='bold', fontsize=14)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(k_values)

    # 2. Heatmap
    ax2 = plt.subplot(2, 2, 2)
    im = ax2.imshow(data_pct, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax2.set_xticks(range(len(k_values)))
    ax2.set_xticklabels(k_values)
    ax2.set_yticks(range(len(all_fault_types)))
    ax2.set_yticklabels(all_fault_types, fontsize=9)
    ax2.set_xlabel('Number of Clusters (K)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Fault Type', fontweight='bold', fontsize=12)
    ax2.set_title('Fault Type Percentage Heatmap', fontweight='bold', fontsize=14)

    # Add percentage text
    for i in range(len(all_fault_types)):
        for j in range(len(k_values)):
            if data_pct[i, j] > 0:
                text = ax2.text(j, i, f'{data_pct[i, j]:.1f}%',
                              ha="center", va="center", color="black" if data_pct[i, j] < 50 else "white",
                              fontsize=7)

    plt.colorbar(im, ax=ax2, label='Percentage (%)')

    # 3. Line plot for major fault types
    ax3 = plt.subplot(2, 2, 3)
    # Plot top 5 most common fault types
    fault_totals = data_matrix.sum(axis=1)
    top_indices = np.argsort(fault_totals)[-5:][::-1]

    for idx in top_indices:
        fault_type = all_fault_types[idx]
        ax3.plot(k_values, data_pct[idx], marker='o', linewidth=2, markersize=8,
                label=fault_type)

    ax3.set_xlabel('Number of Clusters (K)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
    ax3.set_title('Top 5 Fault Types Trend', fontweight='bold', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_xticks(k_values)

    # 4. Bar chart for each K
    ax4 = plt.subplot(2, 2, 4)
    x = np.arange(len(k_values))
    width = 0.08

    for i, fault_type in enumerate(all_fault_types):
        offset = (i - len(all_fault_types)/2) * width
        values = [all_results[k].get(fault_type, 0) for k in k_values]
        ax4.bar(x + offset, values, width, label=fault_type, alpha=0.8)

    ax4.set_xlabel('Number of Clusters (K)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Number of Windows', fontweight='bold', fontsize=12)
    ax4.set_title('Fault Type Counts by Cluster Count', fontweight='bold', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(k_values)
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax4.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('cluster_count_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: cluster_count_analysis.png")

    # Create detailed comparison table
    fig2, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    header = ['Fault Type'] + [f'K={k}' for k in k_values]
    table_data.append(header)

    for fault_type in all_fault_types:
        row = [fault_type]
        for k in k_values:
            count = all_results[k].get(fault_type, 0)
            pct = (count / totals[k_values.index(k)] * 100) if count > 0 else 0
            row.append(f'{count}\n({pct:.1f}%)')
        table_data.append(row)

    # Add total row
    total_row = ['TOTAL']
    for k in k_values:
        total_row.append(f'{int(totals[k_values.index(k)])}')
    table_data.append(total_row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15] + [0.095]*len(k_values))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(header)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style total row
    for i in range(len(total_row)):
        table[(len(table_data)-1, i)].set_facecolor('#E7E6E6')
        table[(len(table_data)-1, i)].set_text_props(weight='bold')

    # Style fault type column
    for i in range(1, len(table_data)-1):
        table[(i, 0)].set_facecolor('#D9E1F2')
        table[(i, 0)].set_text_props(weight='bold')

    plt.title('Detailed Cluster Count Comparison Table\n(Count and Percentage)',
             fontweight='bold', fontsize=16, pad=20)
    plt.savefig('cluster_count_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved table: cluster_count_table.png")

def main():
    """
    Main analysis function
    """
    print("="*80)
    print("CLUSTER COUNT ANALYSIS - K=4 to K=12")
    print("="*80)

    # Load data and extract features
    print("\n1. Loading data and extracting features...")
    df_raw = pd.read_csv(Config.INPUT_FILE)
    print(f"   Loaded {len(df_raw):,} readings")

    print("\n2. Extracting window features...")
    df_features = extract_window_features(df_raw)
    print(f"   ✓ Extracted features for {len(df_features):,} windows")

    # Select feature columns
    feature_cols = [col for col in df_features.columns
                   if col not in ['Sensor_Code', 'Window_Start', 'Window_End',
                                 'Total_Readings', 'Days_In_Window', 'Error_Count',
                                 'High_Error_Count', 'Low_Error_Count',
                                 'FCB_On_Count', 'FCB_Off_Count',
                                 'AC_On_Count', 'AC_Off_Count']]

    print(f"   Using {len(feature_cols)} features for clustering")

    # Analyze different cluster counts
    print("\n3. Analyzing cluster counts K=4 to K=12...")
    all_results = {}

    for k in range(4, 13):  # 4 to 12 inclusive
        df_copy = df_features.copy()
        fault_dist = analyze_cluster_count(df_copy, feature_cols, k)
        all_results[k] = fault_dist

    # Create visualizations
    print("\n4. Creating visualizations...")
    visualize_cluster_analysis(all_results)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Fault Type Coverage by K")
    print("="*80)
    for k in range(4, 13):
        n_fault_types = len(all_results[k])
        print(f"K={k:2d}: {n_fault_types} different fault types detected")

    print("\n✓ Analysis complete!")
    print("\nGenerated files:")
    print("  - cluster_count_analysis.png  (4 charts showing trends)")
    print("  - cluster_count_table.png     (detailed comparison table)")

if __name__ == "__main__":
    main()
