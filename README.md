# Sensor Fault Detection with Clustering

Automated sensor fault detection system using unsupervised learning (clustering) to identify and label fault patterns in sensor temperature readings compared to meteorological reference data.

## Overview

This system automatically detects and classifies 8 types of sensor faults without requiring manual labeling:

### Fault Types Detected

1. **Sürekli yüksek okuyor** (Consistently reading high)
   - Sensor calibration or positioning issue causing persistent high readings

2. **Sürekli düşük okuyor** (Consistently reading low)
   - Sensor calibration or positioning issue causing persistent low readings

3. **FCB devrede değilken yüksek okuyor** (High when FCB off)
   - Sensor reads high when FCB is disabled → FCB not engaging properly

4. **Klima devredeyken düşük okuyor** (Low when AC on)
   - Sensor reads low when AC is running → FCB engaging too early

5. **Gündüz yüksek okuyor** (High during daytime)
   - Sunlight exposure or shading issues causing high readings during day

6. **Yüksek nemde hatalı okuyor** (Error at high humidity)
   - Sensor deviation when humidity >90% (weak sensor protection)

7. **Yağışlı havada hatalı okuyor** (Error during rain)
   - Measurement errors during precipitation (water effect)

8. **Düzensiz (rastgele) hatalı okuyor** (Irregular/random errors)
   - Inconsistent readings from noise, cable contact, or communication errors

## How It Works

```
Raw Sensor Data → Feature Extraction → Clustering → Fault Type Mapping → Labeled Dataset
```

1. **Feature Extraction**: Analyzes 15-day windows of sensor data
   - Error characteristics (rate, direction, variability)
   - Operational state analysis (FCB on/off, AC on/off)
   - Environmental conditions (humidity, precipitation, time of day)
   - Consistency metrics

2. **Clustering**: Uses K-Means to group similar fault patterns
   - Automatically finds optimal number of clusters (7-12 range)
   - Uses multiple validation metrics (Silhouette, Davies-Bouldin, Elbow)

3. **Fault Mapping**: Intelligent rule-based mapping of clusters to fault types
   - Analyzes cluster characteristics
   - Assigns fault type with confidence score
   - Provides bilingual descriptions (Turkish/English)

4. **Output**: Labeled dataset ready for supervised learning
   - Each window labeled with fault type
   - Confidence scores for each label
   - Sensor-level fault summaries

## Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Input Data

Place your `clean_joined_dataset.csv` file in the project directory. Required columns:
- `FC_BOX_CODE` - Sensor identifier
- `Met_Timestamp` - Meteorological timestamp
- `Met_Temperature` - Reference temperature
- `Met_NEM` - Humidity
- `HADISE` - Weather condition
- `Sensor_Temperature` - Sensor reading
- `Elevation_Difference_m` - Elevation difference
- `BX_CHZ_STT_PR` - Operational state (FCB/AC)

### Run the Analysis

```bash
python clustering_approach_sensor_faults.py
```

### Output Files

1. **ml_15day_windows_clustered.csv** - Main output with fault labels
   - All window features
   - Cluster assignments
   - Fault type labels
   - Confidence scores
   - Descriptions in TR/EN

2. **cluster_profiles.csv** - Cluster characteristics
   - Statistical profiles of each cluster
   - Fault type mappings
   - Average metrics

3. **sensor_fault_summary.csv** - Sensor-level summary
   - Fault type distribution per sensor
   - Window counts by fault type

4. **Visualizations**:
   - `cluster_optimization.png` - Optimal cluster selection
   - `cluster_visualization_pca.png` - PCA visualization
   - `fault_type_distribution.png` - Fault type distribution

## Configuration

Edit `Config` class in the script:

```python
class Config:
    # Files
    INPUT_FILE = "clean_joined_dataset.csv"
    OUTPUT_FILE = "ml_15day_windows_clustered.csv"

    # Window parameters
    WINDOW_DAYS = 15          # Analysis window size
    WINDOW_STEP = 1           # Window sliding step
    MIN_READINGS = 150        # Minimum readings per window
    MIN_DAYS = 15             # Minimum days with data

    # Error calculation
    ERROR_TOLERANCE = 4.0     # Error threshold (°C)
    LAPSE_RATE = -0.65        # Temperature lapse rate

    # Clustering
    N_CLUSTERS_RANGE = range(7, 13)  # Try 7-12 clusters
```

## Next Steps

After generating labeled data, you can:

1. **Train Supervised Models**: Use fault labels for classification
   ```python
   # Example: Train Random Forest
   from sklearn.ensemble import RandomForestClassifier

   X = df[feature_cols]
   y = df['Fault_Type']

   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   ```

2. **Manual Refinement**: Review and adjust cluster-to-fault mappings
   - Check confidence scores
   - Validate with domain knowledge
   - Adjust thresholds in `map_cluster_to_fault_type()`

3. **Deploy for Real-time Detection**: Use trained model for live monitoring

## Data Privacy

⚠️ **Important**: Your sensor data is protected
- All CSV files are excluded from git (see `.gitignore`)
- Data never leaves your local system
- No data is shared or uploaded

## Troubleshooting

**Issue**: Too few clusters detected
- Adjust `N_CLUSTERS_RANGE` to try more clusters
- Check if you have enough varied fault patterns in data

**Issue**: Low confidence scores
- Review cluster profiles manually
- Adjust thresholds in `map_cluster_to_fault_type()`
- May need more data or different window parameters

**Issue**: Fault types not separating well
- Try adjusting `WINDOW_DAYS` (larger = more stable patterns)
- Check `ERROR_TOLERANCE` threshold
- Review feature engineering in `extract_window_features()`

## Technical Details

- **Algorithm**: K-Means clustering with standardized features
- **Dimensionality**: 30+ features per window
- **Validation**: Silhouette score, Davies-Bouldin index, Elbow method
- **Mapping**: Rule-based decision tree with confidence scoring

## License

This is a private sensor fault detection system. Keep your data confidential.