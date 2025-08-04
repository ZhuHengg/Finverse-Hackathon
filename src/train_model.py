import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


def compute_user_similarity(df, feature_cols):
    similarity_scores = []
    for idx, row in df.iterrows():
        user_id = row['user_id']
        user_df = df[df['user_id'] == user_id].drop(index=idx)
        if len(user_df) < 3:
            similarity_scores.append(np.nan)
            continue
        avg_vector = user_df[feature_cols].mean().values.reshape(1, -1)
        current_vector = row[feature_cols].values.reshape(1, -1)
        score = cosine_similarity(current_vector, avg_vector)[0][0]
        similarity_scores.append(score)
    return similarity_scores


def compute_user_zscore_deviation(df, numeric_cols):
    z_df = df.copy()
    for col in numeric_cols:
        z_df[f'{col}_z'] = z_df.groupby('user_id')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-5)
        )
    z_df['user_deviation_score'] = z_df[[f"{c}_z" for c in numeric_cols]].abs().mean(axis=1)
    return z_df['user_deviation_score']


def main():
    # === Load Data ===
    df = pd.read_csv("Data/ghostpattern_sessions.csv")
    print("Raw data loaded.")

    # === Define columns - 'vpn', 'recent_device_change', and 'browser_language' features are removed ===
    categorical_cols = ['device_os', 'ip_country',
                        'login_day_of_week', 'device_id']
    numerical_cols = ['login_hour', 'typing_speed_cpm',
                      'session_duration_sec', 'mouse_movement_rate',
                      'ip_consistency_score', 'geo_distance_from_usual',
                      'failed_login_attempts_last_24h']
    binary_cols = []

    # === No amplified VPN feature, as requested ===
    
    # === Frequency encoding ===
    print("Encoding categorical features with frequency encoding...")
    frequency_maps = {}
    for col in categorical_cols:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[f"{col}_encoded"] = df[col].map(freq_map).fillna(0.0)
        frequency_maps[col] = freq_map

    os.makedirs("models", exist_ok=True)
    joblib.dump(frequency_maps, "models/frequency_encodings.pkl")

    # Drop original categorical columns
    df.drop(columns=categorical_cols, inplace=True)

    # === Standardize numerical columns ONLY ===
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    joblib.dump(scaler, "models/feature_scaler.pkl")

    # === Cast binary columns to float without standardization ===
    # No binary columns to cast now

    # === Define feature columns ===
    encoded_cols = [f"{col}_encoded" for col in categorical_cols]
    base_features = numerical_cols + binary_cols + encoded_cols

    # === Add behavioral features, but not user_deviation_score, as requested ===
    print("Computing behavioral features...")
    df['user_similarity_score'] = compute_user_similarity(df, base_features)
    df['user_similarity_score'] = df['user_similarity_score'].fillna(df['user_similarity_score'].mean())
    # The 'user_deviation_score' feature is not computed as requested

    # Final features
    feature_cols = base_features + ['user_similarity_score']
    
    # === Feature selection based on correlation matrix ===
    print("\nChecking for highly correlated features...")
    
    # Create a DataFrame for correlation analysis
    correlation_df = df[feature_cols].copy()
    
    # Calculate the correlation matrix
    corr_matrix = correlation_df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than a threshold (e.g., 0.9)
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        
    if to_drop:
        print(f"Features to consider dropping due to high correlation (>0.9): {to_drop}")
        # Drop the identified features from the final feature list
        feature_cols_filtered = [f for f in feature_cols if f not in to_drop]
        print(f"Original feature count: {len(feature_cols)}")
        print(f"Filtered feature count: {len(feature_cols_filtered)}")
        X = df[feature_cols_filtered]
        joblib.dump(feature_cols_filtered, "models/feature_columns.pkl")
    else:
        print("No features with high correlation found. All features will be used.")
        X = df[feature_cols]
        joblib.dump(feature_cols, "models/feature_columns.pkl")
    
    # === Plotting the final correlation matrix ===
    print("\nGenerating correlation heatmap for the filtered features...")
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Feature Correlation Matrix (Filtered)')
    plt.tight_layout()
    plt.savefig('plots/feature_correlation_heatmap.png')
    plt.show()

    y = df['label'] if 'label' in df.columns else None

    # === Train model ===
    print("Training Isolation Forest...")
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    model.fit(X)
    joblib.dump(model, "models/isolation_forest_behavioral.pkl")
    print("Model saved to models/isolation_forest_behavioral.pkl")
    
    # === Evaluate model ===
    scores = model.decision_function(X)

    def score_to_risk(score):
        if score <= -0.05:
            return 'High'
        elif score <= -0.005:
            return 'Medium'
        elif score <= 0.005:
            return 'Low'
        else:
            return 'None'

    risk_levels = [score_to_risk(s) for s in scores]
    predicted_labels = np.array([-1 if rl == 'High' else 1 for rl in risk_levels])

    plot_df = pd.DataFrame({
        'score': scores,
        'risk_level': risk_levels,
        'predicted_label': predicted_labels,
        'label': y if y is not None else predicted_labels
    })

    print("\nRisk level distribution:")
    print(plot_df['risk_level'].value_counts())

    if y is not None:
        print("\nClassification Report:")
        print(classification_report(y, predicted_labels, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y, predicted_labels))

    plt.figure(figsize=(10, 6))
    sns.histplot(data=plot_df, x='score', hue='label', kde=True, bins=30,
                  palette='coolwarm', multiple='stack')
    plt.title("Anomaly Score Distribution by Label")
    plt.xlabel("Score (Higher = More Normal)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig("plots/anomaly_score_plot.png")
    plt.show()

    print(f"\nAnomaly detection complete. Detected {sum(predicted_labels == -1)} anomalies out of {len(X)} sessions.")
    
    return model, X, y

def plot_partial_dependence(model, X, feature, num_points=20):
    values = np.linspace(X[feature].min(), X[feature].max(), num_points)
    scores = []
    for val in values:
        X_temp = X.copy()
        X_temp[feature] = val
        scores.append(model.decision_function(X_temp).mean())

    plt.figure(figsize=(8, 4))
    plt.plot(values, scores, marker='o')
    plt.title(f"PDP: {feature} vs Mean Anomaly Score")
    plt.xlabel(feature)
    plt.ylabel("Mean Anomaly Score")
    plt.grid(True)
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"plots/pdp_{feature}.png")
    plt.show()


if __name__ == "__main__":
    model, X, y = main()
    # No partial dependence plots are generated in this version due to feature removals
    