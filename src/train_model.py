# src/train_model.py
print("train_model.py is starting...")

import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from data_loader import load_csv_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Load and preprocess data
    X, y = load_csv_dataset(r"C:\Users\zhuheng\Documents\VSC\Machine Learning\Finverse\Data\ghostpattern_sessions.csv")
    
    print("Data loaded and preprocessed.")
    # Train Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(X)
    print ("Model trained.")
    # Predict anomaly scores
    scores = model.decision_function(X)
    print("Scores predicted.")
    predictions = model.predict(X)  # 1 = normal, -1 = anomaly

    print ("Done predicting scores.")
    # Show distribution of scores
    
    #sns.histplot(scores, kde=True, hue=(y if y is not None else predictions))
    #plt.title("Anomaly Score Distribution")
    #plt.xlabel("Score (higher = more normal)")
    #plt.grid()
    #plt.show()
    
    # Prepare data for plotting
    plot_df = pd.DataFrame({
        'score': scores,
        'label': y if y is not None else predictions
    })

    print("Preparing data for plotting...")
    # Plot using seaborn
    sns.histplot(plot_df['score'], kde=True, bins=30)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score (higher = more normal)")
    plt.grid()
    plt.savefig("anomaly_score_plot.png")
    plt.show()

    
    # Print quick stats
    print("\nTraining complete.")
    print(f"Model predicts {sum(predictions == -1)} anomalies out of {len(X)} sessions.")

    # Save model to file
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/isolation_forest.pkl")
    print("Model saved to models/isolation_forest.pkl")

if __name__ == "__main__":
    main()
