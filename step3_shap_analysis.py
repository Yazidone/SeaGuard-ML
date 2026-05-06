import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def perform_shap_analysis(data_file='processed_maritime_data.csv'):
    # Load model and data
    try:
        model = joblib.load('xgboost_model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        df = pd.read_csv(data_file)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please make sure step1 and step2 are completed.")
        return

    # Extract a portion of data for explanation (to save time and memory)
    # We'll take the latest 500 samples as background data for SHAP
    df = df.sort_values('datetime').reset_index(drop=True)
    X = df[feature_columns]
    X_sample = X.tail(500)

    # 1. Initialize shap.TreeExplainer for the trained model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 2. Generate Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Comprehensive Feature Importance - Summary Plot", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    print("Summary plot saved to shap_summary_plot.png")
    
    """
    Comment for Decision Maker regarding Summary Plot:
    - This plot shows the importance of each feature and its impact on the prediction (incidents).
    - Variables at the top are the most impactful (e.g., wave height, visibility, wind speed).
    - Colors represent the variable's value (Red = High, Blue = Low).
    - The position of the points (left or right of zero) shows how the value affects the prediction.
    - Strategy: Based on these top features, rescue teams can focus resources on days exceeding critical thresholds.
    """

    # 3. Generate Force Plot for a specific single case
    # Select the case with the highest predicted incidents as a peak example
    predictions = model.predict(X_sample)
    peak_idx = predictions.argmax()
    
    # Force Plot for the worst-case scenario
    force_plot = shap.force_plot(explainer.expected_value, shap_values[peak_idx,:], X_sample.iloc[peak_idx,:])
    shap.save_html('shap_force_plot.html', force_plot)
    print("Force plot saved to shap_force_plot.html")
    
    """
    Comment for Decision Maker regarding Force Plot:
    - This plot explains why the model made this specific prediction.
    - Red arrows (Pushing higher) drive the incident prediction up.
    - Blue arrows (Pushing lower) pull the prediction down (safety factors).
    - Strategy: If "wave height" is the biggest driver, IoT systems can be directed to monitor waves and alert ports in real-time.
    """

if __name__ == "__main__":
    perform_shap_analysis()
