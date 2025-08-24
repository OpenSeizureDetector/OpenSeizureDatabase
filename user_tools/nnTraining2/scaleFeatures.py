import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def scale_features(feature_list):
    """
    Applies Min-Max scaling to a list of feature dictionaries.

    Args:
        feature_list (list): A list of dictionaries, where each dictionary
                             contains the features for a single epoch.

    Returns:
        tuple: A tuple containing:
            - A pandas DataFrame with the scaled features.
            - A list of the original feature names.
    """
    if not feature_list:
        return pd.DataFrame(), []

    # Convert the list of dictionaries into a Pandas DataFrame
    df = pd.DataFrame(feature_list)
    
    # Store the feature names for later use
    feature_names = df.columns.tolist()
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    
    # Fit and transform the data
    scaled_features = scaler.fit_transform(df)
    
    # Convert the scaled features back to a DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
    
    return scaled_df, feature_names

# --- Example Usage ---
if __name__ == '__main__':
    # We will use dummy data that mimics the output of the previous function.
    # Each dictionary represents a single epoch.
    dummy_features = [
        {'activity_count_x': 7.30, 'total_power_x_seizure_main': 101.00, 'peak_psd_x_seizure_main': 67.45},
        {'activity_count_x': 15.12, 'total_power_x_seizure_main': 450.50, 'peak_psd_x_seizure_main': 301.20},
        {'activity_count_x': 8.55, 'total_power_x_seizure_main': 120.80, 'peak_psd_x_seizure_main': 80.60},
    ]

    print("Original Features:")
    for features in dummy_features:
        print(features)

    # Scale the features using the new function
    scaled_df, feature_names = scale_features(dummy_features)

    print("\nScaled Features:")
    print(scaled_df)

    # You can now use 'scaled_df' and 'feature_names' for model training.
    # For instance, you could convert it back to a list of dictionaries if needed.
    scaled_features_list = scaled_df.to_dict('records')
    print("\nScaled Features (as list of dictionaries):")
    for features in scaled_features_list:
        print(features)