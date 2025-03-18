"""
Basic example demonstrating how to import CGM data and calculate features using glucose360.
This example uses the sample data provided in the datasets directory.
"""

import os
import pandas as pd
from glucose360 import preprocessing, features

# Path to the datasets directory (relative to this script)
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")

def main():
    # Import the CGM data from the datasets directory
    # The data is in Dexcom format, so we'll use the default sensor type
    print("Importing CGM data...")
    cgm_data = preprocessing.import_data(
        path=DATASETS_DIR,
        sensor="dexcom",  # This is the default, but we'll specify it for clarity
        interval=5,       # 5-minute intervals (default)
        max_gap=45       # Maximum gap of 45 minutes to interpolate (default)
    )
    
    print(f"\nImported data shape: {cgm_data.shape}")
    print(f"Number of patients: {len(cgm_data.index.unique())}")
    
    # Calculate features for all patients
    print("\nCalculating features...")
    feature_df = features.create_features(cgm_data)
    
    # Display the features
    print("\nFeatures calculated for each patient:")
    print(feature_df)
    
    # Save the features to a CSV file
    output_file = "cgm_features.csv"
    feature_df.to_csv(output_file)
    print(f"\nFeatures saved to {output_file}")
    
    # Example of calculating individual metrics for the first patient
    first_patient = cgm_data.index[0]
    patient_data = cgm_data.loc[first_patient]
    
    print(f"\nDetailed metrics for patient {first_patient}:")
    print(f"Mean glucose: {features.mean(patient_data):.1f} mg/dL")
    print(f"Standard deviation: {features.SD(patient_data):.1f} mg/dL")
    print(f"Coefficient of variation: {features.CV(patient_data):.1f}%")
    print(f"Time in range (70-180 mg/dL): {features.percent_time_in_range(patient_data):.1f}%")
    print(f"Time in hypoglycemia (<70 mg/dL): {features.percent_time_in_hypoglycemia(patient_data):.1f}%")
    print(f"Time in hyperglycemia (>180 mg/dL): {features.percent_time_in_hyperglycemia(patient_data):.1f}%")
    print(f"Estimated A1C: {features.eA1c(patient_data):.1f}%")
    print(f"Glucose Management Indicator: {features.GMI(patient_data):.1f}%")

if __name__ == "__main__":
    main() 