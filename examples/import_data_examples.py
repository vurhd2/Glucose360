"""
This module demonstrates various ways to import CGM (Continuous Glucose Monitor) data using the glucose360 package.
It provides five different examples showing how to use the import_data function with different configurations:

1. Basic Import: Import a single Dexcom CSV file using default settings
2. Directory Import: Import all Dexcom CSV files from a directory and combine them
3. Custom Column Format: Import data with custom column names and format
4. Custom ID Template: Create patient IDs from CSV content using a template (e.g., last__first)
5. Filename ID: Use filenames (minus .csv) as patient IDs

Each example demonstrates a different use case and configuration option of the import_data function.
The examples are designed to show how to handle different data formats.

Usage:
    Run this file directly to see all examples in action:
    python import_data_examples.py
"""

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os
import pandas as pd
from pathlib import Path
import glob
from glucose360.preprocessing import import_data

# 1. Import a single Dexcom CSV file
# 2. Import all Dexcom CSV files from a directory
# 3. Import data with custom column format
# 4. Import Dexcom data with custom ID template
# 5. Import Dexcom data with ID from filename

def example_1_single_dexcom():
    """
    Example 1: Import a single Dexcom CSV file
    Demonstrates the basic usage of import_data with a single Dexcom CSV file.
    """
    # Get the absolute path to the datasets directory
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    
    # Find the first CSV file in the datasets directory
    csv_files = glob.glob(os.path.join(datasets_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in the datasets directory!")
        return
    
    # Use the first CSV file found
    input_file = csv_files[0]
    print(f"\nImporting file: {os.path.basename(input_file)}")
    
    try:
        # Import the data using default settings
        df = import_data(input_file)
        
        print("\nSuccessfully imported data!")
        print("\nFirst 5 rows of imported data:")
        print(df.head())
        print("\nDataFrame Info:")
        print(df.info())
        print("-" * 80)
        
    except Exception as e:
        print(f"Error importing file: {str(e)}")

def example_2_directory_dexcom():
    """
    Example 2: Import all Dexcom CSV files from a directory
    Demonstrates how to import multiple Dexcom CSV files using import_data's built-in directory handling.
    """
    # Get the absolute path to the datasets directory
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    
    print(f"\nImporting all CSV files from: {datasets_dir}")
    
    try:
        # Import all CSV files from the directory using import_data
        df = import_data(datasets_dir)
        
        print("\n=== Import Summary ===")
        print(f"Total number of rows: {len(df)}")
        print(f"Number of unique patients: {df.index.unique().size}")
        print("\nFirst 5 rows of imported data:")
        print(df.head())
        print("\nDataFrame Info:")
        print(df.info())
        print("-" * 80)
        
    except Exception as e:
        print(f"Error importing directory: {str(e)}")

def example_3_custom_column_format():
    """
    Example 3: Import data with custom column format
    """
    # Get the absolute path to the datasets directory
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    
    # Create a directory for converted files
    converted_dir = os.path.join(datasets_dir, "converted")
    os.makedirs(converted_dir, exist_ok=True)
    
    # Convert Dexcom Clarity exports to custom format as an example
    print("\n=== Step 1: Convert Dexcom Clarity exports to custom format ===")
    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(datasets_dir, "*.csv"))
    
    converted_files = []
    for file in csv_files:
        try:
            print(f"\nConverting file: {os.path.basename(file)}")
            # Convert the file to custom format
            converted_df = convert_dexcom_clarity(file)
            
            # Save the converted file in the converted directory
            output_file = os.path.join(converted_dir, f"converted_{os.path.basename(file)}")
            converted_df.to_csv(output_file, index=False)
            converted_files.append(output_file)
            
            print(f"Successfully converted and saved to: {output_file}")
            print("\nFirst 5 rows of converted data:")
            print(converted_df.head())
            print("-" * 80)
        except Exception as e:
            print(f"Error converting {file}: {str(e)}")
    
    print("\n=== Step 2: Import converted files using custom column names ===")
    for file in converted_files:
        try:
            print(f"\nImporting converted file: {os.path.basename(file)}")
            # Import using custom column names
            df = import_data(
                file,
                sensor="columns",
                id_template="patient_id",
                glucose="blood_sugar",
                time="measurement_time"
            )
            print("\nFirst 5 rows of imported data:")
            print(df.head())
            print("-" * 80)
        except Exception as e:
            print(f"Error importing {file}: {str(e)}")

def example_4_custom_id_template():
    """
    Example 4: Import Dexcom data with custom ID template
    Demonstrates how to use a custom ID template to create patient IDs from the CSV file's
    patient information fields. In this example, we use the format 'last__first' where
    the last name and first name are separated by double underscores.
    """
    # Get the absolute path to the datasets directory
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    
    print(f"\nImporting data with custom ID template (last__first)")
    
    try:
        # Import data using the custom ID template
        # The template '{last}__{first}' will be formatted with the patient's
        # last name and first name from the CSV file. Be careful with PHI if it is PHI in your dataset
        df = import_data(
            datasets_dir,
            id_template="{last}__{first}",  # Template to format last and first names
            sensor="dexcom"  # Specify sensor type explicitly
        )
        
        print("\n=== Import Summary ===")
        print(f"Total number of rows: {len(df)}")
        print(f"Number of unique patients: {df.index.unique().size}")
        print("\nUnique patient IDs:")
        print(df.index.unique())
        print("\nFirst 5 rows of imported data:")
        print(df.head())
        print("\nDataFrame Info:")
        print(df.info())
        print("-" * 80)
        
    except Exception as e:
        print(f"Error importing data with custom ID template: {str(e)}")

def example_5_id_from_filename():
    """
    Example 5: Import Dexcom data with ID from filename
    Demonstrates how to use the entire filename (minus .csv extension) as the patient ID.
    """
    # Get the absolute path to the datasets directory
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    
    print(f"\nImporting data with IDs from filenames")
    
    try:
        # Import data using the entire filename (minus .csv) as the ID
        # The pattern '(?P<id>.+)\.csv$' will:
        # - Capture everything before '.csv' at the end of the filename as the ID
        df = import_data(
            datasets_dir,
            id_template="(?P<id>.+)\.csv$"  # Capture everything before .csv as the ID
        )
        
        print("\n=== Import Summary ===")
        print(f"Total number of rows: {len(df)}")
        print(f"Number of unique patients: {df.index.unique().size}")
        print("\nUnique patient IDs:")
        print(df.index.unique())
        print("\nFirst 5 rows of imported data:")
        print(df.head())
        print("\nDataFrame Info:")
        print(df.info())
        print("-" * 80)
        
    except Exception as e:
        print(f"Error importing data with IDs from filenames: {str(e)}")

def convert_dexcom_clarity(input_file):
    """
    Convert a Dexcom Clarity export file into a custom format that can be imported
    using the import_data function with custom column names.
    
    Args:
        input_file (str): Path to the Dexcom Clarity export CSV file
    
    Returns:
        pd.DataFrame: The converted DataFrame with custom column names
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Extract patient info from the first few rows
    patient_info = {}
    for _, row in df.iloc[:4].iterrows():
        if pd.notna(row['Event Type']):
            if row['Event Type'] == 'FirstName':
                patient_info['FirstName'] = row['Patient Info']
            elif row['Event Type'] == 'LastName':
                patient_info['LastName'] = row['Patient Info']
    
    # Create patient ID
    patient_id = f"{patient_info.get('FirstName', 'Unknown')}_{patient_info.get('LastName', 'Unknown')}"
    
    # Get the actual glucose data (skip the first 4 rows which contain patient info)
    glucose_data = df.iloc[4:].copy()
    
    # Filter only rows where Event Type is 'EGV' (Estimated Glucose Value)
    glucose_data = glucose_data[glucose_data['Event Type'] == 'EGV']
    
    # Convert timestamp column to datetime
    glucose_data['Timestamp (YYYY-MM-DDThh:mm:ss)'] = pd.to_datetime(glucose_data['Timestamp (YYYY-MM-DDThh:mm:ss)'])
    
    # Create the output DataFrame with custom column names
    output_df = pd.DataFrame({
        'patient_id': patient_id,
        'blood_sugar': glucose_data['Glucose Value (mg/dL)'],
        'measurement_time': glucose_data['Timestamp (YYYY-MM-DDThh:mm:ss)']
    })
    
    # Remove any rows with missing values
    output_df = output_df.dropna()
    
    # Sort by timestamp
    output_df = output_df.sort_values('measurement_time')
    
    return output_df

def demonstrate_all_examples():
    """
    Demonstrate all import data examples
    """
    print("\n=== Example 1: Import a single Dexcom CSV file ===")
    example_1_single_dexcom()
    
    print("\n=== Example 2: Import all Dexcom CSV files from a directory ===")
    example_2_directory_dexcom()
    
    print("\n=== Example 3: Import data with custom column format ===")
    example_3_custom_column_format()
    
    print("\n=== Example 4: Import Dexcom data with custom ID template ===")
    example_4_custom_id_template()
    
    print("\n=== Example 5: Import Dexcom data with ID from filename ===")
    example_5_id_from_filename()

if __name__ == "__main__":
    demonstrate_all_examples() 