#features.py

import pandas as pd
import numpy as np
from scipy.integrate import trapezoid


def resample_cgm_data(cgm_df, time_column='datetime_local', id_column='id'):
    """
    Resample CGM data to 5-minute intervals for each subject.
    
    Parameters:
    cgm_df (DataFrame): The original CGM DataFrame
    time_column (str): The column name for the datetime
    id_column (str): The column name for the subject ID
    
    Returns:
    DataFrame: A new DataFrame with resampled data
    """
    
    # Sort the DataFrame by subject ID and datetime
    cgm_df.sort_values(by=[id_column, time_column], inplace=True)
    
    # Set the datetime column as the index
    cgm_df.set_index(time_column, inplace=True)
    
    # Group by each subject and resample to 5-minute intervals
    resampled_df = cgm_df.groupby(id_column).resample('5T').mean()

    
    # Drop the 'id' level from the index
    resampled_df.index = resampled_df.index.droplevel(id_column)
    
    # Reset only the time index to make it a column
    resampled_df.reset_index(level=0, inplace=True)
    
    # Rename the time column to its original name ('datetime_local')
    resampled_df.rename(columns={time_column: 'datetime_local'}, inplace=True)
    
    # Now reset the 'id' index to make it a regular column
    resampled_df.reset_index(inplace=True)

    resampled_df.drop(columns=['index'], inplace=True)
    resampled_df.dropna(axis= 'index', subset = ['id'], inplace=True)
    
    return resampled_df


# Helper functions
def post_event_filter(cgm_df, glucose_col = 'glucose'):
    return cgm_df[cgm_df['mins_since_start']>=0]

# Generic glucose features
def AUC(cgm_df, glucose_col = 'glucose'):
    return trapezoid(cgm_df[glucose_col],dx = 5)

def AUC_above_x(cgm_df, x, glucose_col = 'glucose'):
    new_df = cgm_df.copy()
    new_df[glucose_col] = cgm_df[glucose_col]-x
    new_df.loc[new_df[glucose_col]<0,glucose_col] = 0
    return AUC(new_df)

def AUC_above_140(cgm_df, glucose_col = 'glucose'):
    return AUC_above_x(cgm_df,140,glucose_col=glucose_col)

def AUC_above_180(cgm_df, glucose_col = 'glucose'):
    return AUC_above_x(cgm_df,180,glucose_col=glucose_col)

def CV(cgm_df, glucose_col = 'glucose'):
    return np.std(cgm_df[glucose_col]) / np.mean(cgm_df[glucose_col])

def average_glucose(cgm_df, glucose_col = 'glucose'):
    return np.mean(cgm_df[glucose_col])

def count_samples(cgm_df, glucose_col = 'glucose'):
    return len(cgm_df)

def time_in_range(cgm_df, glucose_col = 'glucose'):
    in_range_df = cgm_df[(cgm_df[glucose_col] >= 70) & (cgm_df[glucose_col] <= 180)]
    time_in_range = len(in_range_df)
    total_time = len(cgm_df)
    return time_in_range / total_time if total_time > 0 else np.nan

def time_above_range(cgm_df, glucose_col = 'glucose'):
    above_range_df = cgm_df[cgm_df[glucose_col] > 180]
    time_above_range = len(above_range_df)
    total_time = len(cgm_df)
    return time_above_range / total_time if total_time > 0 else np.nan

def time_below_range(cgm_df, glucose_col = 'glucose'):
    below_range_df = cgm_df[cgm_df[glucose_col] < 70]
    time_below_range = len(below_range_df)
    total_time = len(cgm_df)
    return time_below_range / total_time if total_time > 0 else np.nan

def percent_coverage(cgm_df, glucose_col = 'glucose'):
    return count_samples(cgm_df)/4032

def days_with_data(cgm_df, glucose_col = 'glucose'):
    return len(cgm_df)/288

def time_in_range_low(cgm_df, glucose_col = 'glucose'):
    in_range_df = cgm_df[(cgm_df[glucose_col] >= 70) & (cgm_df[glucose_col] <= 140)]
    time_in_range = len(in_range_df)
    total_time = len(cgm_df)
    return time_in_range / total_time if total_time > 0 else np.nan

def time_in_range_high(cgm_df, glucose_col = 'glucose'):
    in_range_df = cgm_df[(cgm_df[glucose_col] >= 140) & (cgm_df[glucose_col] <= 180)]
    time_in_range = len(in_range_df)
    total_time = len(cgm_df)
    return time_in_range / total_time if total_time > 0 else np.nan

def maximum_glucose(cgm_df, glucose_col='glucose'):
    return np.max(cgm_df[glucose_col])

def average_maximum_glucose_per_day(cgm_df, glucose_col='glucose'):
    # Resampling the data per day and calculating the maximum for each day
    cgm_df = cgm_df.set_index('datetime_local')
    max_glucose_per_day = cgm_df.resample('D').apply(lambda x: np.max(x[glucose_col]) if len(x) >= 200 else np.nan)
    
    # Drop NaN values (days with less than 200 datapoints)
    max_glucose_per_day.dropna(inplace=True)
    
    # Calculating the average of these maximum values
    return np.mean(max_glucose_per_day)

def variability_of_maximum_per_day(cgm_df, glucose_col='glucose', time_col='datetime_local'):
    """
    Calculate the variability of the maximum glucose value per day.

    Parameters:
    cgm_df (DataFrame): The CGM DataFrame
    glucose_col (str): The column name for the glucose
    time_col (str): The column name for the datetime

    Returns:
    float: The variability of the maximum glucose values per day
    """
    # Set datetime as the index and resample to find the maximum value per day
    cgm_df = cgm_df.set_index(time_col)
    max_glucose_per_day = cgm_df[glucose_col].resample('D').max()
    
    # Calculate and return the variability (standard deviation) of the max glucose values
    variability = np.std(max_glucose_per_day, ddof=1) # using 1 degree of freedom (ddof=1) for sample standard deviation
    
    return variability


# Excursion specific features

def AUC_after_event(cgm_df, glucose_col = 'glucose'):
    return AUC(cgm_df)

def total_length(cgm_df, glucose_col = 'glucose'):
    return len(cgm_df)

def length_after_0min(cgm_df, glucose_col = 'glucose'):
    return len(cgm_df[cgm_df['mins_since_start']>=0])

def peak_value(cgm_df, glucose_col = 'glucose'):
    return np.max(cgm_df[glucose_col])

def time_to_peak(cgm_df, glucose_col = 'glucose'):
    peak_idx = np.argmax(cgm_df[glucose_col])
    return cgm_df.iloc[peak_idx]['mins_since_start']

def baseline_glucose(cgm_df, glucose_col = 'glucose'):
    idx = (abs(cgm_df['mins_since_start'])).idxmin()
    return cgm_df.loc[idx][glucose_col]

def AUC_above_baseline(cgm_df, glucose_col = 'glucose'):
    return AUC_above_x(cgm_df,baseline_glucose(cgm_df,glucose_col=glucose_col),glucose_col=glucose_col)

def glucose_at_x_mins(cgm_df, x, glucose_col = 'glucose'):
    idx = (abs(cgm_df['mins_since_start']-x)).idxmin()
    return cgm_df.loc[idx][glucose_col]

def glucose_at_60_mins(cgm_df, glucose_col = 'glucose'):
    return glucose_at_x_mins(cgm_df, 60, glucose_col=glucose_col)

def glucose_at_120_mins(cgm_df, glucose_col = 'glucose'):
    return glucose_at_x_mins(cgm_df, 120, glucose_col=glucose_col)

def glucose_at_170_mins(cgm_df, glucose_col = 'glucose'):
    return glucose_at_x_mins(cgm_df, 170, glucose_col=glucose_col)

def return_to_baseline_time(cgm_df, glucose_col = 'glucose'):
    bg = baseline_glucose(cgm_df,glucose_col=glucose_col)
    peak_time = time_to_peak(cgm_df,glucose_col=glucose_col)
    new_df = cgm_df[cgm_df['mins_since_start']>peak_time]
    below_baseline = new_df[new_df[glucose_col]<=bg]
    if len(below_baseline)==0:
        return 170
    else:
        return below_baseline.iloc[0]['mins_since_start']

def return_to_baseline(cgm_df, glucose_col = 'glucose'):
    return return_to_baseline_time(cgm_df)<170

def slope_baseline_to_peak(cgm_df, glucose_col = 'glucose'):
    bg = baseline_glucose(cgm_df, glucose_col=glucose_col)
    peak_glucose = peak_value(cgm_df, glucose_col=glucose_col)
    denominator = time_to_peak(cgm_df, glucose_col=glucose_col)
    return (peak_glucose-bg)/denominator

# Call all functions
def calculate_all_features(cgm_df,glucose_col = 'glucose'):
    functions = {
        'whole_time_to_peak':time_to_peak,
        'total_length':total_length,
        'length_after_0min':length_after_0min
        }
    return {f:functions[f](cgm_df) for f in functions.keys()}

def calculate_all_after_event_features(cgm_df, glucose_col = 'glucose'):
    functions = {
        'AUC':AUC,
        'AUC_above_140':AUC_above_140,
        'AUC_above_180':AUC_above_180,
        'CV':CV,
        'average_glucose':average_glucose,
        'peak_value':peak_value,
        'time_to_peak':time_to_peak,
        'baseline_glucose':baseline_glucose,
        'AUC_above_baseline':AUC_above_baseline,
        'glucose_at_60_mins':glucose_at_60_mins,
        'glucose_at_120_mins':glucose_at_120_mins,
        'glucose_at_170_mins':glucose_at_170_mins,
        'return_to_baseline_time':return_to_baseline_time,
        'return_to_baseline':return_to_baseline,
        'slope_baseline_to_peak':slope_baseline_to_peak
    }
    return {f:functions[f](cgm_df) for f in functions.keys()}

def calculate_all_generic_features(cgm_df, glucose_col = 'glucose'):
    functions = {
        'AUC':AUC,
        'AUC_above_140':AUC_above_140,
        'AUC_above_180':AUC_above_180,
        'CV':CV,
        'average_glucose':average_glucose,
        'count_samples':count_samples,
        'time_in_range':time_in_range,
        'time_above_range':time_above_range,
        'time_below_range':time_below_range,
        'percent_coverage':percent_coverage,
        'days_with_data':days_with_data,
        'time_in_range_low':time_in_range_low,
        'time_in_range_high':time_in_range_high,
        'maximum_glucose': maximum_glucose,
        'average_maximum_glucose_per_day': average_maximum_glucose_per_day,
        'variability_of_maximum_per_day': variability_of_maximum_per_day
    }
    return {f:functions[f](cgm_df) for f in functions.keys()}

def calculate_food_challenge_features(cgm_df,glucose_col = 'glucose'):
    result = cgm_df.groupby(['subject','foods','rep','food','mitigator'], dropna=False).apply(calculate_all_features)
    result = result.apply(pd.Series).reset_index()
    filtered_df = post_event_filter(cgm_df, glucose_col=glucose_col)
    post_event_result = filtered_df.groupby(['subject','foods','rep','food','mitigator'], dropna=False).apply(calculate_all_after_event_features)
    post_event_result = post_event_result.apply(pd.Series).reset_index()
    return pd.merge(result,post_event_result, on =['subject','foods','rep','food','mitigator'])

def calculate_AGP_features(cgm_df,glucose_col = 'glucose'):
    result = cgm_df.groupby(['id'], dropna=False).apply(calculate_all_generic_features)
    result = result.apply(pd.Series).reset_index()
    return result


def main():
    cgm_df = pd.read_csv('binge_days_data.csv')
    #cgm_df = pd.read_csv('non_binge_days_data.csv')
    #cgm_df = pd.read_csv('intermediate_data/combined.csv')

    #cgm_df['glucose'].replace('Low', '40', inplace=True)
    
    cgm_df['glucose'] = pd.to_numeric(cgm_df['glucose'], errors='coerce')
    cgm_df['datetime_local'] = pd.to_datetime(cgm_df['datetime_local'])

    #cgm_df.to_csv('cleaned_combined.csv')
   
    #resampled_cgm_df = resample_cgm_data(cgm_df)


    generic_features = calculate_AGP_features(cgm_df)

    # Save the features to a CSV file
    generic_features.to_csv('binge_days_features.csv', index=False)
    #generic_features.to_csv('non_binge_days_features.csv', index=False)
    #generic_features.to_csv('features.csv', index=False)


    print(generic_features)

if __name__ == "__main__":
    main()

# After resampling we are getting nans in cgm data for the ids.
# [301, 308, 313, 314, 315, 325, 326, 328, 334, 336, 337, 338, 340]