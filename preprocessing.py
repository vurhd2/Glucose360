import os
import glob
import pandas as pd
import numpy as np
import configparser

# globals for glucose values to replace "Low" and "High" with in the CGM data
LOW = 40
HIGH = 400

def import_directory(
    path: str,
    glucose: str = "Glucose Value (mg/dL)",
    time: str = "Timestamp (YYYY-MM-DDThh:mm:ss)",
    interval: int = 5,
    max_gap: int = 45,
) -> pd.DataFrame:
    """
    Returns a Multiindexed Pandas DataFrame containing all of the csv data found in the directory at the given path.
    The DataFrame holds columns for DateTime and Glucose Value, and is indexed by 'id'
    @param path    the path of the directory to be parsed through
    @param glucose_col   the header of the column containing the glucose values
    """
    if not os.path.isdir(path):
        raise ValueError("Directory does not exist")

    global glucose_name
    glucose_name = glucose

    global time_name
    time_name = time

    global resample_interval
    resample_interval = interval

    config = configparser.ConfigParser()
    config['variables'] = {'glucose': glucose, 'time': time, 'interval': interval}
    
    with open('config.ini', 'w') as configfile:
      config.write(configfile)

    csv_files = glob.glob(path + "/*.csv")

    data = pd.concat(import_data(file, interval, max_gap) for file in csv_files)
    data = data.set_index(["id"])

    print(f"{len(csv_files)} .csv files were found in the specified directory.")

    return data

def import_data(path: str, interval: int = 5, max_gap = int) -> pd.DataFrame:
    """
    Returns a pre-processed Pandas DataFrame containing the timestamp and glucose data for the csv file at the given path.
    The DataFrame returned has three columns, the DateTime, Glucose Value, and 'id' of the patient
    @param path    the path of the csv file to be pre-processed and read into a Pandas Dataframe
    """
    df = pd.read_csv(path)

    id = df["Patient Info"].iloc[0] + df["Patient Info"].iloc[1]
    df["id"] = id

    df = df.dropna(subset=[glucose_name])

    df = df.replace("Low", LOW)
    df = df.replace("High", HIGH)

    df[time_name] = pd.to_datetime(df[time_name], format="%Y-%m-%dT%H:%M:%S")

    df[glucose_name] = pd.to_numeric(df[glucose_name])

    df = df[[time_name, glucose_name]].copy()
    df = resample_data(df, interval, max_gap)
    df = df.dropna(subset=[glucose_name])
    df = chunk_day(chunk_time(df))
    df["id"] = id

    return df

def resample_data(df: pd.DataFrame, minutes: int = 5, max_gap = int) -> pd.DataFrame:
    """
    Resamples and (if needed) interpolates the given default-indexed DataFrame.
    Used mostly to preprocess the data in the csv files being imported in import_data().
    @param df         the DataFrame to be resampled and interpolated
    @param minutes    the length of the interval to be resampled into (in minutes)
    """
    # Sort the DataFrame by datetime
    resampled_df = df.sort_values(by=[time_name])
    resampled_df = resampled_df.set_index(time_name)

    interval = str(minutes) + "T"
    
    # generate the times that match the frequency
    resampled_df = resampled_df.asfreq(interval)
    # add in the original points that don't match the frequency (just for linear time-based interpolation)
    resampled_df.reset_index(inplace=True)
    resampled_df = (pd.concat([resampled_df, df])).drop_duplicates(subset=[time_name])
    resampled_df.sort_values(by=[time_name], inplace=True)

    # interpolate the missing values
    resampled_df.set_index(time_name, inplace=True)
    resampled_df = interpolate_data(resampled_df, max_gap)
    
    # now that the values have been interpolated, remove the points that don't match the frequency
    resampled_df = resampled_df.asfreq(interval)
    resampled_df.reset_index(inplace=True)

    return resampled_df

def interpolate_data(df: pd.DataFrame, max_gap = int) -> pd.DataFrame:
    """
    Only linearly interpolates NaN glucose values for time gaps that are less than the given number of minutes.
    Used mainly in preprocessing for csv files that are being imported in import_data().
    @param df         a DataFrame with only two columns, DateTime and Glucose Value
    @param max_gap    the maximum minute length of gaps that should be interpolated
    """

    # based heavily on https://stackoverflow.com/questions/67128364/how-to-limit-pandas-interpolation-when-there-is-more-nan-than-the-limit

    s = df[glucose_name].notnull()
    s = s.ne(s.shift()).cumsum()

    m = df.groupby([s, df[glucose_name].isnull()])[glucose_name].transform('size').where(df[glucose_name].isnull())
    interpolated_df = df.interpolate(method="time", limit_area="inside").mask(m >= int(max_gap / resample_interval))

    return interpolated_df

def chunk_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column specifying whether the values occur during a waking or sleeping period
    """
    times = df[time_name] - df[time_name].dt.normalize()
    is_waking = (times >= pd.Timedelta(hours=8)) & (times <= pd.Timedelta(hours=22))
    df["Time Chunking"] = is_waking.replace({True: "Waking", False: "Sleeping"})
    return df

def chunk_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column specifying whether the values occur during the week or during the weekend
    """
    is_weekend = df[time_name].dt.dayofweek > 4
    df["Day Chunking"] = is_weekend.replace({True: "Weekend", False: "Weekday"})
    return df