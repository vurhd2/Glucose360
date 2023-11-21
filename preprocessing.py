import os
import glob
import pandas as pd
import numpy as np

# globals for glucose values to replace "Low" and "High" with in the CGM data
LOW = 40
HIGH = 400

"""
Returns a Multiindexed Pandas DataFrame containing all of the csv data found in the directory at the given path.
The DataFrame holds columns for DateTime and Glucose Value, and is indexed by 'id'
@param path    the path of the directory to be parsed through
@param glucose_col   the header of the column containing the glucose values
"""
def import_directory(path: str, glucose_col: str = "Glucose Value (mg/dL)", 
                     time_col: str = "Timestamp (YYYY-MM-DDThh:mm:ss)", interval: int = 5) -> pd.DataFrame:
   
   if not os.path.isdir(path):
      raise ValueError("Directory does not exist") 
   
   global glucose_name
   glucose_name = glucose_col

   global time_name
   time_name = time_col

   global resample_interval
   resample_interval = interval

   csv_files = glob.glob(path + "/*.csv")

   data = pd.concat(import_data(file, interval) for file in csv_files)
   data = data.set_index(['id'])

   print(f"{len(csv_files)} .csv files were found in the specified directory.")

   return data

"""
Returns a pre-processed Pandas DataFrame containing the timestamp and glucose data for the csv file at the given path.
The DataFrame returned has three columns, the DateTime, Glucose Value, and 'id' of the patient
@param path    the path of the csv file to be pre-processed and read into a Pandas Dataframe
"""
def import_data(path: str, interval: int = 5) -> pd.DataFrame:
   df = pd.read_csv(path)

   #id = df["Patient Info"].iloc[0] + df["Patient Info"].iloc[1] + df["Patient Info"].iloc[2]
   id = df["Patient Info"].iloc[0] + df["Patient Info"].iloc[1]
   df['id'] = id

   df = df.dropna(subset=[glucose_name])

   df = df.replace("Low", LOW)
   df = df.replace("High", HIGH)

   df[time_name] = pd.to_datetime(df[time_name], format='%Y-%m-%dT%H:%M:%S')

   df[glucose_name] = pd.to_numeric(df[glucose_name])

   df = df[[time_name, glucose_name]].copy()
   df = resample_data(df, interval)
   df = chunk_day(chunk_time(df))
   df['id'] = id

   return df

"""
Resamples and (if needed) interpolates the given default-indexed DataFrame.
Used mostly to preprocess the data in the csv files being imported in import_data().
@param df         the DataFrame to be resampled and interpolated
@param minutes    the length of the interval to be resampled into (in minutes)
"""
def resample_data(df: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
   # Sort the DataFrame by datetime
   df.sort_values(by=[time_name], inplace=True)
   
   interval = str(minutes) + 'T'

   df = df.set_index(time_name)

   resampled_df = df.resample(interval, origin='start').mean()
   resampled_df = interpolate_data(resampled_df) 

   resampled_df.reset_index(inplace=True)

   return resampled_df

"""
Only linearly interpolates NaN glucose values for time gaps that are less than the given number of minutes.
Used mainly in preprocessing for csv files that are being imported in import_data().
@param df         a DataFrame with only two columns, DateTime and Glucose Value
@param max_gap    the maximum minute length of gaps that should be interpolated
"""
def interpolate_data(df: pd.DataFrame, max_gap: int = 45) -> pd.DataFrame:
   df.reset_index(inplace=True)

   gaps = []
   index_range = {}
   for i in range(len(df) - 1):
      is_nan = lambda x : np.isnan(df.iloc[x][glucose()])
      timestamp = lambda x : df.iloc[x][time()]
      length = len(index_range)

      # start of time gap needing interpolation
      if length == 0 and not is_nan(i) and is_nan(i+1):
         index_range['start'] = timestamp(i)

      # end of time gap needing interpolation
      elif length == 1 and is_nan(i) and not is_nan(i+1):
         index_range['end'] = timestamp(i+1)
         if index_range['end'] - index_range['start'] <= pd.Timedelta(minutes=max_gap):
            gaps.append(index_range)
         index_range = {}
   
   df.set_index(time(), inplace=True)
   # interpolate only the necessary gaps of time
   for gap in gaps:
      df.loc[gap['start']:gap['end']] = df.loc[gap['start']:gap['end']].copy().interpolate('linear', limit_area='inside')
   
   return df

"""
Adds a new column specifying whether the values occur during a waking or sleeping period
"""
def chunk_time(df: pd.DataFrame) -> pd.DataFrame:
   times = df[time()] - df[time()].dt.normalize()
   is_waking = (times >= pd.Timedelta(hours=8)) & (times <= pd.Timedelta(hours=22))
   df["Time Chunking"] = is_waking.replace({True: "Waking", False: "Sleeping"})
   return df

"""
Adds a new column specifying whether the values occur during the week or during the weekend
"""
def chunk_day(df: pd.DataFrame) -> pd.DataFrame:
   is_weekend = df[time()].dt.dayofweek > 4
   df["Day Chunking"] = is_weekend.replace({True: "Weekend", False: "Weekday"})
   return df


# ---------------- Global Variables representing the DateTime, Glucose Value, and Resampling Interval column names ------------

def glucose() -> str:
   return glucose_name

def time() -> str:
   return time_name

def interval() -> int:
   return resample_interval