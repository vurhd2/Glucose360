import glob
import pandas as pd
import numpy as np

"""
Returns a Multiindexed Pandas DataFrame containing all of the csv data found in the directory at the given path
@param path    the path of the directory to be parsed through
@param glucose_col   the header of the column containing the glucose values
"""
def import_directory(path, glucose_col="Glucose Value (mg/dL)", time_col="Timestamp (YYYY-MM-DDThh:mm:ss)", interval=5):
   global glucose_name
   glucose_name = glucose_col

   global time_name
   time_name = time_col

   global resample_interval
   resample_interval = interval

   csv_files = glob.glob(path + "/*.csv")

   data = pd.DataFrame()
   for file in csv_files:
      df = import_data(file, interval)

      data = pd.concat([data, df])

   data = data.set_index(['id'])

   return data

"""
Returns a pre-processed Pandas DataFrame containing the timestamp and glucose data for the csv file at the given path
@param path    the path of the csv file to be pre-processed and read into a Pandas Dataframe
"""
def import_data(path, interval=5):
   df = pd.read_csv(path)

   #id = df["Patient Info"].iloc[0] + df["Patient Info"].iloc[1] + df["Patient Info"].iloc[2]
   id = df["Patient Info"].iloc[0] + df["Patient Info"].iloc[1]
   df['id'] = id

   df = df.dropna(subset=[glucose_name])
   df = df.replace("Low", 40)
   df = df.replace("High", 400)

   df[time_name] = pd.to_datetime(df[time_name], format='%Y-%m-%dT%H:%M:%S')

   df[glucose_name] = pd.to_numeric(df[glucose_name])

   df = df[[time_name, glucose_name]].copy()
   df = resample_data(df, interval)
   df['id'] = id

   return df

"""
Resamples and (if needed) interpolates the given default-indexed DataFrame
@param df         the DataFrame to be resampled and interpolated
@param minutes    the length of the interval to be resampled into (in minutes)
"""
def resample_data(df, minutes=5):
   # Sort the DataFrame by datetime
   df.sort_values(by=[time_name], inplace=True)
   
   interval = str(minutes) + 'T'

   df = df.set_index(time_name)

   resampled_df = df.resample(interval, origin='start').mean()
   resampled_df = interpolate_data(resampled_df) 

   resampled_df.reset_index(inplace=True)

   return resampled_df

"""
Only interpolates NaN glucose values for time gaps that are less than the given number of minutes
@param df         a DataFrame with only two columns, DateTime and Glucose Value
@param max_gap    the maximum minute length of gaps that should be interpolated
"""
def interpolate_data(df, max_gap=30):
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

def glucose():
   return glucose_name

def time():
   return time_name

def interval():
   return resample_interval