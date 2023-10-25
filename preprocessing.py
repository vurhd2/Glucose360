import glob
import pandas as pd

"""
Returns a Multiindexed Pandas DataFrame containing all of the csv data found in the directory at the given path
@param path    the path of the directory to be parsed through
@param glucose_col   the header of the column containing the glucose values
"""
def import_directory(path, glucose_col="Glucose Value (mg/dL)", time_col="Timestamp (YYYY-MM-DDThh:mm:ss)"):
   global glucose_name
   glucose_name = glucose_col

   global time_name
   time_name = time_col

   csv_files = glob.glob(path + "/*.csv")

   data = pd.DataFrame()
   for file in csv_files:
      df = import_data(file)

      data = pd.concat([data, df])

   data = data.set_index(['id'])
   print(data)

   return data

"""
Returns a pre-processed Pandas DataFrame containing the timestamp and glucose data for the csv file at the given path
@param path    the path of the csv file to be pre-processed and read into a Pandas Dataframe
"""
def import_data(path):
   df = pd.read_csv(path)

   id = df["Patient Info"].iloc[0] + df["Patient Info"].iloc[1] + df["Patient Info"].iloc[2]
   df['id'] = id

   df = df.dropna(subset=[glucose_name])
   df = df.replace("Low", 40)
   df = df.replace("High", 400)

   df[time_name] = pd.to_datetime(df[time_name], format='%Y-%m-%dT%H:%M:%S')

   df[glucose_name] = pd.to_numeric(df[glucose_name])

   df = df[[time_name, glucose_name]].copy()
   df = resample_data(df)
   df['id'] = id

   return df

"""
Resample CGM data to 5-minute intervals for each subject.

Parameters:
cgm_df (DataFrame): The original CGM DataFrame
time_column (str): The column name for the datetime
id_column (str): The column name for the subject ID

Returns:
DataFrame: A new DataFrame with resampled data
"""
def resample_data(df, minutes=5):
   # Sort the DataFrame by subject ID and datetime
   df.sort_values(by=[time_name], inplace=True)
   
   interval = str(minutes) + 'T'

   df = df.set_index(time_name)

   resampled_df = df.resample(interval, origin='start').mean()

   resampled_df.interpolate('linear', axis=1, inplace=True) 

   resampled_df.reset_index(inplace=True)

   return resampled_df

def glucose():
   return glucose_name

def time():
   return time_name