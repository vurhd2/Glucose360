import pandas as pd
import glob

"""
Returns a Multiindexed Pandas DataFrame containing all of the csv data found in the directory at the given path
@param path    the path of the directory to be parsed through
@param glucose_col   the header of the column containing the glucose values
"""
def import_directory(path, glucose_col="Glucose Value (mg/dL)", time_col="Timestamp (YYYY-MM-DDThh:mm:ss)"):
   global glucose
   glucose = glucose_col

   global time
   time = time_col

   csv_files = glob.glob(path + "/*.csv")

   data = pd.DataFrame()
   for file in csv_files:
      df = import_data(file)

      data = pd.concat([data, df])

   data = data.set_index(['id', time])

   return data

"""
Returns a pre-processed Pandas DataFrame containing the timestamp and glucose data for the csv file at the given path
@param path    the path of the csv file to be pre-processed and read into a Pandas Dataframe
"""
def import_data(path):
   df = pd.read_csv(path)

   id = df["Patient Info"].iloc[0] + df["Patient Info"].iloc[1] + df["Patient Info"].iloc[2]
   df['id'] = id

   df = df.dropna(subset=[glucose])
   df = df.replace("Low", 40)
   df = df.replace("High", 400)

   df[time] = pd.to_datetime(df[time], format='%Y-%m-%d %H:%M:%S')

   df[glucose] = pd.to_numeric(df[glucose])

   return df[['id', time, glucose]]

"""
Takes in a multiindexed Pandas DataFrame containing CGM data for multiple patients/datasets, and
returns a single indexed Pandas DataFrame containing summary metrics in the form of one row per patient/dataset
"""
def create_features(data):
   df = pd.DataFrame()

"""
Returns a multiindexed Pandas DataFrame containing only the patient data during their respective 'events'
@param data    a multiindexed Pandas DataFrame containing all the relevant patient data
@param events  a single indexed Pandas DataFrame, with each row specifying a single event in the form of
               an id, a datetime, # of hours before the datetime to include, # of hours after to include, and a desc
"""
def retrieve_event_data(df, events, before="before", after="after", desc="description"):
   event_data = pd.DataFrame()

   for row in events:
      id = row['id']
      datetime = row[time]

      data = df[id][df[id][time] >= datetime - before and df[id][time] <= datetime + after]
      data[desc] = row[desc]

      event_data = pd.concat([event_data, data])

   event_data = event_data.set_index([desc, 'id', time])

   return event_data

def ave_glucose(df):
   return df[glucose].mean()

def summary(df):
   min = df[glucose].min()
   first = df[glucose].quantile(0.25)
   median = df[glucose].median()
   third = df[glucose].quantile(0.75)
   max = df[glucose].max()

   return [min, first, median, third, max]

def intersd(df):
   return df[glucose].std()

def a1c(df):
   return (46.7 + ave_glucose(df)) / 28.7

def gmi(df):
   return (0.02392 * ave_glucose(df)) + 3.31

def std(df):
   return df[glucose].std()

"""
Returns the approximate total amount of minutes the glucose levels were between the given lower and upper bounds (inclusive)
@param df: the data in the form of a Pandas DataFrame
@param low: the lower bound of the acceptable glucose values
@param high: the upper bound of the acceptable glucose values
"""
def time_in_range(df, low, high):
   temp_df = df[df[glucose] <= high and df[glucose] >= low]

def main():
   df = import_directory("datasets")

   for id, data in df.groupby(level=0):
      print("ID: " + str(id))
      print("summary: " + str(summary(data)))
      print("mean: " + str(ave_glucose(data)))
      print("a1c: " + str(a1c(data)))
      print("gmi: " + str(gmi(data)))
      print("std: " + str(std(data)))
   
   events = pd.DataFrame()
   event_data = retrieve_event_data()

if __name__ == "__main__":
   main()


