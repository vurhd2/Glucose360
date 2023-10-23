import pandas as pd
import numpy as np
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

   data = data.set_index(['id'])

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

   df[time] = pd.to_datetime(df[time], format='%Y-%m-%dT%H:%M:%S')

   df[glucose] = pd.to_numeric(df[glucose])

   return df[['id', time, glucose]]

"""
Takes in a multiindexed Pandas DataFrame containing CGM data for multiple patients/datasets, and
returns a single indexed Pandas DataFrame containing summary metrics in the form of one row per patient/dataset
"""
def create_features(dataset):
   df = pd.DataFrame()

   for id, data in dataset.groupby(level=0):
      features = {}
      summary = summary_stats(data)

      features['id'] = id
      
      features['mean'] = mean(data)
      features['min'] = summary[0]
      features['first quartile'] = summary[1]
      features['median'] = summary[2]
      features['third quartile'] = summary[3]
      features['max'] = summary[4]

      features['intrasd'] = std(data)
      features['intersd'] = std(dataset)

      features['a1c'] = a1c(data)
      features['gmi'] = gmi(data)
      features['percent time in range'] = percent_time_in_range(data)
      
      df = pd.concat([df, pd.DataFrame.from_records([features])])

   df = df.set_index(['id'])

   return df

"""
Returns a multiindexed Pandas DataFrame containing only the patient data during their respective 'events'
@param df      a multiindexed Pandas DataFrame containing all the relevant patient data
@param events  a single indexed Pandas DataFrame, with each row specifying a single event in the form of
               an id, a datetime, # of hours before the datetime to include, # of hours after to include, and a desc
@param before  name of the column specifying the amount of hours before to include
@param after   name of the column specifying the amount of hours after to include
@param desc    name of the column describing this particular event
"""
def retrieve_event_data(df, events, before="before", after="after", desc="description"):
   event_data = pd.DataFrame()

   for index, row in events.iterrows():
      id = row['id']
      datetime = pd.Timestamp(row[time])
      initial = datetime - pd.Timedelta(row[before], 'h')
      final = datetime + pd.Timedelta(row[after], 'h')

      patient_data = df.loc[id]
      data = patient_data[(patient_data[time] >= initial) & (patient_data[time] <= final)].copy()
      
      data['id'] = id
      data[desc] = row[desc]

      event_data = pd.concat([event_data, data])

   event_data = event_data.set_index(['id'])

   return event_data

"""
Returns a multiindexed Pandas DataFrame containing metrics for the patient data during their respective 'events'
@param df      a multiindexed Pandas DataFrame containing all the relevant patient data
@param events  a single indexed Pandas DataFrame, with each row specifying a single event in the form of
               an id, a datetime, # of hours before the datetime to include, # of hours after to include, and a desc
@param before  name of the column specifying the amount of hours before to include
@param after   name of the column specifying the amount of hours after to include
@param desc    name of the column describing this particular event 
"""
def create_event_features(df, events, before="before", after="after", desc="description"):
   event_data = retrieve_event_data(df, events, before, after, desc)
   return create_features(event_data)

def mean(df):
   return df[glucose].mean()

def summary_stats(df):
   min = df[glucose].min()
   first = df[glucose].quantile(0.25)
   median = df[glucose].median()
   third = df[glucose].quantile(0.75)
   max = df[glucose].max()

   return [min, first, median, third, max]

def std(df):
   return df[glucose].std()

def a1c(df):
   return (46.7 + mean(df)) / 28.7

def gmi(df):
   return (0.02392 * mean(df)) + 3.31

"""
Returns the percent of total time the glucose levels were between the given lower and upper bounds (inclusive)
@param df: the data in the form of a Pandas DataFrame
@param low: the lower bound of the acceptable glucose values
@param high: the upper bound of the acceptable glucose values
"""
def percent_time_in_range(df, low=70, high=180):
   in_range_df = df[(df[glucose] <= high) & (df[glucose] >= low)]
   time_in_range = len(in_range_df)
   total_time = len(df)
   return (100 * time_in_range / total_time) if total_time > 0 else np.nan

def main():
   df = import_directory("datasets")

   for id, data in df.groupby(level=0):
      print("ID: " + str(id))
      print("summary: " + str(summary_stats(data)))
      print("mean: " + str(mean(data)))
      print("a1c: " + str(a1c(data)))
      print("gmi: " + str(gmi(data)))
      print("std: " + str(std(data)))
   
   event1 = pd.DataFrame.from_records([{'id': "NathanielBarrow9/8/88", time: '2023-08-08 7:45:35', 'before': 2, 'after': 3, 'description': 'testing1'}])
   event2 = pd.DataFrame.from_records([{'id': "ElizaSutherland2/23/68", time: '2023-07-03 3:15:17', 'before': 6, 'after': 9, 'description': 'testing2'}])
   event3 = pd.DataFrame.from_records([{'id': "PenelopeFitzroy1/5/52", time: '2023-06-24 0:26:10', 'before': 3, 'after': 1, 'description': 'testing3'}])
   events = pd.concat([event1, event2, event3])
   
   #event_data = retrieve_event_data(df, events)

   print(create_event_features(df, events))

if __name__ == "__main__":
   main()