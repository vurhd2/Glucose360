import pandas as pd
from preprocessing import time
from features import create_features

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

      datetime = pd.Timestamp(row[time()])
      initial = datetime - pd.Timedelta(row[before], 'h')
      final = datetime + pd.Timedelta(row[after], 'h')

      patient_data = df.loc[id]
      data = patient_data[(patient_data[time()] >= initial) & (patient_data[time()] <= final)].copy()
      
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