import pandas as pd
import numpy as np
from features import create_features
import configparser
import math

config = configparser.ConfigParser()
config.read('config.ini')
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
INTERVAL = config['variables'].getint('interval')

def episodes_helper(
   df: pd.DataFrame, 
   id: str, 
   type: str, 
   threshold: int, 
   level: int, 
   min_length: int, 
   end_length: int
   ):
   timegap = lambda timedelta: timedelta.total_seconds() / 60
   episodes = pd.DataFrame()

   data = df.copy(); data.reset_index(drop=True, inplace=True)
   episode_df = df[(df[GLUCOSE] <= threshold)].copy() if type == "hypo" else df[df[GLUCOSE] >= threshold].copy()
   episode_df.reset_index(drop=True, inplace=True)
   episode_df["gap"] = episode_df[TIME].diff().apply(timegap)

   edges = episode_df.index[episode_df["gap"] != INTERVAL].to_list()
   edges.append(-1)

   get = lambda loc, col: episode_df.iloc[loc][col]
   index = 0
   while index < len(edges) - 1:
      offset = 0 if (index == len(edges) - 2) else 1
      end_i = edges[index + 1] - offset # index of the end of the episode (inclusive! - that's what the offset is for)
      start_i = edges[index] # index of the start of the episode
      start_time = get(start_i, TIME)
      end_time = get(end_i, TIME)
      episode_length = timegap(end_time - start_time)

      if episode_length >= min_length: # check if episode lasts longer than 15 min
         if offset != 0: # not the very last episode
            end_counts = math.ceil(end_length / INTERVAL)
            
            end_index = data.index[data[TIME] == end_time].to_list()[0]
            end_data = data.iloc[end_index + 1 : end_index + 1 + end_counts][GLUCOSE]
            outside_threshold = np.where(end_data >= threshold, True, False) if type == "hypo" else np.where(end_data <= threshold, True, False)
            if False in outside_threshold: # check if episode ends within 15 min
               edges.pop(index + 1) # this episode does not end within 15 min, so combine this episode with the next
               continue

         description = f"{type}glycemic episode of level {level} occurring from {start_time} to {end_time}"
         event = pd.DataFrame.from_records([{"id": id, TIME: start_time, "before": 0, "after": episode_length, 
                                             "type": f"{type} level {level} episode", "description": description}])
         episodes = pd.concat([episodes, event]) 
      
      index += 1

   return episodes

def get_episodes(
    df: pd.DataFrame,
    hypo_lvl1: int = 70,
    hypo_lvl2: int = 54,
    hyper_lvl1: int = 180,
    hyper_lvl2: int = 250,
    min_length: int = 15,
    end_length: int = 15
) -> pd.DataFrame:
   output = pd.DataFrame()
   for id, data in df.groupby('id'):
      episodes = pd.concat([episodes_helper(data, id, "hyper", hyper_lvl1, 1, min_length, end_length),
                            episodes_helper(data, id, "hyper", hyper_lvl2, 2, min_length, end_length),
                            episodes_helper(data, id, "hypo", hypo_lvl1, 1, min_length, end_length),
                            episodes_helper(data, id, "hypo", hypo_lvl2, 2, min_length, end_length)])
      
      episodes.sort_values(by=[TIME], inplace=True)
      output = pd.concat([output, episodes])

   return output 

def get_excursions(df: pd.DataFrame, z: int = 2) -> pd.DataFrame:
   excursions = pd.DataFrame()
   for id, data in df.groupby('id'):
      sd = data[GLUCOSE].std()
      mean = data[GLUCOSE].mean()
      upper = mean + (z * sd)
      lower = mean - (z * sd)

      peaks = data[(data[GLUCOSE].shift(1) < data[GLUCOSE]) & (data[GLUCOSE].shift(-1) < data[GLUCOSE])][TIME].copy()
      peaks.reset_index(drop=True, inplace=True)
      nadirs = data[(data[GLUCOSE].shift(1) > data[GLUCOSE]) & (data[GLUCOSE].shift(-1) > data[GLUCOSE])][TIME].copy()
      nadirs.reset_index(drop=True, inplace=True)


      outliers = data[(data[GLUCOSE] >= upper) | (data[GLUCOSE] <= lower)].copy()
      outliers.reset_index(drop=True, inplace=True)

      # calculate the differences between each of the timestamps
      timegap = lambda timedelta: timedelta.total_seconds() / 60
      outliers["gaps"] = outliers[TIME].diff().apply(timegap)

      edges = outliers.index[outliers["gaps"] != INTERVAL].to_list()
      edges.append(-1)

      for i in range(len(edges) - 1):
         type = "hyper" if outliers.iloc[edges[i]][GLUCOSE] > mean else "hypo"
         offset = 0 if i == len(edges) - 2 else 1
         start_time = outliers.iloc[edges[i]][TIME]
         start_index = data.index[data[TIME] == start_time].to_list()[0]
         end_time = outliers.iloc[edges[i+1] - offset][TIME]
         end_index = data.index[data[TIME] == end_time].to_list()[0]

         if start_index != 0:
            extrema = peaks if type == "hypo" else nadirs
            extrema_index = (abs(extrema - start_time)).idxmin()
            print(extrema)
            print(extrema_index)
            start_time = extrema.iloc[extrema_index]

         if end_index != data.shape[0] - 1:
            extrema = peaks if type == "hypo" else nadirs
            extrema_index = (abs(extrema - end_time)).idxmin()
            print(extrema)
            print(extrema_index)
            end_time = extrema.iloc[extrema_index]

         episode_length = timegap(end_time - start_time)
         description = f"{type}glycemic excursion occurring from {start_time} to {end_time}"
         event = pd.DataFrame.from_records([{"id": id, TIME: start_time, "before": 0, "after": episode_length, 
                                             "type": f"{type} excursion", "description": description}])
         excursions = pd.concat([excursions, event])

   return excursions

def event_summary(events: pd.DataFrame, type: str = "type") -> pd.Series:
    return events[type].value_counts()

def retrieve_event_data(
    df: pd.DataFrame,
    events: pd.DataFrame,
    before: str = "before",
    after: str = "after",
    type: str = "type",
    desc: str = "description",
) -> pd.DataFrame:
    """
    Returns a multiindexed Pandas DataFrame containing only the patient data during their respective 'events'
    @param df      a multiindexed Pandas DataFrame containing all the relevant patient data
    @param events  a single indexed Pandas DataFrame, with each row specifying a single event in the form of
                   an id, a datetime, # of hours before the datetime to include, # of hours after to include, and a desc
    @param before  name of the column specifying the amount of minutes before to include
    @param after   name of the column specifying the amount of minutes after to include
    @param type    name of the column specifying the 'type' of event (like 'meal', 'exercise', etc.)
    @param desc    name of the column describing this particular event
    """
    event_data = pd.DataFrame()

    for index, row in events.iterrows():
        id = row["id"]

        datetime = pd.Timestamp(row[TIME])
        initial = datetime - pd.Timedelta(row[before], "m")
        final = datetime + pd.Timedelta(row[after], "m")

        patient_data = df.loc[id]
        data = patient_data[
            (patient_data[TIME] >= initial) & (patient_data[TIME] <= final)
        ].copy()

        data["id"] = id
        data[desc] = row[desc]

        event_data = pd.concat([event_data, data])

    event_data = event_data.set_index(["id"])

    return event_data

def create_event_features(
    df: pd.DataFrame,
    events: pd.DataFrame,
    before: str = "before",
    after: str = "after",
    type: str = "type",
    desc: str = "description",
) -> pd.DataFrame:
    """
    Returns a multiindexed Pandas DataFrame containing metrics for the patient data during their respective 'events'
    @param df      a multiindexed Pandas DataFrame containing all the relevant patient data
    @param events  a single indexed Pandas DataFrame, with each row specifying a single event in the form of
                   an id, a datetime, # of hours before the datetime to include, # of hours after to include, and a desc
    @param before  name of the column specifying the amount of hours before to include
    @param after   name of the column specifying the amount of hours after to include
    @param type    name of the column specifying the 'type' of event (like 'meal', 'exercise', etc.)
    @param desc    name of the column describing this particular event 
    """
    event_data = retrieve_event_data(df, events, before, after, type, desc)
    return create_features(event_data, events=True)
