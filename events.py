import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
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
) -> pd.DataFrame:
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

def get_excursions(
   df: pd.DataFrame, 
   z: int = 2, 
   min_length: int = 15,
   end_length: int = 15
) -> pd.DataFrame:
   excursions = pd.DataFrame()
   for id, data in df.groupby('id'):
      sd = data[GLUCOSE].std()
      mean = data[GLUCOSE].mean()
      upper = mean + (z * sd)
      lower = mean - (z * sd) #75.6

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
      i = 0
      while i < len(edges) - 1:
         type = "hyper" if outliers.iloc[edges[i]][GLUCOSE] > mean else "hypo"
         offset = 0 if i == len(edges) - 2 else 1
         start_time = outliers.iloc[edges[i]][TIME]
         start_index = data.index[data[TIME] == start_time].to_list()[0]
         end_time = outliers.iloc[edges[i+1] - offset][TIME]
         end_index = data.index[data[TIME] == end_time].to_list()[0]

         excursion_length = timegap(end_time - start_time)
         if excursion_length >= min_length:
            if offset != 0: # not the very last episode
               end_counts = math.ceil(end_length / INTERVAL)
               
               last_index = data.reset_index().index[data[TIME] == end_time].to_list()[0]
               last_data = data.iloc[last_index + 1 : last_index + 1 + end_counts][GLUCOSE]
               print(last_data)
               outside_threshold = np.where(last_data <= upper if type == "hyper" else last_data >= lower, True, False)
               print(outside_threshold)
               if False in outside_threshold: # check if excursion ends within 15 min
                  edges.pop(i + 1) # this excursion does not end within 15 min, so combine this episode with the next
                  continue
               
            outliers.set_index(TIME, inplace=True)
            last_point = edges[i+1] if offset != 0 else None
            timestamp = outliers.iloc[edges[i]:last_point][GLUCOSE].idxmax() if type == "hyper" else outliers.iloc[edges[i]:last_point][GLUCOSE].idxmin()
            outliers.reset_index(inplace=True)

            extrema = peaks if type == "hypo" else nadirs
            if start_index != 0:
               start_time = extrema[extrema < timestamp].iloc[-1]
            if end_index != data.shape[0] - 1:
               end_time = extrema[extrema > timestamp].iloc[0]
            
            description = f"{type}glycemic excursion occurring from {start_time} to {end_time}"
            event = pd.DataFrame.from_records([{"id": id, TIME: timestamp, "before": timegap(timestamp - start_time), 
                                                "after": timegap(end_time - timestamp), 
                                                "type": f"{type} excursion", "description": description}])
            excursions = pd.concat([excursions, event])
         
         i += 1

   return excursions

def get_curated_events(df: pd.DataFrame) -> pd.DataFrame:
   return pd.concat([get_episodes(df), get_excursions(df)])

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
        data = patient_data[(patient_data[TIME] >= initial) & (patient_data[TIME] <= final)].copy()

        data["id"] = id
        data[desc] = row[desc]

        event_data = pd.concat([event_data, data])

    if event_data.shape[0] != 0:
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

def event_summary(events: pd.DataFrame, type: str = "type") -> pd.Series:
   return events[type].value_counts()

def episode_statistics(
   df: pd.DataFrame,
   events: pd.DataFrame,
   id: str,
   before: str = "before",
   after: str = "after",
   type: str = "type",
   desc: str = "description"
) -> pd.DataFrame:
   "Calculates episode-specific metrics on the given DataFrame (helper function)"
   hypo_lvl1 = events[events[type] == "hypo level 1 episode"]
   hypo_lvl2 = events[events[type] == "hypo level 2 episode"]
   hyper_lvl1 = events[events[type] == "hyper level 1 episode"]
   hyper_lvl2 = events[events[type] == "hyper level 2 episode"]

   total_days = (df.iloc[-1][TIME] - df.iloc[0][TIME]).total_seconds() / (3600 * 24)

   # mean episodes per day
   hypo_lvl1_day = hypo_lvl1.shape[0] / total_days
   hypo_lvl2_day = hypo_lvl2.shape[0] / total_days
   hyper_lvl1_day = hyper_lvl1.shape[0] / total_days
   hyper_lvl2_day = hypo_lvl2.shape[0] / total_days

   # mean episode duration per day
   hypo_lvl1_duration = hypo_lvl1[after].mean() / total_days or np.nan
   hypo_lvl2_duration = hypo_lvl2[after].mean() / total_days or np.nan
   hyper_lvl1_duration = hyper_lvl1[after].mean() / total_days or np.nan
   hyper_lvl2_duration = hyper_lvl2[after].mean() / total_days or np.nan

   # mean glucose per episode (hypo / hyper)
   hypo_mean_glucose = np.nan
   if hypo_lvl1_day != 0:
      hypo_glucose_data = retrieve_event_data(df, hypo_lvl1).loc[id]
      hypo_mean_glucose = np.mean([data[GLUCOSE].mean() for description, data in hypo_glucose_data.groupby(desc)])

   hyper_mean_glucose = np.nan
   if hyper_lvl1_day != 0:
      hyper_glucose_data = retrieve_event_data(df, hyper_lvl1).loc[id]
      hyper_mean_glucose = np.mean([data[GLUCOSE].mean() for description, data in hyper_glucose_data.groupby(desc)])

   return pd.DataFrame.from_records([{"mean hypoglycemic level 1 episodes per day": hypo_lvl1_day,
                                      "mean hypoglycemic level 2 episodes per day": hypo_lvl2_day,
                                      "mean hyperglycemic level 1 episodes per day": hyper_lvl1_day,
                                      "mean hyperglycemic level 2 episodes per day": hyper_lvl2_day,
                                      "mean hypoglycemic level 1 duration per day": hypo_lvl1_duration,
                                      "mean hypoglycemic level 2 duration per day": hypo_lvl2_duration,
                                      "mean hyperglycemic level 1 duration per day": hyper_lvl1_duration,
                                      "mean hyperglycemic level 2 duration per day": hyper_lvl2_duration,
                                      "mean hypoglycemic glucose value (level 1) per day": hypo_mean_glucose,
                                      "mean hyperglycemic glucose value (level 1) per day": hyper_mean_glucose,}])

def AUC(df: pd.DataFrame) -> float:
    return trapezoid(df[GLUCOSE], dx=INTERVAL)

def iAUC(df: pd.DataFrame, level: int = 70) -> float:
    data = df.copy()
    data[GLUCOSE] = data[GLUCOSE] - level
    data.loc[data[GLUCOSE] < 0, GLUCOSE] = 0
    return AUC(data)

def baseline(df: pd.DataFrame) -> float:
    return df[TIME].iloc[0]

def peak(df: pd.DataFrame) -> float:
    return np.max(df[GLUCOSE])

def delta(df: pd.DataFrame) -> float:
    return peak(df) - baseline(df)

def excursion_statistics(
   df: pd.DataFrame,
   events: pd.DataFrame, 
   id: str,
   before: str = "before",
   after: str = "after",
   type: str = "type",
   desc: str = "description"
) -> pd.DataFrame:
   patient = events[events['id'] == id].copy()
   total_days = (data.iloc[-1][TIME] - data.iloc[0][TIME]).total_seconds() / (3600 * 24)

   num_hypo_excursions = patient[patient[type] == "hypo excursion"].shape[0]
   num_hyper_excursions = patient[patient[type] == "hyper excursion"].shape[0]
   hypo_excursions_per_day = num_hypo_excursions / total_days
   hyper_excursions_per_day = num_hyper_excursions / total_days
   excursions_per_day = hypo_excursions_per_day + hyper_excursions_per_day

   return

def event_statistics(
   df: pd.DataFrame,
   events: pd.DataFrame, 
   before: str = "before",
   after: str = "after",
   type: str = "type",
   desc: str = "description"
) -> pd.DataFrame:
   
   statistics = pd.DataFrame()

   for id, data in df.groupby('id'):
      patient = events[events['id'] == id]
      stats = pd.concat([pd.DataFrame.from_records([{'id': id}]), episode_statistics(data, patient, id)], axis=1)
      statistics = pd.concat([statistics, stats])
   
   statistics.set_index('id', inplace=True)
   return statistics
