import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
import configparser
import glob, os, zipfile, tempfile
import math

config = configparser.ConfigParser()
config.read('config.ini')
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
INTERVAL = config['variables'].getint('interval')

BEFORE = "Before"
AFTER = "After"
TYPE = "Type"
DESCRIPTION = "Description"

def import_events(
   path: str, 
   id: str,
   name: str = None,
   day_col: str = "Day",
   time_col: str = "Time",
   before: int = 60,
   after: int = 60,
   type: str = "imported event"
):
   """
   Bulk imports events from .csv files within the given directory
   @param path       the path of the directory to import from
   @param id         the ID of the patient that the imported events belong to
   @param day_col    the name of the column specifying the day the event occurred (year, month, and specific day)
   @param time_col   the name of the column specifying what time during the day the event occurred
   @param before     the amount of minutes to also look at before the event timestamp
   @param after      the amount of minutes to also look at after the event timestamp
   @param type       the type of event all the imported events are
   """
   ext = os.path.splitext(path)[1]

   # path leads to directory
   if ext == "":
      if not os.path.isdir(path):
         raise ValueError("Directory does not exist")
      else:
         return import_events_directory(path, id, day_col, time_col, before, after, type)
   
   # check if path leads to .zip or .csv
   if ext.lower() in [".csv", ".zip"]:
      if not os.path.isfile(path):
         raise ValueError("File does not exist")
   else:
      raise ValueError("Invalid file type")

   # path leads to .csv
   if ext.lower() == ".csv":
      return import_events_csv(path, id, day_col, time_col, before, after, type)

   # otherwise has to be a .zip file
   with zipfile.ZipFile(path, 'r') as zip_ref:
   # create a temporary directory to pull from
      with tempfile.TemporaryDirectory() as temp_dir:
         zip_ref.extractall(temp_dir)
         dir = name or path.split("/")[-1].split(".")[0]
         return import_events_directory((temp_dir + "/" + dir), id, day_col, time_col, before, after, type)

def import_events_directory(
   path: str,
   id: str,
   day_col: str = "Day",
   time_col: str = "Time",
   before: int = 60,
   after: int = 60,
   type: str = "imported event"
) -> pd.DataFrame:
    """
    Returns a Multiindexed Pandas DataFrame containing all of the csv data found in the directory at the given path.
    The DataFrame holds columns for DateTime and Glucose Value, and is indexed by 'id'
    @param path    the path of the directory to be parsed through
    @param glucose_col   the header of the column containing the glucose values
    """
    csv_files = glob.glob(path + "/*.csv")

    if len(csv_files) == 0:
       raise Exception("No CSV files found.")

    return pd.concat(import_events_csv(file, id, day_col, time_col, before, after, type) for file in csv_files)

def import_events_csv(
   path: str,
   id: str,
   day_col: str = "Day",
   time_col: str = "Time",
   before: int = 60,
   after: int = 60,
   type: str = "imported event"
):
   """
   Imports events from a single given .csv file
   @param path       the path of the .csv file to import from
   @param id         the ID of the patient that the imported events belong to
   @param day_col    the name of the column specifying the day the event occurred (year, month, and specific day)
   @param time_col   the name of the column specifying what time during the day the event occurred
   @param before     the amount of minutes to also look at before the event timestamp
   @param after      the amount of minutes to also look at after the event timestamp
   @param type       the type of event all the imported events are
   """
   df = pd.read_csv(path)
   csv_name = os.path.splitext(path)[0]

   events = pd.DataFrame()
   events[TIME] = pd.to_datetime(df[day_col] + " " + df[time_col])
   events[BEFORE] = before
   events[AFTER] = after
   events[TYPE] = type
   events[DESCRIPTION] = df["Food Name"] if "Food Name" in df.columns else ("imported event #" + (events.index + 1).astype(str) + f"from {csv_name}")
   events.insert(0, 'id', id)

   return events.dropna(subset=[TIME])

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

         description = f"{start_time} to {end_time} level {level} {type}glycemic episode"
         event = pd.DataFrame.from_records([{"id": id, TIME: start_time, BEFORE: 0, AFTER: episode_length, 
                                             TYPE: f"{type} level {level} episode", DESCRIPTION: description}])
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
               outside_threshold = np.where(last_data <= upper if type == "hyper" else last_data >= lower, True, False)
               if False in outside_threshold: # check if excursion ends within 15 min
                  edges.pop(i + 1) # this excursion does not end within 15 min, so combine this episode with the next
                  continue
               
            outliers.set_index(TIME, inplace=True)
            last_point = edges[i+1] if offset != 0 else None
            timestamp = outliers.iloc[edges[i]:last_point][GLUCOSE].idxmax() if type == "hyper" else outliers.iloc[edges[i]:last_point][GLUCOSE].idxmin()
            outliers.reset_index(inplace=True)

            extrema = peaks if type == "hypo" else nadirs
            if start_index != 0:
               start_time = extrema[extrema <= start_time].iloc[-1]
            if end_index != data.shape[0] - 1:
               end_time = extrema[extrema >= end_time].iloc[0]
            
            description = f"{start_time} to {end_time} {type}glycemic excursion"
            event = pd.DataFrame.from_records([{"id": id, TIME: timestamp, BEFORE: timegap(timestamp - start_time), 
                                                AFTER: timegap(end_time - timestamp), 
                                                TYPE: f"{type} excursion", DESCRIPTION: description}])
            excursions = pd.concat([excursions, event])
         
         i += 1

   return excursions

def backwards_ROC(df: pd.DataFrame) -> pd.Series:
   data = df.copy().reset_index(drop=True).set_index(TIME)
   return ((3 * data[GLUCOSE]) - (4 * data[GLUCOSE].shift()) + data[GLUCOSE].shift(2)) / (2 * INTERVAL)

def get_meals(df: pd.DataFrame, threshold: float, cooldown: int = 20) -> pd.DataFrame:
   meals = pd.DataFrame()
   for id, data in df.groupby('id'):
      roc = backwards_ROC(data)
      meal_times = roc[roc >= threshold].index.tolist()
      for time in meal_times:
         if meals.shape[0] == 0 or ((time - meals[TIME].iloc[-1]) >= pd.Timedelta(minutes = cooldown)):
            meal = pd.DataFrame.from_records([{"id": id, TIME: time, BEFORE: 0, AFTER: 0, 
                                                      TYPE: "meal", DESCRIPTION: f"meal occurring at {time}"}])
            meals = pd.concat([meals, meal])
   return meals

def get_curated_events(df: pd.DataFrame) -> pd.DataFrame:
   return pd.concat([get_episodes(df), get_excursions(df)])

def retrieve_event_data(
    df: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a multiindexed Pandas DataFrame containing only the patient data during their respective 'events'
    @param df      a multiindexed Pandas DataFrame containing all the relevant patient data
    @param events  a single indexed Pandas DataFrame, with each row specifying a single event in the form of
                   an id, a datetime, # of hours before the datetime to include, # of hours after to include, and a desc
    """
    event_data = pd.DataFrame()

    for index, row in events.iterrows():
        id = row["id"]

        datetime = pd.Timestamp(row[TIME])
        initial = datetime - pd.Timedelta(row[BEFORE], "m")
        final = datetime + pd.Timedelta(row[AFTER], "m")

        patient_data = df.loc[id]
        data = patient_data[(patient_data[TIME] >= initial) & (patient_data[TIME] <= final)].copy()

        data["id"] = id
        data[DESCRIPTION] = row[DESCRIPTION]

        event_data = pd.concat([event_data, data])

    #if event_data.shape[0] != 0:
      #event_data = event_data.set_index(["id"])

    return event_data

def create_event_features(
    df: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
   """
   Returns a multiindexed Pandas DataFrame containing metrics for the patient data during their respective 'events'
   @param df      a multiindexed Pandas DataFrame containing all the relevant patient data
   @param events  a single indexed Pandas DataFrame, with each row specifying a single event in the form of
                  an id, a datetime, # of hours before the datetime to include, # of hours after to include, and a desc
   """
   event_data = retrieve_event_data(df, events)
   return create_features(event_data, events=True)

def event_summary(events: pd.DataFrame) -> pd.Series:
   return events[TYPE].value_counts()

def episode_statistics(
   df: pd.DataFrame,
   events: pd.DataFrame,
   id: str,
) -> pd.DataFrame:
   "Calculates episode-specific metrics on the given DataFrame (helper function)"
   hypo_lvl1 = events[events[TYPE] == "hypo level 1 episode"]
   hypo_lvl2 = events[events[TYPE] == "hypo level 2 episode"]
   hyper_lvl1 = events[events[TYPE] == "hyper level 1 episode"]
   hyper_lvl2 = events[events[TYPE] == "hyper level 2 episode"]

   total_days = (df.iloc[-1][TIME] - df.iloc[0][TIME]).total_seconds() / (3600 * 24)

   # mean episodes per day
   hypo_lvl1_day = hypo_lvl1.shape[0] / total_days
   hypo_lvl2_day = hypo_lvl2.shape[0] / total_days
   hyper_lvl1_day = hyper_lvl1.shape[0] / total_days
   hyper_lvl2_day = hypo_lvl2.shape[0] / total_days

   # mean episode duration per day
   hypo_lvl1_duration = hypo_lvl1[AFTER].mean() / total_days or np.nan
   hypo_lvl2_duration = hypo_lvl2[AFTER].mean() / total_days or np.nan
   hyper_lvl1_duration = hyper_lvl1[AFTER].mean() / total_days or np.nan
   hyper_lvl2_duration = hyper_lvl2[AFTER].mean() / total_days or np.nan

   # mean glucose per episode (hypo / hyper)
   hypo_mean_glucose = np.nan
   if hypo_lvl1_day != 0:
      hypo_glucose_data = retrieve_event_data(df, hypo_lvl1).loc[id]
      hypo_mean_glucose = np.mean([data[GLUCOSE].mean() for description, data in hypo_glucose_data.groupby(DESCRIPTION)])

   hyper_mean_glucose = np.nan
   if hyper_lvl1_day != 0:
      hyper_glucose_data = retrieve_event_data(df, hyper_lvl1).loc[id]
      hyper_mean_glucose = np.mean([data[GLUCOSE].mean() for description, data in hyper_glucose_data.groupby(DESCRIPTION)])

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

def iAUC(df: pd.DataFrame, level = float) -> float:
    data = df.copy()
    data[GLUCOSE] = abs(data[GLUCOSE] - level)
    data.loc[data[GLUCOSE] < 0, GLUCOSE] = 0
    return AUC(data)

def baseline(df: pd.DataFrame) -> float:
    return df[GLUCOSE].iloc[0]

def peak(df: pd.DataFrame) -> float:
    return np.max(df[GLUCOSE])

def nadir(df: pd.DataFrame) -> float:
   return np.min(df[GLUCOSE])

def delta(df: pd.DataFrame) -> float:
    return abs(peak(df) - baseline(df))

def excursion_statistics(
   df: pd.DataFrame,
   events: pd.DataFrame, 
   id: str,
) -> pd.DataFrame:
   total_days = (df.iloc[-1][TIME] - df.iloc[0][TIME]).total_seconds() / (3600 * 24)
   mean = df[GLUCOSE].mean()
   sd = df[GLUCOSE].std()
   upper = mean + (2 * sd)
   lower = mean - (2 * sd)

   hypo_excursions = events[events[TYPE] == "hypo excursion"].copy()
   num_hypo_excursions = hypo_excursions.shape[0]
   hyper_excursions = events[events[TYPE] == "hyper excursion"].copy()
   num_hyper_excursions = hyper_excursions.shape[0]
   hypo_data = retrieve_event_data(df, hypo_excursions)
   hyper_data = retrieve_event_data(df, hyper_excursions)

   hypo_excursions_per_day = num_hypo_excursions / total_days
   hyper_excursions_per_day = num_hyper_excursions / total_days

   hypo_mean_duration = np.mean(hypo_excursions[BEFORE] + hypo_excursions[AFTER])
   mean_hypo_iAUC = 0
   mean_hypo_nadir = 0
   mean_hypo_delta = 0
   mean_hypo_upwards = 0
   mean_hypo_downwards = 0
   if num_hypo_excursions != 0:
      for description, hypo_excursion in hypo_data.groupby(DESCRIPTION):
         mean_hypo_iAUC += iAUC(hypo_excursion, level=lower)
         nadir = np.min(hypo_excursion[GLUCOSE]); mean_hypo_nadir += nadir
         delta = abs(lower - nadir); mean_hypo_delta += delta

         event = events[events[DESCRIPTION] == description]
         downwards = delta / event[BEFORE]; mean_hypo_downwards += downwards
         upwards = delta / event[AFTER]; mean_hypo_upwards += upwards
      mean_hypo_iAUC /= num_hypo_excursions
      mean_hypo_nadir /= num_hypo_excursions
      mean_hypo_delta /= num_hypo_excursions
      mean_hypo_upwards /= num_hypo_excursions
      mean_hypo_downwards /= num_hypo_excursions

   hyper_mean_duration = np.mean(hyper_excursions[BEFORE] + hyper_excursions[AFTER])
   mean_hyper_iAUC = 0
   mean_hyper_peak = 0
   mean_hyper_delta = 0
   mean_hyper_upwards = 0
   mean_hyper_downwards = 0
   if num_hyper_excursions != 0:
      for description, hyper_excursion in hyper_data.groupby(DESCRIPTION):
         mean_hyper_iAUC += iAUC(hyper_excursion, level=upper)
         peak = np.max(hyper_excursion[GLUCOSE]); mean_hyper_peak += peak
         delta = abs(peak - upper); mean_hyper_delta += delta

         event = events[events[DESCRIPTION] == description]
         downwards = delta / event[AFTER]; mean_hyper_downwards += downwards
         upwards = delta / event[BEFORE]; mean_hyper_upwards += upwards
      mean_hyper_iAUC /= num_hyper_excursions
      mean_hyper_peak /= num_hyper_excursions
      mean_hyper_delta /= num_hyper_excursions
      mean_hyper_upwards /= num_hyper_excursions
      mean_hyper_downwards /= num_hyper_excursions

   return pd.DataFrame.from_records([{"mean hypoglycemic excursions per day": hypo_excursions_per_day,
                                      "mean hyperglycemic excursions per day": hyper_excursions_per_day,
                                      "mean hypoglycemic excursion duration": hypo_mean_duration,
                                      "mean hyperglycemic excursion duration": hyper_mean_duration,
                                      "mean hypoglycemic excursion incremental area above curve (iAAC)": mean_hypo_iAUC,
                                      "mean hyperglycemic excursion incremental area under curve (iAUC)": mean_hyper_iAUC,
                                      "mean hypoglycemic excursion minimum": mean_hypo_nadir,
                                      "mean hyperglycemic excursion maximum": mean_hyper_peak,
                                      "mean hypoglycemic excursion amplitude": mean_hypo_delta,
                                      "mean hyperglycemic excursion amplitude": mean_hyper_delta,
                                      "mean hypoglycemic excursion downwards slope (mg/dL per min)": mean_hypo_downwards,
                                      "mean hyperglycemic excursion downwards slope (mg/dL per min)": mean_hyper_downwards,
                                      "mean hypoglycemic excursion upwards slope (mg/dL per min)": mean_hypo_upwards,
                                      "mean hyperglycemic excursion upwards slope (mg/dL per min)": mean_hyper_upwards}])

def event_statistics(
   df: pd.DataFrame,
   events: pd.DataFrame, 
) -> pd.DataFrame:
   statistics = pd.DataFrame()

   for id, data in df.groupby('id'):
      patient = events[events['id'] == id]
      stats = pd.concat([pd.DataFrame.from_records([{'id': id}]), 
                         episode_statistics(data, patient, id),
                         excursion_statistics(data, patient, id)], axis=1)
      statistics = pd.concat([statistics, stats])
   
   statistics.set_index('id', inplace=True)
   return statistics

def event_metrics(
   df: pd.DataFrame,
   event: pd.Series
) -> pd.DataFrame:
   id = event["id"]

   datetime = pd.Timestamp(event[TIME])
   initial = datetime - pd.Timedelta(event[BEFORE], "m")
   final = datetime + pd.Timedelta(event[AFTER], "m")

   patient_data = df.loc[id]
   data = patient_data[(patient_data[TIME] >= initial) & (patient_data[TIME] <= final)].copy()

   metrics = pd.Series()
   metrics["Baseline"] = baseline(data)
   metrics["Peak"] = peak(data)
   metrics["Delta"] = delta(data)
   metrics["iAUC"] = iAUC(data, baseline(data))

   return metrics.to_frame().T