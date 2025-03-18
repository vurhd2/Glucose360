import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
import configparser
import glob, os, zipfile, tempfile
import math

dir_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(dir_path, "config.ini")
config = configparser.ConfigParser()
config.read(config_path)
ID = config['variables']['id']
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
BEFORE = config['variables']['before']
AFTER = config['variables']['after']
TYPE = config['variables']['type']
DESCRIPTION = config['variables']['description']

def import_events(
   path: str, 
   id: str,
   name: str = None,
   day_col: str = "Day",
   time_col: str = "Time",
   before: int = 60,
   after: int = 60,
   type: str = "imported event"
) -> pd.DataFrame:
   """Bulk imports events from standalone .csv files or from those within a given directory or .zip file
   
   :param path: the path of the directory/zip/csv to import from
   :type path: str
   :param id: the identification of the patient that the imported events belong to
   :type id: str
   :param day_col: the name of the column specifying the day the event occurred (year, month, and specific day), defaults to 'Day'
   :type day_col: str, optional
   :param time_col: the name of the column specifying what time during the day the event occurred, defaults to 'Time'
   :type time_col: str, optional
   :param before: the amount of minutes to also look at before the event timestamp, defaults to 60
   :type before: int, optional
   :param after: the amount of minutes to also look at after the event timestamp, defaults to 60
   :type after: int, optional
   :param type: the type of event to classify all the imported events as, defaults to 'imported event'
   :type type: str, optional
   :return: a Pandas DataFrame containing all the imported events
   :rtype: 'pandas.DataFrame'
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
    """Bulk imports events from .csv files within a given directory
   
   :param path: the path of the directory to import from
   :type path: str
   :param id: the identification of the patient that the imported events belong to
   :type id: str
   :param day_col: the name of the column specifying the day the event occurred (year, month, and specific day), defaults to 'Day'
   :type day_col: str, optional
   :param time_col: the name of the column specifying what time during the day the event occurred, defaults to 'Time'
   :type time_col: str, optional
   :param before: the amount of minutes to also look at before the event timestamp, defaults to 60
   :type before: int, optional
   :param after: the amount of minutes to also look at after the event timestamp, defaults to 60
   :type after: int, optional
   :param type: the type of event to classify all the imported events as, defaults to 'imported event'
   :type type: str, optional
   :return: a Pandas DataFrame containing all the imported events
   :rtype: 'pandas.DataFrame'
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
) -> pd.DataFrame:
   """Bulk imports events from a single .csv file
   
   :param path: the path of the .csv file to import from
   :type path: str
   :param id: the identification of the patient that the imported events belong to
   :type id: str
   :param day_col: the name of the column specifying the day the event occurred (year, month, and specific day), defaults to 'Day'
   :type day_col: str, optional
   :param time_col: the name of the column specifying what time during the day the event occurred, defaults to 'Time'
   :type time_col: str, optional
   :param before: the amount of minutes to also look at before the event timestamp, defaults to 60
   :type before: int, optional
   :param after: the amount of minutes to also look at after the event timestamp, defaults to 60
   :type after: int, optional
   :param type: the type of event to classify all the imported events as, defaults to 'imported event'
   :type type: str, optional
   :return: a Pandas DataFrame containing all the imported events
   :rtype: 'pandas.DataFrame'
   """
   df = pd.read_csv(path)
   csv_name = os.path.splitext(path)[0]

   events = pd.DataFrame()
   events[TIME] = pd.to_datetime(df[day_col] + " " + df[time_col])
   events[BEFORE] = before
   events[AFTER] = after
   events[TYPE] = type
   events[DESCRIPTION] = df["Food Name"] if "Food Name" in df.columns else ("imported event #" + (events.index + 1).astype(str) + f"from {csv_name}")
   events.insert(0, ID, id)

   return events.dropna(subset=[TIME])

def _episodes_helper(
   df: pd.DataFrame, 
   id: str, 
   type: str, 
   threshold: int, 
   level: int, 
   min_length: int, 
   end_length: int
) -> pd.DataFrame:
   """Retrieves all episodes of a specific type/level for a specific patient within the given CGM data

   :param df: Pandas DataFrame containing preprocessed CGM data
   :type df: pandas.DataFrame
   :param id: identification of the patient to retrieve episodes for
   :type id: str
   :param type: type of episode ('hypo' or 'hyper')
   :type type: str
   :param threshold: threshold (in mg/dL) above/below which glucose values are considered as part of an episode
   :type threshold: int
   :param level: the level the retrieved episodes are (0, 1, or 2)
   :type level: int
   :param min_length: minimum duration (in minutes) required for excursions, defaults to 15
   :type min_length: int, optional
   :param end_length: minimum amount of time (in minutes) that the glucose values must be within typical ranges 
      at the end of an excursion, defaults to 15
   :type end_length: int, optional
   :return: a Pandas DataFrame containing all episodes of a specific type/level for a specific patient within the given CGM data
   :rtype: pandas.DataFrame
   """

   config.read('config.ini')
   interval = int(config["variables"]["interval"])
   timegap = lambda timedelta: timedelta.total_seconds() / 60
   episodes = pd.DataFrame()

   data = df.copy(); data.reset_index(drop=True, inplace=True)
   episode_df = df[(df[GLUCOSE] <= threshold)].copy() if type == "hypo" else df[df[GLUCOSE] >= threshold].copy()
   episode_df.reset_index(drop=True, inplace=True)
   episode_df["gap"] = episode_df[TIME].diff().apply(timegap)

   edges = episode_df.index[episode_df["gap"] != interval].to_list()
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
            end_counts = math.ceil(end_length / interval)
            
            end_index = data.index[data[TIME] == end_time].to_list()[0]
            end_data = data.iloc[end_index + 1 : end_index + 1 + end_counts][GLUCOSE]
            outside_threshold = np.where(end_data >= threshold, True, False) if type == "hypo" else np.where(end_data <= threshold, True, False)
            if False in outside_threshold: # check if episode ends within 15 min
               edges.pop(index + 1) # this episode does not end within 15 min, so combine this episode with the next
               continue

         description = f"{start_time} to {end_time} level {level} {type}glycemic episode"
         event = pd.DataFrame.from_records([{ID: id, TIME: start_time, BEFORE: 0, AFTER: episode_length, 
                                             TYPE: f"{type} level {level} episode", DESCRIPTION: description}])
         episodes = pd.concat([episodes, event]) 
      
      index += 1

   return episodes

def get_episodes(
    df: pd.DataFrame,
    hypo_lvl2: int = 54,
    hypo_lvl1: int = 70,
    hyper_lvl0: int = 140,
    hyper_lvl1: int = 180,
    hyper_lvl2: int = 250,
    min_length: int = 15,
    end_length: int = 15
) -> pd.DataFrame:
   """Retrieves all episodes within the given CGM data

   :param df: Pandas DataFrame containing preprocessed CGM data
   :type df: pandas.DataFrame
   :param hypo_lvl2: threshold (in mg/dL) below which glucose values are considered level 2 hypoglycemic, defaults to 54
   :type hypo_lvl2: int, optional
   :param hypo_lvl1: threshold (in mg/dL) below which glucose values are considered level 1 hypoglycemic, defaults to 70
   :type hypo_lvl1: int, optional
   :param hyper_lvl0: threshold (in mg/dL) above which glucose values are considered level 0 hyperglycemic, defaults to 140
   :type hyper_lvl0: int, optional
   :param hyper_lvl1: threshold (in mg/dL) above which glucose values are considered level 1 hyperglycemic, defaults to 180
   :type hyper_lvl1: int, optional
   :param hyper_lvl2: threshold (in mg/dL) above which glucose values are considered level 2 hyperglycemic, defaults to 250
   :type hyper_lvl2: int, optional
   :param min_length: minimum duration (in minutes) required for excursions, defaults to 15
   :type min_length: int, optional
   :param end_length: minimum amount of time (in minutes) that the glucose values must be within typical ranges 
      at the end of an excursion, defaults to 15
   :type end_length: int, optional
   :return: a Pandas DataFrame containing all episodes within the given CGM data
   :rtype: pandas.DataFrame
   """
   output = pd.DataFrame()
   for id, data in df.groupby(ID):
      episodes = pd.concat([_episodes_helper(data, id, "hyper", hyper_lvl0, 0, min_length, end_length),
                            _episodes_helper(data, id, "hyper", hyper_lvl1, 1, min_length, end_length),
                            _episodes_helper(data, id, "hyper", hyper_lvl2, 2, min_length, end_length),
                            _episodes_helper(data, id, "hypo", hypo_lvl1, 1, min_length, end_length),
                            _episodes_helper(data, id, "hypo", hypo_lvl2, 2, min_length, end_length)])
      
      episodes.sort_values(by=[TIME], inplace=True)
      output = pd.concat([output, episodes])

   return output 

def get_excursions(
   df: pd.DataFrame, 
   z: int = 2, 
   min_length: int = 15,
   end_length: int = 15
) -> pd.DataFrame:
   """Retrieves all excursions within the given CGM data

   :param df: Pandas DataFrame containing preprocessed CGM data
   :type df: pandas.DataFrame
   :param z: the number of standard deviations away from the mean that should define an 'excursion', defaults to 2
   :type z: int, optional
   :param min_length: minimum duration (in minutes) required for excursions, defaults to 15
   :type min_length: int, optional
   :param end_length: minimum amount of time (in minutes) that the glucose values must be within typical ranges 
      at the end of an excursion, defaults to 15
   :type end_length: int, optional
   :return: a Pandas DataFrame containing all excursions within the given CGM data
   :rtype: pandas.DataFrame
   """

   excursions = pd.DataFrame()

   config.read('config.ini')
   interval = int(config["variables"]["interval"])
   for id, data in df.groupby(ID):
      data.reset_index(drop=True, inplace=True)
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

      edges = outliers.index[outliers["gaps"] != interval].to_list()
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
               end_counts = math.ceil(end_length / interval)
               
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
               if not extrema[extrema <= start_time].empty: start_time = extrema[extrema <= start_time].iloc[-1]
            if end_index != data.shape[0] - 1:
               if not extrema[extrema >= end_time].empty: end_time = extrema[extrema >= end_time].iloc[0]
            
            description = f"{start_time} to {end_time} {type}glycemic excursion"
            event = pd.DataFrame.from_records([{ID: id, TIME: timestamp, BEFORE: timegap(timestamp - start_time), 
                                                AFTER: timegap(end_time - timestamp), 
                                                TYPE: f"{type} excursion", DESCRIPTION: description}])
            excursions = pd.concat([excursions, event])
         
         i += 1

   return excursions

def get_curated_events(df: pd.DataFrame) -> pd.DataFrame:
   """Retrieves all curated events (episodes and excursions) for all the patients within the given DataFrame

   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :return: a Pandas DataFrame (in the usual event structure defined by the package) containing all curated events for all the patients within the given DataFrame
   :rtype: 'pandas.DataFrame'
   """
   return pd.concat([get_episodes(df), get_excursions(df)])

def retrieve_event_data(
    df: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Returns a multiindexed Pandas DataFrame containing only patient data during the respective given events
    :param df: a Pandas DataFrame containing the preprocessed CGM traces to retrieve event subsets from
    :type df: 'pandas.DataFrame'
    :param events: a single indexed Pandas DataFrame, with each row specifying a single event in the form of
                   an id, a datetime, # of hours before the datetime to include, # of hours after to include, and a description
    :type events: 'pandas.DataFrame'
    :return: a multi-indexed Pandas DataFrame, with each index referring to a subset of CGM trace that was found within 'df' and occurs during a single event within 'events' 
    :rtype: 'pandas.DataFrame'
    """
    event_data = pd.DataFrame()
    for index, row in events.to_frame().T.iterrows():
      id = row[ID]
      if id in df.index:
         datetime = pd.Timestamp(row[TIME])
         initial = datetime - pd.Timedelta(row[BEFORE], "m")
         final = datetime + pd.Timedelta(row[AFTER], "m")

         patient_data = df.loc[id]
         data = patient_data[(patient_data[TIME] >= initial) & (patient_data[TIME] <= final)].copy()

         data[ID] = id
         data[DESCRIPTION] = row[DESCRIPTION]

         event_data = pd.concat([event_data, data])

    #if event_data.shape[0] != 0:
      #event_data = event_data.set_index(["id"])

    return event_data

def event_summary(events: pd.DataFrame) -> pd.Series:
   """Returns the number of events per unique event type found within 'events'

   :param events: a Pandas DataFrame containing events (as per package guidelines)
   :type events: 'pandas.DataFrame'
   :return: a Pandas Series containing the number of events per unique event type found within 'events'
   :rtype: 'pandas.Series'
   """
   return events[TYPE].value_counts()

def AUC(df: pd.DataFrame) -> float:
    """Calculates the total Area-Under-Curve (AUC) for the given CGM trace

    :param df: a Pandas DataFrame containing the CGM trace to calculate the AUC of
    :type df: 'pandas.DataFrame'
    :return: the AUC of the given CGM trace
    :rtype: float
    """
    config.read('config.ini')
    interval = int(config["variables"]["interval"])
    return trapezoid(df[GLUCOSE], dx=interval)

def iAUC(df: pd.DataFrame, level: float) -> float:
    """Calculates the incremental Area-Under-Curve (iAUC) for the given CGM trace

    :param df: a Pandas DataFrame containing the CGM trace to calculate the AUC of
    :type df: 'pandas.DataFrame'
    :param level: the threshold above which to calculate iAUC
    :type level: float
    :return: the iAUC of the given CGM trace
    :rtype: float
    """
    data = df.copy()
    data[GLUCOSE] = abs(data[GLUCOSE] - level)
    data.loc[data[GLUCOSE] < 0, GLUCOSE] = 0
    return AUC(data)

def baseline(df: pd.DataFrame) -> float:
    """Returns the baseline glucose level for the given CGM trace

    :param df: a Pandas DataFrame containing the CGM trace to retrieve the baseline glucose level for
    :type df: 'pandas.DataFrame'
    :return: the baseline glucose level of the given CGM trace
    :rtype: float
    """
    return df[GLUCOSE].iloc[0]

def peak(df: pd.DataFrame) -> float:
    """Returns the maximum glucose level for the given CGM trace

    :param df: a Pandas DataFrame containing the CGM trace to retrieve the maximum glucose level for
    :type df: 'pandas.DataFrame'
    :return: the maximum glucose level of the given CGM trace
    :rtype: float
    """
    return np.max(df[GLUCOSE])

def nadir(df: pd.DataFrame) -> float:
   """Returns the minimum glucose level for the given CGM trace

    :param df: a Pandas DataFrame containing the CGM trace to retrieve the minimum glucose level for
    :type df: 'pandas.DataFrame'
    :return: the minimum glucose level of the given CGM trace
    :rtype: float
    """
   return np.min(df[GLUCOSE])

def delta(df: pd.DataFrame) -> float:
    """Returns the difference in maximum and baseline glucose levels (delta) for the given CGM trace

    :param df: a Pandas DataFrame containing the CGM trace to retrieve the delta for
    :type df: 'pandas.DataFrame'
    :return: the delta of the given CGM trace
    :rtype: float
    """
    return abs(peak(df) - baseline(df))

def post_event_glucoses(data: pd.DataFrame, event_time: pd.Timestamp, times: list[int], glucose_col: str = GLUCOSE) -> dict:
    """
    Returns the glucose values closest to the specified times (in minutes) after the given event_time.

    :param data: Pandas DataFrame containing the CGM data
    :type data: pd.DataFrame
    :param event_time: The time of the event
    :type event_time: pd.Timestamp
    :param times: A list of integers representing the number of minutes after event_time for which to find the glucose values
    :type times: list[int]
    :param glucose_col: The name of the glucose column in the data, defaults to GLUCOSE
    :type glucose_col: str, optional
    :return: A dictionary where keys are strings like "X-min Post Event" and values are the corresponding glucose readings or np.nan if not found
    :rtype: dict
    """
    result = {}

    # Always include 0-min to have a reference point
    if 0 not in times:
       times = [0] + times

    for t in times:
        key = f"{t}-min Post Event"
        result[key] = np.nan
        post_time = event_time + pd.Timedelta(minutes=t)

        # Check if the desired time is within the range of the data
        if not data.empty and data[TIME].min() <= post_time <= data[TIME].max():
            closest_idx = (data[TIME] - post_time).abs().idxmin()
            result[key] = data.loc[closest_idx, glucose_col]

    return result

def post_event_aucs(data: pd.DataFrame, event_time: pd.Timestamp, durations: list[int], glucose_col: str = GLUCOSE) -> dict:
    """
    Calculates AUC values for multiple durations (in minutes) starting from the given event_time.
    
    :param data: Pandas DataFrame containing the CGM data
    :type data: pd.DataFrame
    :param event_time: The time of the event
    :type event_time: pd.Timestamp
    :param durations: A list of integers representing the number of minutes after event_time for which to calculate the AUC
    :type durations: list[int]
    :param glucose_col: The name of the glucose column in the data, defaults to GLUCOSE
    :type glucose_col: str, optional
    :return: A dictionary where keys are strings like "X-min AUC" and values are the corresponding AUC readings or np.nan if no data is available
    :rtype: dict
    """
    result = {}
    for d in durations:
        key = f"{d}-min AUC"
        end_time = event_time + pd.Timedelta(minutes=d)
        subset = data[(data[TIME] >= event_time) & (data[TIME] <= end_time)].copy()

        if subset.empty:
            result[key] = np.nan
        else:
            result[key] = AUC(subset)
    return result

def event_metrics(
      df: pd.DataFrame,
      event: pd.Series,
      post_times: list[int] = [60, 120],
      post_auc_times: list[int] = [120]
) -> pd.DataFrame:
   """Calculates basic metrics for events (baseline, peak, delta, iAUC, and
   0-h, 1-h, and 2-h post event glucose values, and 2-h post event AUC)

   :param df: Pandas DataFrame containing preprocessed CGM data
   :type df: pandas.DataFrame
   :param event: Pandas Series with fields that represent an 'event'
   :type event: pandas.Series
   :param post_times: A list of integers representing the number of minutes after event_time for which to find the glucose values
   :type post_times: list[int], optional (defaults to [0, 60, 120] for 0-h, 1-h and 2-h post event)
   :return: Pandas DataFrame containing the basic metrics for the given event
   :rtype: pandas.DataFrame
   """
   id = event[ID]

   datetime = pd.Timestamp(event[TIME])
   initial = datetime - pd.Timedelta(event[BEFORE], "m")
   final = datetime + pd.Timedelta(event[AFTER], "m")

   patient_data = df.loc[id]
   data = patient_data[(patient_data[TIME] >= initial) & (patient_data[TIME] <= final)].copy()

   metrics = pd.Series()
   metrics["Baseline"] = baseline(data)
   metrics["Peak"] = peak(data)
   metrics["Delta"] = delta(data)
   metrics["AUC"] = AUC(data)
   metrics["iAUC"] = iAUC(data, baseline(data))

   # Get post-event glucose values (including 0-min)
   post_values = post_event_glucoses(data, datetime, post_times, GLUCOSE)
   for k, v in post_values.items():
      metrics[k] = v
   
   # Compute deltas from 0-min Post Event
   zero_min_val = metrics["0-min Post Event"]
   for t in post_times:
      if t == 0:
         continue
      post_key = f"{t}-min Post Event"
      delta_key = f"{t}-min Delta"
      if post_key in metrics and not np.isnan(metrics[post_key]) and not np.isnan(zero_min_val):
         metrics[delta_key] = metrics[post_key] - zero_min_val
      else:
         metrics[delta_key] = np.nan

   auc_values = post_event_aucs(data, datetime, post_auc_times, GLUCOSE)
   for k, v in auc_values.items():
      metrics[k] = v

   return metrics.to_frame().T

def create_event_features(
    df: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
   """Returns a multi-indexed Pandas DataFrame containing metrics for the patient data during their respective 'events'
   
   :param df: a Pandas DataFrame containing all the relevant patient CGM data to generate event metrics for
   :type df: 'pandas.Series'
   :param events: a single indexed Pandas DataFrame, with each row specifying a single event in the form of
                  an id, a datetime, # of hours before the datetime to include, # of hours after to include, and a desc
   :type events: 'pandas.DataFrame'
   :return: a multi-indexed Pandas DataFrame containing metrics for the patient data during their respective 'events'
   """
   event_features = {}
   for id in df.index.unique():
      sub_features = {}
      for type, sub_events in events[events[ID] == id].groupby(TYPE):
         sub_features.update(create_event_features_helper(df.loc[id], sub_events, type))
      event_features[id] = sub_features

   return pd.DataFrame(event_features).T

def create_event_features_helper(
   df: pd.DataFrame,
   sub_events: pd.DataFrame,
   type: str,     
) -> dict[str, float]:
   """Calculates aggregate event-based metrics for a single patient and type of event. Helper method for 'create_event_features()'.

   :param df: Pandas DataFrame containing the CGM trace for a single patient
   :type df: 'pandas.DataFrame'
   :param sub_events: Pandas DataFrame containing events of only one type solely for the patient whose CGM trace is also given
   :type sub_events: 'pandas.DataFrame'
   :param type: the type of event that 'sub_events' contains
   :type type: str
   :return: a dictionary with str-type keys that refer to the name of the calculated features and float-type values
   :rtype: dict[str, float]
   """
   
   features = {
      f"Mean {type} Duration": [],
      f"Mean Glucose During {type}s": [],
      f"Mean Upwards Slope of {type}s (mg/dL per min)": [],
      f"Mean Downwards Slope of {type}s (mg/dL per min)": [],
      f"Mean Minimum Glucose of {type}s": [],
      f"Mean Maximum Glucose of {type}s": [],
      f"Mean Amplitude of {type}s": [],
      f"Mean iAUC of {type}s": []
   }
   
   for _, event in sub_events.iterrows():
      event_data = retrieve_event_data(df, event)

      duration = event[AFTER] - event[BEFORE]
      features[f"Mean {type} Duration"].append(duration)

      features[f"Mean Glucose During {type}s"].append(event_data[GLUCOSE].mean())
      features[f"Mean Minimum Glucose of {type}s"] = nadir(event_data)
      features[f"Mean Maximum Glucose of {type}s"] = peak(event_data)

      event_time = event[TIME]
      closest_idx = (event_data[TIME] - event_time).abs().idxmin()
      event_glucose = event_data.loc[closest_idx, GLUCOSE]

      peak_glucose = peak(event_data)
      peak_time = event_data.loc[event_data[GLUCOSE].idxmax(), TIME]
      amplitude = peak_glucose - event_glucose
      features[f"Mean Amplitude of {type}s"].append(abs(amplitude))

      time_diff_to_peak = (peak_time - event_time).total_seconds() / 60.0
      slope_to_peak = (peak_glucose - event_glucose) / time_diff_to_peak if time_diff_to_peak != 0 else np.nan
      features[f"Mean Upwards Slope of {type}s (mg/dL per min)"].append(slope_to_peak)

      end_time = event_data[TIME].iloc[-1]
      end_glucose = event_data[GLUCOSE].iloc[-1]
      time_diff_peak_to_end = (end_time - peak_time).total_seconds() / 60.0
      slope_peak_to_end = (end_glucose - peak_glucose) / time_diff_peak_to_end if time_diff_peak_to_end != 0 else np.nan
      features[f"Mean Downwards Slope of {type}s (mg/dL per min)"].append(slope_peak_to_end)

      features[f"Mean iAUC of {type}s"].append(iAUC(event_data, event_glucose))

   features = {k: np.mean(v) for k, v in features.items()}
   features[f"Mean # of {type}s per day"] = sub_events.shape[0] / len(df[TIME].dt.date.unique())
   return features