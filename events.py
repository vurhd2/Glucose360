import pandas as pd
from features import create_features
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
INTERVAL = config['variables'].getint('interval')

def get_episodes(
    df: pd.DataFrame,
    hypo_lvl1: int = 70,
    hypo_lvl2: int = 54,
    hyper_lvl1: int = 180,
    hyper_lvl2: int = 250,
    min_length: int = 15
) -> pd.DataFrame:
   episodes = pd.DataFrame()
   for id, data in df.groupby('id'):
      timegap = lambda timedelta: timedelta.total_seconds() / 60

      episode_df = data[(data[GLUCOSE] <= hypo_lvl1) | (data[GLUCOSE] >= hyper_lvl1)].copy()
      print(episode_df)
      episode_df.reset_index(drop=True, inplace=True)
      episode_df["gap"] = episode_df[TIME].diff().apply(timegap)

      edges = episode_df.index[episode_df["gap"] != INTERVAL].to_list()
      edges.insert(0,0); edges.append(-1)

      get = lambda loc, col: episode_df.iloc[loc][col]
      for index in range(1, len(edges)):
         offset = 0 if (index == len(edges) - 1) else 1
         end_i = edges[index] - offset # index of the end of the episode (inclusive! - that's what the offset is for)
         start_i = edges[index - 1] # index of the start of the episode
         episode_length = timegap(get(end_i, TIME) - get(start_i, TIME))

         if episode_length >= min_length:
            type = "hyper" if (get(start_i, GLUCOSE) >= hyper_lvl1) else "hypo"

            level = 1
            current_episode = episode_df.iloc[start_i:end_i+offset]
            if ((type == "hyper" and (current_episode[current_episode[GLUCOSE] >= hyper_lvl2].shape[0] > (min_length / INTERVAL))) or 
                (type == "hypo" and (current_episode[current_episode[GLUCOSE] <= hypo_lvl2].shape[0] > (min_length / INTERVAL)))):
               level = 2

            description = f"{type}glycemic episode of level {level} occurring from {get(start_i, TIME)} to {get(end_i, TIME)}"
            event = pd.DataFrame.from_records([{"id": id, TIME: get(start_i, TIME), "before": 0, "after": episode_length, 
                                               "type": f"{type} level {level} episode", "description": description}])
            episodes = pd.concat([episodes, event]) 

   return episodes

def excursions(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Pandas Series containing the Timestamps of glucose excursions
    """
    sd = std(df)
    ave = mean(df)

    outlier_df = df[(df[GLUCOSE] >= ave + (2 * sd)) | (df[GLUCOSE] <= ave - (2 * sd))].copy()

    # calculate the differences between each of the timestamps
    outlier_df.reset_index(inplace=True)
    outlier_df["timedeltas"] = outlier_df[TIME].diff()[1:]

    # find the gaps between the times
    gaps = outlier_df[outlier_df["timedeltas"] > pd.Timedelta(minutes=INTERVAL)][TIME]

    # adding initial and final timestamps so excursions at the start/end are included
    initial = pd.Series(df[TIME].iloc[0] - pd.Timedelta(seconds=1))
    final = pd.Series(df[TIME].iloc[-1] + pd.Timedelta(seconds=1))
    gaps = pd.concat([initial, gaps, final])

    # getting the timestamp of the peak within each excursion
    excursions = []
    for i in range(len(gaps) - 1):
        copy = outlier_df[
            (outlier_df[TIME] >= gaps.iloc[i])
            & (outlier_df[TIME] < gaps.iloc[i + 1])
        ][[TIME, GLUCOSE]].copy()
        copy.set_index(TIME, inplace=True)
        if np.min(copy) > ave:
            # local max
            excursions.append(copy.idxmax())
        else:
            # local min
            excursions.append(copy.idxmin())

    return pd.Series(excursions)

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


def create_event_features(
    df: pd.DataFrame,
    events: pd.DataFrame,
    before: str = "before",
    after: str = "after",
    type: str = "type",
    desc: str = "description",
) -> pd.DataFrame:
    event_data = retrieve_event_data(df, events, before, after, type, desc)
    return create_features(event_data, events=True)
