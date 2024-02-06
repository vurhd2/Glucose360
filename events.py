import pandas as pd
from features import create_features
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
INTERVAL = config['variables'].getint('interval')

def episodes(
    df: pd.DataFrame,
    hypo_lvl1: int = 70,
    hypo_lvl2: int = 54,
    hyper_lvl1: int = 180,
    hyper_lvl2: int = 250,
    min_length: int = 15
) -> pd.DataFrame:
   episode_df = pd.DataFrame()
   for id, data in df.groupby('id'):
      timegap = lambda timedelta: timedelta.total_seconds() / 60

      # hypo
      hypo_df = data[data[GLUCOSE] <= hypo_lvl1]
      hypo_df["gap"] = hypo_df[TIME].diff().apply(timegap)

      edges = list(hypo_df[hypo_df["gap"] != INTERVAL][TIME])
      edges.insert(0, hypo_df[TIME].iloc[0]); edges.append(hypo_df[TIME].iloc[-1])

      for index in range(1, len(edges)):
         episode_length = timegap(edges[index] - edges[index - 1])
         if episode_length >= min_length:
            level = 2 if (hypo_df[hypo_df[GLUCOSE] <= hypo_lvl2].shape[0] > min_length / INTERVAL) else 1
            description = f"hypoglycemic episode of level {level} occurring from {edges[index - 1]} to {edges[index]}"
            event = pd.DataFrame.from_records([{"id": id, TIME: edges[index - 1], "before": 0, "after": episode_length, 
                                               "type": f"hypo level {level} episode", "description": description}])
            episode_df = pd.concat([episode_df, event]) 
             
      # hyper
      """
      hyper_df = data[data[GLUCOSE] >= hyper_lvl1]
      hyper_df["gap"] = hyper_df[TIME].diff().apply(timegap)

      edges = list(hyper_df[hyper_df["gap"] != INTERVAL][TIME])
      edges.insert(0, hyper_df[TIME].iloc[0]); edges.append(hyper_df[TIME].iloc[-1])

      for index in range(1, len(edges)):
         episode_length = timegap(edges[index] - edges[index - 1])
         if episode_length >= min_length:
            level = 2 if (hyper_df[hyper_df[GLUCOSE] >= hyper_lvl2].shape[0] > min_length / INTERVAL) else 1
            description = f"hyperglycemic episode of level {level} occurring from {edges[index - 1]} to {edges[index]}"
            event = pd.DataFrame.from_records([{"id": id, TIME: edges[index - 1], "before": 0, "after": episode_length, 
                                               "type": f"hyper level {level} episode", "description": description}])
            episode_df = pd.concat([episode_df, event])       
   """
   return episode_df

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
