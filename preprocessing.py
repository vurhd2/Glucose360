import os, glob
import pandas as pd
import configparser
import zipfile, tempfile

config = configparser.ConfigParser()
config.read('config.ini')
ID = config['variables']['id']
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']

# globals for glucose values to replace "Low" and "High" with in the CGM data
LOW = 40
HIGH = 400

ex = r"(([\d]+-)(?P<id>[\d]+).(?P<section>[\w]+)(.+))"

def import_data(
   path: str,
   name: str = None,
   id_template: str = None,
   glucose: str = "Glucose Value (mg/dL)",
   time: str = "Timestamp (YYYY-MM-DDThh:mm:ss)",
   interval: int = 5,
   max_gap: int = 45,
) -> pd.DataFrame:
    """
    Returns a Multiindexed Pandas DataFrame containing all of the csv data found at the given path.
    The path can lead to a directory, .zip file, or a .csv file.
    The returned DataFrame holds columns for Timestamps and Glucose Value, and is indexed by 'id'
    @param path         the path of the directory/zip/csv to be parsed through
    @param glucose      the name of the column containing the glucose values in the .csv files
    @param time         the name of the column containing the timestamps in the .csv files
    @param interval     the resampling interval that the data should be in
    @param max_gap      the maximum amount of minutes a gap in the data can be interpolated 
                        (filling in a gap with a longer duration would be considered extrapolation)
    """
    global resample_interval
    resample_interval = interval

    ext = os.path.splitext(path)[1]

    # path leads to directory
    if ext == "":
       if not os.path.isdir(path):
          raise ValueError("Directory does not exist")
       else:
          return import_directory(path, id_template, glucose, time, interval, max_gap)
    
    # check if path leads to .zip or .csv
    if ext.lower() in [".csv", ".zip"]:
       if not os.path.isfile(path):
          raise ValueError("File does not exist")
    else:
       raise ValueError("Invalid file type")
   
    # path leads to .csv
    if ext.lower() == ".csv":
       df = import_csv(path, id_template, glucose, time, interval, max_gap)
       return df.set_index(ID)

    # otherwise has to be a .zip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
      # create a temporary directory to pull from
      with tempfile.TemporaryDirectory() as temp_dir:
         zip_ref.extractall(temp_dir)
         dir = name or path.split("/")[-1].split(".")[0]
         return import_directory((temp_dir + "/" + dir), id_template, glucose, time, interval, max_gap)
    
def import_directory(
    path: str,
    id_template: str = None,
    glucose: str = "Glucose Value (mg/dL)",
    time: str = "Timestamp (YYYY-MM-DDThh:mm:ss)",
    interval: int = 5,
    max_gap: int = 45,
) -> pd.DataFrame:
    """
    Returns a Multiindexed Pandas DataFrame containing all of the csv data found in the directory at the given path.
    The DataFrame holds columns for Timestamp and Glucose Value, and is indexed by 'id'
    @param path         the path of the directory to be parsed through
    @param interval     the resampling interval that the data should be in
    @param max_gap      the maximum amount of minutes a gap in the data can be interpolated 
                        (filling in a gap with a longer duration would be considered extrapolation)
    """
    csv_files = glob.glob(path + "/*.csv")

    if len(csv_files) == 0:
       raise Exception("No CSV files found.")

    data = pd.concat(import_csv(file, id_template, glucose, time, interval, max_gap) for file in csv_files)
    data = data.set_index([ID])

    print(f"{len(csv_files)} .csv files were found in the specified directory.")

    return data

def import_csv(
    path: str, 
    id_template: str = None,
    glucose: str = "Glucose Value (mg/dL)",
    time: str = "Timestamp (YYYY-MM-DDThh:mm:ss)",
    interval: int = 5, 
    max_gap: int = 45) -> pd.DataFrame:
    """
    Returns a pre-processed Pandas DataFrame containing the timestamp and glucose data for the csv file at the given path.
    The DataFrame returned has three columns: Timestamps, Glucose Values, and 'id' of the patient
    @param path      the path of the csv file to be pre-processed and read into a Pandas Dataframe
    @param interval  the resampling interval that the data should be in
    @param max_gap   the maximum amount of minutes a gap in the data can be interpolated 
                     (filling in a gap with a longer duration would be considered extrapolation)
    """
    df = pd.read_csv(path)
    
    id = retrieve_id(path.split("/")[-1], df, id_template)

    df = df.dropna(subset=[glucose])
    df = df.replace("Low", LOW)
    df = df.replace("High", HIGH)

    df[TIME] = pd.to_datetime(df[time], format="%Y-%m-%dT%H:%M:%S")

    df[GLUCOSE] = pd.to_numeric(df[glucose])

    df = df[[TIME, GLUCOSE]].copy()
    df = resample_data(df, interval, max_gap)
    df = df.loc[df[GLUCOSE].first_valid_index():df[GLUCOSE].last_valid_index()]
    df = chunk_day(chunk_time(df))
    df[ID] = id

    return df

def retrieve_id(name: str, df: pd.DataFrame, id_template: str = None):
   if id_template and "first" not in id_template and "last" not in id_template and "patient_identifier" not in id_template:
      # need to parse file name for id
      import re
      pattern = re.compile(fr"{id_template}")
      match = pattern.match(name)
      id = str(match.group("id"))
      try: 
         id += str(match.group("section"))
      except:
         print(f"'Section' not defined for patient {id}.")
      return id

   # use csv fields for id instead
   first = df["Patient Info"].iloc[0]
   last = df["Patient Info"].iloc[1]
   patient_identifier = df["Patient Info"].iloc[2]
   id = df["Patient Info"].iloc[0] + df["Patient Info"].iloc[1] 
   if id_template: id = id_template.format(first=first, last=last, patient_identifier=patient_identifier)
   return id

def resample_data(df: pd.DataFrame, minutes: int = 5, max_gap: int = 45) -> pd.DataFrame:
    """
    Resamples and (if needed) interpolates the given default-indexed DataFrame.
    Used mostly to preprocess the data in the csv files being imported in import_csv().
    @param df         the DataFrame to be resampled and interpolated
    @param minutes    the length of the interval to be resampled into (in minutes)
    """
    # Sort the DataFrame by datetime
    resampled_df = df.sort_values(by=[TIME])
    resampled_df = resampled_df.set_index(TIME)

    interval = str(minutes) + "T"
    
    # generate the times that match the frequency
    resampled_df = resampled_df.asfreq(interval)
    # add in the original points that don't match the frequency (just for linear time-based interpolation)
    resampled_df.reset_index(inplace=True)
    resampled_df = (pd.concat([resampled_df, df])).drop_duplicates(subset=[TIME])
    resampled_df.sort_values(by=[TIME], inplace=True)

    # interpolate the missing values
    resampled_df.set_index(TIME, inplace=True)
    resampled_df = interpolate_data(resampled_df, max_gap)
    
    # now that the values have been interpolated, remove the points that don't match the frequency
    resampled_df = resampled_df.asfreq(interval)
    resampled_df.reset_index(inplace=True)

    return resampled_df

def interpolate_data(df: pd.DataFrame, max_gap = int) -> pd.DataFrame:
    """
    Only linearly interpolates NaN glucose values for time gaps that are less than the given number of minutes.
    Used mainly in preprocessing for csv files that are being imported in import_csv().
    @param df         a DataFrame with only two columns, DateTime and Glucose Value
    @param max_gap    the maximum minute length of gaps that should be interpolated
    """

    # based heavily on https://stackoverflow.com/questions/67128364/how-to-limit-pandas-interpolation-when-there-is-more-nan-than-the-limit

    s = df[GLUCOSE].notnull()
    s = s.ne(s.shift()).cumsum()

    m = df.groupby([s, df[GLUCOSE].isnull()])[GLUCOSE].transform('size').where(df[GLUCOSE].isnull())
    interpolated_df = df.interpolate(method="time", limit_area="inside").mask(m >= int(max_gap / resample_interval))

    return interpolated_df

def chunk_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column specifying whether the values occur during a waking or sleeping period
    """
    times = df[TIME] - df[TIME].dt.normalize()
    is_waking = (times >= pd.Timedelta(hours=8)) & (times <= pd.Timedelta(hours=22))
    df["Time Chunking"] = is_waking.replace({True: "Waking", False: "Sleeping"})
    return df

def chunk_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column specifying whether the values occur during the week or during the weekend
    """
    is_weekend = df[TIME].dt.dayofweek > 4
    df["Day Chunking"] = is_weekend.replace({True: "Weekend", False: "Weekday"})
    return df

def get_interval():
   """
   Accessor function for the resampling interval (for use within other modules)
   """
   return resample_interval