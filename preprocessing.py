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

def import_data(
   path: str,
   name: str = None,
   id_template: str = None,
   glucose: str = "Glucose Value (mg/dL)",
   time: str = "Timestamp (YYYY-MM-DDThh:mm:ss)",
   interval: int = 5,
   max_gap: int = 45,
   output = print
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
    # update the config with the resampling interval the user chose
    updated_config = config['variables'] 
    updated_config['interval'] = str(interval)
    config["variables"] = updated_config
    with open('config.ini', 'w') as configfile:
      config.write(configfile)

    # get file extension of where the given path points
    ext = os.path.splitext(path)[1]

    # path leads to directory
    if ext == "":
       if not os.path.isdir(path):
          raise ValueError("Directory does not exist")
       else:
          return import_directory(path, id_template, glucose, time, interval, max_gap, output)
    
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
         return import_directory((temp_dir + "/" + dir), id_template, glucose, time, interval, max_gap, output)
    
def import_directory(
    path: str,
    id_template: str = None,
    glucose: str = "Glucose Value (mg/dL)",
    time: str = "Timestamp (YYYY-MM-DDThh:mm:ss)",
    interval: int = 5,
    max_gap: int = 45,   
    output = print
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
    num_files = len(csv_files)

    if num_files == 0:
       raise Exception("No CSV files found.")
    
    output(f"{num_files} .csv files were found in the specified directory.")
    
    data = []
    num_valid_files = num_files
    for file in csv_files:
       try:
          data.append(import_csv(file, id_template, glucose, time, interval, max_gap))
       except:
          num_valid_files -= 1
   
    output(f"{num_valid_files} .csv files were successfully imported.")

    if len(data) == 0: raise Exception("CSV files found, but none were valid.")  
    df = pd.concat(data)
    df.set_index([ID], inplace=True)

    output(f"{df.index.unique().size} sections were found in the imported data.")

    return df

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
      match = pattern.search(name)
      if match is None:
         raise Exception("The RegEx ID template passed does not match the file name.")
      id = str(match.group("id"))
      try: 
         section = str(match.group("section"))
         id += f" ({section})"
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
    config.read('config.ini')
    interval = int(config["variables"]["interval"])

    # based heavily on https://stackoverflow.com/questions/67128364/how-to-limit-pandas-interpolation-when-there-is-more-nan-than-the-limit

    s = df[GLUCOSE].notnull()
    s = s.ne(s.shift()).cumsum()

    m = df.groupby([s, df[GLUCOSE].isnull()])[GLUCOSE].transform('size').where(df[GLUCOSE].isnull())
    interpolated_df = df.interpolate(method="time", limit_area="inside").mask(m >= int(max_gap / interval))

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

def segment_data(path: str, df: pd.DataFrame) -> pd.DataFrame:
   """
   Splits patients' data into multiple segments based on a given .csv file containing ID's and DateTimes
   """
   segments = pd.read_csv(path)
   segments[TIME] = pd.to_datetime(segments[TIME])
   segments.set_index(ID, inplace=True)

   segmented_df = df.reset_index(drop=False)

   for id, locations in segments.groupby(ID):
      locations.reset_index(drop=True, inplace=True)
      locations.sort_values(TIME, ascending=False, inplace=True)
      for index, row in locations.iterrows():
         mask = (segmented_df[ID] == id) & (segmented_df[TIME] >= row[TIME])
         segmented_df.loc[mask, ID] = f"{id} ({index})"
   
   segmented_df.set_index(ID, inplace=True)
   return segmented_df