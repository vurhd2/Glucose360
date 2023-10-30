import pandas as pd
from features import *
from preprocessing import *
from events import *
from plots import *

def main():
   df = import_directory("datasets")
   
   event1 = pd.DataFrame.from_records([{'id': "NathanielBarrow", time(): '2023-08-08 7:45:35', 'before': 2, 'after': 3, 'description': 'testing1'}])
   event2 = pd.DataFrame.from_records([{'id': "NathanielBarrow", time(): '2023-08-13 2:55:35', 'before': 10, 'after': 10, 'description': 'testing2'}])
   event3 = pd.DataFrame.from_records([{'id': "ElizaSutherland", time(): '2023-07-03 3:15:17', 'before': 6, 'after': 9, 'description': 'testing3'}])
   event4 = pd.DataFrame.from_records([{'id': "PenelopeFitzroy", time(): '2023-06-24 0:26:10', 'before': 3, 'after': 1, 'description': 'testing4'}])
   events = pd.concat([event1, event2, event3, event4])
   
   event_df = retrieve_event_data(df, events)
   #create_event_features(df, events)
   #daily_plot(df, events, "NathanielBarrow", save=False)
   daily_plot_all(df, events=None)
   #spaghetti_plot_all(df)

if __name__ == "__main__":
   main()