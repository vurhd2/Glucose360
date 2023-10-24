import pandas as pd
from features import *
from preprocessing import *
from events import *

def main():
   df = import_directory("datasets")

   for id, data in df.groupby('id'):
      print("ID: " + str(id))
      print("summary: " + str(summary_stats(data)))
      print("mean: " + str(mean(data)))
      print("a1c: " + str(a1c(data)))
      print("gmi: " + str(gmi(data)))
      print("std: " + str(std(data)))
   
   event1 = pd.DataFrame.from_records([{'id': "NathanielBarrow9/8/88", time(): '2023-08-08 7:45:35', 'before': 2, 'after': 3, 'description': 'testing1'}])
   event2 = pd.DataFrame.from_records([{'id': "ElizaSutherland2/23/68", time(): '2023-07-03 3:15:17', 'before': 6, 'after': 9, 'description': 'testing2'}])
   event3 = pd.DataFrame.from_records([{'id': "PenelopeFitzroy1/5/52", time(): '2023-06-24 0:26:10', 'before': 3, 'after': 1, 'description': 'testing3'}])
   events = pd.concat([event1, event2, event3])
   
   print(retrieve_event_data(df, events))

   print(create_event_features(df, events))

if __name__ == "__main__":
   main()