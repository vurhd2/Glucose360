import pandas as pd
from preprocessing import *
from features import *
from events import *
from plots import *

def main():
   df = import_directory("datasets")
   for index, row in create_features(df).iterrows():
      print(row)

if __name__ == "__main__":
   main()