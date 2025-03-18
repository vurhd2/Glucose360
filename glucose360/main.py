import pandas as pd
from glucose360.preprocessing import *
from glucose360.features import *
from glucose360.events import *
from glucose360.plots import *

def main():
   df = import_data("datasets")
   for index, row in create_features(df).iterrows():
      print(row)

if __name__ == "__main__":
   main()