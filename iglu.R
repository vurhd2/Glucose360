library(iglu)

temp_df <- read_raw_data("datasets/Clarity_Export_00000_Sutherland_Eliza_2023-10-16_235810.csv", sensor = "dexcom", id = "read")
#temp_df <- read_raw_data("datasets/Clarity_Export_00001_Fitzroy_Penelope_2023-10-16_235810.csv", sensor = "dexcom", id = "read")
#temp_df <- read_raw_data("datasets/Clarity_Export_00002_Barrow_Nathaniel_2023-10-16_235810.csv", sensor = "dexcom", id = "read")

df <- process_data(temp_df, id = "id", timestamp = "time", glu = "gl")

#print(agp(df, inter_gap = 30))
print(modd(df))
#print(mage(df, plot=TRUE))