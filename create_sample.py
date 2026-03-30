import pandas as pd

df = pd.read_csv("data/US_Accidents_March23.csv", nrows=5000)
df.to_csv("data/small_data.csv", index=False)

print("Done")