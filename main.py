import pandas as pd

df = pd.read_csv("dataset.csv")

feats = [ "danceability","energy","loudness","tempo","valence","instrumentalness","speechiness"]
pred_popularity = "popularity"

df = df[feats + [pred_popularity]]      #gets us only the cols that we want
df = df.dropna()

df["popular"] = (df["popularity"] >= 70).astype(int)
y = df[feats]
y = df["popular"]


print(df.head())
print(df.columns)