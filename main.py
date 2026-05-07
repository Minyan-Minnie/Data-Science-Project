import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset.csv")

feats = [ "danceability","energy","loudness","tempo","valence","instrumentalness","speechiness"]
pred_popularity = "popularity"

df = df[feats + [pred_popularity]]      #gets us only the cols that we want
df = df.dropna()

df["popular"] = (df["popularity"] >= 70).astype(int)      #a song is considered popular if it has a rating of 70 or more
x = df[feats]
y = df["popular"]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state=42, stratify=y)

scale = StandardScaler()

train_scaled_x = scale.fit_transform(train_x)
test_scaled_x= scale.transform(test_x)

print(df.head())
print(df.columns)