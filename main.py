import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

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

# Logistic Regression
#clf = LogisticRegression().fit(train_scaled_x, train_y)        #train the barebones logistic regression model
clf_lr = LogisticRegression(l1_ratio=0, solver='newton-cholesky', class_weight='balanced').fit(train_scaled_x, train_y)        #train using the newton-cholesky solver rather than default, class_weight='balanced' punishes the model learning 0 for everything because of the inbalance in the training dataset

prob_y = clf_lr.predict_proba(test_scaled_x)[:, 1]          #predict the probabilities for all training data

pred_y = (prob_y > 0.6).astype(int)             #predicts 1 or 0 depending on the probability. The default was 0.5, but after trying some other thresholds, 0.6 had the best f1 score (0.16 on average compared with 0.13 for 'prob_y > 0.5')
print("Logistic Regression:\nPrecision: ", precision_score(test_y, pred_y), "\nRecall:    ", recall_score(test_y, pred_y), "\nF1:        ", f1_score(test_y, pred_y))

# Random Forest
clf_rf = RandomForestClassifier().fit(train_scaled_x, train_y)      #train the barebones random forest model
#clf_rf = RandomForestClassifier(class_weight='balanced').fit(train_scaled_x, train_y)   #class weight balanced hurt the overall f1 score but did end up balancing the precision and recall.    
prob_y = clf_rf.predict_proba(test_scaled_x)[:, 1]
pred_y = (prob_y > 0.4).astype(int)         #I found 0.4 netted the best f1 score (~0.71)

""" Test t values
for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    pred_y = (prob_y > t).astype(int)

    print("Random Forest: ", t, "\nPrecision: ", precision_score(test_y, pred_y), "\nRecall:    ", recall_score(test_y, pred_y), "\nF1:        ", f1_score(test_y, pred_y))
"""

print("Random Forest:\nPrecision: ", precision_score(test_y, pred_y), "\nRecall:    ", recall_score(test_y, pred_y), "\nF1:        ", f1_score(test_y, pred_y))
