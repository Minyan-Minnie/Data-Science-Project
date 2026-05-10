import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import joblib
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
print("Accuracy:  ", accuracy_score(test_y, pred_y))

#Logistic Regression scores
lr_accuracy = accuracy_score(test_y, pred_y)
lr_precision = precision_score(test_y, pred_y)
lr_recall = recall_score(test_y, pred_y)
lr_f1 = f1_score(test_y, pred_y)

# Random Forest
clf_rf = RandomForestClassifier().fit(train_scaled_x, train_y)      #train the barebones random forest model
#clf_rf = RandomForestClassifier(class_weight='balanced').fit(train_scaled_x, train_y)   #class weight balanced hurt the overall f1 score but did end up balancing the precision and recall.    
prob_y = clf_rf.predict_proba(test_scaled_x)[:, 1]
pred_y = (prob_y > 0.4).astype(int)         #I found 0.4 netted the best f1 score (~0.71)

""" Test t values
for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    pred_y = (prob_y > t).astype(int)

    print("Random Forest: ", t, "\nPrecision: ", precision_score(test_y, pred_y), "\nRecall:    ", recall_score(test_y, pred_y), "\nF1:        ", f1_score(test_y, pred_y), "\nAccuracy:  ", accuracy_score(test_y, pred_y))
"""

print("Random Forest:\nPrecision: ", precision_score(test_y, pred_y), "\nRecall:    ", recall_score(test_y, pred_y), "\nF1:        ", f1_score(test_y, pred_y), "\nAccuracy:  ", accuracy_score(test_y, pred_y))

#Random Forest Scores
rf_accuracy = accuracy_score(test_y, pred_y)
rf_precision = precision_score(test_y, pred_y)
rf_recall = recall_score(test_y, pred_y)
rf_f1 = f1_score(test_y, pred_y)

#===================================================
clf_rf_tuned = RandomForestClassifier(
    n_estimators=300,
    max_depth=12
).fit(train_scaled_x, train_y)

prob_y_tuned = clf_rf_tuned.predict_proba(test_scaled_x)[:, 1]
pred_y_tuned = (prob_y_tuned > 0.12).astype(int)                  #used a threshold of 0.12 because I found it works best here, still only made the f1 score worse

print("Random Forest w/ Hyperparameters:\nPrecision: ", precision_score(test_y, pred_y_tuned), "\nRecall:    ", recall_score(test_y, pred_y_tuned), "\nF1:        ", f1_score(test_y, pred_y_tuned), "\nAccuracy:  ", accuracy_score(test_y, pred_y_tuned))

#Random Forest with Hyperparameters Scores
rf_hyper_accuracy = accuracy_score(test_y, pred_y_tuned)
rf_hyper_precision = precision_score(test_y, pred_y_tuned)
rf_hyper_recall = recall_score(test_y, pred_y_tuned)
rf_hyper_f1 = f1_score(test_y, pred_y_tuned)


#now trying with additional features (liveness and duration)
more_feats = [
    "danceability",
    "energy",
    "loudness",
    "tempo",
    "valence",
    "instrumentalness",
    "speechiness",
    "liveness",
    "duration_ms"
]

df_more = pd.read_csv("dataset.csv")

df_more = df_more[more_feats + [pred_popularity]]
df_more = df_more.dropna()

df_more["popular"] = (df_more["popularity"] >= 70).astype(int)

x_more = df_more[more_feats]
y_more = df_more["popular"]

train_x_more, test_x_more, train_y_more, test_y_more = train_test_split(
    x_more,
    y_more,
    test_size=0.2,
    random_state=42,
    stratify=y_more
)

scale_more = StandardScaler()

train_scaled_more = scale_more.fit_transform(train_x_more)
test_scaled_more = scale_more.transform(test_x_more)

clf_rf_more = RandomForestClassifier().fit(train_scaled_more, train_y_more)

prob_y_more = clf_rf_more.predict_proba(test_scaled_more)[:, 1]
pred_y_more = (prob_y_more > 0.4).astype(int)

print(
    "Additional Features Random Forest:"
    "\nPrecision: ", precision_score(test_y_more, pred_y_more),
    "\nRecall:    ", recall_score(test_y_more, pred_y_more),
    "\nF1:        ", f1_score(test_y_more, pred_y_more),
    "\nAccuracy:  ", accuracy_score(test_y_more, pred_y_more)
)
#adding features didn't change anything, best score still around 0.7

#Random Forest with Added Features Scores
rf_feats_accuracy = accuracy_score(test_y_more, pred_y_more)
rf_feats_precision = precision_score(test_y_more, pred_y_more)
rf_feats_recall = recall_score(test_y_more, pred_y_more)
rf_feats_f1 = f1_score(test_y_more, pred_y_more)


#=======================================================================================================================
#Below are a bunch of graphs/visuals

#Graph comparing the amount of songs that are popular vs not popular
df["popular"].value_counts().sort_index().plot(kind = "bar")

plt.title("Popular Songs vs Unpopular Songs")
plt.xlabel("Popularity Group")
plt.ylabel("Number of Songs")
plt.xticks([0, 1], ["Unpopular", "Popular"], rotation = 0)
plt.show()

#Using the random forest model to help us figure out which feature played a big role in deciding popularity
most_useful_feat = clf_rf.feature_importances_

plt.barh(feats, most_useful_feat)
plt.title("Importance of Features Based on Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Graph showing if Logistic Regression is pushing model towards Popular or Unpopular
lr_coefficients = clf_lr.coef_[0]

coef_df = pd.DataFrame({ "Feature": feats, "Coefficient": lr_coefficients }).sort_values(by = "Coefficient", ascending = True)

plt.barh(coef_df["Feature"], coef_df["Coefficient"])
plt.title("Logistic Regression Feature Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.axvline(0)
plt.show()

# ===================================================================================

# Graph comparing Logistic Regression vs Random Forest vs RF w/ Hyperparameter vs RF w/ Added Feats performance
metrics_df = pd.DataFrame({ "Measurement Type": ["Accuracy", "Precision", "Recall", "F1"],
                           "Logistic Regression": [lr_accuracy, lr_precision, lr_recall, lr_f1],
                           "Random Forest": [rf_accuracy, rf_precision, rf_recall, rf_f1],
                           "Random Forest w/ Hyperparameters": [rf_hyper_accuracy, rf_hyper_precision, rf_hyper_recall, rf_hyper_f1],
                           "Random Forest w/ Added Features": [rf_feats_accuracy, rf_feats_precision, rf_feats_recall, rf_feats_f1]})


metrics_df.set_index("Measurement Type").plot(kind = "bar")

plt.title("Logistic Regression vs. Random Forest[Base, w/ Hyperparameters, w/ Added Features]")
plt.xlabel("Selected Measurement")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation = 0)
plt.show()



joblib.dump(clf_rf, "spotify_model.pkl")      #saves the training data so that we could use it for the UI
joblib.dump(scale, "spotify_scaler.pkl")      #saves the training data so that we could use it for the UI