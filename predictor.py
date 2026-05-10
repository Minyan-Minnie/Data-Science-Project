import joblib
import pandas as pd

model = joblib.load("spotify_model.pkl")
scaler = joblib.load("spotify_scaler.pkl")

print("Spotify Popularity Predictor")
print()

danceability = float(input("Danceability (0-1): "))
energy = float(input("Energy (0-1): "))
loudness = float(input("Loudness (-60 to 0): "))
tempo = float(input("Tempo (BPM): "))
valence = float(input("Valence (0-1): "))
instrumentalness = float(input("Instrumentalness (0-1): "))
speechiness = float(input("Speechiness (0-1): "))

features = pd.DataFrame([{
    "danceability": danceability,
    "energy": energy,
    "loudness": loudness,
    "tempo": tempo,
    "valence": valence,
    "instrumentalness": instrumentalness,
    "speechiness": speechiness
}])

scaled_features = scaler.transform(features)

prob = model.predict_proba(scaled_features)[0][1]

prediction = "Popular" if prob > 0.4 else "Not Popular"

print()
print("Prediction:", prediction)
print(f"Confidence: {prob:.2%}")
