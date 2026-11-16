from matplotlib import Path
import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "rf_model.joblib"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude",]

model = joblib.load(MODEL_PATH)

def predict_price(features: dict) -> float:
    X = pd.DataFrame([features], columns=FEATURES)
    return float(model.predict(X)[0]) * 100_000

if __name__ == "__main__":
    sample = {
        "MedInc": 8.3252, "HouseAge": 41, "Averooms": 6.98, "AveBedrms":1.02 , 
        "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23
    }
    print(f"Predicted price: ${predict_price(sample):,.0f}")