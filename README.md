# House Price Prediction

Predicting median house values in California districts with scikit-learn.

## Data

California Housing dataset, loaded through sklearn.datasets.fetch_california_housing. It has 8 features per district (median income, house age, average rooms, location, etc.). The target is MedHouseVal, in units of $100,000.

## Results

Models were trained on 80% of the data and evaluated on the held-out 20%.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.533 | 0.746 | 0.576 |
| Random Forest | 0.328 | 0.506 | 0.804 |
| Random Forest (tuned) | 0.327 | 0.505 | 0.806 |

The tuned model (`n_estimators=200`, `max_depth=None`) is off by about $33k on average. Tuning barely changed the score, so the gain came from switching to a non-linear model, not from the parameters.

Median income is by far the most important feature (about 0.52 importance), followed by average occupancy and latitude.

## Limitations

- House values are capped at $500k in the dataset, so the model can't learn prices above that.
- The random forest overfits a bit: R² is 0.97 on the training set and 0.80 on the test set.

## Usage

pip install -r requirements.txt
Run the notebooks in notebooks/ in order. The last cells save the trained model to models/rf_model.joblib (not tracked in git because of its size). Then:

python src/predict.py
## Structure

notebooks/   exploration,  modeling
src/         predict.py
models/      saved model (generated)