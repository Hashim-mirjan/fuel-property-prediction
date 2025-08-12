import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import os


# Change working directory to the folder where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.Data import loadData
from src.Data import remove_nulls, add_weighted_features, compute_entropy, add_weighted_mean_properties, add_constraint_aware_features, add_normalized_entropy_features, add_polynomial_features

X, y, X_test = loadData("train.csv", "test.csv")

#### Feature engineering #########################################################################################################

X = remove_nulls(X)

X = add_weighted_features(X)
print(type(X), type(X_test))

# Add Shannon entropy to both train and test
X['blend_entropy'] = X.apply(compute_entropy, axis=1)
print(type(X), type(X_test))

X = add_weighted_mean_properties(X)
print(type(X), type(X_test))

# Add to both training and test sets
X = add_constraint_aware_features(X)
print(type(X), type(X_test))

# Add to both training and test sets
X = add_normalized_entropy_features(X)
print(type(X), type(X_test))

#### Model #########################################################################################################

def XGB_model(X, y, X_test):
    # Splitting the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    # Training the XGB model
    xgb_model = MultiOutputRegressor(
        XGBRegressor(
        objective='reg:squarederror',  # Standard regression loss
        n_estimators=1200,              # Number of trees (increase for better performance)
        learning_rate=0.01,            # Low learning rate for smoother convergence                 # Controls tree complexity (6–8 is typical)                     # Minimum loss reduction to make a further split
        random_state=42,
        verbosity=0
))
    xgb_model.fit(X_train, y_train)

    # Predicting on validation set
    val_pred = xgb_model.predict(X_val)

    return val_pred, y_val  

    # Predicting on validation set
    val_pred = lgb_model.predict(X_val)

    return val_pred, y_val

def evaluate_model(final_pred, y_val, show_split):

    # Evaluating the final predictions
    mape = mean_absolute_percentage_error(y_val, final_pred)
    r2 = r2_score(y_val, final_pred)
    rmse = mean_squared_error(y_val, final_pred, squared=False)

    print(f"📊 Validation MAPE: {mape:.4f}")
    print(f"📈 R2 Score: {r2:.4f}")
    print(f"📉 RMSE: {rmse:.4f}")

    # Ensure final_pred is a DataFrame with proper columns and index
    y_pred_df = pd.DataFrame(final_pred, columns=y_val.columns, index=y_val.index)

    if show_split:
        # Compute MAPE per blend property
        mape_per_property = {
            col: mean_absolute_percentage_error(y_val[col], y_pred_df[col])
            for col in y_val.columns
        }

        # Format as DataFrame
        mape_df = pd.DataFrame.from_dict(mape_per_property, orient='index', columns=['MAPE'])
        mape_df = mape_df.sort_values(by='MAPE', ascending=True)

        # Display
        print("📊 MAPE per property:")
        print(mape_df.round(4))

pred,y_val = XGB_model(X, y, X_test)

evaluate_model(pred, y_val, False)
