import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import os

# Change working directory to the folder where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.data import loadData
from src.data import remove_nulls, add_weighted_features, compute_entropy, add_weighted_mean_properties, add_constraint_aware_features, add_normalized_entropy_features, add_polynomial_features

X, y, X_test = loadData("train.csv", "test.csv")

#################################################################################################################
#### Feature engineering ########################################################################################
#################################################################################################################

X = remove_nulls(X)

X = add_weighted_features(X)
print(type(X), type(X_test))

X['blend_entropy'] = X.apply(compute_entropy, axis=1)
print(type(X), type(X_test))

X = add_weighted_mean_properties(X)
print(type(X), type(X_test))

X = add_constraint_aware_features(X)
print(type(X), type(X_test))

corr = X.corr(numeric_only=True).abs()
high_pairs = np.column_stack(np.where((corr.values>0.9) & (corr.values<1)))
print("Highly correlated pairs (abs(corr) > 0.9):")
for i, j in high_pairs: 
    if i < j:  # Avoid duplicates
        print(f"{corr.index[i]} and {corr.columns[j]}: {corr.values[i, j]:.4f}")    

# Dropping redundant Component 4 features
to_drop = [c for c in X.columns
           if "component4" in c.lower() and "weighted" in c.lower()]
X = X.drop(columns=to_drop)

##############################################################################################################
#### Stacked emsemble model ##################################################################################
##############################################################################################################

def stackedensemble_model(X, y, X_test):
    # Splitting the data into training and validation sets
    
    # To avoid lookahead bias data used in base and meta models should not be used in the final validation set.
    # Splitting off 40% (meta + final val)
    X_base, X_temp, y_base, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # Splitting that 40% into 20% meta-training and 20% final validation
    # the final validation is done on the output of the meta model
    X_meta, X_val, y_meta, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Printing to verify
    print(f"X_base shape: {X_base.shape}")
    print(f"X_meta shape: {X_meta.shape}")
    print(f"X_val shape: {X_val.shape}")

    from sklearn.linear_model import Ridge
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    # Training the base models : LGBM XGBoost CatBoost and Ridge
    lgb_model = MultiOutputRegressor(
            LGBMRegressor(
            objective='huber', alpha=0.9,             
            n_estimators=1000,
            learning_rate=0.01,
            random_state=42,
            max_depth=-1,
            importance_type='gain' 
        ))
    lgb_model.fit(X_base, y_base)

    # Train Ridge (multi-output)
    ridge_model = Ridge()
    ridge_model.fit(X_base, y_base)

    cat_model = MultiOutputRegressor(
        CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            random_strength=1,
            bagging_temperature=1,
            early_stopping_rounds=30,
            verbose=0,
            random_seed=42
        ))
    cat_model.fit(X_base, y_base)

    xgb_model = MultiOutputRegressor(XGBRegressor(
    objective='reg:pseudohubererror',  
        n_estimators=1000,            
        learning_rate=0.01,                                           
        random_state=42,
        verbosity=0
    ))
    xgb_model.fit(X_base, y_base)

    # Getting base predictions on meta data to use in the meta model
    pred_lgb = lgb_model.predict(X_meta)
    pred_ridge = ridge_model.predict(X_meta)
    pred_cat = cat_model.predict(X_meta)
    pred_xgb = xgb_model.predict(X_meta)
    # Stacking predictions horizontally
    import numpy as np
    meta_X = np.hstack([pred_lgb, pred_ridge, pred_cat, pred_xgb])  

    # Training meta-model
    meta_model = Ridge()
    meta_model.fit(meta_X, y_meta)

    # Predicting with base models on X_val
    val_pred_lgb = lgb_model.predict(X_val)
    val_pred_ridge = ridge_model.predict(X_val)
    val_pred_cat = cat_model.predict(X_val)
    val_pred_xgb = xgb_model.predict(X_val)

    # Stacking val predictions
    val_meta_X = np.hstack([val_pred_lgb,val_pred_ridge, val_pred_cat, val_pred_xgb])

    # Final stacked prediction
    final_pred = meta_model.predict(val_meta_X)

    return final_pred, y_val, lgb_model, ridge_model, cat_model, xgb_model, meta_model

def evaluate_model(final_pred, y_val, show_split):

    # Evaluating the final predictions
    mape = mean_absolute_percentage_error(y_val, final_pred)
    r2 = r2_score(y_val, final_pred)
    rmse = mean_squared_error(y_val, final_pred, squared=False)

    print(f"📊 Validation MAPE: {mape:.4f}")
    print(f"📈 R2 Score: {r2:.4f}")
    print(f"📉 RMSE: {rmse:.4f}")

    
    if show_split:
        # Ensure final_pred is a DataFrame with proper columns and index
        y_pred_df = pd.DataFrame(final_pred, columns=y_val.columns, index=y_val.index)

        # Compute MAPE per blend property
        mape_per_property = {
            col: mean_absolute_percentage_error(y_val[col], y_pred_df[col])
            for col in y_val.columns
        }

        # Format as Dataframe
        mape_df = pd.DataFrame.from_dict(mape_per_property, orient='index', columns=['MAPE'])
        mape_df = mape_df.sort_values(by='MAPE', ascending=True)

        print("📊 MAPE per property:")
        print(mape_df.round(4))

final_pred, y_val, lgbm, ridge, catboost, xgboost, meta = stackedensemble_model(X, y, X_test)

evaluate_model(final_pred, y_val, True)

