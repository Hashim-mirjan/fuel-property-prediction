import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def loadData(trainpath="train.csv", testpath="test.csv"):
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    X_test = test.copy()

    # Separate features and target
    X = train.iloc[:, :55].copy()  
    y = train.iloc[:, 55:].copy()
    X_test = test.copy()

    print(X.head())

    return X, y, X_test

def remove_nulls(df):
    # Remove rows with any null values
    df = df.dropna()
    # Reset index after dropping rows
    df.reset_index(drop=True, inplace=True)
    return df

# function to multiply component feature by component percentage to get a weighted amount for that feature
def add_weighted_features(df):
    df = df.copy()  
    for i in range(1, 6):  # Components 1 to 5
        frac_col = f"Component{i}_fraction"
        for j in range(1, 11):  # Properties 1 to 10
            prop_col = f"Component{i}_Property{j}"
            new_col = f"{prop_col}_weighted"
            df.loc[:, new_col] = df[prop_col] * df[frac_col]
    return df

# Calculate blend entropy as new feature
def compute_entropy(row):
    fractions = row[[f"Component{i}_fraction" for i in range(1, 6)]].values
    # Avoid log(0) by adding a small epsilon
    fractions = np.clip(fractions, 1e-10, 1)
    return -np.sum(fractions * np.log(fractions))

# functions to add 10 weighted mean properties by summing the weighted properties of each component
def add_weighted_mean_properties(df):
    # Add weighted mean property features
    for j in range(1, 11):  # Property1 to Property10
        weighted_sum = sum(df[f"Component{i}_Property{j}"] * df[f"Component{i}_fraction"] for i in range(1, 6))
        df[f"Property{j}_weighted_mean"] = weighted_sum
    return df

# Adding features measuring the ratio of specific component fractions
def add_constraint_aware_features(df):
    df["fraction_ratio_51"] = df["Component5_fraction"] / (df["Component1_fraction"] + 1e-6)
    df["fraction_ratio_54"] = df["Component5_fraction"] / (df["Component4_fraction"] + 1e-6)
    df["fraction_ratio_42"] = df["Component4_fraction"] / (df["Component2_fraction"] + 1e-6)
    df["fraction_ratio_41"] = df["Component4_fraction"] / (df["Component1_fraction"] + 1e-6)
    return df


def add_normalized_entropy_features(df):
    # Normalized entropy-like feature over fractions
    fractions = [f"Component{i}_fraction" for i in range(1, 6)]
    df["fraction_entropy_proxy"] = -sum(df[col] * np.log(df[col] + 1e-6) for col in fractions)
    return df

# Adding features for ratio of property n to other properties
# Only used when modelling property 8, 7 and 5 alone later in the code
def add_property_ratios(df, propertynum):
    for j in range(1, 11):
        if j != propertynum:  # Exclude self-ratio
            df[f"P{propertynum}_to_P{j}_weighted_mean"] = df[f"Property{propertynum}_weighted_mean"] / (df[f"Property{j}_weighted_mean"] + 1e-6)
    return df
    
# Adding polynomial features to show more complex relationships
def add_polynomial_features(X, X_test, columns, degree=2):
    # Extract only the specified columns for transformation
    X_sub = X[columns]
    X_test_sub = X_test[columns]
    # Fit on training subset only to avoid data leakage
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly.fit(X_sub)
    # Transform both train and test
    X_poly_vals = poly.transform(X_sub)
    X_test_poly_vals = poly.transform(X_test_sub)
    # Use compatible feature name extraction
    poly_col_names = poly.get_feature_names_out(columns)
    # Create DataFrames
    X_poly_df = pd.DataFrame(X_poly_vals, columns=poly_col_names, index=X.index)
    X_test_poly_df = pd.DataFrame(X_test_poly_vals, columns=poly_col_names, index=X_test.index)
    # Exclude original features (degree-1 terms)
    degree_1_terms = set(columns)
    new_cols = [col for col in X_poly_df.columns if col not in degree_1_terms]
    X_poly_filtered = X_poly_df[new_cols]
    X_test_poly_filtered = X_test_poly_df[new_cols]
    # Concatenate only new interaction/squared features
    X_out = pd.concat([X, X_poly_filtered], axis=1)
    X_test_out = pd.concat([X_test, X_test_poly_filtered], axis=1)
    return X_out, X_test_out

