## Project Overview

This project focuses on predicting 10 unknown fuel properties of a Sustainable Aviation Fuel (SAF) blend from blend component properties.

SAFs are very important for reducing the environmental impact of the aviation sector, but crafting the perfect belnd from different sustainable fuels and traditional fuels is a challenge. Hence, this project aims to predict blend properties given the properties and blend fractions of the components to enable rapid evaluation of many blend comobinations.

Context: This project was developed during Shell.ai Hackathon for Sustainable and Affordable Energy 2025.

## Data

The training data has 65 columns with the last 10 being the 10 target properties. There are 2000 samples in the training data.

Each row represents a unique fuel blend with a total of 65 columns, which are organized into three distinct groups:
- Blend Composition (first 5 columns) : A decimal percentage between 0 and 1
- Component Properties (next 50 columns) : . The column names are structured using the format {component_number}_{property_number}. 5 Components with 10 properties each = 50
- Final Blend Properties - Targets (last 10 columns): The column names for these target properties are Blend{property_number}

The identity of the properties are unknown. All property values in the dataset have been pre-standardized so they are all on the same scale (centered around 0) and each feature has a unit variance. Example below showing property 1 distributions:

|       |   Component1_Property1 |   Component2_Property1 |   Component3_Property1 |   Component4_Property1 |   Component5_Property1 |
|-------|------------------------|------------------------|------------------------|------------------------|------------------------|
| count |                2000    |                2000    |                2000    |                2000    |                2000    |
| mean  |                   0    |                  -0.02 |                   0    |                  -0    |                  -0.02 |
| std   |                   1    |                   1.01 |                   1    |                   1.01 |                   1.01 |
| min   |                  -2.94 |                  -1.72 |                  -3.01 |                  -3.03 |                  -3.57 |
| 25%   |                  -0.69 |                  -0.77 |                  -0.7  |                  -0.69 |                  -0.71 |
| 50%   |                   0.01 |                  -0.03 |                   0.02 |                   0.02 |                   0.19 |
| 75%   |                   0.69 |                   0.65 |                   0.67 |                   0.66 |                   1.03 |
| max   |                   2.98 |                   3.05 |                   2.87 |                   2.98 |                   1.03 |

## Feature Engineering

To enhance the quality of the data I engineering some additional features.
The following features were added:

- *Weighted Properties*: 50 weighted properties caluclated by multiplying component percentage by component properties. If component percentage is lower then the resulting weighted property will be lower. If a component percentage is zero then the resulting weighted property will be zero as there is no contribution.
- *Mean Weighted Properties*: The weighted average of each property in the fuel. Calculated by summing the 5 weighted values for each property giving us 10 weighted average properties.
- *Shannon Entropy*: Measures the diversity of the blend. A lower entropy suggests the blend is dominated by one/two components. Blend entropy in a practical sense can affect combustion characteristics of fuels.
- *Component Ratios*: The ratio of one component fraction to another, eg. (Component5% / Component1%). These interaction features are another way to add non-linearity into the features and show relative influence of components.

### Feature Redundancy

Features pairs with a pearson correlation coefficient over 0.9 were flagged for collinearity:

| Feature 1                    | Feature 2                          | Correlation |
|-----------------------------|------------------------------------|-------------|
| Component4_Property1        | Component4_Property1_weighted      | 0.9255      |
| Component4_Property2        | Component4_Property2_weighted      | 0.9250      |
| Component4_Property3        | Component4_Property3_weighted      | 0.9252      |
| Component4_Property4        | Component4_Property4_weighted      | 0.9263      |
| Component4_Property5        | Component4_Property5_weighted      | 0.9264      |
| Component4_Property6        | Component4_Property6_weighted      | 0.9221      |
| Component4_Property7        | Component4_Property7_weighted      | 0.9203      |
| Component4_Property8        | Component4_Property8_weighted      | 0.9202      |
| Component4_Property9        | Component4_Property9_weighted      | 0.9255      |
| Component4_Property10       | Component4_Property10_weighted     | 0.9234      |

These can be explained by component 4's distribution. Unlike the other compnents, component_4_fraction is never 0 so it is always present and for the majority of mixtures it is equal to or close to 0.5. Its lower variance means that the weighted features don't add much value and may harm the model.

<img width="391" height="281" alt="image" src="https://github.com/user-attachments/assets/795623fb-5e92-41a0-91f1-34f716b603ca" />


Result: Dropped Component 4 weighted properties but maintained the others

## Modelling

A stacked ensemble model was used. Outputs from 4 base models are fed into a meta model to combine the individual predictions and give a final prediction.

Base Models:
- XGBoost, LightGBM, CatBoost, Ridge

Stacking Strategy:
- To prevent lookahead bias and data leakage, a careful data splitting strategy was followed:

1. Split 1 — Base vs. Meta/Validation

  - 60% of the data was used to train the base models (X_base, y_base)

  - 40% held out for meta training and final validation (X_temp, y_temp)

2. Split 2 — Meta vs. Final Validation

  - The 40% held-out set was split again:

    - 20% for training the meta model (X_meta, y_meta)

    - 20% for final evaluation (X_val, y_val) — only predictions from the meta-model were used here

This strategy ensures that the final validation set is completely unseen by both base and meta models, enabling an unbiased evaluation of the full ensemble.

Meta Model:
- A Ridge regression model was used to combine predictions from the base models. The meta-model was trained using predictions from the base layer.

## Results

Metrics used: Mean Absolute Percentage Error (this was the competiton metric), R² Score (Coefficient of Determination, max = 1.0)

First comparing the individual models to the stacked model:

|         | Avg MAPE | Avg R2 |
|---------|----------|--------|
| LGBM    | 1.31     | 0.971  |
| XGB     | 1.70     | 0.960  |
| Cat     | 0.791    | 0.980  |
| Stacked | **0.581**    | **0.985**  |

On average the MAPEs seem very high but this is explained by the majority of datapoints being close to zero causing mape to explode.

The stacked model outperforms all of the individual models. While the training process was slightly different we can still conclude that the stacked model gives a stronger prediction from these results.

Breaking down the final model average MAPE into individual properties:

| Blend Property   | MAPE   |
|------------------|--------|
| BlendProperty1   | 0.1801 |
| BlendProperty2   | 0.2171 |
| BlendProperty10  | 0.2698 |
| BlendProperty4   | 0.3079 |
| BlendProperty5   | 0.4105 |
| BlendProperty6   | 0.4790 |
| BlendProperty3   | 0.5351 |
| BlendProperty9   | 0.6376 |
| BlendProperty8   | 0.8565 |
| BlendProperty7   | 1.9174 |

