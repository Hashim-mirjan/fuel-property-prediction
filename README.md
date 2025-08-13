## Project Overview

This project focuses on predicting 10 unknown fuel properties of a Sustainable Aviation Fuel blend from blend component properties.

The names of the properties and compnents and unknown and the values are all standardized (mean ≈ 0, std ≈ 1)

Context: This project was developed during Shell.ai Hackathon for Sustainable and Affordable Energy 2025.

## Data

The training data has 65 columns with the last 10 being the 10 target properties.

Each row represents a unique fuel blend with a total of 65 columns, which are organized into three distinct groups:
- Blend Composition (first 5 columns) : A decimal percentage between 0 and 1
- Component Properties (next 50 columns) : . The column names are structured using the format {component_number}_{property_number}. 5 Components with 10 properties each = 50
- Final Blend Properties - Targets (last 10 columns): The column names for these target properties are Blend{property_number}

The identity of the properties are unknown.

## Feature Engineering

To enhance the quality of the data I engineering some additional features.
The following features were added:

- *Weighted Properties*: 50 weighted properties caluclated by multiplying component percentage by component properties. If component percentage is lower then the resulting weighted property will be lower. If a component percentage is zero then the resulting weighted property will be zero as there is no contribution.
- *Mean Weighted Properties*: The weighted average of each property in the fuel. Calculated by summing the 5 weighted values for each property giving us 10 weighted average properties.
- *Shannon Entropy*: Measures the diversity of the blend. A lower entropy suggests the blend is dominated by one/two components. Blend entropy in a practical sense can affect combustion characteristics of fuels.
- *Component Ratios*: The ratio of one component fraction to another, eg. (Component5% / Component1%). These interaction features are another way to add non-linearity into the features and show relative influence of components.




