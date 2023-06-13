![license
status](https://img.shields.io/github/license/johnwslee/extrucal_machine-learning)

# Study on Machine Learning Using Data Generated by Analytical Solution

**Author:** John W.S. Lee

## Introduction

This study was motivated from a simple question, "Can a trained machine learning model perform as well as an analytical solution?". In order to find out, this study was carried out using data in the field of polymer extrusion processes. 

First, in order to prepare a dataset for machine learning, throughput data were generated using [extrucal](https://github.com/johnwslee/extrucal) library with various extruder size, screw geometries, polymer melt density, and screw RPMs.

Second, basic exploratory data analysis was performed to check on the distribution of features and target, and log transformation was applied to the skewed features and target.

Third, cross-validation was carried out using multiple machine learning models, and the best model was selected based on the cross-validation score, which was `mean_squared_error`.

Fourth, hyperparameter optimization was performed for the selected machine learning model, and performance of the model before and after optimization were compared for the extruders with the size ranging from 25 mm to 250 mm.

## Approaches

### 1. Generation of Data



### 2. Exploratory Data Analysis



### 3. Cross-Validation of Multiple Machine Learning Models




### 4. Hyperparameter Optimization




### 5. Conclusion


