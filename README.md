![license
status](https://img.shields.io/github/license/johnwslee/extrucal_machine-learning)

# Study on Machine Learning Using Data Generated by Analytical Solution

**Author:** John W.S. Lee

## 1. Introduction

This study was motivated from a simple question, "Can a trained machine learning model perform as well as an analytical solution?". In order to find out, this study was carried out using data in the field of polymer extrusion processes. 

First, in order to prepare a dataset for machine learning, throughput data was generated using [extrucal](https://github.com/johnwslee/extrucal) library with various extruder size, screw geometries, polymer melt density, and screw RPMs. Then, basic exploratory data analysis was performed to check on the distribution of features and target, and log transformation was applied to the skewed ones. With the transformed data, cross-validation was carried out using multiple machine learning models, and the best model was selected based on the cross-validation score, which was based on `mean_squared_error`. Once the best model was chosen, hyperparameter optimization was performed, and the performance of the selected machine learning model before and after optimization were compared for the extruders with the size ranging from 25 mm to 250 mm. The evaluation of the model's performance revealed significant disparities between the throughputs predicted by machine learning model and the analytical solution, [extrucal](https://github.com/johnwslee/extrucal) for certain extruder sizes. The followings are the summarized report of this study, and actual codes used for this study can be found in [notebook folder](https://github.com/johnwslee/extrucal_machine-learning/tree/main/notebooks).

## 2. Summary of Study

### 2.1. Generation of Data

Extrusion throughput dataset was generated using `extrucal.throughput_cal()` function and the following 7 parameters.

- `extruder_size`: Sizes of extruders ranging from 20mm to 250mm with an increment of 10mm.
- `metering_depth_percent`: Depths of metering section of extrusion screws ranging from 2% to 10% of extruder sizes
- `polymer_density`: Melt density of polymer materials ranging from 800 to 1500 kg/m^3
- `screw_pitch_percent`: Screw pitch ranging from 0.6D to 2D
- `flight_width_percent`: Flight width of screws ranging from 0.06D to 0.2D
- `number_flight`: number of flights with a choice of 1 or 2
- `rpm`: Screw RPMs ranging from 0 to 90

In order to apply randomness to the throughputs in the dataset, +/- 5% variation was applied to the throughputs calculated by `extrucal.throughput_cal()`.

### 2.2. Exploratory Data Analysis

The following graphs show the distribution of features. `metering_depth`, `screw_pitch`, and `flight_width` show skewness to a certain degree.

<img src="https://github.com/johnwslee/extrucal_machine-learning/blob/main/img/distribution_of_features.png" style="width:700px;height:400px;background-color:white">

Log-transformation was applied to the 3 features, and the following are the results.

<img src="https://github.com/johnwslee/extrucal_machine-learning/blob/main/img/distribution_of_log_features.png" style="width:700px;height:200px;background-color:white">

The target, throughput, also showed a strong skewness as shown below. Therefore log-transformation was applied to it.

<img src="https://github.com/johnwslee/extrucal_machine-learning/blob/main/img/distribution_of_target.png" style="width:400px;height:300px;background-color:white">

After log-transformation of the target, the skewness disappeared. However, since there were many zero throughput data for the screw RPM of zero, there was a sharp peak in the graph as shown below.

<img src="https://github.com/johnwslee/extrucal_machine-learning/blob/main/img/distribution_of_log_target.png" style="width:400px;height:300px;background-color:white">

### 2.3. Cross-Validation of Multiple Machine Learning Models

Cross-validation was carried out using 6 different machine learning models: `Ridge`, `Lasso`, `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`, and `CatBoostRegressor`. `mean_squared_error` was used as the metric, and the following table shows the results.

<img src="https://github.com/johnwslee/extrucal_machine-learning/blob/main/img/CV_result_table.png" style="width:700px;height:180px;background-color:white">

`CatBoostRegressor` performed best among the models.

### 2.4. Hyperparameter Optimization

`Optuna` library was used for the hyperparameter optimization of the `CatBoostRegressor` model. The following shows the throughput results predicted by the `CatBoostRegressor` models before/after optimization and the analytical solution (with `extrucal` library).

<img src="https://github.com/johnwslee/extrucal_machine-learning/blob/main/img/opt_comparison.png" style="width:700px;height:1200px;background-color:white">

The prediction was not that good for 25mm extruder for both models before and after hyperparameter optimization.

### 2.5. Comparison of Predictions for Different Extruder Sizes

The `CatBoostRegressor` model was trained with `extruder_size` in the range from 20mm to 250mm with 10mm increment. The previous results showed that the model didn't perform well for 25mm extruder, which was not the size used for training the model. So, it was tested to see if the model would perform any better for the extruder sizes that were included in the train data.

- #### For `extruder_size` in Train Data

<img src="https://github.com/johnwslee/extrucal_machine-learning/blob/main/img/size_comp_in.png" style="width:700px;height:1200px;background-color:white">

- #### For `extruder_size` not in Train Data

<img src="https://github.com/johnwslee/extrucal_machine-learning/blob/main/img/size_comp_not_in.png" style="width:700px;height:1200px;background-color:white">

There are clear disparities between the throughputs predicted by the model and those by the analytical solution(i.e. by `extrucal` library) for the `extruder_size` that were not in the Train Data. The disparity was bigger for the smallest extruder(i.e. 25mm) maybe because its throughputs were order of magnitude smaller than other sizes, and `mean_squared_error` was used as the evaluation metric. On the other hand, the predicted throughputs predicted for the `extruder_size` that were in the Train Data were almost identical to those calculated by the analytical solution(i.e. by `extrucal` library).

When the two cases were compared using `mean_absolute_percentage_error`, it was 1.14% for the `extruder_size` present in Train Data, whereas it was 6.92% for the `extruder_size` that were not in Train Data.

### 2.6. Feature Importances 

In order to find out if the machine learning model correctly learned the effect of each extrusion parameter on the throughput, the feature importances of a machine learning model were investigated by using `shap` library. Just to save the computation time, the optimized `LightGBM` model (whose optimization process is shown in Appendix 2) was used to check the feature importances.

- #### Rank of Features

<img src="https://github.com/johnwslee/extrucal_machine-learning/blob/main/img/feature_rank.png" style="width:700px;height:400px;background-color:white">

Similarly to actual extrusion processes, `rpm` and `extruder_size` were two biggest processing parameter for the model. The rank for the rest of the processing parameters also made sense.

- #### Effect of Each Processing Parameter on Throughput

<img src="https://github.com/johnwslee/extrucal_machine-learning/blob/main/img/feature_effect.png" style="width:700px;height:400px;background-color:white">

The effect of each processing parameter on the throughput was correctly displayed. For example, the throughput increased with increasing `rpm`, `extruder_size`, `metering_depth`, `screw_pitch`, and `polymer_density`, whereas it decreased with increasing `number_flight` and `flight_width`.

## 3. Conclusion

In the beginning, this study started with a simple purpose of just demonstrating that machine learning model can learn very complicated pattern and can perform as well as an analytical solution. However, while I was working on modeling, I found out that the model didn't perform well for the smallest extruder (i.e. 25mm). Initially, I thought that it was due to the fact that the throughputs at zero screw RPM were included in the train data. I also suspected that either the log transformation of the throughput might have affected the performance of the model (because the distribution of throughputs after log transformation looked really weird) or the throughputs of the 25mm extruder were just too small to be considered significant by the model. In the end, it was clear that, since `CatBoostRegresser`, which is a tree-based model, was used, the errors for the `extruder_size` that were not included in the train data were higher than those sizes that were included in the train data. Moreover, the feature importances showed that the trained model correctly learned the effect of each processing parameter in extrusion. For example, the throughput increased with increasing rpm, extruder_size, metering_depth, screw_pitch, and polymer_density, whereas it decreased with increasing number_flight and flight_width.

In conclusion, this study clearly demonstrated that it might be possible to train machine learning models with the datasets generated by an analytical solution. It would be also interesting to apply machine learning to learn the patterns of the dataset that are generated by more sophisticated computational methods, which would be one of my future works.

## How to Run the Notebooks Locally

To download the contents of this GitHub page on to your local machine, follow these steps:

1. Copy and paste the following link: `git clone https://github.com/johnwslee/extrucal_machine-learning.git` to your Terminal.

2. On your terminal, type: `cd extrucal_machine-learning`.

3. Create a virtualenv by typing: `conda env create -f env.yml`

4. Activate the virtualenv by typing: `conda activate extrucal_ml`

5. Run the notebooks in notebook folder in order.