from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from extrucal.extrusion import throughput_cal


def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def mape(true, pred):
    """
    Returns mean absolute percentage error
    """
    mape = 100.0 * np.mean(np.abs((pred - true) / (true + 0.1)))  # 0.1 for avoid ZeroDivisionError
    return mape


def generate_dataframe_for_evaluation():
    """
    Returns dataframe for evaluation of the ML models.
    The columns of the dataframe are "extruder_size", "metering_depth", "polymer_density",
    "rpm", "screw_pitch", "flight_width", "number_flight", and the throughput calculated by
    "extrucal"

    Returns
    ----------
        pandas Dataframe for evaluation
    """
    extruder_size = []
    for i in range(25, 251, 25):
        extruder_size.extend([i] * 10)

    metering_depth_percent = [0.05] * 100
    polymer_density = [1000] * 100
    screw_pitch_percent = [1] * 100
    flight_width_percent = [0.1] * 100
    number_flight = [1] * 100
    rpm = [
        r for r in range(0, 92, 10)
    ] * 10

    df = pd.DataFrame(
        {
            "extruder_size": extruder_size,
            "metering_depth_percent": metering_depth_percent,
            "polymer_density": polymer_density,
            "screw_pitch_percent": screw_pitch_percent,
            "flight_width_percent": flight_width_percent,
            "number_flight": number_flight,
            "rpm": rpm,
        }
    )

    df["metering_depth"] = df["extruder_size"] * df["metering_depth_percent"]
    df["screw_pitch"] = df["extruder_size"] * df["screw_pitch_percent"]
    df["flight_width"] = df["extruder_size"] * df["flight_width_percent"]

    df["extrucal"] = df.apply(
        lambda row: throughput_cal(
            row["extruder_size"],
            row["metering_depth"],
            row["polymer_density"],
            row["rpm"],
            row["screw_pitch"],
            row["flight_width"],
            int(row["number_flight"]),
        ),
        axis=1,
    )

    new_col_order = [
        "extruder_size",
        "metering_depth",
        "polymer_density",
        "rpm",
        "screw_pitch",
        "flight_width",
        "number_flight",
        "extrucal",
    ]

    df = df[new_col_order]

    return df
