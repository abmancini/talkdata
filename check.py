import scipy.signal.signaltools

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
scipy.signal.signaltools._centered = _centered
import numpy as np
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sklearn.ensemble import RandomForestRegressor
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split


if __name__ == '__main__':
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)
    regressor = make_pipeline(
        TSFreshFeatureExtractor(show_warnings=False, disable_progressbar=True),
        RandomForestRegressor(),
    )
    forecaster = make_reduction(
        regressor, scitype="time-series-regressor", window_length=12
    )
    forecaster.fit(y_train)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred = forecaster.predict(fh)
    print(y_pred)