"""The curve-distance metrics, as `Metric` instances.

`WASSERSTEIN` is the default. `DTW` is built lazily so importing this module
does not pull in the DTW backend unless the metric is actually used.
"""
from aemo_curves.core.metric import Metric
from aemo_curves.stages.evaluate.hc_mae import curve_hc_mae_distance
from aemo_curves.stages.evaluate.wasserstein import curve_wasserstein_distance

WASSERSTEIN = Metric("wasserstein", curve_wasserstein_distance)
HC_MAE = Metric("hc_mae", curve_hc_mae_distance)


def _build_dtw() -> Metric:
    from aemo_curves.stages.evaluate.dtw import curve_dtw_distance

    return Metric("dtw", curve_dtw_distance)


def __getattr__(name):
    if name == "DTW":
        return _build_dtw()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
