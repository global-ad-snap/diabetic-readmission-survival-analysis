import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored, brier_score

def c_index_cox(cox_model):
    """Return concordance index from a fitted CoxPH model."""
    return cox_model.concordance_index_

def rsf_c_index(rsf, X, y):
    """Compute RSF concordance index using cumulative hazard risk."""
    try:
        chf = rsf.predict_cumulative_hazard_function(X, return_array=True)
        risk = np.asarray(chf).sum(axis=1)
        return concordance_index_censored(y['event'], y['time'], risk)[0]
    except Exception:
        # fallback to predict()
        rsf_pred = rsf.predict(X)
        if hasattr(rsf_pred, "shape") and rsf_pred.ndim > 1:
            rsf_pred = rsf_pred[:, 0]
        return concordance_index_censored(y['event'], y['time'], rsf_pred)[0]

def brier_score_at_time(model, df_cox, eval_time):
    """Compute Brier score for Cox model at a given evaluation time."""
    y_cox = np.array(list(zip(df_cox["event"].astype(bool), df_cox["time"].astype(float))),
                     dtype=[('event', bool), ('time', float)])
    surv_df = model.predict_survival_function(df_cox)
    surv_array = surv_df.values
    time_index = np.abs(surv_df.index.values - eval_time).argmin()
    pred_surv = surv_array[time_index, :].reshape(-1, 1)
    times = np.array([eval_time])
    return brier_score(y_cox, y_cox, pred_surv, times)[1][0]