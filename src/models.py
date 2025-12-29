import joblib
from pathlib import Path
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[1]

def fit_and_save_cox(df_cox, model_path=ROOT/"models"/"cox_model.pkl"):
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col='time', event_col='event')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cph, model_path)
    return cph

def fit_and_save_rsf(X, y, model_path=ROOT/"models"/"rsf_model.pkl", **kwargs):
    model = RandomSurvivalForest(**kwargs)
    model.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model
