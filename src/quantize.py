import joblib
import numpy as np

def quantize_model():
    model = joblib.load("src/model.joblib")
    coef = model.coef_
    intercept = model.intercept_

    joblib.dump((coef, intercept), "src/unquant_params.joblib")

    coef_q = np.uint8((coef - coef.min()) / (coef.max() - coef.min()) * 255)
    intercept_q = np.uint8((intercept - coef.min()) / (coef.max() - coef.min()) * 255)

    joblib.dump((coef_q, intercept_q), "src/quant_params.joblib")

if __name__ == "__main__":
    quantize_model()
