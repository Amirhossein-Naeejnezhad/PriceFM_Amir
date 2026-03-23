import numpy as np
from .data import *

def inverse_scale_y_true(y_true_scaled, y_scaler):
    # y_true_scaled: (N, y_dim)
    N, D = y_true_scaled.shape
    y_inv = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(N, D)
    return y_inv

def inverse_scale_y_pred(y_pred_scaled, y_scaler):
    # y_pred_scaled: (N, y_dim, Q)
    N, D, Q = y_pred_scaled.shape
    y_inv = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(N, D, Q)
    return y_inv

def AQL_metric(y_true, y_pred, quantiles):
    # y_true: (N, D)
    # y_pred: (N, D, Q)
    qs = np.asarray(quantiles, dtype=np.float64)
    if qs.max() > 1.0:
        qs = qs / 100.0

    yt = y_true[..., None]                     # (N, D, 1)
    e  = yt - y_pred                           # (N, D, Q)
    qs_ = qs.reshape(1, 1, -1)                 # (1, 1, Q)
    loss = np.maximum(qs_ * e, (qs_ - 1.0) * e)
    return float(loss.mean())


def _median_idx(quantiles):
    qs = np.asarray(quantiles, dtype=np.float64)
    if qs.max() > 1.0:
        qs = qs / 100.0
    return int(np.argmin(np.abs(qs - 0.5)))


def rmse(y_true, y_pred, quantiles):
    mid = _median_idx(quantiles)
    yhat = y_pred[..., mid]  # (N, D)
    return float(np.sqrt(np.mean((y_true - yhat) ** 2)))


def mae(y_true, y_pred, quantiles):
    mid = _median_idx(quantiles)
    yhat = y_pred[..., mid]
    return float(np.mean(np.abs(y_true - yhat)))




def evaluation(y_pred_scaled, y_true_scaled, quantiles, y_scaler):
    # inverse scale first
    y_true = inverse_scale_y_true(y_true_scaled, y_scaler)        # (N, D)
    y_pred = inverse_scale_y_pred(y_pred_scaled, y_scaler)        # (N, D, Q)

    return {
        "AQL":  AQL_metric(y_true, y_pred, quantiles),
        "RMSE": rmse(y_true, y_pred, quantiles),
        "MAE":  mae(y_true, y_pred, quantiles),
    }


def evaluate_countries(model, split, input_countries, output_countries, gate_fn, quantiles, y_scalers, batch_size=256):
    results = {}
    for c in output_countries:
        Xlag, Xlead, G, Y = pack_dataset(split, input_countries, targets=[c], gate_fn=gate_fn)

        pred = model.predict(
            {"X_lag_all": Xlag, "X_lead_all": Xlead, "graph_gate": G},
            batch_size=batch_size,
            verbose=0,
        )
        results[c] = evaluation(pred, Y, quantiles, y_scalers[c])


        print(f"\n=== Evaluation for {c} ===")
        for k, v in results[c].items():
            print(f"{k:5s}: {v:.6f}")

            
    # EU-level average (mean over countries)
    # ---------------------------
    metric_keys = list(next(iter(results.values())).keys())  # keys from first country
    eu_avg = {k: float(np.mean([results[c][k] for c in output_countries])) for k in metric_keys}

    print("\n=== EU-level Average (mean over countries) ===")
    for k, v in eu_avg.items():
        print(f"{k:20s}: {v:.6f}")

    return results

