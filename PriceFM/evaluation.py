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

def AQCR_metric(y_pred):
    # y_pred: (N, D, Q)
    if y_pred.shape[-1] < 2:
        return 0.0
    crossed = (y_pred[..., :-1] > y_pred[..., 1:])  # (N, D, Q-1)
    return float(crossed.mean())




def AQCE_metric_percent(y_true, y_pred, quantiles, absolute=False):
    """
    Average Quantile Coverage Error (AQCE) in percentage points (%).

    y_true: (N, D)
    y_pred: (N, D, Q) aligned with `quantiles`

    For each symmetric pair (q_low, q_high):
      coverage = mean( y_pred_low <= y_true <= y_pred_high )
      target   = q_high - q_low
      error    = coverage - target

    If absolute=True -> mean(|error|) across symmetric pairs.
    Output is multiplied by 100 (% points).
    """
    qs = np.asarray(quantiles, dtype=np.float64)
    if qs.max() > 1.0:
        qs = qs / 100.0

    # sort quantiles and reorder y_pred accordingly
    order = np.argsort(qs)
    qs = qs[order]
    y_pred = y_pred[..., order]

    Q = len(qs)
    if Q < 2:
        return 0.0

    errs = []
    for i in range(Q // 2):
        q_low = qs[i]
        q_high = qs[-(i + 1)]

        # same symmetry assumption as your AIW_metric
        if not np.isclose(q_low + q_high, 1.0, atol=1e-6):
            raise ValueError(f"Quantiles not symmetric: {q_low}, {q_high}")

        low  = y_pred[..., i]           # (N, D)
        high = y_pred[..., -(i + 1)]    # (N, D)

        covered = (y_true >= low) & (y_true <= high)
        coverage = float(covered.mean())
        target = float(q_high - q_low)

        err = coverage - target
        if absolute:
            err = abs(err)

        errs.append(err)

    return float(np.mean(errs) * 100.0)  # % points





def AIW_metric(y_pred, quantiles):
    """
    Average Interval Width (AIW) over all symmetric quantile pairs.

    y_pred: (N, D, Q)
    quantiles: list like [0.1, 0.25, 0.5, 0.75, 0.9]
    """
    qs = np.asarray(quantiles, dtype=np.float64)
    if qs.max() > 1.0:
        qs = qs / 100.0

    Q = len(qs)
    assert Q >= 2, "Need at least two quantiles for AIW"

    widths = []

    for i in range(Q // 2):
        q_low = qs[i]
        q_high = qs[-(i + 1)]

        # sanity: ensure symmetry
        if not np.isclose(q_low + q_high, 1.0, atol=1e-6):
            raise ValueError(f"Quantiles not symmetric: {q_low}, {q_high}")

        w = y_pred[..., -(i + 1)] - y_pred[..., i]  # (N, D)
        widths.append(np.mean(w))

    return float(np.mean(widths))


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


def r2(y_true, y_pred, quantiles):
    mid = _median_idx(quantiles)
    yhat = y_pred[..., mid]
    ss_res = np.sum((y_true - yhat) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    return float(1.0 - (ss_res / (ss_tot + 1e-12)))


def evaluation(y_pred_scaled, y_true_scaled, quantiles, y_scaler):
    # inverse scale first
    y_true = inverse_scale_y_true(y_true_scaled, y_scaler)        # (N, D)
    y_pred = inverse_scale_y_pred(y_pred_scaled, y_scaler)        # (N, D, Q)

    return {
        "AQL":  AQL_metric(y_true, y_pred, quantiles),
        "AQCR": AQCR_metric(y_pred),
        "AIW":  AIW_metric(y_pred, quantiles),
        "AQCE": AQCE_metric_percent(y_true, y_pred, quantiles, absolute=True),   # absolute
        "RMSE": rmse(y_true, y_pred, quantiles),
        "MAE":  mae(y_true, y_pred, quantiles),
        "R2":   r2(y_true, y_pred, quantiles),
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


def load_corresponding_date_data(
    df,
    forecast_date,
    forecast_target_country,
    input_countries,
    lag_features,
    lead_features,
    label_column,
    lag_window,
    lead_window,
):
    """
    Load one forecast anchor from the raw dataframe and return the exact lag/lead
    slices expected by the model pipeline.
    """
    import pandas as pd

    forecast_anchor = pd.Timestamp(forecast_date)
    if forecast_anchor.tzinfo is None:
        forecast_anchor = forecast_anchor.tz_localize("UTC")
    else:
        forecast_anchor = forecast_anchor.tz_convert("UTC")

    if forecast_anchor.hour != 0 or forecast_anchor.minute != 0:
        raise ValueError("forecast_date must be a 00:00 UTC timestamp.")

    freq = df.index.to_series().diff().dropna().mode().iloc[0]
    lag_index = pd.date_range(end=forecast_anchor - freq, periods=lag_window, freq=freq)
    lead_index = pd.date_range(start=forecast_anchor, periods=lead_window, freq=freq)

    missing_index = lag_index.difference(df.index).union(lead_index.difference(df.index))
    if len(missing_index) > 0:
        raise ValueError(
            f"Missing rows for forecast_date {forecast_anchor}. "
            f"First missing timestamps: {list(missing_index[:5])}"
        )

    loaded_data = {
        "forecast_anchor": forecast_anchor,
        "forecast_frequency": freq,
        "lag_index": lag_index,
        "lead_index": lead_index,
        "lag_window_raw": {},
        "lead_window_raw": {},
        "target_window_raw": df.loc[lead_index, [f"{forecast_target_country}-{label_column}"]].copy(),
    }

    for country in input_countries:
        lag_cols = [
            f"{country}-{label_column}" if feature == label_column else f"{country}-{feature}"
            for feature in lag_features
        ]
        lead_cols = [f"{country}-{feature}" for feature in lead_features]

        loaded_data["lag_window_raw"][country] = df.loc[lag_index, lag_cols].copy()
        loaded_data["lead_window_raw"][country] = df.loc[lead_index, lead_cols].copy()

    return loaded_data


def normalize_and_forecast(
    loaded_data,
    phase2_model_path,
    forecast_target_country,
    forecast_graph_degree,
    adjacency_dict,
    input_countries,
    features,
    lag_features,
    lead_features,
    label_column,
    x_scalers,
    y_scalers,
    quantiles,
):
    """
    Normalize one prepared forecast sample, run the saved model, and return both
    scaled and original-scale outputs.
    """
    import pandas as pd
    from .model import load_model

    active_countries = get_k_hop_countries(
        adjacency_dict,
        input_countries,
        forecast_target_country,
        forecast_graph_degree,
    )
    gate_vector = np.array(
        [[1.0 if country in active_countries else 0.0 for country in input_countries]],
        dtype="float32",
    )

    x_lag_all = []
    x_lead_all = []
    for country in input_countries:
        x_cols = [f"{country}-{feature}" for feature in features]
        y_col = f"{country}-{label_column}"

        lag_raw = loaded_data["lag_window_raw"][country]
        lead_raw = loaded_data["lead_window_raw"][country]

        lag_x_scaled = x_scalers[country].transform(lag_raw[x_cols])
        lead_x_scaled = x_scalers[country].transform(lead_raw[x_cols])
        lag_y_scaled = y_scalers[country].transform(lag_raw[[y_col]])

        lag_scaled = np.concatenate([lag_x_scaled, lag_y_scaled], axis=1)

        lag_col_order = [
            f"{country}-{label_column}" if feature == label_column else f"{country}-{feature}"
            for feature in lag_features
        ]
        lead_col_order = [f"{country}-{feature}" for feature in lead_features]
        lag_lookup = {
            col: idx for idx, col in enumerate(x_cols + [y_col])
        }
        lead_lookup = {col: idx for idx, col in enumerate(x_cols)}

        x_lag_all.append(
            np.stack([lag_scaled[:, lag_lookup[col]] for col in lag_col_order], axis=1).astype("float32")
        )
        x_lead_all.append(
            np.stack([lead_x_scaled[:, lead_lookup[col]] for col in lead_col_order], axis=1).astype("float32")
        )

    x_lag_all = np.stack(x_lag_all, axis=0)[None, ...]
    x_lead_all = np.stack(x_lead_all, axis=0)[None, ...]

    y_col = f"{forecast_target_country}-{label_column}"
    y_true_raw = loaded_data["target_window_raw"][y_col].to_numpy(dtype="float32")
    y_true_scaled = y_scalers[forecast_target_country].transform(
        loaded_data["target_window_raw"][[y_col]]
    ).reshape(1, -1)

    model = load_model(phase2_model_path)
    pred_scaled = model.predict(
        {"X_lag_all": x_lag_all, "X_lead_all": x_lead_all, "graph_gate": gate_vector},
        verbose=0,
    )
    pred_unscaled = inverse_scale_y_pred(pred_scaled, y_scalers[forecast_target_country])[0]

    qs = np.asarray(quantiles, dtype=float)
    if qs.max() > 1.0:
        qs = qs / 100.0

    forecast_df_data = {
        "forecast_time_utc": loaded_data["lead_index"],
        "true_price": y_true_raw,
    }
    for idx, q in enumerate(qs):
        forecast_df_data[f"q{int(round(q * 100)):02d}"] = pred_unscaled[:, idx]

    return {
        "model_path": phase2_model_path,
        "forecast_target_country": forecast_target_country,
        "forecast_graph_degree": forecast_graph_degree,
        "forecast_anchor": loaded_data["forecast_anchor"],
        "active_countries": active_countries,
        "quantiles": qs,
        "pred_scaled": pred_scaled,
        "y_true_scaled": y_true_scaled,
        "pred_unscaled": pred_unscaled,
        "y_true_unscaled": y_true_raw,
        "forecast_df": pd.DataFrame(forecast_df_data),
    }


def produce_testing_metrics(forecast_result, y_scalers):
    """
    Compute evaluation metrics for a single prepared forecast result.
    """
    import pandas as pd

    metrics = evaluation(
        forecast_result["pred_scaled"],
        forecast_result["y_true_scaled"],
        forecast_result["quantiles"],
        y_scalers[forecast_result["forecast_target_country"]],
    )
    return pd.DataFrame([metrics])


def visualize_forecast(forecast_result, figsize=(14, 6)):
    """
    Plot the median forecast, true series, and all symmetric prediction
    intervals for a single forecast result.
    """
    import matplotlib.pyplot as plt

    quantiles_sorted_order = np.argsort(forecast_result["quantiles"])
    quantiles_sorted = forecast_result["quantiles"][quantiles_sorted_order]
    pred_sorted = forecast_result["pred_unscaled"][:, quantiles_sorted_order]
    median_idx = int(np.argmin(np.abs(quantiles_sorted - 0.5)))
    forecast_df = forecast_result["forecast_df"]

    fig, ax = plt.subplots(figsize=figsize)
    alphas = np.linspace(0.10, 0.28, len(quantiles_sorted) // 2)
    for i in range(len(quantiles_sorted) // 2):
        low_q = quantiles_sorted[i]
        high_q = quantiles_sorted[-(i + 1)]
        ax.fill_between(
            forecast_df["forecast_time_utc"],
            pred_sorted[:, i],
            pred_sorted[:, -(i + 1)],
            color="tab:blue",
            alpha=float(alphas[i]),
            label=(
                f"{int(round((high_q - low_q) * 100))}% interval "
                f"(q{int(round(low_q * 100)):02d}-q{int(round(high_q * 100)):02d})"
            ),
        )

    ax.plot(
        forecast_df["forecast_time_utc"],
        forecast_result["y_true_unscaled"],
        color="black",
        linewidth=1.5,
        label="True price",
    )
    ax.plot(
        forecast_df["forecast_time_utc"],
        pred_sorted[:, median_idx],
        color="tab:blue",
        linewidth=1.5,
        label="Median forecast",
    )
    ax.set_title(
        f"{forecast_result['forecast_target_country']} forecast vs true price from "
        f"{forecast_result['forecast_anchor']:%Y-%m-%d %H:%M UTC}"
    )
    ax.set_xlabel("Forecast time (UTC)")
    ax.set_ylabel("Price")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    return fig, ax
