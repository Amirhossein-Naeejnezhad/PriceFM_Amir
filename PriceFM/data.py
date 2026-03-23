import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import RobustScaler

def graph_adj_matrix():
    adjacency_dict = {
        'AT': ['AT', 'CZ', 'DE_LU', 'HU', 'IT_NORD', 'SI'],
        'BE': ['BE', 'DE_LU', 'FR', 'NL'],
        'BG': ['BG', 'GR', 'RO'],
        'CZ': ['AT', 'CZ', 'DE_LU', 'PL', 'SK'],
        'DE_LU': ['AT', 'BE', 'CZ', 'DK_1', 'DK_2', 'DE_LU', 'FR', 'NL', 'NO_2', 'PL', 'SE_4'],
        'DK_1': ['DE_LU', 'DK_1', 'DK_2', 'NL', 'NO_2', 'SE_3'],
        'DK_2': ['DE_LU', 'DK_1', 'DK_2', 'SE_4'],
        'EE': ['EE', 'FI', 'LV'],
        'ES': ['ES', 'FR', 'PT'],
        'FI': ['EE', 'FI', 'NO_4', 'SE_1', 'SE_3'],
        'FR': ['BE', 'DE_LU', 'ES', 'FR', 'IT_NORD'],
        'GR': ['BG', 'GR', 'IT_SUD'],
        'HR': ['HR', 'HU', 'SI'],
        'HU': ['AT', 'HR', 'HU', 'RO', 'SI', 'SK'],
        'IT_CALA': ['IT_CALA', 'IT_SICI', 'IT_SUD'],
        'IT_CNOR': ['IT_CNOR', 'IT_CSUD', 'IT_NORD'],
        'IT_CSUD': ['IT_CNOR', 'IT_CSUD', 'IT_SARD', 'IT_SUD'],
        'IT_NORD': ['AT', 'FR', 'IT_CNOR', 'IT_NORD', 'SI'],
        'IT_SARD': ['IT_CSUD', 'IT_SARD'],
        'IT_SICI': ['IT_CALA', 'IT_SICI'],
        'IT_SUD': ['GR', 'IT_CALA', 'IT_CSUD', 'IT_SUD'],
        'LT': ['LT', 'LV', 'PL', 'SE_4'],
        'LV': ['EE', 'LT', 'LV'],
        'NL': ['BE', 'DK_1', 'DE_LU', 'NL', 'NO_2'],
        'NO_1': ['NO_1', 'NO_2', 'NO_3', 'NO_5', 'SE_3'],
        'NO_2': ['DE_LU', 'DK_1', 'NL', 'NO_1', 'NO_2', 'NO_5'],
        'NO_3': ['NO_1', 'NO_3', 'NO_4', 'NO_5', 'SE_2'],
        'NO_4': ['FI', 'NO_3', 'NO_4', 'SE_1', 'SE_2'],
        'NO_5': ['NO_1', 'NO_2', 'NO_3', 'NO_5'],
        'PL': ['CZ', 'DE_LU', 'LT', 'PL', 'SE_4', 'SK'],
        'PT': ['ES', 'PT'],
        'RO': ['BG', 'HU', 'RO'],
        'SE_1': ['FI', 'NO_4', 'SE_1', 'SE_2'],
        'SE_2': ['NO_3', 'NO_4', 'SE_1', 'SE_2', 'SE_3'],
        'SE_3': ['DK_1', 'FI', 'NO_1', 'SE_2', 'SE_3', 'SE_4'],
        'SE_4': ['DE_LU', 'DK_2', 'LT', 'PL', 'SE_3', 'SE_4'],
        'SI': ['AT', 'HR', 'HU', 'IT_NORD', 'SI'],
        'SK': ['CZ', 'HU', 'PL', 'SK'],
    }

    return adjacency_dict

def create_raw_dataframe(freq, seed=42, noise_std=5.0):
    rng = np.random.default_rng(seed)

    # timestamps
    idx = pd.date_range(
        "2025-01-01",
        "2025-07-01",
        freq=freq,
        inclusive="left"
    )
    df = pd.DataFrame(index=idx)

    countries = ["AT", "DE", "NL"]

    features = {
        "AT": ["Feature1", "Feature2", "Feature3"],
        "DE": ["Feature1", "Feature2", "Feature3"],
        "NL": ["Feature1", "Feature2", "Feature3"]
    }

    # ------------------------------------------------------------------
    # 1. Generate base latent drivers (shared temporal structure)
    # ------------------------------------------------------------------
    t = np.arange(len(df))

    common_trend = 0.05 * t
    daily_season = 10 * np.sin(2 * np.pi * t / 24)
    weekly_season = 5 * np.sin(2 * np.pi * t / (24 * 7))

    latent = common_trend + daily_season + weekly_season

    # ------------------------------------------------------------------
    # 2. Generate features as noisy transforms of latent signal
    # ------------------------------------------------------------------
    for c in countries:
        for i, f in enumerate(features[c]):
            df[f"{c}-{f}"] = (
                100
                + (i + 1) * latent
                + rng.normal(0, 20, len(df))
            )

    # ------------------------------------------------------------------
    # 3. Generate labels as KNOWN functions of features
    # ------------------------------------------------------------------
    for c in countries:
        f1 = df[f"{c}-Feature1"]
        f2 = df[f"{c}-Feature2"]
        f3 = df[f"{c}-Feature3"]

        df[f"{c}-Label"] = (
            0.6 * f1
            + 0.3 * f2
            - 0.2 * f3
            + 10 * np.tanh(f1 / 100)
            + rng.normal(0, noise_std, len(df))
        )

    return df



# split into train, val, test
def split_dataframe(
    df,
    train_start, train_end,
    val_start,   val_end,
    test_start,  test_end
):

    df_train = df.loc[train_start : train_end]
    df_val   = df.loc[val_start   : val_end]
    df_test  = df.loc[test_start  : test_end]

    return df_train, df_val, df_test


def read_dataset(file_path, time_col="time_utc"):
    df = pd.read_csv(file_path)

    # parse UTC
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # keep UTC tz-aware index
    df = df.set_index(time_col).sort_index()
    return df


def split_dataframe(df, train_start, train_end, val_start, val_end, test_start, test_end):
    # force bounds into UTC timestamps (works if input is string or Timestamp)
    train_start = pd.to_datetime(train_start, utc=True)
    train_end   = pd.to_datetime(train_end,   utc=True)
    val_start   = pd.to_datetime(val_start,   utc=True)
    val_end     = pd.to_datetime(val_end,     utc=True)
    test_start  = pd.to_datetime(test_start,  utc=True)
    test_end    = pd.to_datetime(test_end,    utc=True)

    df_train = df.loc[train_start:train_end]
    df_val   = df.loc[val_start:val_end]
    df_test  = df.loc[test_start:test_end]
    return df_train, df_val, df_test


# scale data
def scale_dataframe(df_train, df_val, df_test):
    """
    Scale X and Y separately using RobustScaler.
    
    Inputs:
        df_train, df_val, df_test: DataFrames
        x_cols: list of feature columns
        y_cols: list of label columns

    Returns:
        (df_train_scaled, df_val_scaled, df_test_scaled, x_scaler, y_scaler)
    """
    # -----------------------
    # Fit scalers only on TRAIN
    # -----------------------
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    x_cols = [c for c in df_train.columns if not c.endswith("Label")]
    y_cols = [c for c in df_train.columns if c.endswith("Label")]

    x_scaler.fit(df_train[x_cols])   # KEEP feature names
    y_scaler.fit(df_train[y_cols])

    # -----------------------
    # Transform all splits
    # -----------------------
    def transform_df(df):
        df_x = pd.DataFrame(
            x_scaler.transform(df[x_cols]),
            index=df.index,
            columns=x_cols
        )
        df_y = pd.DataFrame(
            y_scaler.transform(df[y_cols]),
            index=df.index,
            columns=y_cols
        )
        # keep exact same column order: X first then Y
        return pd.concat([df_x, df_y], axis=1)

    df_train_s = transform_df(df_train)
    df_val_s   = transform_df(df_val)
    df_test_s  = transform_df(df_test)

    return df_train_s, df_val_s, df_test_s, x_scaler, y_scaler


def scale_dataframe_per_country(df_train, df_val, df_test, countries, features, label_column):
    """
    Scale X (features) and Y (label) separately per country.

    Parameters:
        df_train, df_val, df_test : original dataframes (not separated yet)
        countries : ["AT", "DE", "NL"]
        features : ["Feature1", "Feature2", "Feature3"]

    Returns:
        df_train_s, df_val_s, df_test_s : scaled dataframes (same shape as input)
        x_scalers : dict[country] = RobustScaler for features
        y_scalers : dict[country] = RobustScaler for labels
    """

    # Make deep copies to preserve index & columns
    df_train_s = df_train.copy()
    df_val_s   = df_val.copy()
    df_test_s  = df_test.copy()

    x_scalers = {}
    y_scalers = {}
    #features = features+ [label_column]
    for c in countries:
        # -----------------------------------
        # 1. Identify columns for that country
        # -----------------------------------
        x_cols = [f"{c}-{f}" for f in features if f"{c}-{f}" in df_train.columns]
        y_cols = [f"{c}-{label_column}"]

        # Skip if country has no columns
        if len(x_cols) == 0 or len(y_cols) == 0:
            continue

        # -----------------------------------
        # 2. Fit per-country scalers on TRAIN only
        # -----------------------------------
        x_scaler = RobustScaler()
        y_scaler = RobustScaler()

        x_scaler.fit(df_train[x_cols])
        y_scaler.fit(df_train[y_cols])

        x_scalers[c] = x_scaler
        y_scalers[c] = y_scaler

        # -----------------------------------
        # 3. Transform train/val/test for this country
        # -----------------------------------
        df_train_s[x_cols] = x_scaler.transform(df_train[x_cols])
        df_train_s[y_cols] = y_scaler.transform(df_train[y_cols])

        df_val_s[x_cols]   = x_scaler.transform(df_val[x_cols])
        df_val_s[y_cols]   = y_scaler.transform(df_val[y_cols])

        df_test_s[x_cols]  = x_scaler.transform(df_test[x_cols])
        df_test_s[y_cols]  = y_scaler.transform(df_test[y_cols])

    return df_train_s, df_val_s, df_test_s, x_scalers, y_scalers





def scale_dataframe_eu_level(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    x_contains=(
        "Forecasted Load",
        "day_ahead_Wind Onshore",
        "day_ahead_Wind Offshore",
        "day_ahead_Solar",
    ),
    y_contains=("DA_price",),
):
    # Copies
    df_train_s = df_train.copy()
    df_val_s   = df_val.copy()
    df_test_s  = df_test.copy()

    def _match_cols(df, contains):
        contains = tuple(contains)
        cols = [c for c in df.columns if any(sub in c for sub in contains)]
        # de-duplicate while preserving order (important if duplicates exist)
        seen = set()
        out = []
        for c in cols:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    x_cols = _match_cols(df_train_s, x_contains)
    y_cols = _match_cols(df_train_s, y_contains)

    if not x_cols:
        raise ValueError(f"No X columns matched substrings {x_contains}.")
    if not y_cols:
        raise ValueError(f"No Y columns matched substrings {y_contains}.")

    # Ensure val/test have these columns
    for name, df_ in [("val", df_val_s), ("test", df_test_s)]:
        missing_x = [c for c in x_cols if c not in df_.columns]
        missing_y = [c for c in y_cols if c not in df_.columns]
        if missing_x or missing_y:
            raise ValueError(
                f"{name} split missing required columns selected from train.\n"
                f"Missing X: {missing_x}\n"
                f"Missing Y: {missing_y}"
            )

    # Fit EU-level scalers using vertical stacking
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    X_train_stack = df_train_s[x_cols].to_numpy(dtype="float64").reshape(-1, 1)
    y_train_stack = df_train_s[y_cols].to_numpy(dtype="float64").reshape(-1, 1)

    x_scaler.fit(X_train_stack)
    y_scaler.fit(y_train_stack)

    # --- transform helper: returns float64 DataFrame with same cols/index ---
    def _transform_df(df_in, cols, scaler):
        arr = df_in[cols].to_numpy(dtype="float64")
        arr_scaled = scaler.transform(arr.reshape(-1, 1)).reshape(arr.shape)
        return pd.DataFrame(arr_scaled, index=df_in.index, columns=cols, dtype="float64")

    X_train_scaled = _transform_df(df_train_s, x_cols, x_scaler)
    X_val_scaled   = _transform_df(df_val_s,   x_cols, x_scaler)
    X_test_scaled  = _transform_df(df_test_s,  x_cols, x_scaler)

    y_train_scaled = _transform_df(df_train_s, y_cols, y_scaler)
    y_val_scaled   = _transform_df(df_val_s,   y_cols, y_scaler)
    y_test_scaled  = _transform_df(df_test_s,  y_cols, y_scaler)

    # --- assign per-column to guarantee dtype conversion ---
    for c in x_cols:
        df_train_s[c] = X_train_scaled[c].astype("float64")
        df_val_s[c]   = X_val_scaled[c].astype("float64")
        df_test_s[c]  = X_test_scaled[c].astype("float64")

    for c in y_cols:
        df_train_s[c] = y_train_scaled[c].astype("float64")
        df_val_s[c]   = y_val_scaled[c].astype("float64")
        df_test_s[c]  = y_test_scaled[c].astype("float64")

    return df_train_s, df_val_s, df_test_s, x_scaler, y_scaler



# 4. separate Europe df into individual country df
def separate_countries(df, countries, features, label_column):
    """
    Return dict[country] = df_country
    df_country contains:
       {f"{c}-{requested_feature}"} ∩ existing columns
       + {f"{c}-Label"}
    """
    separated = {}
    for c in countries:
        cols = [f"{c}-{f}" for f in features if f"{c}-{f}" in df.columns]
        cols.append(f"{c}-{label_column}")
        separated[c] = df[cols].copy()
    return separated


# 5. rolling window, lag, lead, label
def make_rolling_window_samples(
    df_country,
    country,
    lag_features,
    lead_features,
    label_column,
    M_steps,    # number of rows for lag window
    N_steps     # number of rows for lead window
):
    """
    Resolution-agnostic rolling window.
    df_country: indexed by datetime (any freq: 15min, 30min, 1h...)

    Returns:
      X_lag:   (num_samples, M_steps, num_lag_features)
      X_lead:  (num_samples, N_steps, num_lead_features)
      Y:       (num_samples, N_steps)
      timestamps: list of anchor timestamps t (only 00:00)
    """

    # --------------------------------------
    # 1. infer usable columns
    # --------------------------------------
    lag_cols = [
        f"{country}-{f}" for f in lag_features
        if f"{country}-{f}" in df_country.columns
    ]

    lead_cols = [
        f"{country}-{f}" for f in lead_features
        if f"{country}-{f}" in df_country.columns
    ]

    label_col = f"{country}-{label_column}"

    # --------------------------------------
    # 2. anchors at 00:00 (same logic)
    # --------------------------------------
    anchors = df_country.index[
        (df_country.index.hour == 0) &
        (df_country.index.minute == 0) 
    ]

    X_lag_list  = []
    X_lead_list = []
    Y_list      = []
    anchor_list = []

    # convert index to a list for integer slicing
    idx = df_country.index
    values = df_country

    # --------------------------------------
    # 3. main rolling loop (row-based slicing)
    # --------------------------------------
    for t in anchors:
        # find integer index of anchor
        try:
            anchor_pos = idx.get_loc(t)
        except KeyError:
            continue

        # positions
        lag_start_pos  = anchor_pos - M_steps
        lead_end_pos   = anchor_pos + N_steps

        if lag_start_pos < 0:
            continue
        if lead_end_pos >= len(idx):
            continue

        # row-based slices (this works for ANY resolution)
        lag_window  = values.iloc[lag_start_pos:anchor_pos][lag_cols]
        lead_window = values.iloc[anchor_pos:lead_end_pos][lead_cols]
        label_window= values.iloc[anchor_pos:lead_end_pos][label_col]

        if len(lag_window) != M_steps or len(lead_window) != N_steps:
            continue

        X_lag_list.append(lag_window.values)
        X_lead_list.append(lead_window.values)
        Y_list.append(label_window.values)
        anchor_list.append(t)

    return (
        np.array(X_lag_list, dtype="float32"),
        np.array(X_lead_list, dtype="float32"),
        np.array(Y_list, dtype="float32"),
        anchor_list,
    )


def graph_mask(
    target_country: str,
    countries: list[str],
    graph_distance: int,
    adjacency_obj,   # can be dict OR (N,) object array/list of dicts
):
    """
    Returns a 0/1 mask aligned with `countries`.

    - If adjacency_obj is a dict: returns shape (C,)
    - If adjacency_obj is a per-sample container (len N): returns shape (N, C)
    """
    # pick the actual dict (works for dict, list-of-dicts, np object array)
    adj = adjacency_obj if isinstance(adjacency_obj, dict) else adjacency_obj[0]

    # BFS up to depth = graph_distance
    visited = {target_country: 0}
    q = deque([target_country])

    while q:
        u = q.popleft()
        du = visited[u]
        if du == graph_distance:
            continue
        for v in adj.get(u, []):
            if v not in visited:
                visited[v] = du + 1
                q.append(v)

    reachable = set(visited.keys())
    mask_1d = np.array([1 if c in reachable else 0 for c in countries], dtype=np.int8)  # (C,)

    # if per-sample adjacency provided, expand to (N, C)
    if not isinstance(adjacency_obj, dict):
        N = len(adjacency_obj)
        return np.repeat(mask_1d[None, :], repeats=N, axis=0)

    return mask_1d





def add_adj_dict_to_rollings(
    adjacency_dict,
    rolling_train,
    rolling_val,
    rolling_test,
    countries,
    key="graph_adjacency",
):
    if isinstance(adjacency_dict, dict) and adjacency_dict and all(
        isinstance(v, dict) for v in adjacency_dict.values()
    ):
        adjacency_map = adjacency_dict
    else:
        adjacency_map = {c: adjacency_dict for c in countries}

    for c in countries:
        if c not in adjacency_map:
            raise KeyError(f"Missing adjacency for country {c}.")
        adj = adjacency_map[c]

        num_train = rolling_train[c]["Y"].shape[0]
        num_val   = rolling_val[c]["Y"].shape[0]
        num_test  = rolling_test[c]["Y"].shape[0]

        rolling_train[c][key] = np.array([adj] * num_train, dtype=object)
        rolling_val[c][key]   = np.array([adj] * num_val,   dtype=object)
        rolling_test[c][key]  = np.array([adj] * num_test,  dtype=object)

    return rolling_train, rolling_val, rolling_test


def make_gate_vec(input_countries, active_countries):
    active = set(active_countries)
    return np.array([1.0 if c in active else 0.0 for c in input_countries], dtype="float32")


def pack_dataset(split, input_countries, targets, gate_fn):
    N = split[targets[0]]["Y"].shape[0]

    X1_all = np.stack([split[c]["X_lag"] for c in input_countries], axis=1)  # (N, n_in, 24, 3)
    X2_all = np.stack([split[c]["X_lead"] for c in input_countries], axis=1)  # (N, n_in, 48, 2)

    X1s, X2s, Gs, Ys = [], [], [], []
    for t in targets:
        gate = np.repeat(make_gate_vec(input_countries, gate_fn(t))[None, :], repeats=N, axis=0)  # (N, n_in)
        y    = split[t]["Y"]  # (N, y_dim)

        X1s.append(X1_all); X2s.append(X2_all); Gs.append(gate); Ys.append(y)

    return (np.concatenate(X1s, 0), np.concatenate(X2s, 0), np.concatenate(Gs, 0), np.concatenate(Ys, 0))



def get_k_hop_countries(adjacency_dict, input_countries, target_country, degree):
    """
    Return k-hop neighbor set for target_country, restricted to input_countries.

    degree = 0 -> [target]
    degree = 1 -> [target + 1-hop neighbors in input_countries]
    degree = 2 -> [target + 2-hop neighbors in input_countries]
    ...

    Output order:
      - starts with target_country
      - then others in the order of input_countries
    """
    if degree < 0:
        raise ValueError("degree must be >= 0")

    allowed = set(input_countries)

    if target_country not in allowed:
        raise ValueError(f"target_country '{target_country}' must be in input_countries")

    # BFS over the adjacency graph, but only keep nodes inside `allowed`
    visited = set([target_country])
    q = deque([(target_country, 0)])

    while q:
        node, dist = q.popleft()
        if dist == degree:
            continue

        for nb in adjacency_dict.get(node, []):
            if nb in allowed and nb not in visited:
                visited.add(nb)
                q.append((nb, dist + 1))

    # Stable output: target first, then follow input_countries order
    out = [target_country] + [c for c in input_countries if c in visited and c != target_country]
    return out



def build_degree_getters(adjacency_dict, input_countries):
    allowed = set(input_countries)

    def _bfs_dist(src):
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adjacency_dict.get(u, []):
                if v in allowed and v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    # precompute "max degree needed" (eccentricity) for every target
    max_deg = {}
    for tgt in input_countries:
        dist = _bfs_dist(tgt)
        if len(dist) != len(allowed):
            missing = [c for c in input_countries if c not in dist]
            raise ValueError(f"Graph disconnected for target {tgt}; missing {missing}")
        max_deg[tgt] = max(dist.values())

    def get_max_degree(target_country: str) -> int:
        return max_deg[target_country]

    def get_graph_degrees(target_country: str):
        return list(range(max_deg[target_country] + 1))

    return get_max_degree, get_graph_degrees