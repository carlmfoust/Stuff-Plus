"""Shared pipeline for MLB Stuff+ primary pitch-type models.

Port of the logic shared across primary-models/fastball-models.Rmd,
primary-models/breakingball-models.Rmd, and primary-models/offspeed-models.Rmd.

The three driver scripts (``fastball_models.py``, ``breakingball_models.py``,
``offspeed_models.py``) call into this module with different ``PitchGroup``
filters and feature sets.
"""

from __future__ import annotations

import csv
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 1015

# Description codes, mirroring fastball-models.Rmd:175-178.
SWING_DESCRIPTIONS = {
    "hit_into_play",
    "foul",
    "foul_tip",
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_bunt",
    "missed_bunt",
    "bunt_foul_tip",
}
WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "bunt_foul_tip",
}
FOUL_DESCRIPTIONS = {"foul", "foul_bunt"}

# Feature lists for the secondary VAA/HAA models, from
# secondary-models/approach-angle-model-script.Rmd and arm-angle-model-script.Rmd.
VAA_FEATURES = [
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "a_horz",
    "a_vert",
    "plate_x",
    "plate_z",
]
HAA_FEATURES = [
    "release_pos_x",
    "release_pos_z",
    "a_horz",
    "a_vert",
    "spin_axis",
]

# Columns pulled from Statcast, matching fastball-models.Rmd:55-59.
STATCAST_COLUMNS = [
    "player_name",
    "pitcher",
    "pitch_name",
    "at_bat_number",
    "pitch_number",
    "description",
    "launch_speed_angle",
    "plate_x",
    "plate_z",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "spin_axis",
    "release_pos_x",
    "release_pos_z",
    "pfx_x",
    "pfx_z",
    "p_throws",
    "stand",
    "game_year",
    "game_date",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "arm_angle",
]

# Features used by the primary models, from fastball-models.Rmd:200-203.
# The fastball driver drops the three relative-diff columns.
BASE_MODEL_FEATURES = [
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "a_horz",
    "a_vert",
    "spin_axis",
    "arm_angle",
    "stand",
    "xVAA",
    "xHAA",
]
RELATIVE_FEATURES = ["RelSpeedDiff", "Horz_Accel_Diff", "Vert_Accel_Diff"]


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TRAINING_SQL = (
    "SELECT * FROM statcast_pitching "
    "WHERE game_year >= 2021 AND game_year < 2025 AND game_type = 'R'"
)
TESTING_SQL = (
    "SELECT * FROM statcast_pitching "
    "WHERE game_year >= 2025 AND game_type = 'R'"
)


def load_statcast(
    user: str = "root",
    password: str = "",
    host: str = "localhost",
    database: str = "baseball-research",
    port: int = 3306,
) -> pd.DataFrame:
    """Pull training-era pitch-by-pitch rows from MySQL.

    Mirrors fastball-models.Rmd:37-48. Uses SQLAlchemy + PyMySQL instead of
    RMySQL.
    """
    from sqlalchemy import create_engine

    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(url)
    with engine.connect() as conn:
        return pd.read_sql(TRAINING_SQL, conn)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VAA, HAA, and tangent/horz/vert acceleration projections.

    Port of fastball-models.Rmd:54-114 (up to PitchGroup assignment).
    """
    keep = [c for c in STATCAST_COLUMNS if c in df.columns]
    df = df[keep].copy()

    # drop_na(!launch_speed_angle): drop rows missing any column besides
    # launch_speed_angle (fastball-models.Rmd:60).
    non_lsa = [c for c in df.columns if c != "launch_speed_angle"]
    df = df.dropna(subset=non_lsa).reset_index(drop=True)

    vx0 = df["vx0"].to_numpy()
    vy0 = df["vy0"].to_numpy()
    vz0 = df["vz0"].to_numpy()
    ax = df["ax"].to_numpy()
    ay = df["ay"].to_numpy()
    az = df["az"].to_numpy()

    # Physics block (fastball-models.Rmd:62-89).
    vy_f = -np.sqrt(vy0**2 - (2 * ay * (50 - (17 / 12))))
    t = (vy_f - vy0) / ay
    vz_f = vz0 + az * t
    vx_f = vx0 + ax * t

    vaa = -np.arctan(vz_f / vy_f) * (180 / np.pi)
    haa = -np.arctan(vx_f / vy_f) * (180 / np.pi)

    v_mag = np.sqrt(vx0**2 + vy0**2 + vz0**2)
    tang_x = vx0 / v_mag
    tang_y = vy0 / v_mag
    tang_z = vz0 / v_mag

    # temp vector is (0, 0, 1) in the R code, so horz = temp x tang reduces to:
    #   horz_x = -tang_y, horz_y = tang_x, horz_z = 0 (before normalization).
    horz_x_raw = -tang_y
    horz_y_raw = tang_x
    horz_z_raw = np.zeros_like(tang_x)
    horz_mag = np.sqrt(horz_x_raw**2 + horz_y_raw**2 + horz_z_raw**2)
    horz_x = horz_x_raw / horz_mag
    horz_y = horz_y_raw / horz_mag
    horz_z = horz_z_raw / horz_mag

    vert_x = tang_y * horz_z - tang_z * horz_y
    vert_y = tang_z * horz_x - tang_x * horz_z
    vert_z = tang_x * horz_y - tang_y * horz_x

    a_horz = ax * horz_x + ay * horz_y + az * horz_z
    a_vert = ax * vert_x + ay * vert_y + az * vert_z

    df["VAA"] = vaa
    df["HAA"] = haa
    df["a_horz"] = a_horz
    df["a_vert"] = a_vert

    # Drop the now-unneeded raw velocity/acceleration columns (mirrors
    # select(-c(vx0:az, vy_f:vx_f, v_mag:a_tang)) at line 90).
    df = df.drop(columns=["vx0", "vy0", "vz0", "ax", "ay", "az"])

    # Unit conversions + plate-location normalization (lines 92-99).
    df["pfx_x"] = df["pfx_x"] * 12
    df["pfx_z"] = df["pfx_z"] * 12
    df["stand"] = (df["stand"].astype("category").cat.codes).astype(int)
    df["plate_x_og"] = df["plate_x"]
    df["plate_z_og"] = df["plate_z"]
    df["plate_x"] = 0.0
    df["plate_z"] = 2.5

    # PitchGroup (lines 101-107).
    fastball_names = {"4-Seam Fastball", "Sinker", "Cutter"}
    breakingball_names = {
        "Slider",
        "Sweeper",
        "Curveball",
        "Slurve",
        "Knuckle Curve",
        "Slow Curve",
        "Screwball",
    }
    offspeed_names = {"Changeup", "Split-Finger", "Forkball"}
    other_names = {"Eephus", "Knuckleball", "Other"}

    def _group(name: str) -> str | None:
        if name in fastball_names:
            return "Fastball"
        if name in breakingball_names:
            return "BreakingBall"
        if name in offspeed_names:
            return "Offspeed"
        if name in other_names:
            return "Other"
        return None

    df["PitchGroup"] = df["pitch_name"].map(_group)

    return df


# ---------------------------------------------------------------------------
# Secondary model (VAA / HAA) loading + prediction
# ---------------------------------------------------------------------------


@dataclass
class SecondaryModels:
    vaa: xgb.Booster
    haa: xgb.Booster


def load_secondary_models(vaa_path: str | os.PathLike, haa_path: str | os.PathLike) -> SecondaryModels:
    """Load the pre-trained VAA / HAA XGBoost boosters from .ubj files.

    The user maintains these .ubj files locally. They are not in the repo.
    Pass their paths via the driver CLI flags --vaa-model / --haa-model.
    """
    vaa_p = Path(vaa_path)
    haa_p = Path(haa_path)
    if not vaa_p.exists():
        raise FileNotFoundError(
            f"VAA model not found at {vaa_p}. "
            "Pass --vaa-model /path/to/VAA.ubj to the driver script."
        )
    if not haa_p.exists():
        raise FileNotFoundError(
            f"HAA model not found at {haa_p}. "
            "Pass --haa-model /path/to/HAA.ubj to the driver script."
        )
    vaa_booster = xgb.Booster()
    vaa_booster.load_model(str(vaa_p))
    haa_booster = xgb.Booster()
    haa_booster.load_model(str(haa_p))
    return SecondaryModels(vaa=vaa_booster, haa=haa_booster)


def predict_xvaa_xhaa(df: pd.DataFrame, models: SecondaryModels) -> pd.DataFrame:
    """Add xVAA and xHAA columns using the pre-trained secondary boosters.

    Port of fastball-models.Rmd:109-114.
    """
    vaa_matrix = xgb.DMatrix(df[VAA_FEATURES].to_numpy(), feature_names=VAA_FEATURES)
    haa_matrix = xgb.DMatrix(df[HAA_FEATURES].to_numpy(), feature_names=HAA_FEATURES)
    df = df.copy()
    df["xVAA"] = models.vaa.predict(vaa_matrix)
    df["xHAA"] = models.haa.predict(haa_matrix)
    return df


# ---------------------------------------------------------------------------
# Cutter reclassification + relative-to-fastball features
# ---------------------------------------------------------------------------


def reclassify_cutters(df: pd.DataFrame) -> pd.DataFrame:
    """Reassign cutters to Fastball or BreakingBall per-pitcher.

    Port of fastball-models.Rmd:119-161.
    """
    fb = (
        df[df["pitch_name"].isin(["4-Seam Fastball", "Sinker"])]
        .groupby(["player_name", "pitcher"], as_index=False)
        .agg(
            FBvelo=("release_speed", "mean"),
            FBvbreak=("pfx_z", "mean"),
            FBhbreak=("pfx_x", "mean"),
        )
    )

    cutter = (
        df[df["pitch_name"] == "Cutter"]
        .groupby(["player_name", "pitcher"], as_index=False)
        .agg(CTvelo=("release_speed", "mean"), CTvert=("pfx_z", "mean"))
    )

    merged = cutter.merge(fb, on=["player_name", "pitcher"], how="left")
    merged["VeloRatio"] = merged["CTvelo"] / merged["FBvelo"]

    def _new_group(row: pd.Series) -> str:
        fb_velo = row["FBvelo"]
        ct_vert = row["CTvert"]
        ratio = row["VeloRatio"]
        if pd.isna(fb_velo) and ct_vert > 5:
            return "Fastball"
        if pd.isna(fb_velo):
            return "BreakingBall"
        if ratio >= 0.95:
            return "Fastball"
        if ct_vert <= 5:
            return "BreakingBall"
        if ratio <= 0.935:
            return "BreakingBall"
        return "Fastball"

    merged["new_group"] = merged.apply(_new_group, axis=1)
    merged["pitch_name"] = "Cutter"

    labeled = merged[["player_name", "pitcher", "pitch_name", "new_group"]]
    df = df.merge(labeled, on=["player_name", "pitcher", "pitch_name"], how="left")
    cutter_mask = df["pitch_name"] == "Cutter"
    df.loc[cutter_mask, "PitchGroup"] = df.loc[cutter_mask, "new_group"]
    df = df.drop(columns=["new_group"])
    return df


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each pitcher's fastball averages and compute relative diffs.

    Port of fastball-models.Rmd:163-169.
    """
    fb = (
        df[df["pitch_name"].isin(["4-Seam Fastball", "Sinker"])]
        .groupby(["player_name", "pitcher"], as_index=False)
        .agg(
            FBvelo=("release_speed", "mean"),
            FBvbreak=("pfx_z", "mean"),
            FBhbreak=("pfx_x", "mean"),
        )
    )
    df = df.merge(fb, on=["player_name", "pitcher"], how="left")
    df["RelSpeedDiff"] = df["release_speed"] - df["FBvelo"]
    df["Horz_Accel_Diff"] = df["a_horz"] - df["FBhbreak"]
    df["Vert_Accel_Diff"] = df["a_vert"] - df["FBvbreak"]
    df["launch_speed_angle"] = df["launch_speed_angle"].fillna(0)
    df = df.dropna().reset_index(drop=True)
    return df


def add_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Build Swing / Whiff / Foul / BIP outcome columns.

    Port of fastball-models.Rmd:175-186.
    """
    desc = df["description"]
    df = df.copy()
    df["Swing"] = desc.isin(SWING_DESCRIPTIONS).astype(int)
    df["Whiff"] = desc.isin(WHIFF_DESCRIPTIONS).astype(int)
    df["Foul"] = desc.isin(FOUL_DESCRIPTIONS).astype(int)
    df["BIP"] = df["launch_speed_angle"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Model data prep
# ---------------------------------------------------------------------------


def build_model_data(
    df: pd.DataFrame,
    pitch_group: str,
    include_relative: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter to a pitch group and split into RHP / LHP model frames.

    Port of fastball-models.Rmd:190-220. Returns (rhp, lhp) frames, each
    already stripped of ``p_throws``.
    """
    features = list(BASE_MODEL_FEATURES)
    if include_relative:
        features += RELATIVE_FEATURES
    outcome_cols = ["Swing", "Whiff", "Foul", "BIP"]
    keep = ["player_name", "p_throws"] + features + outcome_cols

    sub = df[df["PitchGroup"] == pitch_group][keep].copy()

    rhp = sub[sub["p_throws"] == "R"].drop(columns=["p_throws"]).reset_index(drop=True)
    lhp = sub[sub["p_throws"] == "L"].drop(columns=["p_throws"]).reset_index(drop=True)
    return rhp, lhp


def prep_model_data(df: pd.DataFrame, outcome_type: str) -> pd.DataFrame:
    """Build the modeling frame with a ``Label`` column for one outcome.

    Port of ``modelPrep`` at fastball-models.Rmd:226-249.
    """
    if outcome_type == "Whiff":
        data = df.copy()
        data["Label"] = data["Whiff"].astype(int)
    elif outcome_type == "Foul":
        data = df.copy()
        data["Label"] = data["Foul"].astype(int)
    elif outcome_type == "BIP":
        data = df[df["BIP"] != 0].copy()
        data["Label"] = (data["BIP"] - 1).astype(int)
    else:
        raise ValueError(f"Invalid outcome type: {outcome_type!r}. Expected Whiff, Foul, or BIP.")

    drop_cols = ["Swing", "Whiff", "Foul", "BIP", "player_name"]
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])
    return data.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _stratified_train_val_test_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
]:
    """Two-stage 80/10/10 stratified split.

    Mirrors ``initial_validation_split(prop = c(0.8, 0.1), strata = Label)``
    at fastball-models.Rmd:265.
    """
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _log_message(msg: str, pitch_type: str, outcome_type: str, p_hand: str, models_dir: Path) -> None:
    """Append a timestamped line to the per-model log.

    Port of ``log_message`` at fastball-models.Rmd:251-258.
    """
    log_path = models_dir / p_hand / outcome_type / f"{pitch_type}_{outcome_type}_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} - {msg}\n")


class _EarlyStoppingCallback:
    """Stop an Optuna study after ``patience`` non-improving trials."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self._best: float | None = None
        self._stale = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = trial.value
        if value is None:
            return
        if self._best is None or value > self._best:
            self._best = value
            self._stale = 0
        else:
            self._stale += 1
            if self._stale >= self.patience:
                study.stop()


def train_stuff_model(
    df: pd.DataFrame,
    pitch_type: str,
    outcome_type: str,
    p_hand: str,
    models_dir: Path,
    n_trials: int = 100,
    n_startup_trials: int = 20,
    early_stopping_patience: int = 5,
    nthread: int = 16,
) -> xgb.XGBClassifier:
    """Tune + fit an XGBoost classifier for one (pitch, outcome, hand) combo.

    Port of ``trainStuffModels`` at fastball-models.Rmd:260-340.
    """
    if outcome_type not in {"Whiff", "Foul", "BIP"}:
        _log_message(
            "Invalid Outcome Type: (Whiff, Foul, BIP)",
            pitch_type,
            outcome_type,
            p_hand,
            models_dir,
        )
        raise ValueError(f"Invalid outcome type: {outcome_type!r}")

    set_global_seed(RANDOM_SEED)

    y = df["Label"].astype(int)
    X = df.drop(columns=["Label"])
    feature_names = list(X.columns)

    X_train, X_val, X_test, y_train, y_val, y_test = _stratified_train_val_test_split(X, y)

    num_classes = int(y.nunique())
    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"

    def _objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 40),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        clf = xgb.XGBClassifier(
            objective=objective,
            eval_metric="auc" if num_classes == 2 else "mlogloss",
            nthread=nthread,
            random_state=RANDOM_SEED,
            tree_method="hist",
            **params,
        )
        clf.fit(X_train, y_train, verbose=False)
        val_proba = clf.predict_proba(X_val)
        if num_classes == 2:
            return float(roc_auc_score(y_val, val_proba[:, 1]))
        return float(
            roc_auc_score(y_val, val_proba, multi_class="ovo", average="macro", labels=list(range(num_classes)))
        )

    _log_message("Starting Tuning", pitch_type, outcome_type, p_hand, models_dir)
    print(f"Starting Tuning: {datetime.now().isoformat(timespec='seconds')}")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        _objective,
        n_trials=n_trials,
        callbacks=[_EarlyStoppingCallback(patience=early_stopping_patience)],
        show_progress_bar=False,
    )

    best_params = study.best_params
    _log_message(f"Best params: {best_params}", pitch_type, outcome_type, p_hand, models_dir)
    _log_message("Starting Fitting", pitch_type, outcome_type, p_hand, models_dir)
    print(f"Starting Fitting: {datetime.now().isoformat(timespec='seconds')}")

    final_model = xgb.XGBClassifier(
        objective=objective,
        eval_metric="auc" if num_classes == 2 else "mlogloss",
        nthread=nthread,
        random_state=RANDOM_SEED,
        tree_method="hist",
        **best_params,
    )
    final_model.fit(X_train, y_train, verbose=False)

    _log_message("Ended Fitting", pitch_type, outcome_type, p_hand, models_dir)
    print(f"Ended Fitting: {datetime.now().isoformat(timespec='seconds')}")

    # Save outputs.
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = models_dir / p_hand / outcome_type
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{pitch_type}_{outcome_type}_{stamp}"

    final_model.save_model(str(out_dir / f"{base}.ubj"))

    test_proba = final_model.predict_proba(X_test)
    if num_classes == 2:
        auc = float(roc_auc_score(y_test, test_proba[:, 1]))
        auc_row = {"metric": "roc_auc", "estimator": "binary", "estimate": auc}
    else:
        # roc_auc with multi_class='ovo' + average='macro' is the closest
        # sklearn equivalent to the Hand-Till multiclass estimator used in R.
        auc = float(
            roc_auc_score(
                y_test,
                test_proba,
                multi_class="ovo",
                average="macro",
                labels=list(range(num_classes)),
            )
        )
        auc_row = {"metric": "roc_auc", "estimator": "hand_till", "estimate": auc}

    with (out_dir / f"{base}_Error.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(auc_row.keys()))
        writer.writeheader()
        writer.writerow(auc_row)

    _save_vip_plot(
        final_model,
        feature_names,
        title=f"{pitch_type} - {outcome_type}:  Feature Importance",
        out_path=out_dir / f"{base}_VIP.png",
    )

    return final_model


def _save_vip_plot(
    model: xgb.XGBClassifier,
    feature_names: Iterable[str],
    title: str,
    out_path: Path,
    top_n: int = 40,
) -> None:
    importances = model.feature_importances_
    names = list(feature_names)
    order = np.argsort(importances)[::-1][:top_n]
    top_names = [names[i] for i in order][::-1]
    top_values = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_names, top_values)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# High-level driver helper
# ---------------------------------------------------------------------------


def run_pipeline(
    pitch_group: str,
    pitch_type_label: str,
    include_relative: bool,
    vaa_path: str,
    haa_path: str,
    models_dir: Path,
    db_kwargs: dict | None = None,
) -> dict[tuple[str, str], xgb.XGBClassifier]:
    """End-to-end: load data, engineer features, tune + fit all 6 models.

    Returns a dict keyed by ``(handedness, outcome)``.
    """
    logging.basicConfig(level=logging.INFO)
    set_global_seed(RANDOM_SEED)

    secondary = load_secondary_models(vaa_path, haa_path)

    df = load_statcast(**(db_kwargs or {}))
    df = engineer_features(df)
    df = predict_xvaa_xhaa(df, secondary)
    df = reclassify_cutters(df)
    df = add_relative_features(df)
    df = add_outcome_columns(df)

    rhp, lhp = build_model_data(df, pitch_group, include_relative=include_relative)

    results: dict[tuple[str, str], xgb.XGBClassifier] = {}
    for hand_label, frame in (("RHP", rhp), ("LHP", lhp)):
        for outcome in ("Whiff", "Foul", "BIP"):
            prepped = prep_model_data(frame, outcome)
            model = train_stuff_model(
                prepped,
                pitch_type=pitch_type_label,
                outcome_type=outcome,
                p_hand=hand_label,
                models_dir=models_dir,
            )
            results[(hand_label, outcome)] = model
    return results
