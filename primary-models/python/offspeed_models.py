"""Train Stuff+ offspeed models (Whiff / Foul / BIP, RHP + LHP).

Python port of primary-models/offspeed-models.Rmd. Keeps the three
relative-to-fastball features (RelSpeedDiff / Horz_Accel_Diff /
Vert_Accel_Diff), matching offspeed-models.Rmd:201.

Prerequisites: this script loads pre-trained VAA / HAA XGBoost boosters
saved as ``.ubj`` files. Those models are produced outside this script and
are not committed to the repo; pass their paths via --vaa-model / --haa-model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stuff_pipeline import run_pipeline

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_VAA = DEFAULT_MODELS_DIR / "VAA.ubj"
DEFAULT_HAA = DEFAULT_MODELS_DIR / "HAA.ubj"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vaa-model", default=str(DEFAULT_VAA), help="Path to the pre-trained VAA .ubj model.")
    p.add_argument("--haa-model", default=str(DEFAULT_HAA), help="Path to the pre-trained HAA .ubj model.")
    p.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Directory where trained primary models / plots / logs are written.",
    )
    p.add_argument("--db-user", default="root")
    p.add_argument("--db-password", default="")
    p.add_argument("--db-host", default="localhost")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-name", default="baseball-research")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    db_kwargs = {
        "user": args.db_user,
        "password": args.db_password,
        "host": args.db_host,
        "port": args.db_port,
        "database": args.db_name,
    }
    run_pipeline(
        pitch_group="Offspeed",
        pitch_type_label="Offspeed",
        include_relative=True,
        vaa_path=args.vaa_model,
        haa_path=args.haa_model,
        models_dir=Path(args.models_dir),
        db_kwargs=db_kwargs,
    )


if __name__ == "__main__":
    main()
