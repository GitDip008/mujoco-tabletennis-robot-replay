"""
Split tt_synthetic_*.csv into 4 per-class CSVs.

Each output file contains the timeseries rows for one stroke class only,
in the same column format as the input, so you can feed them to the
realtime pipeline (realtimeapp_imu.py) one at a time and verify that the
classifier predicts the expected class.
"""
import argparse
import pathlib
import pandas as pd

HERE   = pathlib.Path(__file__).resolve().parent
OUTDIR = HERE.parent / "output" / "per_class"
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str,
                   default=str(HERE.parent / "output" / "tt_synthetic_100reps.csv"))
    args = p.parse_args()

    src = pathlib.Path(args.input)
    print(f"Loading {src} ...")
    df = pd.read_csv(src)
    print(f"  rows: {len(df):,}   classes: {sorted(df['stroke_name'].unique())}")

    for label, sub in df.groupby("stroke_name"):
        out = OUTDIR / f"only_{label}.csv"
        sub.to_csv(out, index=False)
        print(f"  → {out}    ({len(sub):,} rows  ≈ {len(sub)//60} strokes)")

    print("\nDone.")


if __name__ == "__main__":
    main()
