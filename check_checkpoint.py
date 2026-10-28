"""Midsemester checkpoint checker, COSC 4371 Security Analytics, Fall 2026.

Reads the two files your code writes, split.csv and scores.csv, and prints:
  - rows and class counts per dataset and split,
  - the metrics table for each test set, with 95% bootstrap intervals,
  - warnings.

Usage:
  python check_checkpoint.py --split split.csv --scores scores.csv --threshold 0.5

File formats (column names must match exactly; extra columns are ignored). Either file may be
gzip-compressed (split.csv.gz); the checker reads it the same way.

split.csv, one row per row of data you kept, in every dataset you use:
  row_id   unique id of the row, the same id used in scores.csv
  dataset  dataset name, for example safeguard or deepset
  split    train, validation or test
  label    1 for the positive class, 0 for the negative class

scores.csv, one row per test row:
  row_id    id from split.csv
  test_set  the test set's name, as in your results table
  label     1 or 0, the same label as in split.csv
  score     the classifier's score; higher means more likely positive

A row is predicted positive when score >= threshold.
Needs Python 3.9 or later with pandas, numpy and scikit-learn.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SPLITS = ["train", "validation", "test"]
CEILING = 0.99
MIN_CLASS = 30
BIG = 100_000
METRICS = ["precision", "recall", "false-positive rate", "AP", "AUROC"]

warnings = []


def warn(msg):
    warnings.append(msg)


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def read_csv(path, needed, name):
    try:
        df = pd.read_csv(path, dtype={"row_id": str, "dataset": str,
                                      "split": str, "test_set": str})
    except Exception as exc:  # noqa: BLE001
        fail(f"cannot read {name} ({path}): {exc}")
    missing = [c for c in needed if c not in df.columns]
    if missing:
        fail(f"{name} is missing column(s): {', '.join(missing)}. "
             f"Columns found: {', '.join(df.columns)}")
    return df


def check_binary(series, name):
    values = pd.to_numeric(series, errors="coerce")
    bad = values.isna() | ~values.isin([0, 1])
    if bad.any():
        fail(f"{name} must be 0 or 1 in every row; {int(bad.sum())} row(s) are not")
    return values.astype(int)


def section(title):
    print()
    print(title)
    print("-" * len(title))


def split_report(split):
    section("Rows per dataset and split")
    rows = []
    for (ds, sp), g in split.groupby(["dataset", "split"]):
        pos = int((g["label"] == 1).sum())
        rows.append({"dataset": ds, "split": sp, "rows": len(g), "positive": pos,
                     "negative": len(g) - pos,
                     "positive %": round(100 * pos / len(g), 2)})
    order = {s: i for i, s in enumerate(SPLITS)}
    table = pd.DataFrame(rows)
    table["_o"] = table["split"].map(order)
    print(table.sort_values(["dataset", "_o"]).drop(columns="_o").to_string(index=False))


def reps_for(n, reps):
    """1000 resamples by default; 200 for a test set over 100,000 rows."""
    if reps is not None:
        return reps
    return 200 if n > BIG else 1000


def rates(y, pred):
    out = {}
    out["recall"] = pred[y == 1].mean() if (y == 1).any() else None
    out["false-positive rate"] = pred[y == 0].mean() if (y == 0).any() else None
    out["precision"] = y[pred == 1].mean() if (pred == 1).any() else None
    return out


def all_metrics(y, s, pred):
    out = rates(y, pred)
    two = 0 < y.sum() < len(y)
    out["AP"] = average_precision_score(y, s) if two else None
    out["AUROC"] = roc_auc_score(y, s) if two else None
    return out


def bootstrap(y, s, pred, rng, reps):
    """One resample per repetition, shared by every metric."""
    n = len(y)
    draws = {k: [] for k in METRICS}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        for k, v in all_metrics(y[idx], s[idx], pred[idx]).items():
            if v is not None:
                draws[k].append(v)
    ci = {}
    for k, vals in draws.items():
        ci[k] = (None if len(vals) < reps // 2
                 else (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))))
    return ci


def fmt(v, ci=None):
    if v is None:
        return "n/a"
    s = f"{v:.4f}"
    if ci is None:
        s += " [no interval: the metric was undefined in too many resamples]"
    else:
        s += f" [{ci[0]:.4f}, {ci[1]:.4f}]"
    return s


def classifier_metrics(scores, threshold, reps, seed):
    section(f"Metrics per test set (threshold {threshold}; positive when score >= threshold)")
    print("Each value is followed by its 95% bootstrap interval in brackets.")
    rng = np.random.default_rng(seed)
    for ts, g in scores.groupby("test_set", sort=False):
        y = g["label"].to_numpy()
        s = g["score"].to_numpy(dtype=float)
        pred = (s >= threshold).astype(int)
        n, pos = len(y), int(y.sum())
        neg = n - pos
        r = reps_for(n, reps)
        print()
        print(f"Test set: {ts}")
        print(f"  test rows                 {n}")
        print(f"  positive rows             {pos} ({100 * pos / n:.2f}%)")
        print(f"  negative rows             {neg} ({100 * neg / n:.2f}%)")
        print(f"  bootstrap resamples       {r}")
        values = all_metrics(y, s, pred)
        ci = bootstrap(y, s, pred, rng, r)

        if pos == 0 or neg == 0:
            only = "positive" if neg == 0 else "negative"
            name = "recall" if neg == 0 else "false-positive rate"
            print(f"  one class only ({only}): report {name} only")
            print(f"  {name:<25} {fmt(values[name], ci[name])}")
            if n < MIN_CLASS:
                warn(f"{ts}: only {n} rows. The interval will be wide; keep this in mind for the "
                     f"yes difference in section 5(b).")
            if neg == 0 and values[name] >= CEILING:
                warn(f"{ts}: recall at or above {CEILING}. Name the leakage check you ran and its "
                     f"result in section 3(e).")
            continue

        print(f"  random-scorer AP          {pos / n:.4f} (equals the positive share)")
        print("  random-scorer AUROC       0.5000")
        for name in METRICS:
            if values[name] is None:
                print(f"  {name:<25} undefined (no row scored at or above the threshold)")
            else:
                print(f"  {name:<25} {fmt(values[name], ci[name])}")

        if values["AUROC"] < 0.5:
            warn(f"{ts}: AUROC {values['AUROC']:.4f} is below 0.5. The labels or the score "
                 f"direction may be inverted; check both before submitting.")
        if values["AP"] <= pos / n or values["AUROC"] <= 0.5:
            warn(f"{ts}: the baseline does not beat a random scorer on AP or AUROC. "
                 f"Say so and give one reason in section 3(d).")
        high = [k for k in ["precision", "recall", "AP", "AUROC"]
                if values[k] is not None and values[k] >= CEILING]
        if high:
            warn(f"{ts}: {', '.join(high)} at or above {CEILING}. Name the leakage check you ran "
                 f"and its result in section 3(e).")
        if len(np.unique(s)) <= 2:
            warn(f"{ts}: the score column has only {len(np.unique(s))} distinct values. Write the "
                 f"classifier's score or probability, not its predicted label.")
        if pos < MIN_CLASS or neg < MIN_CLASS:
            warn(f"{ts}: fewer than {MIN_CLASS} rows in one class ({pos} positive, {neg} "
                 f"negative). The intervals will be wide; keep this in mind for the yes difference "
                 f"in section 5(b).")


def main():
    ap_ = argparse.ArgumentParser(description="COSC 4371 midsemester checkpoint checker")
    ap_.add_argument("--split", required=True, help="path to split.csv")
    ap_.add_argument("--scores", required=True, help="path to scores.csv")
    ap_.add_argument("--threshold", type=float, help="the threshold chosen on validation")
    args = ap_.parse_args()

    if args.threshold is None:
        fail("give --threshold, the value chosen on your validation set")
    if not np.isfinite(args.threshold):
        fail("--threshold must be a finite number")

    split = read_csv(args.split, ["row_id", "dataset", "split", "label"], "split.csv")
    for col in ["row_id", "dataset"]:
        if split[col].isna().any():
            fail(f"split.csv has {int(split[col].isna().sum())} blank {col} value(s)")
    dup = split["row_id"].duplicated()
    if dup.any():
        fail(f"split.csv has {int(dup.sum())} repeated row_id value(s); each row needs its own id")

    bad_split = ~split["split"].isin(SPLITS)
    if bad_split.any():
        fail(f"split must be train, validation or test; found "
             f"{sorted(split.loc[bad_split, 'split'].unique())}")
    split["label"] = check_binary(split["label"], "split.csv label")
    for sp, section_ref in [("train", "3(a)"), ("validation", "3(b)")]:
        if not (split["split"] == sp).any():
            warn(f"split.csv has no {sp} rows. Every baseline needs a training set and a "
                 f"validation set; see section {section_ref}.")
    scores = read_csv(args.scores, ["row_id", "test_set", "label", "score"], "scores.csv")
    for col in ["row_id", "test_set"]:
        if scores[col].isna().any():
            fail(f"scores.csv has {int(scores[col].isna().sum())} blank {col} value(s)")
    scores["label"] = check_binary(scores["label"], "scores.csv label")
    sc = pd.to_numeric(scores["score"], errors="coerce")
    bad_score = ~np.isfinite(sc)
    if bad_score.any():
        fail(f"scores.csv has {int(bad_score.sum())} blank, non-numeric or infinite score(s)")
    scores["score"] = sc
    if scores.duplicated(["row_id", "test_set"]).any():
        fail("scores.csv lists the same row_id twice in one test set")

    test_rows = split.loc[split["split"] == "test"].set_index("row_id")
    not_test = ~scores["row_id"].isin(test_rows.index)
    if not_test.any():
        fail(f"{int(not_test.sum())} scores.csv row(s) are not test rows in split.csv "
             f"(first: {scores.loc[not_test, 'row_id'].iloc[0]})")
    mismatch = scores["label"].to_numpy() != test_rows.loc[scores["row_id"], "label"].to_numpy()
    if mismatch.any():
        fail(f"{int(mismatch.sum())} row(s) have a different label in scores.csv and split.csv")
    unscored = ~test_rows.index.isin(scores["row_id"])
    if unscored.any():
        warn(f"{int(unscored.sum())} test row(s) in split.csv have no score. Score every "
             f"test row, or explain why in section 2(c).")

    print(f"split.csv: {len(split)} rows. scores.csv: {len(scores)} rows.")
    split_report(split)
    classifier_metrics(scores, args.threshold, None, 0)

    section("Warnings")
    if warnings:
        for w in warnings:
            print(f"- {w}")
    else:
        print("None.")


if __name__ == "__main__":
    main()
