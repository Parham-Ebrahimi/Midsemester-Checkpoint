"""Final project checkpoint checker, COSC 4371 Security Analytics, Fall 2026.

Reads the files your code writes, split.csv, scores.csv and (optionally) errors.csv, and prints:
  - rows and class counts per dataset and split,
  - the metrics for each value of your variable in each column, with 95% bootstrap intervals,
  - your results table for one metric, with each value's difference from the baseline value,
  - a summary of your error sample,
  - warnings.

Usage:
  python check_final_checkpoint.py --split split.csv --scores scores.csv --errors errors.csv --metric recall --reference baseline --direction lower

--metric is your primary metric: precision, recall, fpr, ap or auroc.
--reference is the name of your baseline value, as written in the value column of scores.csv.
--direction is the change your question asks about, from section 1(e): higher (the metric goes up
from the baseline), lower (it goes down) or either (any change).
--errors, --metric, --reference and --direction may be left out while you work; the checkpoint
needs all four.

File formats (column names must match exactly; extra columns are ignored). Any file may be
gzip-compressed (scores.csv.gz); the checker reads it the same way.

split.csv, one row per row of data you kept, in every dataset you use (midsemester format):
  row_id   unique id of the row, the same id used in scores.csv
  dataset  dataset name, for example safeguard or deepset
  split    train, validation or test
  label    1 for the positive class, 0 for the negative class

scores.csv, one row per scored test row, per value of your variable, per column:
  row_id     id from split.csv (a changed copy of a test row keeps the original id)
  value      the value of your variable: the row of your results table
  column     the column of your results table (a test set, a classifier, a training set or a
             defense; see the handout)
  label      1 or 0, the same label as in split.csv
  score      the score; higher means more likely positive
  threshold  the threshold used for this value and column, the same in every row of that pair

errors.csv, one row per misclassified row in your error sample:
  row_id, value, column     as in scores.csv
  category                  the error category from your table in section 4(b)

A row is predicted positive when score >= threshold.
Needs Python 3.9 or later with pandas, numpy and scikit-learn.
"""

import argparse
import sys
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SPLITS = ["train", "validation", "test"]
CEILING = 0.99
MIN_CLASS = 30
BIG = 100_000
METRICS = ["precision", "recall", "false-positive rate", "AP", "AUROC"]
METRIC_ARG = {"precision": "precision", "recall": "recall", "fpr": "false-positive rate",
              "ap": "AP", "auroc": "AUROC"}
SAMPLE_PER_TYPE = 10
CATCH_ALL = {"other", "others", "misc", "miscellaneous", "unknown", "unclear"}
CATCH_ALL_MAX = 5

warnings = []


def warn(msg):
    warnings.append(msg)


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def read_csv(path, needed, name):
    try:
        df = pd.read_csv(path, dtype={"row_id": str, "dataset": str, "split": str,
                                      "column": str, "value": str, "category": str})
    except Exception as exc:  # noqa: BLE001
        fail(f"cannot read {name} ({path}): {exc}")
    missing = [c for c in needed if c not in df.columns]
    if missing:
        fail(f"{name} is missing column(s): {', '.join(missing)}. "
             f"Columns found: {', '.join(df.columns)}")
    for col in needed:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
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


def reps_for(n):
    """1000 resamples by default; 200 for a test set over 100,000 rows."""
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


def fmt(v, ci=None, interval=True):
    if v is None:
        return "n/a"
    s = f"{v:.4f}"
    if not interval:
        return s
    if ci is None:
        s += " [no interval: undefined in too many resamples]"
    else:
        s += f" [{ci[0]:.4f}, {ci[1]:.4f}]"
    return s


def group_rng(value, column):
    """The same value and column always get the same resamples; rows are sorted by row_id first,
    so the intervals do not depend on the order of rows in scores.csv."""
    return np.random.default_rng([zlib.crc32(value.encode()), zlib.crc32(column.encode())])


def classifier_metrics(scores):
    section("Metrics per value and column (positive when score >= the threshold)")
    print("Each value is followed by its 95% bootstrap interval in brackets.")
    results = {}
    for (val, ts), g in scores.groupby(["value", "column"], sort=False):
        g = g.sort_values("row_id", kind="mergesort")
        y = g["label"].to_numpy()
        s = g["score"].to_numpy(dtype=float)
        thr = float(g["threshold"].iloc[0])
        pred = (s >= thr).astype(int)
        n, pos = len(y), int(y.sum())
        neg = n - pos
        r = reps_for(n)
        print()
        print(f"Value: {val}   Column: {ts}")
        print(f"  threshold                 {thr:g}")
        print(f"  test rows                 {n}")
        print(f"  positive rows             {pos} ({100 * pos / n:.2f}%)")
        print(f"  negative rows             {neg} ({100 * neg / n:.2f}%)")
        print(f"  bootstrap resamples       {r}")
        values = all_metrics(y, s, pred)
        ci = bootstrap(y, s, pred, group_rng(val, ts), r)
        where = f"value {val}, column {ts}"

        if pos == 0 or neg == 0:
            only = "positive" if neg == 0 else "negative"
            name = "recall" if neg == 0 else "false-positive rate"
            print(f"  one class only ({only}): report {name} only")
            print(f"  {name:<25} {fmt(values[name], ci[name])}")
            results[(val, ts)] = {"one_class": name, "values": values, "ci": ci, "n": n,
                                  "pos": pos}
            if n < MIN_CLASS:
                warn(f"{where}: only {n} rows. The interval will be wide; keep this in mind "
                     f"for the verdict in section 1(e).")
            if neg == 0 and values[name] >= CEILING:
                warn(f"{where}: recall at or above {CEILING}. Name the leakage check you ran and "
                     f"its result in section 1(c).")
            continue

        results[(val, ts)] = {"one_class": None, "values": values, "ci": ci, "n": n, "pos": pos}
        print(f"  random-scorer AP          {pos / n:.4f} (equals the positive share)")
        print("  random-scorer AUROC       0.5000")
        for name in METRICS:
            if values[name] is None:
                print(f"  {name:<25} undefined (no row scored at or above the threshold)")
            else:
                print(f"  {name:<25} {fmt(values[name], ci[name])}")

        if values["AUROC"] < 0.5:
            warn(f"{where}: AUROC {values['AUROC']:.4f} is below 0.5. The labels or the score "
                 f"direction may be inverted; check both before submitting.")
        if values["AP"] <= pos / n or values["AUROC"] <= 0.5:
            warn(f"{where}: no better than a random scorer on AP or AUROC. Name this cell and "
                 f"give one reason below your results table in section 1(d).")
        high = [k for k in ["precision", "recall", "AP", "AUROC"]
                if values[k] is not None and values[k] >= CEILING]
        if high:
            warn(f"{where}: {', '.join(high)} at or above {CEILING}. Name the leakage check you "
                 f"ran and its result in section 1(c).")
        if len(np.unique(s)) <= 2:
            warn(f"{where}: the score column has only {len(np.unique(s))} distinct values. Write "
                 f"the classifier's score or probability, not its predicted label.")
        if pos < MIN_CLASS or neg < MIN_CLASS:
            warn(f"{where}: fewer than {MIN_CLASS} rows in one class ({pos} positive, {neg} "
                 f"negative). The intervals will be wide; keep this in mind for the verdict in "
                 f"section 1(e).")
    return results


def cell_metric(res, metric):
    """The metric a cell reports: the primary metric, or the one-class metric."""
    if res["one_class"] is not None:
        return res["one_class"]
    return metric


def results_table(results, values_order, tests_order, metric, reference, direction):
    section(f"Results table: {metric} [95% interval] (copy into section 1(d))")
    print("A one-class cell holds the metric the checker prints for it, named in parentheses.")
    width = max(len(v) for v in values_order) + 2
    for ts in tests_order:
        print()
        print(f"Column: {ts}")
        for val in values_order:
            res = results.get((val, ts))
            if res is None:
                cell = "-- (no rows in scores.csv)"
            else:
                m = cell_metric(res, metric)
                cell = fmt(res["values"][m], res["ci"][m])
                if res["values"][m] is None:
                    cell = "undefined (no row scored at or above the threshold)"
                if m != metric:
                    cell += f" ({m})"
            print(f"  {val:<{width}} {cell}")

    if reference is None:
        return
    section(f"Difference from the baseline value '{reference}' (copy into section 1(e))")
    print("Difference = the value's metric minus the baseline value's metric in the same column.")
    if direction == "higher":
        print("Direction of interest: higher. Change = the difference.")
    elif direction == "lower":
        print("Direction of interest: lower. Change = the difference with its sign flipped, so a "
              "drop is positive.")
    elif direction == "either":
        print("Direction of interest: either. Change = the size of the difference.")
    if direction is not None:
        print("The verdict in section 1(e) compares the change with your yes and no differences.")
    any_diff = False
    for ts in tests_order:
        ref = results.get((reference, ts))
        print()
        print(f"Column: {ts}")
        if ref is None:
            print("  no baseline-value rows in this column; no difference")
            continue
        m_ref = cell_metric(ref, metric)
        for val in values_order:
            if val == reference:
                continue
            res = results.get((val, ts))
            if res is None:
                print(f"  {val:<{width}} -- (no rows in scores.csv)")
                continue
            m = cell_metric(res, metric)
            a, b = res["values"][m], ref["values"][m_ref]
            if m != m_ref or a is None or b is None:
                print(f"  {val:<{width}} n/a (metric undefined for one of the two)")
                continue
            any_diff = True
            label = metric if m == metric else m
            diff = a - b
            line = f"  {val:<{width}} difference {diff:+.4f}"
            if direction is not None:
                change = {"higher": diff, "lower": -diff, "either": abs(diff)}[direction]
                line += f"   change {round(change, 4) + 0.0:+.4f}"
            print(f"{line} ({label})")
    if not any_diff:
        warn("No difference could be computed. Section 1(e) needs at least one value besides the "
             "baseline value, scored in a column where the baseline value is also scored.")


def error_report(errors, scores):
    section("Error sample (section 4)")
    key = ["row_id", "value", "column"]
    errors = errors[key + ["category"]]
    if len(errors) == 0:
        pred = (scores["score"] >= scores["threshold"]).astype(int)
        wrong = int((pred != scores["label"]).sum())
        print("errors.csv has no rows.")
        if wrong:
            warn(f"errors.csv has no rows, but scores.csv has {wrong} misclassified row(s). "
                 f"Sample from a cell that has errors (section 4(a)).")
        else:
            print("scores.csv has no misclassified row; say so in section 4(a).")
        return
    if errors["category"].isna().any() or (errors["category"] == "").any():
        warn("errors.csv has a blank category; every sampled row needs one. The error sample "
             "was not checked.")
        return
    if errors.duplicated(key).any():
        warn("errors.csv lists the same row_id twice for one value and column. The error sample "
             "was not checked.")
        return
    merged = errors.merge(scores, on=key, how="left", validate="one_to_one")
    absent = merged["score"].isna()
    if absent.any():
        first = merged.loc[absent, key].iloc[0].tolist()
        warn(f"{int(absent.sum())} errors.csv row(s) have no matching row in scores.csv "
             f"(first: {first}). The error sample was not checked.")
        return
    merged["pred"] = (merged["score"] >= merged["threshold"]).astype(int)
    right = merged["pred"] == merged["label"]
    if right.any():
        first = merged.loc[right, "row_id"].iloc[0]
        warn(f"errors.csv: {int(right.sum())} row(s) are classified correctly at the value's "
             f"threshold (first: {first}). Sample only misclassified rows.")
    merged["error"] = np.where(merged["label"] == 1, "false negative", "false positive")

    pairs = merged[["value", "column"]].drop_duplicates()
    if len(pairs) > 1:
        warn("errors.csv mixes values or columns; section 4(a) samples one value in one "
             "column.")
    for _, (val, ts) in pairs.iterrows():
        g = scores[(scores["value"] == val) & (scores["column"] == ts)]
        pred = (g["score"] >= g["threshold"]).astype(int)
        fp_all = int(((pred == 1) & (g["label"] == 0)).sum())
        fn_all = int(((pred == 0) & (g["label"] == 1)).sum())
        sub = merged[(merged["value"] == val) & (merged["column"] == ts)]
        fp = int(((sub["error"] == "false positive") & ~right.loc[sub.index]).sum())
        fn = int(((sub["error"] == "false negative") & ~right.loc[sub.index]).sum())
        print(f"Value: {val}   Column: {ts}")
        print(f"  misclassified rows available   {fp_all} false positives, "
              f"{fn_all} false negatives")
        print(f"  sampled                        {fp} false positives, {fn} false negatives")
        if g["label"].nunique() < 2:
            warn(f"errors.csv: value {val}, column {ts} has one class; section 4(a) needs a "
                 f"cell with both classes.")
        for kind, got, avail in [("false positives", fp, fp_all), ("false negatives", fn, fn_all)]:
            want = min(SAMPLE_PER_TYPE, avail)
            if got != want:
                warn(f"errors.csv: {got} {kind} sampled for value {val}, column {ts}; the "
                     f"sample needs "
                     f"{want} ({SAMPLE_PER_TYPE}, or all of them when fewer are available).")

    print()
    table = (merged[~right].groupby(["category", "error"]).size().unstack(fill_value=0)
             .reindex(columns=["false positive", "false negative"], fill_value=0))
    table["total"] = table.sum(axis=1)
    print(table.sort_values("total", ascending=False).to_string())
    if len(table) < 2:
        warn("errors.csv: fewer than two categories; section 4(b) needs at least two, or one "
             "reason your errors fit only one.")
    for cat, row in table.iterrows():
        if cat.strip().lower() in CATCH_ALL and row["total"] > CATCH_ALL_MAX:
            warn(f"errors.csv: catch-all category '{cat}' holds {int(row['total'])} rows; "
                 f"section 4(b) allows at most {CATCH_ALL_MAX}.")


def main():
    ap_ = argparse.ArgumentParser(description="COSC 4371 final project checkpoint checker")
    ap_.add_argument("--split", required=True, help="path to split.csv")
    ap_.add_argument("--scores", required=True, help="path to scores.csv")
    ap_.add_argument("--errors", help="path to errors.csv")
    ap_.add_argument("--metric", choices=sorted(METRIC_ARG), help="your primary metric")
    ap_.add_argument("--reference", help="the name of your baseline value in scores.csv")
    ap_.add_argument("--direction", choices=["higher", "lower", "either"],
                     help="the change your question asks about, from section 1(e)")
    args = ap_.parse_args()

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
    for sp in ["train", "validation"]:
        if not (split["split"] == sp).any():
            warn(f"split.csv has no {sp} rows. Every model needs a training set and a validation "
                 f"set.")

    scores = read_csv(args.scores,
                      ["row_id", "value", "column", "label", "score", "threshold"],
                      "scores.csv")
    for col in ["row_id", "value", "column"]:
        if scores[col].isna().any() or (scores[col] == "").any():
            fail(f"scores.csv has blank {col} value(s)")
    scores["label"] = check_binary(scores["label"], "scores.csv label")
    for col in ["score", "threshold"]:
        num = pd.to_numeric(scores[col], errors="coerce")
        bad = ~np.isfinite(num)
        if bad.any():
            fail(f"scores.csv has {int(bad.sum())} blank, non-numeric or infinite {col}(s)")
        scores[col] = num
    per_pair = scores.groupby(["value", "column"], sort=False)["threshold"].nunique()
    bad_pairs = [f"{v} / {c}" for (v, c), k in per_pair.items() if k > 1]
    if bad_pairs:
        fail(f"{len(bad_pairs)} value and column pair(s) have more than one threshold (first: "
             f"{bad_pairs[0]}); a pair uses one threshold in every row")
    if scores.duplicated(["row_id", "value", "column"]).any():
        fail("scores.csv lists the same row_id twice for one value and column")

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
        warn(f"{int(unscored.sum())} test row(s) in split.csv have no score under any value. "
             f"Score every test row, or explain why in section 1(a).")

    for col, g in scores.groupby("column", sort=False):
        sets = {v: set(r) for v, r in g.groupby("value", sort=False)["row_id"]}
        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if sets[a] & sets[b] and sets[a] != sets[b]:
                    warn(f"column {col}: values {a} and {b} share some test rows but not all. "
                         f"Score every value in a column on the same test rows with their "
                         f"original labels (handout, Results table).")

    values_order = list(dict.fromkeys(scores["value"]))
    tests_order = list(dict.fromkeys(scores["column"]))
    if len(values_order) < 2:
        warn("scores.csv has only one value of your variable. Section 1 compares at least two.")
    reference = args.reference
    if reference is not None and reference not in values_order:
        fail(f"--reference '{reference}' is not a value in scores.csv; values found: "
             f"{', '.join(values_order)}")
    if reference is not None:
        values_order = [reference] + [v for v in values_order if v != reference]

    print(f"split.csv: {len(split)} rows. scores.csv: {len(scores)} rows, "
          f"{len(values_order)} value(s), {len(tests_order)} column(s).")
    print(f"Values: {', '.join(values_order)}")
    split_report(split)
    results = classifier_metrics(scores)

    if args.metric is not None:
        results_table(results, values_order, tests_order, METRIC_ARG[args.metric], reference,
                      args.direction)
    else:
        warn("No --metric given; the results table and differences were not printed.")
    if args.metric is not None and reference is None:
        warn("No --reference given; the differences from the baseline value were not printed.")
    if args.metric is not None and reference is not None and args.direction is None:
        warn("No --direction given; the change in the direction of interest was not printed.")

    if args.errors is not None:
        errors = read_csv(args.errors, ["row_id", "value", "column", "category"], "errors.csv")
        error_report(errors, scores)
    else:
        warn("No --errors given; the error sample was not checked.")

    section("Warnings")
    if warnings:
        for w in warnings:
            print(f"- {w}")
    else:
        print("None.")


if __name__ == "__main__":
    main()
