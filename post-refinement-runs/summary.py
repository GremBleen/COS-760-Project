# summary_stats.py
import os
import glob
import csv
import numpy as np


def parse_filename(filename):
    # Assumes filename format: language_model_refinement_timestamp.csv
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) < 4:
        return None  # Unexpected format
    language = parts[0]
    model = parts[1]
    refinement = parts[2]
    return language, model, refinement


def read_csv_data(filepath):
    cers = []
    wers = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cers.append(float(row["CER"]))
            wers.append(float(row["WER"]))
    return cers, wers


def calc_stats(values):
    arr = np.array(values)
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
    }


def main():
    csv_files = glob.glob("*.csv")
    summary = []

    for csv_file in csv_files:
        parsed = parse_filename(csv_file)
        if not parsed:
            continue
        language, model, refinement = parsed
        cers, wers = read_csv_data(csv_file)
        cer_stats = calc_stats(cers)
        wer_stats = calc_stats(wers)
        summary.append(
            {
                "file": csv_file,
                "language": language,
                "model": model,
                "refinement": refinement,
                "cer_mean": cer_stats["mean"],
                "cer_median": cer_stats["median"],
                "cer_std": cer_stats["std"],
                "cer_min": cer_stats["min"],
                "cer_max": cer_stats["max"],
                "wer_mean": wer_stats["mean"],
                "wer_median": wer_stats["median"],
                "wer_std": wer_stats["std"],
                "wer_min": wer_stats["min"],
                "wer_max": wer_stats["max"],
            }
        )

    # Print summary table
    print(
        f"{'File':40} {'CER Mean':8} {'CER Median':10} {'CER Std':8} {'CER Min':8} {'CER Max':8} {'WER Mean':8} {'WER Median':10} {'WER Std':8} {'WER Min':8} {'WER Max':8}"
    )
    for row in summary:
        print(
            f"{row['file']:40} {row['cer_mean']:.4f}   {row['cer_median']:.4f}    {row['cer_std']:.4f}   {row['cer_min']:.4f}   {row['cer_max']:.4f}   {row['wer_mean']:.4f}   {row['wer_median']:.4f}    {row['wer_std']:.4f}   {row['wer_min']:.4f}   {row['wer_max']:.4f}"
        )

    # Print latex table for better formatting
    print("\nLaTeX Table:\n")
    print("\\begin{tabular}{lllrrrrrrrrrr}")
    print("\\toprule")
    print(
        "Language & Model & Refinement & CER Mean & CER Median & CER Std & CER Min & CER Max & WER Mean & WER Median & WER Std & WER Min & WER Max \\\\"
    )
    print("\\midrule")
    for row in summary:
        print(
            f"{row['language']} & {row['model']} & {row['refinement']} & "
            f"{row['cer_mean']:.4f} & {row['cer_median']:.4f} & {row['cer_std']:.4f} & {row['cer_min']:.4f} & {row['cer_max']:.4f} & "
            f"{row['wer_mean']:.4f} & {row['wer_median']:.4f} & {row['wer_std']:.4f} & {row['wer_min']:.4f} & {row['wer_max']:.4f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")

    # Optionally, save to CSV
    with open("results/summary_statistics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
