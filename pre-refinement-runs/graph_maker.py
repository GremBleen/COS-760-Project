import os
import glob
import csv
import matplotlib.pyplot as plt

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
    batches = []
    cers = []
    wers = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batches.append(int(row["Batch"]))
            cers.append(float(row["CER"]))
            wers.append(float(row["WER"]))
    return batches, cers, wers

def main():
    csv_files = glob.glob("*.csv")
    model_groups = {}

    for csv_file in csv_files:
        parsed = parse_filename(csv_file)
        if not parsed:
            continue
        _, model, _ = parsed
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append(csv_file)

    for model, files in model_groups.items():
        plt.figure(figsize=(10, 6))
        for csv_file in files:
            batches, cers, wers = read_csv_data(csv_file)
            label = os.path.splitext(os.path.basename(csv_file))[0]
            plt.scatter(batches, cers, label=f"{label} CER", marker="o")
            plt.scatter(batches, wers, label=f"{label} WER", marker="x")
        plt.title(f"Scatter Plot for Model: {model}")
        plt.xlabel("Batch")
        plt.ylabel("Error Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{model}_scatter_plot.png")
        plt.close()

if __name__ == "__main__":
    main()