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
    cers = []
    wers = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cers.append(float(row["CER"]))
            wers.append(float(row["WER"]))
    return cers, wers


def get_language_code(language):
    language_codes = {
        "afr": "Afrikaans",
        "xho": "Xhosa",
        "zul": "Zulu",
    }
    return language_codes.get(language.lower(), None)


def main():
    csv_files = glob.glob("*.csv")
    language_groups = {}

    for csv_file in csv_files:
        parsed = parse_filename(csv_file)
        if not parsed:
            continue
        language, model, _ = parsed  # Get model name here
        if language not in language_groups:
            language_groups[language] = []
        language_groups[language].append(
            (csv_file, model)
        )  # Store tuple of file and model

    for language, files in language_groups.items():
        plt.figure(figsize=(12, 6))
        cer_data = []
        wer_data = []
        labels = []
        for csv_file, model in files:
            cers, wers = read_csv_data(csv_file)
            cer_data.append(cers)
            wer_data.append(wers)
            labels.append(model)  # Use model name as label

        # Boxplot for CER
        plt.subplot(1, 2, 1)
        plt.boxplot(cer_data, labels=labels, patch_artist=True)
        plt.title(f"{get_language_code(language)} CER")
        plt.ylabel("CER")
        plt.xticks(rotation=45, ha="right")

        # Boxplot for WER
        plt.subplot(1, 2, 2)
        plt.boxplot(wer_data, labels=labels, patch_artist=True)
        plt.title(f"{get_language_code(language)} WER")
        plt.ylabel("WER")
        plt.xticks(rotation=45, ha="right")

        plt.suptitle(
            f"Box and Whisker Plots for Language: {get_language_code(language)}"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{language}_boxplot.png")
        plt.close()


if __name__ == "__main__":
    main()
