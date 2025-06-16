import os
import glob
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
    all_models = set()

    # First pass: collect all models
    for csv_file in csv_files:
        parsed = parse_filename(csv_file)
        if not parsed:
            continue
        language, model, _ = parsed
        all_models.add(model)
        if language not in language_groups:
            language_groups[language] = []
        language_groups[language].append((csv_file, model))

    # Assign a persistent color to each model
    color_map = plt.get_cmap("tab10")
    model_list = sorted(all_models)
    model_to_color = {model: color_map(i % 10) for i, model in enumerate(model_list)}

    for language, files in language_groups.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        cer_data = []
        wer_data = []
        labels = []
        colors = []

        for csv_file, model in files:
            cers, wers = read_csv_data(csv_file)
            cer_data.append(cers)
            wer_data.append(wers)
            labels.append(model)
            colors.append(model_to_color[model])

        # Boxplot for CER
        box1 = ax1.boxplot(cer_data, patch_artist=True)
        for patch, color in zip(box1["boxes"], colors):
            patch.set_facecolor(color)
        for median in box1["medians"]:
            median.set_color("black")
        ax1.set_title(f"{get_language_code(language)} CER")
        ax1.set_ylabel("CER")
        ax1.set_xticklabels([])

        # Boxplot for WER
        box2 = ax2.boxplot(wer_data, patch_artist=True)
        for patch, color in zip(box2["boxes"], colors):
            patch.set_facecolor(color)
        for median in box2["medians"]:
            median.set_color("black")
        ax2.set_title(f"{get_language_code(language)} WER")
        ax2.set_ylabel("WER")
        ax2.set_xticklabels([])

        # Add additional content for readability
        fig.suptitle(
            f"Box and Whisker Plots for Language: {get_language_code(language)}",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        # Add legend below the title, persistent colors
        # Add legend below the title, only for models present in this plot
        models_in_plot = [model for _, model in files]
        unique_models_in_plot = []
        [
            unique_models_in_plot.append(m)
            for m in models_in_plot
            if m not in unique_models_in_plot
        ]
        legend_patches = [
            mpatches.Patch(color=model_to_color[model], label=model)
            for model in unique_models_in_plot
        ]
        fig.legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=len(unique_models_in_plot),
            fontsize=12,
            frameon=False,
        )

        fig.text(
            0.5,
            0.01,
            "Each box shows the distribution of CER/WER for a model on this language.\n"
            "The line in the box is the median, the box edges are the quartiles, and whiskers show the range.",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(
            rect=[0, 0.12, 1, 0.92]
        )  # Adjust top and bottom margins for legend/title

        plt.savefig(f"{language}_boxplot.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
