from datasets import load_dataset, concatenate_datasets
from jiwer import cer, wer
import re
import string
import torch
import torchaudio

def getDataset(opt_lang):
    lang_list = ("afr", "xho", "zul", "ven", "tso", "tsn", "ssw", "nso", "sot")

    if opt_lang in lang_list:
        datasets = load_dataset(f"danielshaps/nchlt_speech_{opt_lang}")
        return concatenate_datasets([split for split in datasets.values()])
    else:
        raise ValueError(f"Invalid `opt_lang`: {opt_lang}")


def normalizeText(predicted_text):
    text = predicted_text.lower()
    text = text.strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


def evaluateTranscription(reference_text, predicted_text, output = False):
    normalized_reference = normalizeText(reference_text)
    normalized_predicted = normalizeText(predicted_text)
    char_err_rate = cer(
        reference=normalized_reference,
        hypothesis=normalized_predicted,
    )
    word_err_rate = wer(
        reference=normalized_reference,
        hypothesis=normalized_predicted,
    )

    if output:
        print(f"Reference: {normalized_reference}")
        print()
        print(f"Prediction: {normalized_predicted}")
        print()
        print(f"CER: {char_err_rate:.4f}")
        print(f"WER: {word_err_rate:.4f}")
        print()

    return char_err_rate, word_err_rate

def resample(waveform, current_sample_rate, required_sample_rate):
    if current_sample_rate != required_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=current_sample_rate, new_freq=required_sample_rate
        )  # ensuring that using 16kHz
        resampled = resampler(
            torch.tensor(waveform, dtype=torch.float32)
        ).numpy()
        return resampled
    else:
        return waveform

# def saveResults_V1(cer, wer, language, model, refinement, filename=None):
#     if filename is None:
#         from datetime import datetime
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{language}_{model}_{refinement}_{timestamp}.txt"
#     # Save the results to a file that displays the model, language, and refinement method as well as the CER and WER over the dataset
#     with open(filename, "a") as f:
#         f.write(f"Model: {model}, Language: {language}, Refinement: {refinement}\n")
#         f.write(f"CER: {cer:.4f}, WER: {wer:.4f}\n")
#         f.write("-" * 40 + "\n")

def saveResults(results_dict, language, model, refinement, filename=None):
    import csv

    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{language}_{model}_{refinement}_{timestamp}.csv"

    # Save the results to a CSV file that displays the model, language, and refinement method as well as the CER and WER for each batch
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Write the header
        writer.writerow(["Batch", "CER", "WER"])
        # Write the results
        for batch, (cer, wer) in results_dict.items():
            writer.writerow([batch, f"{cer:.4f}", f"{wer:.4f}"])