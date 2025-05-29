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
