from datasets import load_dataset, concatenate_datasets
from jiwer import cer, wer
import re
import string
import torch
import torchaudio


def getDataset(opt_lang, option="test"):
    lang_list = ("afr", "xho", "zul", "ven", "tso", "tsn", "ssw", "nso", "sot")

    if opt_lang in lang_list:
        datasets = load_dataset(f"danielshaps/nchlt_speech_{opt_lang}")
        # return concatenate_datasets([split for split in datasets.values()])
        return datasets[option]  # Assuming we want the test split
    else:
        raise ValueError(f"Invalid `opt_lang`: {opt_lang}")


def normalizeText(predicted_text):
    text = predicted_text.lower()
    text = text.strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


def refinementMethod(predicted_text, refinement, word_list):
    from fuzzywuzzy import process

    words = predicted_text.split()
    refined_words = []
    for word in words:
        # Find the closest match in the word list
        match, score = process.extractOne(word, word_list)
        if score >= 80:
            refined_words.append(match)
        else:
            refined_words.append(word)
    return " ".join(refined_words)


def getWordList(language):
    # Load the word list for the specified language, should contain a list of valid words for the language with no duplicates
    dataset = getDataset(language, option="train")
    word_list = set()
    for sample in dataset:
        transcript = sample["text"]
        words = transcript.split()
        for word in words:
            word_list.add(word.lower())
    return list(word_list)


def evaluateTranscription(reference_text, predicted_text, batch_num, output=False):
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
        print(f"Batch Number: {batch_num}")
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
        resampled = resampler(torch.tensor(waveform, dtype=torch.float32)).numpy()
        return resampled
    else:
        return waveform


def trimSilence(waveform, sample_rate):
    import librosa

    # Silence trimming using librosa
    trimmed_waveform, _ = librosa.effects.trim(
        waveform, top_db=50, frame_length=2048, hop_length=512
    )
    # Frame length is the size of a single frame (or section of audio)
    # Hop length is the number sections to break a frame into
    
    return trimmed_waveform

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
