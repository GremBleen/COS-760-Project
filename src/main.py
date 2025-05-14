# Log in to HuggingFace
from huggingface_hub import login

login()

from datasets import load_dataset, concatenate_datasets

"""
As we are not training any models, we are using the entire dataset.
"""


def getDataset(opt_lang):

    lang_list = ("afr", "xho", "zul", "ven", "tso", "tsn", "ssw", "nso", "sot")

    if opt_lang in lang_list:
        datasets = load_dataset(f"danielshaps/nchlt_speech_{opt_lang}")
        return concatenate_datasets([split for split in datasets.values()])
    else:
        raise ValueError(f"Invalid `opt_lang`: {opt_lang}")


import torch

"""
This is the loop called by each of the models in order to do evaluation
"""


def runLoop(processor, model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sample = dataset[0]["audio"]

    input_features = processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features.to(device)

    # Since not training, it is not necessary to include gradients
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(sample["sampling_rate"])
    print(dataset[0]["text"])
    print(transcription)

    return 0

"""
This block has been separated so that the dataset can accessed without redownload across multiple runs
"""

# Run options
opt_lang = "afr"
opt_model = "lelapa"  # 'whisper-medium', 'whisper-large', 'afriwhisper', 'lelapa', 'wav2vec', 'deepspeech', 'all'

# This is getting the dataset specified by `opt_lang` which takes a while
test = getDataset(opt_lang)

# Do not alter anything below this comment

from models.whisper import runWhisperMedium, runWhisperLargeV3, runAfriWhisper
from models.lelapa import runLelapa
from models.wav2vec import runWav2Vec
from models.deep_speech import runDeepSpeech

if opt_model == "whisper-medium":
    runWhisperMedium(test)
elif opt_model == "whisper-large":
    runWhisperLargeV3(test)
elif opt_model == "afriwhisper":
    runAfriWhisper(test)
elif opt_model == "lelapa":
    runLelapa(test)
elif opt_model == "wav2vec":
    runWav2Vec(test)
elif opt_model == "deepspeech":
    runDeepSpeech(test)
elif opt_model == "all":
    runWhisperMedium(test)
    runLelapa(test)
    runWav2Vec(test)
    runDeepSpeech(test)
else:
    raise ValueError(f"Invalid `opt_model`: {opt_model}")
