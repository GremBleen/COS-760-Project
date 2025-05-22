from dotenv import load_dotenv
import os
from huggingface_hub import login


from models.whisper import runWhisperMedium, runWhisperLargeV3, runAfriWhisper
from models.lelapa import runLelapa
from models.wav2vec import runWav2Vec
from models.deep_speech import runDeepSpeech

from common import getDataset


# Log in to HuggingFace
load_dotenv()
login(token = os.getenv("HUGGINGFACE_TOKEN"))

# TODO - chage this so that the run options read from the presets.json instead
opt_lang = "xho"
opt_model = "lelapa"  # 'whisper-medium', 'whisper-large', 'afriwhisper', 'lelapa', 'wav2vec', 'deepspeech', 'all'

# This is getting the dataset specified by `opt_lang` which takes a while
test = getDataset(opt_lang)

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
