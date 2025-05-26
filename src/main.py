from dotenv import load_dotenv
import os
from huggingface_hub import login

from models.whisper import runWhisper, runAfriWhisper
from models.lelapa import runLelapa
from models.wav2vec import runWav2Vec
from models.deep_speech import runDeepSpeech

from common import getDataset

import json
import pathlib

# afr - Afrikaans
# xho - Xhosa
# zul - Zulu
# ven - Venda
# tso - Tsonga
# tsn - Tswana
# ssw - Swati
# nso - Sepedi
# sot - Sotho

# Log in to HuggingFace
load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Get the path to the presets.json file in the parent directory
presets_path = pathlib.Path(__file__).parent.parent / "presets.json"
with open(presets_path, "r") as f:
    presets = json.load(f)

opt_lang = presets["dataset_language"] # 'afr', 'xho', 'zul', 'ven', 'tso', 'tsn', 'ssw', 'nso', 'sot'
opt_model = presets["model"]  # 'whisper-medium', 'whisper-large', 'afriwhisper', 'lelapa', 'wav2vec', 'deepspeech', 'all'
opt_refinement = presets["refinement_method"]
opt_debug = presets["debug"]

# This is getting the dataset specified by `opt_lang` which takes a while
test = getDataset(opt_lang)

if opt_model == "whisper-medium":
    runWhisper("medium", test, language = opt_lang, refinement=opt_refinement, debug=opt_debug)
elif opt_model == "whisper-large":
    runWhisper("large", test, language = opt_lang, refinement=opt_refinement, debug=opt_debug)
elif opt_model == "afriwhisper":
    runAfriWhisper(test, language = opt_lang, refinement=opt_refinement, debug=opt_debug)
elif opt_model == "lelapa":
    runLelapa(test, language = opt_lang, refinement=opt_refinement, debug=opt_debug)
elif opt_model == "wav2vec":
    runWav2Vec(test)
elif opt_model == "deepspeech":
    runDeepSpeech(test)
elif opt_model == "all":
    runWhisper("medium", test, language = opt_lang, refinement=opt_refinement, debug=opt_debug)
    runWhisper("large", test, language = opt_lang, refinement=opt_refinement, debug=opt_debug)
    runAfriWhisper(test, language = opt_lang, refinement=opt_refinement, debug=opt_debug)
    runLelapa(test)
    runWav2Vec(test)
    runDeepSpeech(test)
else:
    raise ValueError(f"Invalid `opt_model`: {opt_model}")
