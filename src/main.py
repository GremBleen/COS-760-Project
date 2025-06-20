from dotenv import load_dotenv
import os
from huggingface_hub import login

from models.whisper import runWhisper, runAfriWhisper
from models.lelapa import runLelapa
from models.wav2vec import runWav2Vec
from models.facebook_mms import runFacebookMMS
from models.SM4T import runSM4T

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

opt_lang = presets[
    "dataset_language"
]  # 'afr', 'xho', 'zul', 'ven', 'tso', 'tsn', 'ssw', 'nso', 'sot'
opt_model = presets[
    "model"
]  # 'whisper-medium', 'whisper-large', 'afriwhisper', 'lelapa', 'wav2vec', 'all'
opt_batch_size = presets["batch_size"]
opt_refinement = presets["refinement_method"]
opt_debug = presets["debug"]

# This is getting the dataset specified by `opt_lang` which takes a while
test = getDataset(opt_lang)

if opt_model == "whisper-medium":
    runWhisper(
        "medium",
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
elif opt_model == "whisper-large":
    runWhisper(
        "large",
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
elif opt_model == "afriwhisper":
    runAfriWhisper(
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
elif opt_model == "lelapa":
    runLelapa(
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
elif opt_model == "facebook-mms":
    runFacebookMMS(
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
elif opt_model == "wav2vec":
    runWav2Vec(
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
elif opt_model == "sm4t":
    runSM4T(
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
elif opt_model == "all":
    runWhisper(
        "medium",
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
    runWhisper(
        "large",
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
    runAfriWhisper(
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
    runLelapa(
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
    runFacebookMMS(
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
    runWav2Vec(
        test,
        batch_size=opt_batch_size,
    )
    runSM4T(
        test,
        batch_size=opt_batch_size,
        language=opt_lang,
        refinement=opt_refinement,
        debug=opt_debug,
    )
else:
    raise ValueError(f"Invalid `opt_model`: {opt_model}")
