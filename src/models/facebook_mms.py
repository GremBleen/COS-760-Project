from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

def getLanguageCode(language):
    language_codes = {
        "afr": "af",
        "xho": "xh",
        "zul": "zu",
        "ven": "ve",
        "tso": "ts",
        "tsn": "tn",
        "ssw": "ss",
        "nso": "nso",
        "sot": "st",
    }
    return language_codes.get(language, None)

def runLoop(processor, model, dataset, refinement=False, debug=False):


def runFacebookMMS(dataset, language=, refinement=False, debug=False):
    