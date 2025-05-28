from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
from math import ceil
import torch
import torchaudio

from common import evaluateTranscription

def runSM4T(dataset, language=None, batch_size=20, refinement=False, debug=False):
    run_model = "facebook/seamless-m4t-v2-large"

    print(f"Running {language} on {run_model} with batch size {batch_size}")

    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large")

