import torch
from transformers import AutoProcessor, AutoModelForCTC
import torchaudio
from math import ceil

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
        "sot": "st"
    }
    return language_codes.get(language, None)

def runLoop(processor, model, dataset, refinement=False, debug=False):
    if hasattr(torch.backends, "mps"):
        try:
            has_mps = torch.backends.mps.is_available()
        except (AttributeError, RuntimeError):
            has_mps = False
    else:
        has_mps = False

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    )
    model = model.to(device)

    # Using mini-batching to make it faster


def runLelapaLoop(processor, model, dataset):
    if hasattr(torch.backends, "mps"):
        try:
            has_mps = torch.backends.mps.is_available()
        except (AttributeError, RuntimeError):
            has_mps = False
    else:
        has_mps = False

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    )
    model = model.to(device)
    model.eval()

    for instance in dataset:
        sample = instance["audio"]
        inputs = processor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(device)

        # Inference (no training)
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the prediction to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # Logging
        print(f"Sample rate: {sample['sampling_rate']}")
        print(f"Reference: {instance['text']}")
        print(f"Prediction: {transcription[0]}")

    # TODO - change this to return metrics
    return 0


def runLelapa(test, language="xho", refinement=False, debug=False):
    run_model = "lelapa/mms-1b-fl102-xho-15"
    print(f"Running on {run_model}")

    processor = AutoProcessor.from_pretrained(run_model)
    model = AutoModelForCTC.from_pretrained(run_model)
    model.config.forced_decoder_ids = (
        None # Disable forced decoder ids for this model
    )

    runLoop(processor=processor, model=model, dataset=test, refinement=refinement, debug=debug)

    print(f"End of run for {run_model}")
