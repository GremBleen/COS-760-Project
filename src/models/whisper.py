from transformers import WhisperProcessor, WhisperForConditionalGeneration
from common import evaluateTranscription
import torch

def runLoop(processor, model, dataset):
    total_reference = ""
    total_hypothesis = ""
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
    model.forced_decoder_ids = None # TODO - change this to see if it improves results

    for instance in dataset:
        sample = instance["audio"]

        input_features = processor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
        ).input_features.to(device)

        # Since not training, it is not necessary to include gradients
        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        curr_reference = instance["text"]
        curr_transcription = ""
        for word in transcription:
            curr_transcription += word + " "

        evaluateTranscription(
            reference_text=curr_reference,
            predicted_text=curr_transcription,
            print=True
        )

        total_reference += curr_reference + "\n"
        total_hypothesis += curr_transcription + "\n"

    return evaluateTranscription(
        reference_text=total_reference, predicted_text=total_hypothesis
    ) # We are returning the error over the whole dataset so that prompts to not have a disproportionate effect on the results

# TODO - may need to resample data as whisper may need it in a different format


def runWhisperMedium(test):
    run_model = "openai/whisper-medium"
    print(f"Running on {run_model}")

    processor = WhisperProcessor.from_pretrained(run_model)
    model = WhisperForConditionalGeneration.from_pretrained(run_model)
    model.config.forced_decoder_ids = None

    runLoop(processor=processor, model=model, dataset=test)

    print(f"End of run for {run_model}")


def runAfriWhisper(test):
    run_model = "intronhealth/afrispeech-whisper-medium-all"
    print(f"Running on {run_model}")

    processor = WhisperProcessor.from_pretrained(run_model)
    model = WhisperForConditionalGeneration.from_pretrained(run_model)
    model.config.forced_decoder_ids = None

    runLoop(processor=processor, model=model, dataset=test)

    print(f"End of run for {run_model}")


def runWhisperLargeV3(test):
    run_model = "openai/whisper-large-v3"
    print(f"Running on {run_model}")

    processor = WhisperProcessor.from_pretrained(run_model)
    model = WhisperForConditionalGeneration.from_pretrained(run_model)
    model.config.forced_decoder_ids = None

    runLoop(processor=processor, model=model, dataset=test)

    print(f"End of run for {run_model}")
