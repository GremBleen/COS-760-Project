from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

def runLoop(processor, model, dataset):

    if hasattr(torch.backends, "mps"):
        try:
            has_mps = torch.backends.mps.is_available()
        except (AttributeError, RuntimeError):
            has_mps = False
    else:
        has_mps = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu")
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
