from transformers import WhisperProcessor, WhisperForConditionalGeneration
from main import runLoop

# TODO - may need to resample data as whisper may need it in a different format

"""
This was included to determine if AfriWhisper is better
"""


def runWhisperMedium(test):
    run_model = "openai/whisper-medium"
    print(f"Running on {run_model}")

    processor = WhisperProcessor.from_pretrained(run_model)
    model = WhisperForConditionalGeneration.from_pretrained(run_model)
    model.config.forced_decoder_ids = None

    runLoop(processor=processor, model=model, dataset=test)

    print(f"End of run for {run_model}")


"""
This makes use of whisper medium
"""


def runAfriWhisper(test):
    run_model = "intronhealth/afrispeech-whisper-medium-all"
    print(f"Running on {run_model}")

    processor = WhisperProcessor.from_pretrained(run_model)
    model = WhisperForConditionalGeneration.from_pretrained(run_model)
    model.config.forced_decoder_ids = None

    runLoop(processor=processor, model=model, dataset=test)

    print(f"End of run for {run_model}")


"""
This is the current flagship, this is included to determine if
the context provided by AfriWhisper results in an improvement
over the flagship
"""


def runWhisperLargeV3(test):
    run_model = "openai/whisper-large-v3"
    print(f"Running on {run_model}")

    processor = WhisperProcessor.from_pretrained(run_model)
    model = WhisperForConditionalGeneration.from_pretrained(run_model)
    model.config.forced_decoder_ids = None

    runLoop(processor=processor, model=model, dataset=test)

    print(f"End of run for {run_model}")
