from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from common import evaluateTranscription
import torch
import torchaudio
from math import ceil

# Explanation:
# - model.config.forced_decoder_ids is used to set the language and task for the Whisper model. By setting it to None, it forces the model into a multi-language mode.
# - Padding is used to ensure that all audio files within a batch are of the same length
# - The attention mask then identifies which tokens within an audio file needs to be processed - ignoring the padded tokens.

# This is mapping to the available languages in the Whisper model to try improve results
def getLanguageCode(language):
    language_codes = {
        "afr": "af",
    }
    return language_codes.get(language, None)


def runLoop(processor, model, dataset, batch_size = 20, refinement=False, debug=False):
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
    num_batches = ceil(len(dataset) / batch_size)

    cer = 0
    wer = 0

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(dataset))

        batch_audio = []
        batch_transcript = []

        for j in range(start_index, end_index):
            sample = dataset[j]
            waveform = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]
            transcript = sample["text"]
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )  # ensuring that using 16kHz
                resampled = resampler(
                    torch.tensor(waveform, dtype=torch.float32)
                ).numpy()
                batch_audio.append(resampled)
            else:
                batch_audio.append(waveform)
            batch_transcript.append(transcript)

        inputs = processor(
            batch_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,  # making all audio files the same length
            return_attention_mask=True,
        )

        input_features = inputs.input_features.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Since not training, it is not necessary to include gradients
        with torch.no_grad():
            forced_decoder_ids = model.config.forced_decoder_ids
            if forced_decoder_ids is None:
                predicted_ids = model.generate(
                    input_features=input_features, attention_mask=attention_mask
                )
            else:
                predicted_ids = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    forced_decoder_ids=forced_decoder_ids,
                )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        reference_text = ""
        predicted_text = ""

        for instance in zip(batch_transcript, transcription):
            reference_text += instance[0] + "\n"
            predicted_text += instance[1] + "\n"

        # We are getting the error over the whole dataset so that prompts to not have a disproportionate effect on the results
        temp_cer, temp_wer = evaluateTranscription(
            reference_text=reference_text, predicted_text=predicted_text, output=debug
        )
        cer += temp_cer
        wer += temp_wer

    cer /= num_batches
    wer /= num_batches

    return cer, wer


def runWhisper(model, test, batch_size = 20, language=None, refinement=False, debug=False):
    if model == "medium":
        run_model = "openai/whisper-medium"
    elif model == "large":
        run_model = "openai/whisper-large-v3"
    else:
        raise ValueError("Invalid model specified. Use 'medium' or 'large'.")

    print(f"Running {language} on {run_model} with batch size {batch_size}")

    processor = WhisperProcessor.from_pretrained(run_model)
    model = WhisperForConditionalGeneration.from_pretrained(run_model)
    language = getLanguageCode(language)
    if language is None:
        model.config.forced_decoder_ids = None
    else:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

    cer, wer = runLoop(
        processor=processor,
        model=model,
        dataset=test,
        batch_size=batch_size,
        refinement=refinement,
        debug=debug,
    )

    print(f"End of run for {run_model}")
    return cer, wer


def runAfriWhisper(test, batch_size = 20, language=None, refinement=False, debug=False):
    run_model = "intronhealth/afrispeech-whisper-medium-all"
    print(f"Running {language} on {run_model} with batch size {batch_size}")

    processor = WhisperProcessor.from_pretrained(run_model)
    model = WhisperForConditionalGeneration.from_pretrained(run_model)
    model.config.forced_decoder_ids = (
        None  # Setting the language is not supported in this model
    )
    model.config.suppress_tokens = None  # Suppressing special tokens - for the AfriSpeech model, the token list is empty causing an error when trying to run if not set to None

    cer, wer = runLoop(
        processor=processor,
        model=model,
        dataset=test,
        batch_size=batch_size,
        refinement=refinement,
        debug=debug,
    )

    print(f"End of run for {run_model}")
    return cer, wer
