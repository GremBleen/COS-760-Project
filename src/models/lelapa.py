import torch
from transformers import AutoProcessor, AutoModelForCTC
import torchaudio
from math import ceil
from common import evaluateTranscription, resample, saveResults


def runLoop(processor, model, dataset, language, batch_size, refinement=False, debug=False):
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

    # Character Error Rate (CER) and Word Error Rate (WER) initialisation
    cer = 0
    wer = 0

    results_dict = {}

    for i in range(len(dataset)):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(dataset))

        batch_audio = []
        batch_transcript = []

        for j in range(start_index, end_index):
            sample = dataset[j]
            waveform = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]
            transcript = sample["text"]
            resampled = resample(waveform, sample_rate, 16000)
            batch_audio.append(resampled)
            batch_transcript.append(transcript)

        inputs = processor(
            batch_audio,
            sampling_rate=sample_rate,  # Ensuring that using 16kHz
            return_tensors="pt",
            padding=True,  # Making all audio files the same length
            return_attention_mask=True,
        )

        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Inference (no training)
        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the prediction to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        reference_text = ""
        predicted_text = ""

        for instance, pred in zip(batch_transcript, transcription):
            reference_text += instance + "\n"
            predicted_text += pred + "\n"

        # Evaluate the transcription
        temp_cer, temp_wer = evaluateTranscription(
            reference_text=reference_text, predicted_text=predicted_text, output=debug
        )

        results_dict[i] = (temp_cer, temp_wer)

        cer += temp_cer
        wer += temp_wer

    cer /= num_batches
    wer /= num_batches

    # saveResults_V1(cer, wer, language=language, model="lelapa", refinement=refinement)
    saveResults(results_dict, language=language, model="lelapa", refinement=refinement)

    return cer, wer


def runLelapa(test, batch_size=20, language="xho", refinement=False, debug=False):
    run_model = "lelapa/mms-1b-fl102-xho-15"
    print(f"Running on {run_model}")

    processor = AutoProcessor.from_pretrained(run_model)
    model = AutoModelForCTC.from_pretrained(run_model)
    model.config.forced_decoder_ids = None  # Disable forced decoder ids for this model

    runLoop(
        processor=processor,
        model=model,
        dataset=test,
        language=language,
        batch_size=batch_size,
        refinement=refinement,
        debug=debug,
    )

    print(f"End of run for {run_model}")
