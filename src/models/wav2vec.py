from math import ceil
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

from common import evaluateTranscription, resample


def getLanguageCode(language):
    language_codes = {
        "afr": "afr",
        "xho": "xho",
        "zul": "zul",
        "ven": "ven",
        "tso": "tso",
        "tsn": "tsn",
        "ssw": "ssw",
        "nso": "nso",
        "sot": "sot",
    }
    return language_codes.get(language, None)


def runWav2Vec(dataset, language=None, batch_size=20, refinement=False, debug=False):
    run_model = "guymandude/MMS-ASR-ZA-11"

    print(f"Running {language} on {run_model} with batch size {batch_size}")

    processor = Wav2Vec2Processor.from_pretrained(run_model)
    model = Wav2Vec2ForCTC.from_pretrained(run_model)

    language = getLanguageCode(language)

    if language is not None:
        model.load_adapter(language)
        processor.tokenizer.set_target_lang(language)

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
                resampled = resample(waveform, sample_rate, 16000)
                batch_audio.append(resampled)
                batch_transcript.append(transcript)

            inputs = processor(
                batch_audio, return_tensors="pt", padding=True, sampling_rate=16000
            )

            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)

            with torch.no_grad():
                logits = model(input_values, attention_mask=attention_mask).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            reference_text = ""
            predicted_text = ""

            for instance in zip(batch_transcript, transcription):
                reference_text += instance[0] + "\n"
                predicted_text += instance[1] + "\n"
            temp_cer, temp_wer = evaluateTranscription(
                reference_text=reference_text,
                predicted_text=predicted_text,
                output=debug,
            )  # We are getting the error over the whole dataset so that prompts to not have a disproportionate effect on the results
            cer += temp_cer
            wer += temp_wer
        cer /= num_batches
        wer /= num_batches

        print(f"End of run for {run_model}")
        return cer, wer

    else:
        raise ValueError(f"Language {language} is not supported for wav2vec")
