from math import ceil
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
import torch
from common import evaluateTranscription, resample, saveResults


def getLanguageCode(language):
    language_codes = {"afr": "afr", "zul": "zul"}
    return language_codes.get(language, None)


def runSM4T(dataset, language=None, batch_size=20, refinement=False, debug=False):
    run_model = "facebook/seamless-m4t-v2-large"

    print(f"Running {language} on {run_model} with batch size {batch_size}")

    language = getLanguageCode(language)

    if language is not None:
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            "facebook/seamless-m4t-v2-large"
        )

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

        results_dict = {}

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

            # TODO - this needs to be changed to accept src and target languages correctly - it also appears to be seeing text and not
            inputs = processor(
                audios=batch_audio,
                src_lang=language,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )

            input_values = inputs.input_features.to(device)
            attention_mask = inputs.attention_mask.to(device)

            with torch.no_grad():
                output_tokens = model.generate(
                    input_features=input_values,
                    attention_mask=attention_mask,
                    tgt_lang=language,
                )

            transcription = processor.batch_decode(
                output_tokens, skip_special_tokens=True
            )

            reference_text = ""
            predicted_text = ""

            for instance in zip(batch_transcript, transcription):
                reference_text += instance[0] + "\n"
                predicted_text += instance[1] + "\n"

            # We are getting the error over the whole dataset so that prompts to not have a disproportionate effect on the results
            temp_cer, temp_wer = evaluateTranscription(
                reference_text=reference_text,
                predicted_text=predicted_text,
                output=debug,
            )

            results_dict[i] = (temp_cer, temp_wer)

            cer += temp_cer
            wer += temp_wer

        cer /= num_batches
        wer /= num_batches

        saveResults(
            results_dict=results_dict,
            language=language,
            model="sm4t",
            refinement=refinement,
        )

        print(f"End of run for {run_model}")
        return cer, wer
    else:
        raise ValueError(f"Language {language} is not supported for SM4T")
