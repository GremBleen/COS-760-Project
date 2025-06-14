from math import ceil
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

from common import evaluateTranscription, getWordList, resample, saveResults


def getLanguageCode(language):
    language_codes = {
        "afr": "afr",
        "xho": "xho",
        "zul": "zul",
    }
    return language_codes.get(language, None)


def runLoop(
    processor, model, dataset, language, batch_size, refinement=False, debug=False
):
    if hasattr(torch.backends, "mps"):
        try:
            has_mps = torch.backends.mps.is_available()
        except (AttributeError, RuntimeError):
            has_mps = False
    else:
        has_mps = False

    # device = torch.device(
    #     "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    # )
    device = torch.device("cpu")
    model = model.to(device)

    # Using mini-batching to make it faster
    num_batches = ceil(len(dataset) / batch_size)

    # Character Error Rate (CER) and Word Error Rate (WER) initialisation
    cer = 0
    wer = 0

    results_dict = {}

    if refinement is not False:
        from common import getWordList

        word_list = getWordList(language)

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
            if refinement is not False:
                from common import trimSilence
                # Trim silence only if refinement is enabled
                resampled = trimSilence(resampled, 16000)
            batch_audio.append(resampled)
            batch_transcript.append(transcript)

        language_code = getLanguageCode(language)
        if language_code is None:
            raise ValueError(
                f"Unsupported language: {language}. Not supported by the facebook MMS model."
            )

        processor.tokenizer.set_target_lang(getLanguageCode(language))
        model.load_adapter(getLanguageCode(language))

        inputs = processor(
            batch_audio,
            sampling_rate=16000,  # Ensuring that using 16kHz
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

        # If refinement is enabled, refine the predicted text
        if refinement is not False:
            from common import refinementMethod

            predicted_text = refinementMethod(
                predicted_text, refinement=refinement, word_list=word_list
            )

        # Evaluate the transcription
        temp_cer, temp_wer = evaluateTranscription(
            reference_text=reference_text,
            predicted_text=predicted_text,
            batch_num=i,
            output=debug,
        )

        results_dict[i] = (temp_cer, temp_wer)

        cer += temp_cer
        wer += temp_wer

    cer /= num_batches
    wer /= num_batches

    # saveResults_V1(cer, wer, language=language, model="lelapa", refinement=refinement)
    saveResults(
        results_dict=results_dict,
        language=language,
        model="facebook-mms",
        refinement=refinement,
    )

    return cer, wer


def runFacebookMMS(dataset, language, batch_size=20, refinement=False, debug=False):
    run_model = "facebook/mms-1b-fl102"
    print(f"Running on {run_model}")

    processor = AutoProcessor.from_pretrained(run_model)
    model = Wav2Vec2ForCTC.from_pretrained(run_model)

    # print(processor.tokenizer.vocab.keys())
    runLoop(
        processor=processor,
        model=model,
        dataset=dataset,
        refinement=refinement,
        language=language,
        batch_size=batch_size,
        debug=debug,
    )

    print(f"End of run for {run_model}")
