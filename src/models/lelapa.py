import torch
from transformers import AutoProcessor, AutoModelForCTC


def runLelapaLoop(processor, model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    sample = dataset[0]["audio"]
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
    print(f"Reference: {dataset[0]['text']}")
    print(f"Prediction: {transcription[0]}")

    return transcription[0]


def runLelapa(test):
    run_model = "lelapa/mms-1b-fl102-xho-15"
    print(f"Running on {run_model}")

    processor = AutoProcessor.from_pretrained(run_model)
    model = AutoModelForCTC.from_pretrained(run_model)

    runLelapaLoop(processor=processor, model=model, dataset=test)

    print(f"End of run for {run_model}")
