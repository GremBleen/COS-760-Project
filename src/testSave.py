from common import saveResults_V2

results_dict = {}

for i in range(10):  # Simulating 10 samples
    # Simulated results for each sample
    cer, wer = 0.01 * i, 0.02 * i
    results_dict[i] = (cer, wer)

saveResults_V2(
    results_dict=results_dict,
    language="afr",
    model="whisper-medium",
    refinement="none"
)