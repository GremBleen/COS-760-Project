import csv
import os

# Change the below to change the file
filename = "af_whisper-large_none_20250604_171212.csv"

file = "pre-refinement-runs" + os.sep + filename

cer_accumulator = 0
wer_accumulator = 0
num_rows = 0

with open(file, "r") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    reader = csv.DictReader(csvfile, fieldnames=header)
    for row in reader:
        cer_accumulator += float(row["CER"])
        wer_accumulator += float(row["WER"])
        num_rows = int(row["Batch"])

    cer_mean = cer_accumulator / num_rows
    wer_mean = wer_accumulator / num_rows
    
    cer_stdev = 0
    wer_stdev = 0

    csvfile.seek(0)
    next(reader)
    reader = csv.DictReader(csvfile, fieldnames=header)
    for row in reader:
        cer_stdev += (float(row["CER"]) - cer_mean)**2
        wer_stdev += (float(row["WER"]) - wer_mean)**2

    cer_stdev = cer_stdev / num_rows
    wer_stdev = wer_stdev / num_rows

print(f"CER Mean: {cer_mean:.4f}, CER Stdev: {cer_stdev:.4f}")
print(f"WER Mean: {wer_mean:.4f}, WER Stdev: {wer_stdev:.4f}")
