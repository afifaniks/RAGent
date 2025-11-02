import argparse
import json

from datasets import load_dataset

# Load SWE-bench Lite (test split)
swebench = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

parser = argparse.ArgumentParser()
parser.add_argument("--ragent_output", type=str)
parser.add_argument("--converted_output", type=str)
args = parser.parse_args()

ragent_output = args.ragent_output
converted_output = args.converted_output

with open(ragent_output, "r") as f:
    ragent_predictions = json.load(f)

# Write JSONL
with open(converted_output, "w") as out_f:
    for idx, data in enumerate(swebench):
        instance_id = data["instance_id"]
        found_files = list(ragent_predictions[idx]["ranked_scores"].keys())
        additional_artifact_loc_file = {}
        file_traj = {}

        # Build dict for JSONL line
        record = {
            "instance_id": instance_id,
            "found_files": found_files,
            "additional_artifact_loc_file": additional_artifact_loc_file,
            "file_traj": file_traj,
        }

        out_f.write(json.dumps(record) + "\n")
