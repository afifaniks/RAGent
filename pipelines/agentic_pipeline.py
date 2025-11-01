import argparse
import json
import sys
import time

from datasets import load_dataset

from ragent.agent.ranker_agent import SWEBenchRankerAgent

MAX_RETRIES = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SWE-bench retrieval results.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature setting for the ranker agent.",
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-oss:120b",
        help="Model name for the LLM.",
    )

    args = parser.parse_args()

    RESULTS_FILE = f"/local/home/amamun/projects/rag_fix/paper_results/agentic_time_test_full_code_{args.llm.replace(':', '_')}_ranked_results_t{args.temperature}.json"

    print("Results will be stored in:", RESULTS_FILE)
    # Load retrieval results (contains swe_data_index and other fields)
    with open(
        "/local/home/amamun/projects/rag_fix/paper_results/agentic_gpt-oss_120b_temp_0.7_ranked_results.json",
        "r",
    ) as f:
        all_retrieved_docs = json.load(f)
        all_retrieved_docs = all_retrieved_docs[:10]

    # Try loading existing results so we can resume
    try:
        with open(RESULTS_FILE, "r") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = []

    # Convert to set of processed indexes for quick lookup
    processed_indexes = {entry["swe_data_index"] for entry in all_results}

    for entry in all_retrieved_docs:
        swe_index = entry["swe_data_index"]

        if swe_index in processed_indexes:
            print(f"Skipping index {swe_index} (already processed).")
            continue

        print(f"\n=== Processing index {swe_index} ===")

        # Load repo content for this index
        with open(f"./swebench_data/swebench_{swe_index}_py.json", "r") as f:
            repo_content = json.load(f)

        retrieved_docs = entry  # This already has all the metadata

        # Retry logic
        attempt = 0
        scores = None
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                start_time = time.time()
                print(f"Attempt {attempt} for index {swe_index}...")
                ranker = SWEBenchRankerAgent(
                    repo_content=repo_content,
                    retrieved_docs=retrieved_docs,
                    model_name=args.llm,
                    temperature=args.temperature,
                )
                scores = ranker.run()
                time_taken = time.time() - start_time
                print(f"Time taken: {time_taken:.2f} seconds")
                if isinstance(scores, dict) and scores:
                    break
            except Exception as e:
                time_taken = time.time() - start_time
                print(f"Error in attempt {attempt} for index {swe_index}: {e}")
                print(f"Time taken before error: {time_taken:.2f} seconds")
                exit()
            time.sleep(2)  # small delay between retries

        # Save results if successful
        if isinstance(scores, dict) and scores:
            result_entry = {
                **entry,  # keep all original SWE-bench + retrieval data
                "ranked_scores": scores,
                "time_taken_seconds": time_taken,
            }
            all_results.append(result_entry)

            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"✅ Saved results for index {swe_index}")
        else:
            print(
                f"❌ Failed to get results for index {swe_index} after {MAX_RETRIES} attempts"
            )
