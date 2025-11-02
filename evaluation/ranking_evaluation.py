import json
import math
from statistics import mean

from tabulate import tabulate

from ragent.util.code_util import extract_patch_file_path, get_code_text_from_path

K_VALUES = [1, 3, 5, 10, 15]  # Cutoffs for evaluation


def ndcg_at_k(relevant_file, ranked_list, k=5):
    """Compute NDCG when there's a single relevant document."""
    try:
        rank = ranked_list.index(relevant_file)
        if rank >= k:
            return 0.0
        return 1 / math.log2(rank + 2)
    except ValueError:
        return 0.0


def evaluate_retrieval(patch_file_path, retrieved_files, top_k=K_VALUES):
    result = {}
    try:
        rank = retrieved_files.index(patch_file_path) + 1
        result["rank"] = rank
        result["reciprocal_rank"] = 1 / rank
    except ValueError:
        result["rank"] = None
        result["reciprocal_rank"] = 0.0

    for k in top_k:
        in_top_k = patch_file_path in retrieved_files[:k]
        result[f"top_{k}"] = in_top_k
        # result[f"ndcg@{k}"] = ndcg_at_k(patch_file_path, retrieved_files, k)
        # result[f"precision@{k}"] = (1.0 / k) if in_top_k else 0.0
        result[f"recall@{k}"] = 1.0 if in_top_k else 0.0

    return result


def compute_aggregate_stats(all_metrics, k_values=K_VALUES):
    stats = {}
    stats["Total Samples Evaluated"] = len(all_metrics)
    stats["MRR"] = (
        mean(m.get("reciprocal_rank", 0.0) for m in all_metrics) if all_metrics else 0.0
    )

    for k in k_values:
        stats[f"Acc@{k}"] = (
            mean(m.get(f"top_{k}", 0.0) for m in all_metrics) if all_metrics else 0.0
        )
        # stats[f"NDCG@{k}"] = (
        #     mean(m.get(f"ndcg@{k}", 0.0) for m in all_metrics) if all_metrics else 0.0
        # )
        # stats[f"Precision@{k}"] = (
        #     mean(m.get(f"precision@{k}", 0.0) for m in all_metrics)
        #     if all_metrics
        #     else 0.0
        # )
        # stats[f"Recall@{k}"] = (
        #     mean(m.get(f"recall@{k}", 0.0) for m in all_metrics) if all_metrics else 0.0
        # )

    return stats


def print_overall_stats(stats, k_values=K_VALUES):
    overall_table = [
        ["Total Samples Evaluated", stats["Total Samples Evaluated"]],
        ["MRR", f"{stats['MRR']:.4f}"],
    ]

    # for k in k_values:
    #     overall_table.append([f"Acc@{k}", f"{stats[f'Acc@{k}']:.4f}"])
    # for k in k_values:
    #     overall_table.append([f"NDCG@{k}", f"{stats[f'NDCG@{k}']:.4f}"])
    # for k in k_values:
    #     overall_table.append([f"Precision@{k}", f"{stats[f'Precision@{k}']:.4f}"])
    for k in k_values:
        overall_table.append([f"Acc@{k}", f"{stats[f'Acc@{k}']:.4f}"])

    print("\n=== Overall Aggregate Statistics ===")
    print(tabulate(overall_table, tablefmt="pretty"))


def evaluate_predictions(
    predictions, pred_list_name, ground_truth, k_values=K_VALUES, limit=None
):
    all_metrics = []

    if limit:
        predictions = predictions[:limit]
    failed_cases = 0
    cases_where_retrieval_failed = 0
    for i, pred in enumerate(predictions):
        patch = None
        if ground_truth == "patch":
            patch = extract_patch_file_path(pred.get("patch", ""))
        elif ground_truth == "patch_file":
            patch = pred.get("patch_file", "")
        else:
            raise ValueError(f"Unknown ground_truth type: {ground_truth}")

        ranked_files = []

        if pred_list_name == "ranked_scores":
            ranked_files = pred.get("ranked_scores", {})
            ranked_files = [
                file for file, _ in sorted(ranked_files.items(), key=lambda x: -x[1])
            ]
        else:
            ranked_files = pred.get(pred_list_name, [])
        # ranked_files = pred.get("retrieved_files_base", [])
        # ranked_files = pred.get("retrieved_files_t1", [])
        patch_file_path = patch

        if patch_file_path not in ranked_files[:10]:
            failed_cases += 1
            t0_files = pred.get("retrieved_files_t0", [])
            t1_files = pred.get("retrieved_files_t1", [])
            if (
                patch_file_path not in t0_files[:15]
                and patch_file_path not in t1_files[:15]
            ):
                cases_where_retrieval_failed += 1

        if not patch_file_path:
            print(f"[Warning] Could not extract patch file in item {i}")
            continue

        metrics = evaluate_retrieval(patch_file_path, ranked_files, top_k=k_values)
        all_metrics.append(metrics)

    stats = compute_aggregate_stats(all_metrics, k_values)
    print_overall_stats(stats, k_values)
    print(
        f"Cases where retrieval failed to include the patch file in top 10: {cases_where_retrieval_failed}/{failed_cases}"
    )
    return all_metrics


if __name__ == "__main__":

    reports = [
        {
            "name": "Agentic RAG - T0+T1 (GPT-OSS 120B)",
            "path": "localization_results/agentic_gpt-oss_120b_temp_0.7_ranked_results.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "name": "Agentic RAG - T0+T1 (QWEN3:32B)",
            "path": "localization_results/agentic_qwen3-32bb_temp_0.7_ranked_results.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "name": "Agentic RAG - T0 (GPT-OSS 120B)",
            "path": "localization_results/agentic_t0_retrieval_gpt-oss_120b_ranked_results_t0.5.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "name": "Agentic RAG - T1 (GPT-OSS 120B)",
            "path": "localization_results/agentic_t1_retrieval_gpt-oss_120b_ranked_results_t0.5.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "name": "Agentic RAG - No Query Transformation (GPT-OSS 120B)",
            "path": "localization_results/agentic_base_retrieval_gpt-oss_120b_ranked_results_t0.5.json",
            "pred_list_name": "ranked_scores",
            "ground_truth": "patch_file",
        },
        {
            "name": "RAG - Text Splitter",
            "path": "localization_results/rag_results_text_splitter.json",
            "pred_list_name": "rag_ranked_files",
            "ground_truth": "patch",
        },
        {
            "name": "RAG - Code Splitter",
            "path": "localization_results/rag_results_code_splitter.json",
            "pred_list_name": "rag_ranked_files",
            "ground_truth": "patch",
        },
        {
            "name": "RAG - File Path Aware Code Splitter",
            "path": "/local/home/amamun/projects/RAGent/localization_results/rag_results_file_path_aware_code_splitter.json",
            "pred_list_name": "rag_ranked_files",
            "ground_truth": "patch",
        },
        {
            "name": "Dense Retrieval - File Path Aware Code Splitter",
            "path": "localization_results/dense_retrieval_result_with_path_aware_chunking.json",
            "pred_list_name": "retrieved_files_base",
            "ground_truth": "patch_file",
        },
    ]

    for report in reports:
        print(f"\nEvaluating report: {report['name']}")
        with open(report["path"], "r") as f:
            predictions = json.load(f)
        evaluate_predictions(
            predictions,
            pred_list_name=report["pred_list_name"],
            ground_truth=report["ground_truth"],
            # limit=86,
        )
