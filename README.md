# RAGent: Agentic RAG for File-Level Bug Localization


## Prerequisites
This repository uses Ollama-based LLMs. Make sure to have Ollama up and running following: https://docs.ollama.com/quickstart

Once Ollama is installed. Pull the LLMs from Ollama library. In our experiments, we use:

`gpt-oss-120b`: https://ollama.com/library/gpt-oss:120b

`qwen3-32b`: https://ollama.com/library/qwen3

These models are required to reproduce the results.

## Setup

Clone the repository and go to the directory
```bash
cd RAGent
```
Setup environment
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OLLAMA_HOST=http://localhost:11434
```
Install dependencies
```bash
pip install -r requirements.txt
```

## Preprocessing
For faster processing, we process SWE-bench-Lite dataset into json files. This can be done by running:

```bash
python preprocess/process_swe_dataset.py
```
It will process the repositories under `swebench_data` directory in the project root. It will create two json files for each repo.  

For example
```bash
swebench_data/swebench_0_raw.json # Containing all file_path and file contents for repository at index 0 of the dataset
swebench_data/swebench_0_py.json # Containing only python file (.py) paths and file contents for repository at index 0 of the dataset
```
Each repository is identified by the number of index in the dataset.

## Running the pipeline
To run the RAG pipeline on swebench dataset in different configuration:

Create a directory to store the results
```bash
mkdir retrieval_results
```

### Dense Retrieval

#### Dense Retrieval with Naive Text Chunking
```bash
python pipelines/rag_pipeline.py --splitter text --results_path retrieval_results/dense_retrieval_text_results.json --chroma_root chroma_data_text
```

#### Dense Retrieval with Code Chunking
```bash
python pipelines/rag_pipeline.py --splitter code --results_path retrieval_results/dense_retrieval_code_results.json --chroma_root chroma_data_code
```

#### Dense Retrieval with Path-aware Code Chunking
```bash
python pipelines/rag_pipeline.py --splitter code --results_path retrieval_results/dense_retrieval_code_with_path_results.json --chroma_root chroma_data_code_with_path --include_path
``` 

### Basic RAG Pipeline

#### Basic RAG Pipeline with Naive Text Chunking
```bash
python pipelines/rag_pipeline.py --splitter text --results_path retrieval_results/rag_text_results.json --chroma_root chroma_data_text --use_rag --llm gpt-oss:120b
```
#### Basic RAG Pipeline with Code Chunking
```bash
python pipelines/rag_pipeline.py --splitter code --results_path retrieval_results/rag_code_results.json --chroma_root chroma_data_code --use_rag --llm gpt-oss:120b
```

#### Basic RAG Pipeline with Path-aware Code Chunking
```bash
python pipelines/rag_pipeline.py --splitter code --results_path retrieval_results/rag_code_with_path_results.json --chroma_root chroma_data_code_with_path --include_path --use_rag  --llm gpt-oss:120b
```

### Query Transformation
```bash
python pipelines/query_transformation.py --results_path retrieval_results/query_transformation_results.json --chroma_root chroma_data_code_with_path --llm gpt-oss:120b
```

### Agentic Pipeline
Since the complete Agentic Pipeline (RAGent) uses query transformation, it is suggested that first query transformation pipeline is executed. Once it is executed we can simply use the output of the query transformation candidates as input.

```bash
python pipelines/agentic_pipeline.py --input_file retrieval_results/query_transformation_results.json --results_file retrieval_results/agentic_ranked_results.json --llm gpt-oss:120b
```

## Localization Results
We provide localization results under different agentic and non-agentic RAG configurations under [localization_results](localization_results). Accuracy metrics like MRR, and Top-k accuracy can be easily calculated by executing:

```bash
python evaluation/ranking_evaluation.py
```
Example output:
```bash
Evaluating report: Agentic RAG - T0+T1 (GPT-OSS 120B)

=== Overall Aggregate Statistics ===
+-------------------------+--------+
| Total Samples Evaluated |  300   |
|           MRR           | 0.7946 |
|          Acc@1          | 0.7100 |
|          Acc@3          | 0.8600 |
|          Acc@5          | 0.9033 |
|         Acc@10          | 0.9433 |
|         Acc@15          | 0.9533 |
+-------------------------+--------+
```

Similarly, evaluation can be ran for new results by updating the ranking_evaluation.py file pointing to the new report file:
```json
{
    "name": "New Agentic RAG",
    "path": "localization_results/new_agentic_ranked_results.json",
    "pred_list_name": "ranked_scores", # Should point to the accurate field name depending on retrieval technique (e.g., ranked_scores, rag_ranked_files, retrieved_files, etc.)
    "ground_truth": "patch_file", # Should be either "patch" or "patch_file"
},
```

