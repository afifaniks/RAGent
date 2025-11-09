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

### APR Setup
To setup Agentless, you can either follow their official documentation where they use proprietary LLM like GPT-4o: https://github.com/OpenAutoCoder/Agentless/blob/main/README_swebench.md

Or, if you want to use it with Ollama, you can use our ported version of Agentless. The instruction is provided later in this documentation.

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

# Repair with Agentless
First, we have to convert the RAGent file-level localization output to suitable Agentless input file in `jsonl` format. For example, if we want to use our previous result file [localization_results/agentic_gpt-oss_120b_temp_0.7_ranked_results.json](localization_results/agentic_gpt-oss_120b_temp_0.7_ranked_results.json), we can do this:

```bash
python preprocess/convert_agentless.py --ragent_output localization_results/agentic_gpt-oss_120b_temp_0.7_ranked_results.json --converted_output ragent_locs.jsonl
```

This will produce `ragent_locs.jsonl` file which is compatible to use in Agentless.

From this stage on https://github.com/OpenAutoCoder/Agentless/blob/main/README_swebench.md, we can follow the Agentless documentation to run the APR specifically from [2. localize to related elements](https://github.com/OpenAutoCoder/Agentless/blob/main/README_swebench.md#2-localize-to-related-elements) stage. However, since our ported Agentless uses Ollama as a backend, we just have to change the LLM and backend whenever an LLM is required to complete an agentless stage.

For example, to localize at the function level using `ragent_locs.jsonl` and with the Ollama-ported Agentless, we can use this:

```bash
python agentless/fl/localize.py --related_level \
                                --output_folder results/swe-bench-lite/related_elements_gpt_oss \
                                --top_n 3 \
                                --compress_assign \
                                --compress \
                                --start_file {DIRECTORY}/ragent_locs.jsonl \                                
                                --num_threads 10 \
                                --model gpt-oss:120b \
                                --backend ollama \
```                                

The last two parameters `(--model and --backend)` are particularly important to enable ollama, otherwise the framework will use its default LLM GPT-4o.

Similarly, for all subsequent steps should follow the official Agentless documentation along with the mentioned parameters.

## Patch Evaluation
Once patches are generated, they can be evaluated using swebench library. We recommend creating a new environment with [swebench 4+ (latest)](https://pypi.org/project/swebench/) as swebench 2.1 used by Agentless is now outdated and we ran into many problems to evaluate with it. The run command is simple:

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path repair_results/run_1/ragent/patches/ragent_majority_voting_and_regression_and_reproduction.jsonl \
    --max_workers 10 \
    --run_id ragent_evaluation
```
Due to the limit to github file size, we provide only the final generated patches on swebench using ragent localization here: [repair_results/run_1/ragent/patches/ragent_majority_voting_and_regression_and_reproduction.jsonl](repair_results/run_1/ragent/patches/ragent_majority_voting_and_regression_and_reproduction.jsonl). 

However, all the intermediate files/outputs using both agentless localization and ragent can be downloaded and evaluated from here: https://drive.google.com/file/d/162p85cxYgGXB2szqkvjwP4CBh7RDuyq2/view?usp=sharing

## Additional Materials

We also provide additional materials like scripts to generate results/diagrams/notebooks under [evaluation/](evaluation/) directory.

For example, [evaluation/analysis.py](evaluation/analysis.py) analyzes how many unique repairs produced by each method fail in the other method due to incorrect localization. Similarly, [evaluation/analyze_incorrect_patch_with_correct_loc.ipynb](evaluation/analyze_incorrect_patch_with_correct_loc.ipynb) explores in which stage (e.g., line-level localization, patch generation) of the APR a particular instance fails, etc.

## Acknowledgements
We thank [Agentless](https://github.com/OpenAutoCoder/Agentless/tree/main) and [CoSIL](https://github.com/ZhonghaoJiang/CoSIL/tree/master) for their work and making it public for others to use.