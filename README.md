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

#### Dense Retrieval with Naive Text Chunking
```bash
python pipelines/rag_pipeline.py --splitter text --results_path retrieval_results/dense_retrieval_text_results.json
```



