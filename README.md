# UMSI RAG System

A Retrieval-Augmented Generation (RAG) system for answering questions about the University of Michigan School of Information (UMSI) using a corpus of program information, handbooks, course details, and other UMSI-related documents.

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer questions about UMSI. The system retrieves relevant information from a corpus of documents and generates natural language answers. Multiple retrieval and generation strategies are implemented and evaluated.

Key features include:
- Document preprocessing and chunking for both text and PDF files
- Multiple retrieval strategies (BM25, embedding-based)
- Reranking of retrieved passages
- Language model-based answer generation
- Comprehensive evaluation framework
- Few-shot prompting support

## Installation

### Requirements

The project requires Python 3.8+ and the following main dependencies:
- farm-haystack (or haystack-ai)
- torch
- transformers
- pandas
- matplotlib
- numpy
- tqdm
```

## Data

The project uses a corpus of UMSI-related documents:
- **25 documents total** (15 text files, 10 PDFs)
- **470 question-answer pairs** (400 training, 85 development, 85 test)
- Documents categorized primarily as MSI Program, About UMSI, Course Schedules, BSI Program

Documents include:
- Program handbooks (MSI, MADS, BSI)
- Internship information
- Course schedules
- UMSI contact and location information
- Degree requirements


### Command Line Arguments

- `--mode`: Execution mode (`dev`, `test`, or `eval`)
- `--data_dir`: Directory containing document data
- `--out_dir`: Directory to save output files
- `--model`: Model type to use (`baseline`, `embed_retriever`, `embed_meta`, `different_reranker`, `squad`)
- `--use_gpu`: Use GPU for computations
- `--few_shot`: Number of few-shot examples to use (0 for zero-shot)
- `--n_doc`: Number of documents to retrieve and rerank

### Available Models

1. **baseline**: BM25 retriever + MiniLM reranker + LaMini-Flan-T5 generator
2. **embed_retriever**: Embedding-based retriever + MiniLM reranker + LaMini-Flan-T5 generator
3. **embed_meta**: Embedding-based retriever that also embeds metadata + reranker + generator
4. **different_reranker**: BM25 retriever + QNLI DistilRoBERTa reranker + generator
5. **squad**: SQuAD-optimized configuration

### Data Analysis

To generate insights about the dataset:
```bash
python3 data_insights.py
```
This will create:
- A comprehensive report (`data_insights_report.txt`)
- Visualizations in the `data_insights_figures` directory

## Evaluation Metrics

The system is evaluated using the following metrics:
- **F1 Score**: Measures word overlap between predictions and reference answers
- **Recall**: Measures if the prediction contains the required information
- **Exact Match (EM)**: Measures exact string match between prediction and reference


## Acknowledgments

- The University of Michigan School of Information
- Libraries used: farm-haystack, transformers, pytorch
