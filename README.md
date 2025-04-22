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

### Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:

There are two methods to run the project:

#### Method 1: Using the compatibility layer

```bash
python3 run_rag.py
```

This script attempts to install the required dependencies and includes a compatibility layer for different haystack versions.

#### Method 2: Direct installation

```bash
# Install Rust compiler (required for tokenizers)
brew install rust

# Install main dependencies
pip install farm-haystack==1.21.0 torch transformers pandas
```

## Project Structure

```
├── data/                       # Document corpus and metadata
│   ├── metadata.json           # Metadata for all documents
│   ├── *.txt                   # Text documents
│   └── *.pdf                   # PDF documents
├── train/                      # Training data
│   ├── questions.txt           # Training questions
│   └── reference_answers.txt   # Reference answers for training
├── dev/                        # Development data
│   ├── questions.txt           # Development questions
│   └── reference_answers.txt   # Reference answers for development
├── test/                       # Test data
│   ├── questions.txt           # Test questions
│   └── reference_answers.txt   # Reference answers for test
├── data_insights_figures/      # Generated data analysis visualizations
├── main.py                     # Main RAG system implementation
├── models.py                   # Pipeline models definitions
├── evaluation.py               # Evaluation metrics implementation
├── fix_bm25.py                 # Fixes for BM25 implementation
├── haystack_adapter.py         # Compatibility layer for haystack versions
├── run_rag.py                  # Helper script to run the system
└── data_insights.py            # Data analysis script
```

## Data

The project uses a corpus of UMSI-related documents:
- **25 documents total** (15 text files, 10 PDFs)
- **635 question-answer pairs** (400 training, 150 development, 85 test)
- Documents categorized primarily as MSI Program, About UMSI, Course Schedules, BSI Program

Documents include:
- Program handbooks (MSI, MADS, BSI)
- Internship information
- Course schedules
- UMSI contact and location information
- Degree requirements

## Usage

### Running the RAG System

The system can be run in different modes:

1. **Development mode** (evaluate on development set):
```bash
python3 run_rag.py --mode dev --data_dir data --dev dev/questions.txt dev/reference_answers.txt --model baseline
```

2. **Test mode** (run on test set):
```bash
python3 run_rag.py --mode test --data_dir data --test test/questions.txt --model embed_retriever
```

3. **Evaluation mode** (evaluate existing predictions):
```bash
python3 run_rag.py --mode eval --eval dev/reference_answers.txt dev/prediction.txt
```

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

## Troubleshooting

### Common Issues

1. **Missing Haystack Module**:
   - Use the `run_rag.py` script which includes a compatibility layer
   - Or install `farm-haystack==1.21.0` with Rust compiler support

2. **Tokenizers Installation Error**:
   - Install Rust: `brew install rust`
   - Update pip: `pip install --upgrade pip`

3. **Dependency Conflicts**:
   - Create a dedicated virtual environment
   - Use the compatibility layer in `haystack_adapter.py`

## License

[Specify your license information here]

## Acknowledgments

- The University of Michigan School of Information
- Libraries used: farm-haystack, transformers, pytorch

## Citation

[Add any citation information if applicable]
