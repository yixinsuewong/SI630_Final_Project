import haystack_adapter  # Import the adapter first to set up compatibility layer
from haystack import Pipeline
from haystack.schema import Document
from haystack.nodes import PreProcessor
from haystack.nodes.file_converter import PDFToTextConverter

from typing import List, Tuple, Optional
import os
import tqdm
import json
import argparse
import pandas as pd
from pathlib import Path

from models import *
from evaluation import evaluate_predictions


def load_documents(data_dir: str) -> List[Document]:
    """
    Load documents from data directory and preprocess them
    """
    print(f"Loading documents from {data_dir}...")
    
    # Load metadata
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata_raw = json.load(f)
    
    # Create metadata lookup dictionaries - one for basenames, one for relative paths
    basename_meta = {}  # For looking up by filename only
    path_meta = {}      # For looking up by relative path
    
    for metadata in metadata_raw:
        m = metadata.copy()
        full_filename = m.pop("filename")
        
        # Handle paths that might start with "data/"
        if full_filename.startswith("data/"):
            rel_path = full_filename[5:]  # Remove "data/" prefix
        else:
            rel_path = full_filename
            
        basename = os.path.basename(full_filename)
        
        # Store metadata for both basename and relative path lookups
        basename_meta[basename] = m
        path_meta[rel_path] = m
    
    # Initialize converters and preprocessors
    pdf_converter = PDFToTextConverter()
    pre = PreProcessor()
    
    # Load text files
    txts = []
    pdfs = []
    
    # Process text files
    txt_files_found = []
    for file_path in Path(data_dir).glob("**/*.txt"):
        rel_path = os.path.relpath(file_path, data_dir)
        basename = file_path.name
        
        # Try to find metadata using multiple lookup methods
        metadata = None
        if basename in basename_meta:
            metadata = basename_meta[basename]
            txt_files_found.append(basename)
        elif rel_path in path_meta:
            metadata = path_meta[rel_path]
            txt_files_found.append(rel_path)
        else:
            print(f"Warning: File {file_path} not in metadata")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        txts.append(Document(content=raw, meta=metadata))
    
    # Process PDF files
    pdf_files_found = []
    for file_path in Path(data_dir).glob("**/*.pdf"):
        rel_path = os.path.relpath(file_path, data_dir)
        basename = file_path.name
        
        # Try to find metadata using multiple lookup methods
        metadata = None
        if basename in basename_meta:
            metadata = basename_meta[basename]
            pdf_files_found.append(basename)
        elif rel_path in path_meta:
            metadata = path_meta[rel_path]
            pdf_files_found.append(rel_path)
        else:
            # Try fuzzy matching for PDFs with slight name differences
            found_match = False
            for meta_name in basename_meta.keys():
                # Check if filenames are similar (ignoring spaces)
                if meta_name.replace(" ", "") == basename.replace(" ", ""):
                    metadata = basename_meta[meta_name]
                    pdf_files_found.append(meta_name)
                    print(f"Matched PDF: {basename} to metadata: {meta_name}")
                    found_match = True
                    break
            
            if not found_match:
                print(f"Warning: File {file_path} not in metadata")
                continue
        
        try:
            pdf_docs = pdf_converter.convert(
                file_path=str(file_path),
                meta=metadata,
                remove_numeric_tables=False,
                valid_languages=["en"],
            )
            pdfs.extend(pdf_docs)
        except Exception as e:
            print(f"Error converting PDF {file_path}: {e}")
    
    # Print statistics about files found vs files in metadata
    print(f"Found {len(txt_files_found)} text files and {len(pdf_files_found)} PDF files in metadata")
    print(f"Loaded {len(txts)} text files and {len(pdfs)} PDF files successfully")
    
    # Print which PDF files were found and loaded
    if pdf_files_found:
        print(f"PDF files found: {', '.join(pdf_files_found)}")
    
    # Split documents into passages
    print("Preprocessing documents...")
    if txts:
        # Process text documents
        preprocessed_txt = pre.process(
            txts,
            split_by="passage",
            split_length=1,
            split_respect_sentence_boundary=False,
        )
    else:
        preprocessed_txt = []
    
    # Combine and further split into chunks
    res = pdfs + preprocessed_txt
    res = pre.process(
        res, 
        split_by="word", 
        split_length=200, 
        split_respect_sentence_boundary=True
    )
    
    print(f"Created {len(res)} document chunks after preprocessing")
    return res


def load_qa(question_file: str, answer_file: str) -> Tuple[List[str], List[str]]:
    """
    Load questions and answers from files
    """
    questions = []
    answers = []
    
    with open(question_file, "r") as fq, open(answer_file, "r") as fa:
        for lineq, linea in zip(fq, fa):
            q, a = lineq.strip(), linea.strip()
            if q and a:  # Only add non-empty questions and answers
                questions.append(q)
                answers.append(a)
    
    return questions, answers


def predict(
    p: Pipeline, 
    queries: List[str], 
    output_file: str, 
    few_shot_p: Optional[Pipeline] = None
) -> List[str]:
    """
    Run predictions on queries and save to output file
    """
    res = []
    with open(output_file, "w") as f:
        for query in tqdm.tqdm(queries, desc="Running predictions"):
            meta = {"few_shot_example": ""}
            
            # Add few-shot examples if available
            if few_shot_p:
                context = few_shot_p.run(query=query)
                meta["few_shot_example"] = "\n".join(
                    f"Example {i+1}:\nQuestion: {document.content}\n\nAnswer: {document.meta['answer']}\n\n"
                    for i, document in enumerate(context["documents"])
                )
            
            # Run query through pipeline
            answer = p.run(query=query, meta=meta)
            
            # Extract first line of answer
            ans = answer["results"][0].strip().split("\n")[0]
            res.append(ans)
            
            # Write to output file
            f.write(ans + "\n")
            f.flush()
    
    return res


def main():
    parser = argparse.ArgumentParser(description="RAG System for Question Answering")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices={"dev", "test", "eval"}, 
        default="dev",
        help="Execution mode: dev (evaluate on dev set), test (run on test set), eval (evaluate predictions)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data",
        help="Directory containing document data"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default=".",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="baseline",
        choices=["baseline", "embed_retriever", "embed_meta", "different_reranker", "squad"],
        help="Model type to use"
    )
    parser.add_argument(
        "--use_gpu", 
        action="store_true",
        help="Use GPU for computations"
    )
    parser.add_argument(
        "--dev",
        type=str,
        nargs=2,
        default=("dev/questions.txt", "dev/reference_answers.txt"),
        help="Paths to dev questions and reference answers"
    )
    parser.add_argument(
        "--test", 
        type=str, 
        default="test/questions.txt",
        help="Path to test questions"
    )
    parser.add_argument(
        "--test_answers", 
        type=str, 
        default="test/reference_answers.txt",
        help="Path to test reference answers (if available)"
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs=2,
        default=("dev/reference_answers.txt", "dev/prediction.txt"),
        help="Paths to reference answers and predictions for evaluation"
    )
    parser.add_argument(
        "--train",
        type=str,
        nargs=2,
        default=("train/questions.txt", "train/reference_answers.txt"),
        help="Paths to training questions and answers for few-shot examples"
    )
    parser.add_argument(
        "--few_shot", 
        type=int, 
        default=1,
        help="Number of few-shot examples to use (0 for zero-shot)"
    )
    parser.add_argument(
        "--n_doc", 
        type=int, 
        default=1,
        help="Number of documents to retrieve and rerank"
    )
    
    args = parser.parse_args()

    # Generate output filename
    output_filename = f"prediction_{args.mode}_{args.model}_{args.few_shot}_shot_k_{args.n_doc}.txt"
    output_path = os.path.join(args.out_dir, output_filename)
    
    print(f"Output will be saved to: {output_path}")

    # Evaluation mode: compare existing predictions
    if args.mode == "eval":
        print("Evaluation mode: Comparing existing predictions...")
        
        answers = []
        with open(args.eval[0], "r", encoding="utf-8-sig") as f:
            for line in f:
                a = line.strip()
                if a != "":
                    answers.append(a)
        
        prediction = []
        with open(args.eval[1], "r", encoding="utf-8-sig") as f:
            for line in f:
                p = line.strip()
                if p != "":
                    prediction.append(p)
        
        f1, recall, em = evaluate_predictions(prediction, answers)
        print(f"Evaluation Results:")
        print(f"F1: {f1:.4f}, Recall: {recall:.4f}, EM: {em:.4f}")
        
        # Save detailed scores to CSV
        metrics_df = pd.DataFrame({
            "Metric": ["F1", "Recall", "EM"],
            "Value": [f1, recall, em]
        })
        metrics_df.to_csv(f"metrics_{os.path.basename(args.eval[1])}.csv", index=False)
        
    else:
        # Setup few-shot pipeline if needed
        few_shot_p = None
        if args.few_shot > 0:
            print(f"Setting up few-shot pipeline with {args.few_shot} examples...")
            train_q, train_a = load_qa(*args.train)
            few_shot_p = few_shot_pipeline(
                train_q, train_a, args.few_shot, args.use_gpu
            )

        # Load documents
        docs = load_documents(args.data_dir)
        
        # Get the specified pipeline
        print(f"Setting up {args.model} pipeline...")
        try:
            if args.model == "baseline":
                p = baseline(docs, use_gpu=args.use_gpu, top_k=args.n_doc)
            elif args.model == "embed_retriever":
                p = embed_retriever(docs, use_gpu=args.use_gpu, top_k=args.n_doc)
            elif args.model == "embed_meta":
                p = embed_meta(docs, use_gpu=args.use_gpu, top_k=args.n_doc)
            elif args.model == "different_reranker":
                p = different_reranker(docs, use_gpu=args.use_gpu, top_k=args.n_doc)
            elif args.model == "squad":
                p = squad(docs, use_gpu=args.use_gpu, top_k=args.n_doc)
            else:
                raise ValueError(f"Unknown model type: {args.model}")
        except Exception as e:
            print(f"Error setting up pipeline: {e}")
            raise

        # Development mode: evaluate on dev set
        if args.mode == "dev":
            print("Dev mode: Evaluating on development set...")
            questions, answers = load_qa(*args.dev)
            
            print(f"Running predictions on {len(questions)} questions...")
            prediction = predict(p, questions, output_path, few_shot_p)

            print("Evaluating predictions...")
            f1, recall, em = evaluate_predictions(prediction, answers)
            
            print(f"Results:")
            print(f"F1: {f1:.4f}, Recall: {recall:.4f}, EM: {em:.4f}")
            
            # Save detailed scores to CSV
            metrics_df = pd.DataFrame({
                "Metric": ["F1", "Recall", "EM"],
                "Value": [f1, recall, em]
            })
            metrics_df.to_csv(f"metrics_{os.path.basename(output_path)}.csv", index=False)

        # Test mode: run on test set
        else:
            print("Test mode: Running on test set...")
            questions = []
            with open(args.test, "r") as fq:
                for lineq in fq:
                    q = lineq.strip()
                    if q != "":
                        questions.append(q)
            
            print(f"Running predictions on {len(questions)} questions...")
            predictions = predict(p, questions, output_path, few_shot_p)
            
            # Evaluate if test answers are available
            try:
                with open(args.test_answers, "r") as fa:
                    answers = [a.strip() for a in fa if a.strip()]
                
                if len(answers) == len(predictions):
                    print("Evaluating predictions against test answers...")
                    f1, recall, em = evaluate_predictions(predictions, answers)
                    
                    print(f"Test Results:")
                    print(f"F1: {f1:.4f}, Recall: {recall:.4f}, EM: {em:.4f}")
                    
                    # Save detailed scores to CSV
                    metrics_df = pd.DataFrame({
                        "Metric": ["F1", "Recall", "EM"],
                        "Value": [f1, recall, em]
                    })
                    metrics_df.to_csv(f"metrics_{os.path.basename(output_path)}.csv", index=False)
            except:
                print("No test answers available for evaluation.")


if __name__ == "__main__":
    main()
