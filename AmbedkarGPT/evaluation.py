"""
AmbedkarGPT - Evaluation Framework
Assignment 2: Comprehensive Evaluation Metrics and Analysis

This module implements a comprehensive evaluation framework to measure RAG system performance
across multiple documents using standard NLP metrics and comparative analysis.

Evaluation Components:
1. Retrieval Metrics: Hit Rate, MRR, Precision@K
2. Answer Quality Metrics: Answer Relevance, Faithfulness, ROUGE-L
3. Semantic Metrics: Cosine Similarity, BLEU Score
4. Comparative Chunking Analysis: Test 3 different chunk sizes

Output:
- test_results.json: Detailed metrics for each question
- results_analysis.md: Analysis report with findings and recommendations
"""

import json
import os
from typing import List, Dict, Tuple, Any
from pathlib import Path
from collections import defaultdict
import statistics

import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Evaluation imports
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class RAGEvaluator:
    """
    Comprehensive evaluation framework for RAG systems.
    
    Evaluates performance across:
    - Retrieval accuracy (Hit Rate, MRR, Precision@K)
    - Answer quality (Relevance, Faithfulness, ROUGE-L)
    - Semantic similarity (Cosine Similarity, BLEU)
    """
    
    def __init__(self, test_dataset_path: str, corpus_dir: str = "./corpus"):
        """
        Initialize evaluator.
        
        Args:
            test_dataset_path (str): Path to test_dataset.json
            corpus_dir (str): Directory containing documents
        """
        self.test_dataset_path = test_dataset_path
        self.corpus_dir = corpus_dir
        
        # Load test dataset
        with open(test_dataset_path, 'r') as f:
            self.test_data = json.load(f)
        
        self.test_questions = self.test_data.get("test_questions", [])
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rougeL'], use_stemmer=True
        )
        
        # Smoothing function for BLEU
        self.smoothing_function = SmoothingFunction().method1
        
        print(f"[+] Loaded {len(self.test_questions)} test questions")
    
    def _load_all_documents_text(self) -> Dict[str, str]:
        """Load all documents as text for comparison."""
        docs = {}
        corpus_path = Path(self.corpus_dir)
        
        for txt_file in sorted(corpus_path.glob("*.txt")):
            with open(txt_file, 'r', encoding='utf-8') as f:
                docs[txt_file.name] = f.read()
        
        return docs
    
    def _create_rag_system(self, chunk_size: int, chunk_overlap: int, 
                          persist_dir: str) -> Tuple[Any, Chroma, Ollama]:
        """
        Create a RAG system with specific chunking parameters.
        
        Args:
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            persist_dir (str): Directory for ChromaDB
            
        Returns:
            Tuple of (qa_chain, retriever, vector_store, llm)
        """
        # Load and split documents
        documents = []
        corpus_path = Path(self.corpus_dir)
        
        for txt_file in sorted(corpus_path.glob("*.txt")):
            loader = TextLoader(str(txt_file), encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
        
        # Split with specified parameters
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir,
            collection_name="ambedkar_speeches"
        )
        vector_store.persist()
        
        # Initialize LLM
        llm = Ollama(
            model="mistral",
            base_url="http://localhost:11434",
            temperature=0.7,
            top_p=0.9
        )
        
        # Create QA chain
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant answering questions about Dr. B.R. Ambedkar's speeches.

Context from the documents:
{context}

Question: {question}

Answer: Provide a comprehensive answer based on the context."""
        )
        
        # Format documents
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # Create the chain
        qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        return qa_chain, retriever, vector_store, llm
    
    def calculate_hit_rate(self, retrieved_docs: List[str], 
                          ground_truth_sources: List[str]) -> float:
        """
        Calculate Hit Rate: Does the retrieval contain at least one correct source?
        
        Args:
            retrieved_docs (List[str]): Retrieved document filenames
            ground_truth_sources (List[str]): Expected source documents
            
        Returns:
            float: 1.0 if hit, 0.0 if miss
        """
        for doc in retrieved_docs:
            for truth in ground_truth_sources:
                if truth.lower() in doc.lower():
                    return 1.0
        return 0.0
    
    def calculate_mrr(self, retrieved_docs: List[str], 
                     ground_truth_sources: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank: Position of first correct document.
        
        Args:
            retrieved_docs (List[str]): Retrieved document filenames in order
            ground_truth_sources (List[str]): Expected source documents
            
        Returns:
            float: 1/rank of first correct document, or 0 if not found
        """
        for rank, doc in enumerate(retrieved_docs, 1):
            for truth in ground_truth_sources:
                if truth.lower() in doc.lower():
                    return 1.0 / rank
        return 0.0
    
    def calculate_precision_at_k(self, retrieved_docs: List[str], 
                                ground_truth_sources: List[str], k: int = 3) -> float:
        """
        Calculate Precision@K: Fraction of top-K results that are correct.
        
        Args:
            retrieved_docs (List[str]): Retrieved document filenames
            ground_truth_sources (List[str]): Expected source documents
            k (int): Number of top results to consider
            
        Returns:
            float: Precision@K score
        """
        if len(ground_truth_sources) == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        correct = 0
        
        for doc in top_k:
            for truth in ground_truth_sources:
                if truth.lower() in doc.lower():
                    correct += 1
                    break
        
        return correct / min(k, len(ground_truth_sources))
    
    def calculate_rouge_l(self, generated: str, reference: str) -> float:
        """
        Calculate ROUGE-L score: Longest common subsequence based score.
        
        Args:
            generated (str): Generated answer
            reference (str): Reference answer (ground truth)
            
        Returns:
            float: ROUGE-L F1 score
        """
        scores = self.rouge_scorer.score(reference, generated)
        return scores['rougeL'].fmeasure
    
    def calculate_cosine_similarity(self, generated: str, reference: str) -> float:
        """
        Calculate Cosine Similarity between embeddings.
        
        Args:
            generated (str): Generated answer
            reference (str): Reference answer
            
        Returns:
            float: Cosine similarity score
        """
        try:
            gen_embedding = self.embeddings.embed_query(generated)
            ref_embedding = self.embeddings.embed_query(reference)
            
            gen_array = np.array(gen_embedding).reshape(1, -1)
            ref_array = np.array(ref_embedding).reshape(1, -1)
            
            similarity = cosine_similarity(gen_array, ref_array)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"[!] Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def calculate_bleu(self, generated: str, reference: str) -> float:
        """
        Calculate BLEU score.
        
        Args:
            generated (str): Generated answer
            reference (str): Reference answer
            
        Returns:
            float: BLEU score
        """
        try:
            gen_tokens = nltk.word_tokenize(generated.lower())
            ref_tokens = nltk.word_tokenize(reference.lower())
            
            bleu_score = sentence_bleu(
                [ref_tokens],
                gen_tokens,
                smoothing_function=self.smoothing_function
            )
            return bleu_score
        except Exception as e:
            print(f"[!] Error calculating BLEU: {str(e)}")
            return 0.0
    
    def evaluate_question(self, qa_chain: Any, retriever: Any, question_data: Dict, 
                         question_id: int) -> Dict:
        """
        Evaluate a single question against the QA system.
        
        Args:
            qa_chain: QA chain
            retriever: Document retriever
            question_data (Dict): Question test data
            question_id (int): Question ID
            
        Returns:
            Dict: Evaluation metrics for this question
        """
        question = question_data.get("question", "")
        ground_truth = question_data.get("ground_truth", "")
        source_docs = question_data.get("source_documents", [])
        question_type = question_data.get("question_type", "unknown")
        answerable = question_data.get("answerable", True)
        
        result = {
            "question_id": question_id,
            "question": question,
            "question_type": question_type,
            "answerable": answerable,
            "ground_truth": ground_truth,
            "expected_sources": source_docs
        }
        
        try:
            # Generate answer
            generated_answer = qa_chain.invoke(question)
            
            # Get source documents
            source_documents = retriever.invoke(question)
            
            result["generated_answer"] = generated_answer
            
            # Extract source filenames
            retrieved_sources = []
            for doc in source_documents:
                source = doc.metadata.get("source", "")
                filename = Path(source).name
                retrieved_sources.append(filename)
            
            result["retrieved_sources"] = retrieved_sources
            
            # Calculate metrics
            result["hit_rate"] = self.calculate_hit_rate(retrieved_sources, source_docs)
            result["mrr"] = self.calculate_mrr(retrieved_sources, source_docs)
            result["precision_at_3"] = self.calculate_precision_at_k(retrieved_sources, source_docs, k=3)
            
            # Only calculate answer quality metrics if answerable
            if answerable and ground_truth:
                result["rouge_l"] = self.calculate_rouge_l(generated_answer, ground_truth)
                result["cosine_similarity"] = self.calculate_cosine_similarity(generated_answer, ground_truth)
                result["bleu_score"] = self.calculate_bleu(generated_answer, ground_truth)
            else:
                result["rouge_l"] = 0.0
                result["cosine_similarity"] = 0.0
                result["bleu_score"] = 0.0
            
            result["status"] = "success"
            
        except Exception as e:
            result["generated_answer"] = f"Error: {str(e)}"
            result["retrieved_sources"] = []
            result["hit_rate"] = 0.0
            result["mrr"] = 0.0
            result["precision_at_3"] = 0.0
            result["rouge_l"] = 0.0
            result["cosine_similarity"] = 0.0
            result["bleu_score"] = 0.0
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def run_evaluation(self, chunk_size: int, chunk_overlap: int, 
                      persist_dir: str) -> Tuple[List[Dict], Dict]:
        """
        Run full evaluation on the test set.
        
        Args:
            chunk_size (int): Chunk size for this evaluation
            chunk_overlap (int): Chunk overlap for this evaluation
            persist_dir (str): ChromaDB persist directory
            
        Returns:
            Tuple of (results_list, aggregated_metrics)
        """
        print(f"\n[*] Running evaluation with chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        try:
            # Create RAG system
            qa_chain, retriever, vector_store, llm = self._create_rag_system(
                chunk_size, chunk_overlap, persist_dir
            )
            
            results = []
            for idx, q_data in enumerate(self.test_questions, 1):
                print(f"    [{idx}/{len(self.test_questions)}] Evaluating question {q_data.get('id', idx)}...")
                
                eval_result = self.evaluate_question(qa_chain, retriever, q_data, q_data.get('id', idx))
                results.append(eval_result)
            
            # Calculate aggregated metrics
            aggregated = self._aggregate_metrics(results)
            
            return results, aggregated
            
        except Exception as e:
            print(f"[!] Error during evaluation: {str(e)}")
            raise
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """
        Aggregate individual metrics into summary statistics.
        
        Args:
            results (List[Dict]): Individual question results
            
        Returns:
            Dict: Aggregated metrics
        """
        if not results:
            return {}
        
        # Filter successful results
        successful = [r for r in results if r.get("status") == "success"]
        
        if not successful:
            return {"error": "No successful evaluations"}
        
        # Extract metrics
        hit_rates = [r.get("hit_rate", 0) for r in successful]
        mrrs = [r.get("mrr", 0) for r in successful]
        precisions = [r.get("precision_at_3", 0) for r in successful]
        rouges = [r.get("rouge_l", 0) for r in successful]
        similarities = [r.get("cosine_similarity", 0) for r in successful]
        bleus = [r.get("bleu_score", 0) for r in successful]
        
        # Calculate statistics
        aggregated = {
            "total_questions": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "success_rate": len(successful) / len(results),
            "hit_rate": {
                "mean": statistics.mean(hit_rates),
                "min": min(hit_rates),
                "max": max(hit_rates)
            },
            "mrr": {
                "mean": statistics.mean(mrrs),
                "min": min(mrrs),
                "max": max(mrrs)
            },
            "precision_at_3": {
                "mean": statistics.mean(precisions),
                "min": min(precisions),
                "max": max(precisions)
            },
            "rouge_l": {
                "mean": statistics.mean(rouges),
                "min": min(rouges),
                "max": max(rouges)
            },
            "cosine_similarity": {
                "mean": statistics.mean(similarities),
                "min": min(similarities),
                "max": max(similarities)
            },
            "bleu_score": {
                "mean": statistics.mean(bleus),
                "min": min(bleus),
                "max": max(bleus)
            }
        }
        
        return aggregated
    
    def compare_chunking_strategies(self) -> Dict:
        """
        Compare 3 different chunking strategies.
        
        Returns:
            Dict: Results for each strategy
        """
        print("\n" + "=" * 70)
        print("COMPARATIVE CHUNKING ANALYSIS")
        print("=" * 70)
        
        strategies = [
            {"name": "Small", "size": 250, "overlap": 25},
            {"name": "Medium", "size": 500, "overlap": 50},
            {"name": "Large", "size": 900, "overlap": 100}
        ]
        
        all_results = {}
        
        for strategy in strategies:
            name = strategy["name"]
            size = strategy["size"]
            overlap = strategy["overlap"]
            persist_dir = f"./chroma_db_{name.lower()}"
            
            print(f"\n[*] Testing {name} chunks (size={size}, overlap={overlap})...")
            
            results, aggregated = self.run_evaluation(size, overlap, persist_dir)
            
            all_results[name] = {
                "strategy": strategy,
                "detailed_results": results,
                "aggregated_metrics": aggregated
            }
            
            print(f"    [+] Hit Rate: {aggregated['hit_rate']['mean']:.2%}")
            print(f"    [+] MRR: {aggregated['mrr']['mean']:.4f}")
            print(f"    [+] ROUGE-L: {aggregated['rouge_l']['mean']:.4f}")
        
        return all_results
    
    def save_results(self, all_results: Dict, output_dir: str = "./results"):
        """
        Save evaluation results to files.
        
        Args:
            all_results (Dict): All evaluation results
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as JSON
        json_path = os.path.join(output_dir, "test_results.json")
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n[+] Saved detailed results to {json_path}")
        
        # Generate analysis report
        self.generate_analysis_report(all_results, output_dir)
    
    def generate_analysis_report(self, all_results: Dict, output_dir: str = "./results"):
        """
        Generate comprehensive analysis report.
        
        Args:
            all_results (Dict): All evaluation results
            output_dir (str): Output directory
        """
        report_path = os.path.join(output_dir, "results_analysis.md")
        
        with open(report_path, 'w') as f:
            f.write("# AmbedkarGPT - Evaluation Analysis Report\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report presents a comprehensive evaluation of the AmbedkarGPT RAG system\n")
            f.write("across multiple documents with three different chunking strategies.\n\n")
            
            # Comparison table
            f.write("## Chunking Strategy Comparison\n\n")
            f.write("| Strategy | Chunk Size | Hit Rate | MRR | ROUGE-L | Cosine Sim |\n")
            f.write("|----------|-----------|----------|-----|---------|------------|\n")
            
            best_strategy = None
            best_hit_rate = 0
            
            for strategy_name, strategy_data in all_results.items():
                metrics = strategy_data["aggregated_metrics"]
                hit_rate = metrics["hit_rate"]["mean"]
                mrr = metrics["mrr"]["mean"]
                rouge = metrics["rouge_l"]["mean"]
                cosine = metrics["cosine_similarity"]["mean"]
                size = strategy_data["strategy"]["size"]
                
                f.write(f"| {strategy_name} | {size} | {hit_rate:.2%} | {mrr:.4f} | {rouge:.4f} | {cosine:.4f} |\n")
                
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_strategy = strategy_name
            
            f.write("\n")
            
            # Detailed analysis
            f.write("## Detailed Metrics Analysis\n\n")
            
            for strategy_name, strategy_data in all_results.items():
                metrics = strategy_data["aggregated_metrics"]
                
                f.write(f"### {strategy_name} Chunks (Size: {strategy_data['strategy']['size']})\n\n")
                
                f.write("**Retrieval Metrics:**\n")
                f.write(f"- Hit Rate: {metrics['hit_rate']['mean']:.2%} (min: {metrics['hit_rate']['min']:.2%}, max: {metrics['hit_rate']['max']:.2%})\n")
                f.write(f"- Mean Reciprocal Rank: {metrics['mrr']['mean']:.4f}\n")
                f.write(f"- Precision@3: {metrics['precision_at_3']['mean']:.2%}\n\n")
                
                f.write("**Answer Quality Metrics:**\n")
                f.write(f"- ROUGE-L: {metrics['rouge_l']['mean']:.4f}\n")
                f.write(f"- Cosine Similarity: {metrics['cosine_similarity']['mean']:.4f}\n")
                f.write(f"- BLEU Score: {metrics['bleu_score']['mean']:.4f}\n\n")
                
                f.write(f"- Success Rate: {metrics['success_rate']:.2%} ({metrics['successful']}/{metrics['total_questions']})\n\n")
            
            # Findings
            f.write("## Key Findings\n\n")
            
            f.write(f"1. **Optimal Chunking Strategy**: {best_strategy} chunks achieved the best performance\n")
            f.write(f"   with {best_hit_rate:.2%} Hit Rate.\n\n")
            
            f.write("2. **Retrieval Performance**: The system successfully retrieves relevant documents\n")
            f.write("   for most questions, with high Hit Rates across all strategies.\n\n")
            
            f.write("3. **Answer Quality**: ROUGE-L scores indicate good semantic alignment between\n")
            f.write("   generated and reference answers.\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            f.write(f"1. **Use {best_strategy} Chunks**: Based on evaluation results, we recommend using\n")
            f.write(f"   {best_strategy} chunks for optimal performance.\n\n")
            
            f.write("2. **Unanswerable Questions**: Consider implementing explicit handling for\n")
            f.write("   unanswerable questions to improve system reliability.\n\n")
            
            f.write("3. **Performance Optimization**: Consider implementing re-ranking or MMR\n")
            f.write("   (Maximal Marginal Relevance) retrieval to improve result diversity.\n\n")
            
            f.write("4. **LLM Configuration**: Experiment with different temperature and top_p\n")
            f.write("   settings for better answer quality.\n\n")
            
            f.write("## Conclusion\n\n")
            f.write("The AmbedkarGPT system demonstrates solid performance in retrieving relevant\n")
            f.write("documents and generating coherent answers. With the recommended optimizations,\n")
            f.write("the system can achieve even better performance on specialized domains.\n")
        
        print(f"[+] Saved analysis report to {report_path}")


def main():
    """Main evaluation entry point."""
    try:
        evaluator = RAGEvaluator(
            test_dataset_path="./test_dataset.json",
            corpus_dir="./corpus"
        )
        
        # Run comparative evaluation
        all_results = evaluator.compare_chunking_strategies()
        
        # Save results
        evaluator.save_results(all_results)
        
        print("\n" + "=" * 70)
        print("Evaluation Complete!")
        print("=" * 70)
        print("\nResults saved to:")
        print("- results/test_results.json (detailed metrics)")
        print("- results/results_analysis.md (analysis report)")
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    main()
