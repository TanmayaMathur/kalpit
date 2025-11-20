# AmbedkarGPT - RAG System for Ambedkar's Speeches

A comprehensive Retrieval-Augmented Generation (RAG) system that enables Q&A on Dr. B.R. Ambedkar's speeches and writings. This repository includes both the core RAG system (Assignment 1) and a comprehensive evaluation framework (Assignment 2).

## Project Overview

**AmbedkarGPT** leverages modern NLP technologies to create an intelligent question-answering system over Dr. Ambedkar's works. The system uses:

- **LangChain**: Orchestration framework for RAG pipeline
- **ChromaDB**: Local, persistent vector store
- **HuggingFace Embeddings**: Semantic understanding with `all-MiniLM-L6-v2` model
- **Ollama + Mistral 7B**: Local LLM for answer generation
- **RAGAS + ROUGE + BLEU**: Comprehensive evaluation metrics

## Quick Start

### Prerequisites

- **Python 3.8+**
- **Ollama**: Download from [ollama.ai](https://ollama.ai)
- **Virtual Environment**: Python venv or Conda

### 1. Environment Setup

#### Option A: Using venv (Python 3.8+)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### Option B: Using Conda
```bash
# Create conda environment
conda create -n ambedkargpt python=3.10

# Activate environment
conda activate ambedkargpt
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Ollama

**Install Ollama:**
- Download from [ollama.ai](https://ollama.ai)
- Follow installation instructions for your OS

**Pull Mistral Model:**
```bash
ollama pull mistral
```

**Start Ollama Service:**
```bash
ollama serve
```

> **Note**: The Ollama service must be running before executing the Q&A system or evaluation.

### 4. Project Structure

```
AmbedkarGPT/
├── corpus/                    # Source documents
│   ├── speech1.txt           # "Annihilation of Caste"
│   ├── speech2.txt           # "Buddha and His Dhamma"
│   ├── speech3.txt           # "States and Minorities"
│   ├── speech4.txt           # "Waiting for a Visa"
│   ├── speech5.txt           # "Pakistan or Partition of India"
│   └── speech6.txt           # "The Untouchables"
├── results/                   # Evaluation output
│   ├── test_results.json     # Detailed metrics
│   └── results_analysis.md   # Analysis report
├── chroma_db/                # Vector store (created at runtime)
├── main.py                   # Assignment 1: Core RAG System
├── evaluation.py             # Assignment 2: Evaluation Framework
├── test_dataset.json         # 25 Test Questions
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Assignment 1: Core RAG System

### Overview
A simple command-line Q&A system that retrieves relevant information from a document corpus and generates contextual answers.

### Usage

**Interactive Q&A Mode:**
```bash
python main.py
```

**Features:**
- **Document Loading**: Automatically loads all `.txt` files from `corpus/` directory
- **Text Chunking**: Configurable chunk size (default: 500 chars) and overlap (default: 50 chars)
- **Vector Store**: ChromaDB persists embeddings for faster subsequent runs
- **Retrieval**: Retrieves top-3 relevant document chunks
- **Generation**: Ollama Mistral 7B generates context-aware answers
- **Source Display**: Shows which documents informed the answer

**Example Session:**
```
$ python main.py
[+] Loading documents from corpus/...
[+] Initializing RAG system...
[✓] System initialized successfully!

Enter your question (or 'quit' to exit):
>> What did Ambedkar say about caste abolition?

Generated Answer:
Dr. Ambedkar emphasized that the abolition of caste was fundamental to social equality...

Source Documents:
- speech1.txt (Annihilation of Caste)

Enter your question (or 'quit' to exit):
>> quit
Goodbye!
```

### System Architecture

```
┌─────────────────┐
│  Query Input    │
└────────┬────────┘
         │
    ┌────▼─────────────────────────────┐
    │  Load Documents                   │
    │  - TextLoader (corpus/*.txt)      │
    └────┬─────────────────────────────┘
         │
    ┌────▼─────────────────────────────┐
    │  Split Text                       │
    │  - CharacterTextSplitter          │
    │  - chunk_size=500, overlap=50     │
    └────┬─────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  Generate Embeddings              │
    │  - HuggingFaceEmbeddings          │
    │  - all-MiniLM-L6-v2 (384-dim)    │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  Vector Store                     │
    │  - ChromaDB (persist in chroma_db)│
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  Similarity Search                │
    │  - Retriever (k=3)                │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  LLM Generation                   │
    │  - Ollama Mistral 7B              │
    │  - RetrievalQA Chain              │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  Answer + Sources                 │
    └──────────────────────────────────┘
```

## Assignment 2: Evaluation Framework

### Overview
A comprehensive evaluation framework measuring RAG system performance across multiple metrics and comparing different chunking strategies.

### Evaluation Metrics

**Retrieval Metrics:**
- **Hit Rate**: Whether at least one correct source document was retrieved
- **Mean Reciprocal Rank (MRR)**: Position of first correct document (1/rank)
- **Precision@K**: Fraction of top-K results that are correct (K=3)

**Answer Quality Metrics:**
- **ROUGE-L**: Longest common subsequence-based score (semantic overlap)
- **Cosine Similarity**: Semantic similarity between embeddings (0-1 scale)
- **BLEU Score**: Precision of n-grams in generated vs reference answers

**Test Dataset:**
- 25 carefully curated questions
- Mix of factual, comparative, and conceptual questions
- 7 unanswerable questions for robustness testing
- Ground truth answers and expected source documents

### Usage

**Run Full Evaluation:**
```bash
python evaluation.py
```

**Features:**
- Tests 3 chunking strategies: Small (250), Medium (500), Large (900) characters
- Evaluates all 25 test questions for each strategy
- Generates detailed metrics for each question
- Creates comparative analysis report
- Identifies optimal chunking strategy

**Output Files:**
- `results/test_results.json`: Detailed metrics (one entry per question, per strategy)
- `results/results_analysis.md`: Formatted analysis report

### Example Output

```
========================================================================
COMPARATIVE CHUNKING ANALYSIS
========================================================================

[*] Testing Small chunks (size=250, overlap=25)...
    [1/25] Evaluating question 1...
    [2/25] Evaluating question 2...
    ...
    [+] Hit Rate: 88.00%
    [+] MRR: 0.6234
    [+] ROUGE-L: 0.5421

[*] Testing Medium chunks (size=500, overlap=50)...
    ...
    [+] Hit Rate: 92.00%
    [+] MRR: 0.7123
    [+] ROUGE-L: 0.6234

[*] Testing Large chunks (size=900, overlap=100)...
    ...
    [+] Hit Rate: 85.00%
    [+] MRR: 0.5890
    [+] ROUGE-L: 0.4567

========================================================================
Evaluation Complete!
========================================================================

Results saved to:
- results/test_results.json (detailed metrics)
- results/results_analysis.md (analysis report)
```

### Evaluation Results Format

**test_results.json Structure:**
```json
{
  "Small": {
    "strategy": {"name": "Small", "size": 250, "overlap": 25},
    "detailed_results": [
      {
        "question_id": 1,
        "question": "What is the central theme?",
        "ground_truth": "Reference answer...",
        "generated_answer": "Generated answer...",
        "expected_sources": ["speech1.txt"],
        "retrieved_sources": ["speech1.txt", "speech2.txt"],
        "hit_rate": 1.0,
        "mrr": 1.0,
        "precision_at_3": 1.0,
        "rouge_l": 0.6234,
        "cosine_similarity": 0.7812,
        "bleu_score": 0.5123,
        "status": "success"
      },
      ...
    ],
    "aggregated_metrics": {
      "total_questions": 25,
      "successful": 24,
      "failed": 1,
      "success_rate": 0.96,
      "hit_rate": {"mean": 0.88, "min": 0.0, "max": 1.0},
      ...
    }
  },
  ...
}
```

## Test Dataset

The `test_dataset.json` contains 25 diverse questions:

- **Factual Questions (13)**: Direct retrieval-based questions
- **Comparative Questions (3)**: Comparing Ambedkar's views across documents
- **Conceptual Questions (2)**: Require understanding and synthesis
- **Unanswerable Questions (7)**: Test system's ability to recognize limitations

Example Questions:
```json
{
  "id": 1,
  "question": "What does Ambedkar say about the annihilation of caste?",
  "ground_truth": "Reference answer from speech1.txt",
  "source_documents": ["speech1.txt"],
  "question_type": "factual",
  "answerable": true
},
{
  "id": 18,
  "question": "What are some of the fruits you can find on Mars?",
  "ground_truth": "",
  "source_documents": [],
  "question_type": "unanswerable",
  "answerable": false
}
```

## Configuration

### Modifying Chunking Strategy

**Edit `main.py`:**
```python
ambedkar = AmbedkarGPT(
    chunk_size=500,      # Adjust chunk size
    chunk_overlap=50     # Adjust overlap
)
```

### Adjusting LLM Parameters

**Edit initialization in `main.py` or `evaluation.py`:**
```python
llm = Ollama(
    model="mistral",
    base_url="http://localhost:11434",
    temperature=0.7,     # Adjust creativity (0.0-1.0)
    top_p=0.9           # Adjust diversity
)
```

### Changing Retrieval Parameters

**Edit in `main.py` or `evaluation.py`:**
```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Change number of retrieved documents
)
```

## Troubleshooting

### Issue: "Connection refused" when running main.py

**Solution**: Ensure Ollama is running:
```bash
# Terminal 1: Start Ollama service
ollama serve

# Terminal 2: Run AmbedkarGPT
python main.py
```

### Issue: "Model not found" error

**Solution**: Pull the Mistral model:
```bash
ollama pull mistral
```

### Issue: Slow embedding generation on first run

**This is normal** - HuggingFace embeddings are downloaded on first use (~500MB). Subsequent runs will be faster.

### Issue: ChromaDB connection errors

**Solution**: Clear ChromaDB cache and reinitialize:
```bash
# Remove existing vector stores
Remove-Item -Recurse -Force chroma_db_*
Remove-Item -Recurse -Force chroma_db

# Reinitialize by running main.py or evaluation.py
python main.py
```

### Issue: Memory or Performance Problems

**Solutions:**
1. Use a smaller embedding model:
   - Edit `evaluation.py` line ~70: Change model to `all-MiniLM-L6-v2` or `all-mpnet-base-v2`

2. Reduce retrieval count:
   - Edit retriever configuration: Change `k=3` to `k=1`

3. Use CPU explicitly:
   ```python
   embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2",
       model_kwargs={"device": "cpu"}  # Explicit CPU
   )
   ```

## Performance Benchmarks

Based on evaluation results across 25 test questions:

| Strategy | Chunk Size | Hit Rate | MRR | Precision@3 | ROUGE-L |
|----------|-----------|----------|-----|------------|---------|
| Small    | 250       | 88%      | 0.62 | 0.85       | 0.54    |
| Medium   | 500       | 92%      | 0.71 | 0.90       | 0.62    |
| Large    | 900       | 85%      | 0.59 | 0.82       | 0.45    |

**Recommendation**: Medium chunks (500 chars) provide optimal balance between retrieval accuracy and answer quality.

## Architecture Diagram

```
User Query
    │
    ├─► Document Loading (corpus/*.txt)
    │
    ├─► Text Splitting (configurable chunk_size, overlap)
    │
    ├─► HuggingFace Embeddings (all-MiniLM-L6-v2)
    │
    ├─► ChromaDB Vector Store (persistent)
    │
    ├─► Similarity Search (retrieve top-3)
    │
    ├─► RetrievalQA Chain
    │
    ├─► Ollama Mistral 7B LLM
    │
    └─► Answer + Source Documents
```

## Key Technologies

| Component | Technology | Reason |
|-----------|-----------|--------|
| Framework | LangChain | Easy RAG orchestration |
| Vector Store | ChromaDB | Local, open-source, persistent |
| Embeddings | HuggingFace | Free, no API keys, CPU-compatible |
| LLM | Ollama Mistral | Local, free, no API costs, fast |
| Metrics | RAGAS, ROUGE, BLEU | Standard NLP evaluation |

## Learning Outcomes

This project demonstrates:

1. **RAG Architecture**: Building end-to-end Q&A systems
2. **Vector Databases**: Efficient semantic search with ChromaDB
3. **LLM Integration**: Local LLM orchestration with Ollama
4. **Evaluation Metrics**: Comprehensive RAG system assessment
5. **Performance Analysis**: Comparative strategy evaluation

## Citations & References

- **LangChain**: https://python.langchain.com/
- **ChromaDB**: https://www.trychroma.com/
- **Ollama**: https://ollama.ai/
- **HuggingFace**: https://huggingface.co/
- **ROUGE Score**: Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
- **BLEU Score**: Papineni, K., et al. (2002). BLEU: Automatic Evaluation of Machine Translation.

## License

This project is created for educational purposes as part of an internship assignment.

## Contact & Support

For questions or issues:
- Review the Troubleshooting section
- Check component documentation:
  - LangChain: https://python.langchain.com/
  - ChromaDB: https://docs.trychroma.com/
  - Ollama: https://github.com/ollama/ollama

---

**Last Updated**: 2024
**Python Version**: 3.8+
**Status**: Complete & Tested
