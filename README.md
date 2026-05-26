# 💼 AI-Powered Finance Policy Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** system that enables employees to query internal finance policy documents using natural language and receive accurate, context-grounded answers.

---

## 🏗️ Architecture

```
Finance Documents (.txt/.pdf/.md)
         │
         ▼
  [Document Loader]          ← LangChain DirectoryLoader / PyPDFLoader
         │
         ▼
  [Text Chunker]             ← RecursiveCharacterTextSplitter (800 tokens, 150 overlap)
         │
         ▼
  [Embedding Model]          ← OpenAI text-embedding-3-small
         │
         ▼
  [FAISS Vector Store]       ← Dense similarity index (in-memory)
         │
         ▼
  [Similarity Retriever]     ← Top-K=4 relevant chunks
         │
         ▼
  [ChatPromptTemplate]       ← Grounding prompt with retrieved context
         │
         ▼
  [GPT-4o Mini]              ← Context-aware answer generation
         │
         ▼
  Answer + Source References ← CLI / Streamlit Web UI
```

---

## 📁 Project Structure

```
finance_policy_assistant/
├── main.py                        # CLI entry point (interactive / demo / single query)
├── app.py                         # Streamlit web UI
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
│
├── src/
│   ├── __init__.py
│   └── rag_pipeline.py            # Core RAG pipeline (ingestion + retrieval + generation)
│
├── data/
│   └── documents/                 # Finance policy documents
│       ├── travel_expense_policy.txt
│       ├── vendor_payment_policy.txt
│       ├── procurement_policy.txt
│       ├── tax_compliance_guidelines.txt
│       └── employee_finance_handbook.txt
│
├── tests/
│   └── test_rag_pipeline.py       # Unit tests
│
└── docs/
    └── technical_documentation.md # System design report
```

---

## ⚡ Quick Start

### 1. Clone and Install

```bash
git clone <repo-url>
cd finance_policy_assistant
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 3. Add Finance Documents (Optional)

Place any additional `.txt`, `.pdf`, or `.md` policy documents into `data/documents/`. Five sample documents are already included.

### 4. Run

**CLI — Interactive Mode:**
```bash
python main.py
```

**CLI — Demo Mode (10 sample queries):**
```bash
python main.py --demo
```

**CLI — Single Query:**
```bash
python main.py --query "What is the reimbursement limit for domestic hotel stays?"
```

**Streamlit Web UI:**
```bash
streamlit run app.py
```

---

## 🔑 Key Features

| Feature | Detail |
|---|---|
| Natural language querying | Employees ask questions in plain English |
| Semantic retrieval | FAISS dense vector similarity (Top-K=4) |
| Grounded generation | GPT-4o Mini answers only from retrieved context |
| Source traceability | Every response shows which documents were used |
| Hallucination guard | LLM instructed to refuse unanswered questions |
| Web UI | Streamlit interface with adjustable RAG parameters |
| Configurable chunking | chunk_size and overlap tunable via CLI/UI |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📚 Finance Documents Included

| Document | Coverage |
|---|---|
| `travel_expense_policy.txt` | Hotel limits, airfare, meals, cash advances, international travel |
| `vendor_payment_policy.txt` | Vendor onboarding, approval matrix, payment terms, 3-way match |
| `procurement_policy.txt` | Purchasing thresholds, RFQ/RFP, PO process, conflict of interest |
| `tax_compliance_guidelines.txt` | W-9/W-8, 1099, VAT, withholding, transfer pricing |
| `employee_finance_handbook.txt` | Petty cash, client entertainment, receipts, budget management |

---

## 📋 Sample Queries

```
- What is the maximum reimbursement limit for domestic business travel?
- What approvals are required for vendor payments above $10,000?
- Can employees claim international travel expenses?
- What documents are mandatory for a procurement request?
- What is the meal expense reimbursement policy for client meetings?
- How long does reimbursement processing take?
- What receipts are required for expense claims under $25?
- What are the tax compliance requirements for vendor payments?
- What is the process for emergency cash advances?
- Are personal days combined with a business trip reimbursable?
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 800 | Token size of each document chunk |
| `chunk_overlap` | 150 | Token overlap between adjacent chunks |
| `top_k` | 4 | Number of chunks retrieved per query |
| `temperature` | 0.0 | LLM temperature (0 = deterministic) |
| `llm_model` | gpt-4o-mini | OpenAI chat model |
| `embedding_model` | text-embedding-3-small | OpenAI embedding model |

---

## 🛡️ Limitations & Future Work

See `docs/technical_documentation.md` for a full discussion of system design decisions, limitations, and planned improvements.

---

## 📄 License

Internal use only. All finance documents are synthetic samples for demonstration purposes.
