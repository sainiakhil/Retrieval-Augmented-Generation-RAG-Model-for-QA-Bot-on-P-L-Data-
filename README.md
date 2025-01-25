# Retrieval Augmented Generation RAG Model for QA Bot on P&LData

---

# **Financial Document Q&A Bot**

This repository provides a two-part solution for querying and analyzing financial data from Profit and Loss (P&L) statements. It utilizes **Retrieval-Augmented Generation (RAG)** for answering complex financial queries and provides an **interactive interface** for users to upload financial documents and query them.

---

## **Repository Structure**

- `Part 1_ (RAG) Model for QA Bot on P&L Data.ipynb`:
  Implements a RAG-based Question Answering (QA) Bot for analyzing financial data from P&L statements.
- `Part 2_ Interactive QA Bot Interface for Financial Data.ipynb`:
  Provides an interactive **Streamlit** interface for the QA Bot with document upload and live querying capabilities.

---

## **Part 1: RAG Model for QA Bot on P&L Data**

### **Overview**
This part builds the backend for the QA Bot using a RAG-based approach. It parses P&L tables from PDF documents, stores them in a vector database, and uses a large language model (LLM) to generate responses based on the context retrieved.

### **Key Features**
- Extracts P&L tables and financial data from PDF files.
- Uses **NousResearch/Meta-Llama-3.1-8B-Instruct** as the LLM with 4-bit quantization for GPU efficiency.
- Stores document embeddings in a vector database (e.g., FAISS or Qdrant).
- Retrieves relevant chunks using similarity search and generates answers.

### **Setup Instructions**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/financial-qa-bot.git
   cd financial-qa-bot
   ```

2. **Install Dependencies**:
   ```bash
   pip install transformers sentence_transformers accelerate bitsandbytes llama-index
   ```

3. **Run the Notebook**:
   Open `Part 1_ (RAG) Model for QA Bot on P&L Data.ipynb` in Jupyter or Google Colab and follow the instructions.

### **Usage**
- Place P&L PDFs in a directory (e.g., `/content/data`).
- Configure the LLM and vector store settings in the notebook.
- Query examples:
  - "What is the net income for Q2 2024?"
  - "How do the operating expenses compare for Q1 2023?"

---

## **Part 2: Interactive QA Bot Interface for Financial Data**

### **Overview**
This part builds an interactive interface for the QA Bot using **Streamlit**. Users can upload financial documents, query the bot, and receive answers in real time.

### **Key Features**
- Web-based interface for uploading and querying financial documents.
- Utilizes **Qwen/Qwen2.5-3B-Instruct** as the LLM for efficient inference.
- Embedding generation via **BAAI/bge-large-en-v1.5**.
- Exposes the app publicly using **ngrok**.

### **Setup Instructions**

#### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/financial-qa-bot.git
cd financial-qa-bot
```

#### **2. Install Dependencies**
```bash
pip install transformers sentence_transformers accelerate bitsandbytes \
    llama-index llama-index-embeddings-huggingface llama-index-llms-huggingface \
    streamlit pyngrok
```

#### **3. Add Your ngrok Token**
- Sign up at [ngrok](https://ngrok.com/).
- Retrieve your **authtoken** and add it to the `app.py` file:
  ```python
  ngrok.set_auth_token("<your-ngrok-token>")
  ```

#### **4. Run the Streamlit App**
```bash
streamlit run app.py
```

### **Usage**
1. Access the Streamlit app URL (printed in the terminal via ngrok).
2. Upload financial documents in PDF format.
3. Enter queries such as:
   - "What is the gross profit for Q3 2024?"
   - "How do net income and expenses compare for Q2 2023?"

### **Example Output**
- **Query**: "What are the employee benefit expenses for Q2 2024?"
- **Response**: "19,527"
- **Source Chunks**:
  - Text and relevance scores for the most relevant document parts.

---

## **Configuration Details**

### **Quantization Configuration**
Both parts utilize 4-bit quantization to reduce GPU memory usage:
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
```

### **LLM and Embedding Models**
- **LLM**:
  - `NousResearch/Meta-Llama-3.1-8B-Instruct` for Part 1.
  - `Qwen/Qwen2.5-3B-Instruct` for Part 2.
- **Embedding**:
  - `BAAI/bge-large-en-v1.5`.

---

## **Troubleshooting**

### **1. GPU Memory Issues**
- Use smaller LLMs (e.g., Qwen2.5-3B or Llama-2-7B).
- Enable offloading for large model layers:
  ```python
  offload_folder="/content/offload"
  offload_state_dict=True
  ```
