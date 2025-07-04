# 📚 CBC Editorial Assistant Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot designed to assist CBC editorial teams by accurately answering editorial policy queries, generating SEO-optimized headlines, and summarizing article content effectively.

## 🚀 Project Overview

This chatbot harnesses advanced AI and Natural Language Processing (NLP) technologies, combining semantic retrieval via FAISS and generative language models from Hugging Face and OpenAI to support editorial workflows at CBC.

## Demo Video  (play CBC_Bot.mp4 in root)

[![FitBeat Demo](./CBC_Bot.gif)](./CBC_Bot.mp4)


## 🌟 Core Features

* **Editorial Policy Assistance**: Answers queries based on CBC's comprehensive editorial guidelines.
* **SEO Optimization**: Generates keyword-rich headlines optimized for search engines.
* **Content Summarization**: Delivers concise and engaging article summaries tailored for social media platforms.
* **Detailed Responses with Citations**: Provides accurate responses supported by transparent references from internal documents.

## 📦 Comprehensive Technical Stack

* **Vector Store**: FAISS for efficient semantic search and retrieval.
* **Embedding Model**: OpenAI's `text-embedding-3-small` model to create high-quality text embeddings.
* **Generative Models**:

  * Primary: **Mistral-7B-Instruct-v0.3** (selected for its optimal balance of performance and efficiency).
  * Alternatives: **Llama-3.1-8B-Instruct** and **OpenAI GPT-3.5 Turbo** (offered for flexibility and performance comparisons).
* **Frontend Interface**: Streamlit for a user-friendly web-based interface.
* **Scraping & Data Extraction Tools**: BeautifulSoup and Playwright to accurately extract and process CBC guideline data from dynamic web content.

## 🛠 Detailed Setup Instructions

### Installation Steps

1. **Clone the Repository**

```bash
git clone https://github.com/Sergendel/cbc-editorial-chatbot
cd CBC
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Environment Configuration**

* Set up your API keys by copying the provided .env-example file to a new .env file at the project's root directory, and insert your API credentials accordingly.

## 🚀 Launching the Project

You have two main options for running the system:

### Option A: Quick Launch (with preprocessed data and indices)

- **Launch Streamlit frontend:**
```bash
streamlit run frontend/app.py
```

- **Run standalone RAG chain (command-line interaction):**
```bash
python -m rag.chains.IntentDrivenRAGChain
```

### Option B: Full Pipeline (ETL → Indexing → Frontend or RAG)

1. **Run the Raw Data ETL Pipeline:**
```bash
python etl/runners/raw_etl_runner.py
```

2. **Run the Embedding ETL Pipeline:**
```bash
python etl/runners/embeddings_etl_runner.py
```

3. **Generate Embeddings and FAISS Indices:**
```bash
python retriever/runner.py
```

4. **Extract Metadata for Enhanced Retrieval:**
```bash
python etl/metadata_etl/extract_metadata.py
```

5. **Start the Frontend Interface:**
```bash
streamlit run frontend/app.py
```

6. **Run RAG chain standalone test:**
```bash
python -m rag.chains.IntentDrivenRAGChain
```

---

---
## Technical Choices and Considerations

### Model Selection
The chosen generative model for this project is **Hugging Face's `meta-llama/Llama-3-8B-Instruct`** due to its strong capability in handling instruction-following tasks, which aligns well with a Retrieval-Augmented Generation (RAG) system designed for assisting editorial tasks. This model was selected primarily for its robustness, ease of integration, and extensive support within the Hugging Face ecosystem.

**Beyond Hugging Face:**  
Potential alternative choices could include:

### Retrieval Enhancement with Cross-Encoder Reranker (BERT/BART)

An additional refinement that could significantly enhance retrieval accuracy and response relevance is the integration of a **cross-encoder reranker**. Specifically, models like **BERT** or **BART** can be leveraged as reranking layers:

- **BERT (Bidirectional Encoder Representations from Transformers)**:
  - Utilizes deep contextual embeddings to understand the nuanced meaning of queries in relation to retrieved document chunks.
  - Ideal for fine-grained reranking of retrieved context to ensure maximum relevance before passing information to the generative model.

- **BART (Bidirectional and Auto-Regressive Transformers)**:
  - Combines the strengths of bidirectional encoding (contextual understanding) and autoregressive decoding (generative capability).
  - Especially beneficial for tasks like headline optimization and summarization due to its generative properties.

### How It Would Work:
- Initial retrieval from FAISS (semantic search) returns a set of candidate chunks.
- A cross-encoder model (BERT or BART) then evaluates these candidates together with the query, assigning relevance scores.
- The chunks are reranked according to these scores, ensuring the most contextually accurate and relevant chunks are provided to the generative model.

Incorporating this additional layer could notably improve the chatbot's precision, context coherence, and overall responsiveness.


### Vector Store
**FAISS** was selected as the vector store for its efficiency, speed, and straightforward integration with Python-based systems. FAISS excels at handling large-scale similarity searches efficiently, which is essential for responsive retrieval in a RAG-based chatbot. Additionally, FAISS supports multiple indexing and querying strategies, providing flexibility in performance tuning.

### Chunking Method
The **Depth-First Search (DFS) chunking strategy** was employed to maintain the semantic coherence and hierarchical integrity of the scraped documents. This method was specifically chosen because:

- It preserves context within each chunk, which significantly improves retrieval accuracy and relevance.
- It leverages the hierarchical nature of the CBC editorial guidelines, ensuring logically grouped sections remain intact.
- It enables fine-grained control over chunk sizes and boundaries, which directly enhances the chatbot’s ability to retrieve contextually appropriate chunks.

### Potential Addition: Memory
In future iterations, the addition of a **memory component** could significantly enhance the user experience by maintaining conversational context over multiple queries. Implementing LangChain’s conversation memory capabilities or other stateful tracking methods would allow the chatbot to refer back to previous interactions, improving the coherence and contextual relevance of responses during extended interactions.

By integrating these technical considerations, the resulting chatbot is robust, contextually accurate, and scalable for practical deployment in an editorial environment.

### Environment and API Management

* API keys securely managed via environment variables for enhanced security and ease of deployment.

## 🎯 Sample Conversations for Demonstration

### Example: Editorial Policy Query

**User Query:**
"What’s CBC’s guideline on citing anonymous sources?"

**Chatbot Response:**
"CBC's editorial guidelines emphasize strict verification standards and careful consideration when citing anonymous sources. Such citations require explicit editorial justification and approval from senior management to ensure accountability and transparency."

**Source Documents:**

1. **Type:** Guidelines  
   **Title:** Investigative Journalism  
   **Section Path:** Investigative Journalism > Verification of User Generated Content in News Stories  
   **URL:** [Investigative Journalism](https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices/investigative-journalism)

2. **Type:** Guidelines  
   **Title:** Use of Social Media  
   **Section Path:** Use of Social Media > Verification of User Generated Content (UGC) in News Stories  
   **URL:** [Use of Social Media](https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices/use-of-social-media)

3. **Type:** Guidelines  
   **Title:** User Generated Content (UGC)  
   **Section Path:** User Generated Content (UGC) > Verification of User Generated Content (UGC) in News Stories  
   **URL:** [User Generated Content](https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices/user-generated-content)

---

### Example: SEO Headline Suggestion

**User Query:**
"Suggest an SEO-optimized headline for article ID 1.6272172."

**Chatbot Response:**
"CBC N.L. Launches Annual 'Make the Season Kind' Campaign to Support Local Food Banks Amid Rising Demand"

**Source Document:**

- **Type:** News  
  **Title:** CBC N.L. launches annual Make the Season Kind campaign to support food banks  
  **ID:** 1.6272172

---

## 📽 Optional Demo Video

play CBC_Bot.mp4 vdeo (in root).

## 📋 Evaluation and Performance Metrics

The chatbot has been meticulously developed considering:

* **Accuracy & Relevance:** Ensuring highly accurate and contextually relevant responses.
* **Generative Quality:** High-quality, readable, and appropriately structured content generation.
* **Code Quality:** Clear, modular, and maintainable code structure, facilitating future improvements.
* **Documentation Quality:** Comprehensive yet succinct documentation clearly outlining setup, usage, and technical decisions.
* **Design Thoughtfulness:** Deliberate technical choices with clear justifications and proactive improvements for user experience and performance.

---

© CBC Editorial Assistant Chatbot | Powered by Mistral-7B-Instruct-v0.3 & FAISS



## Extended Technical Documentation: Raw Data ETL Pipeline

This document describes the Extract-Transform-Load (ETL) pipeline for processing raw data in the CBC Editorial Assistant Chatbot project.

### Overview

All ETL pipeline files are located within the `etl/raw_etl` directory.

The Raw Data ETL pipeline processes news data from a JSON file and editorial guidelines from CBC's dynamic website. The pipeline includes extraction, transformation, and loading stages, orchestrated for automated execution.

### Extraction Phase

* **Abstract Base Class** (`extract_base.py`): Defines the structure for extractor classes.

* **News Extraction** (`extract_raw_news.py`):

  * Loads news data from a provided JSON file.
  * Saves extracted data to an intermediate JSON file.

* **Guidelines Extraction** (`extract_raw_guidelines.py`, `scraper.py`):

  * Utilizes Playwright for dynamic web scraping from CBC's JSP website.
  * Extracts and saves individual HTML pages locally.
  * Handles special cases and errors explicitly.

### Transformation Phase

* **Abstract Base Class** (`transform_base.py`): Defines the structure for transformer classes.

* **News Transformation** (`transform_news.py`):

  * Cleans, validates, and structures individual news articles.
  * Filters incomplete or invalid entries.
  * Normalizes date and text fields to ensure consistency.

* **Guidelines Transformation** (`transform_guidelines.py`):

  * Parses HTML content using BeautifulSoup.
  * Structures guideline content into nested, readable JSON.
  * Adds metadata like URL and timestamps to structured data.

### Loading Phase

* **News Data Loading** (`load_news_json.py`):

  * Saves structured news data to a JSON file.

* **Guidelines Data Loading** (`load_guidelines_json.py`):

  * Saves structured guidelines data to a JSON file.

### ETL Automation Runner (`raw_etl_runner.py`)

* Orchestrates the execution of the ETL scripts for news and guidelines sequentially.
* Automatically logs each step and handles subprocess execution.

### Execution Instructions

To execute the complete Raw Data ETL pipeline, run:

```bash
python etl/runners/raw_etl_runner.py
```

### Logging and Error Handling

* Each ETL stage is comprehensively logged.
* Errors are explicitly caught, logged, and handled to ensure robustness.

This structured ETL pipeline ensures efficient, reliable data processing, providing a solid foundation for further analysis and integration into retrieval and generative systems.





## Technical Documentation: FAISS Vector Data Storage and Retrieval

This documentation describes the implementation details and usage of FAISS (Facebook AI Similarity Search) vector storage for efficient retrieval in the CBC Editorial Assistant Chatbot project.

### Overview

The FAISS integration enables efficient storage and retrieval of precomputed text embeddings, facilitating rapid and relevant information retrieval for chatbot responses.

### Project Structure

All related files are explicitly located in the `retriever/` directory:

* `embeddings.py`: Loads precomputed embeddings.
* `indexing.py`: Creates and stores FAISS indices.
* `runner.py`: Automates the embedding loading and indexing processes.
* `retrieval_demo.py`: Demonstrates retrieval using the FAISS index.

---

### 1. Loading Precomputed Embeddings (`embeddings.py`)

* **Purpose:**
  Loads precomputed text embeddings and corresponding metadata from disk.

* **Functionality:**

  * Loads embeddings from `.npy` NumPy arrays and metadata from JSON.
  * Ensures embeddings and metadata counts match for consistency.

* **Usage:**

```python
from retriever.embeddings import load_embeddings

texts, embeddings, metadata = load_embeddings(
    "path/to/embeddings.npy",
    "path/to/metadata.json"
)
```

---

### 2. Index Creation with FAISS (`indexing.py`)

* **Purpose:**
  Creates and saves a FAISS vector index using precomputed embeddings.

* **Key Components:**

  * `PrecomputedEmbeddings` class: Custom wrapper class.
  * `create_faiss_index` function: Initializes the FAISS index and stores it locally.

* **Usage:**

```python
from retriever.indexing import create_faiss_index

create_faiss_index(
    texts=texts,
    embeddings=embeddings,
    metadata=metadata,
    save_path="path/to/save/faiss/index"
)
```

---

### 3. Index Generation Automation (`runner.py`)

* **Purpose:**
  Automates loading of embeddings and creation of FAISS indices for both guidelines and news datasets.

* **Execution:**

```bash
python retriever/runner.py
```

* **Workflow:**

  * Loads guidelines and news embeddings separately.
  * Creates separate FAISS indices for each data source.

---

### 4. Retrieval Demonstration (`retrieval_demo.py`)

* **Purpose:**
  Demonstrates how to perform semantic retrieval queries against FAISS indices.

* **Key Components:**

  * `QueryEmbeddings` class: Generates embeddings for incoming queries using `embedding_model_function`.
  * `demo_retrieval` function: Executes a query against a specified FAISS index and returns top-k relevant results.

* **Example Usage:**

```bash
python retriever/retrieval_demo.py
```

* **Sample Query:**

```python
demo_retrieval(
    index_name="guidelines_faiss_index",
    query="What does CBC say about privacy?",
    k=3
)
```

---

### Embedding and Index Data Paths

Embeddings and indices follow a consistent directory structure:

```
data/
├── embeddings_storage/
│   ├── guidelines_embeddings.npy
│   ├── guidelines_metadata.json
│   ├── news_embeddings.npy
│   └── news_metadata.json
└── vector_indexes/
    ├── guidelines_faiss_index/
    └── news_faiss_index/
```

---

### Technical Requirements

* Python
* FAISS
* NumPy
* LangChain

---

This detailed documentation outlines the implementation, functionality, and usage of FAISS vector indexing and retrieval components to support efficient semantic information retrieval in the CBC Editorial Assistant Chatbot.


## Technical Documentation: Retrieval-Augmented Generation (RAG) Chain

This document covers the Retrieval-Augmented Generation (RAG) component implemented in the CBC Editorial Assistant Chatbot. It outlines the flow, methods, and key technical details.

### Overview

The RAG chain is designed to dynamically retrieve relevant information using semantic search and metadata matching, guided by user intent classification, to provide accurate and context-aware responses.

### Project Structure

All related files are located in the `rag/` directory:

* `chains/intent_classifier.py`: Classifies user intents.
* `chains/IntentDrivenRAGChain.py`: Core logic integrating retrieval with intent-driven prompts.
* `chains/query_classifier.py`: Categorizes queries into guidelines, news, or mixed categories.
* `utils/prompt_router.py`: Selects prompts based on identified intents.
* `utils/prompts.py`: Defines prompt templates.
* `utils/query_metadata_extractor.py`: Extracts structured metadata from queries.
* `etl/metadata_etl/extract_metadata.etl`: Creates the metadata lookup table (`metadata_lookup.json`) for efficient metadata checks.

### Flow and Methods

#### 1. User Query Handling

* **Intent Classification** (`intent_classifier.py`):

  * User queries are classified into predefined intents (e.g., policy queries, headline requests).
  * Utilizes a generative model with a structured prompt for classification accuracy.

#### 2. Metadata Extraction

* **Query Metadata Extraction** (`query_metadata_extractor.py`):

  * Extracts structured metadata (article IDs, timestamps, URLs) from user queries using regular expressions.

#### 3. Metadata Lookup

* **Metadata Validation** (`metadata_lookup.json`):

  * The lookup table created via `extract_metadata.etl` is used to validate and rapidly check the existence of metadata extracted from queries.

#### 4. Context Retrieval

* **Semantic Retrieval** (`IntentDrivenRAGChain.py`):

  * Loads pre-built FAISS indexes for semantic similarity search.
  * Retrieves the most relevant context based on semantic similarity and explicit metadata matching.
  * Implements fallback mechanisms if initial retrieval is insufficient or unclear.

#### 5. Prompt Routing and Generation

* **Prompt Selection** (`prompt_router.py`, `prompts.py`):

  * Dynamically selects prompt templates based on classified intents to guide the response generation.
  * Prompts are tailored for various intents, such as policy summaries, SEO headlines, or article summaries.

#### 6. Generative Response

* **Generation** (`IntentDrivenRAGChain.py`):

  * Uses a generative model to generate responses based explicitly on retrieved context and selected prompts.

### Technologies Used

* **FAISS**: Efficient semantic search.
* **LangChain**: Prompt templating and vector store management.
* **Hugging Face/OpenAI**: Generative language models for response generation.
* **Regular Expressions**: Structured metadata extraction.

### Usage

To run the RAG chain component, execute:

```bash
python rag/chains/IntentDrivenRAGChain.py
```

Example Query:

```python
response = intent_driven_rag_chain("Summarize this article for a Twitter post.")
```

### Response and Source Metadata

Responses include:

* The generated answer explicitly crafted based on retrieved context.
* Metadata of retrieved documents, providing transparency and traceability of responses.

### Best Practices Followed

* Intent classification to guide accurate context retrieval and prompt selection.
* Explicit fallback strategies ensuring robust retrieval.
* Metadata-driven retrieval for precision.
* Detailed logging and clear modular structure for maintainability.

This comprehensive structure ensures the chatbot delivers highly relevant, accurate, and context-sensitive responses efficiently.


## Technical Documentation: Frontend Component

This document describes the Streamlit-based frontend of the CBC Editorial Assistant Chatbot.

### Overview

The frontend provides a simple, intuitive user interface for interacting with the Retrieval-Augmented Generation (RAG) backend, allowing users to submit queries and receive structured, intent-driven responses.

### Technology

* **Framework**: Streamlit

### File Location

* `frontend/app.py`

### Interface Features

* **Query Input**: Users enter queries in a text area.
* **Response Generation**: Button-triggered generation of chatbot responses.
* **Structured Output**: Clearly formatted responses with categorized source documents (Guidelines, News).
* **Error Handling**: Graceful handling and display of runtime errors.

### How to Run

To launch the frontend, execute the following command from the project root:

```bash
streamlit run frontend/app.py
```


## Technical Documentation: Model Components

This documentation covers the embedding and generative models utilized in the CBC Editorial Assistant Chatbot project.

### Project Structure

The model-related files are located in the `models/` directory:

* `embedding_model.py`: Handles text embeddings.
* `generative_model.py`: Manages various generative models for response generation.

### Embedding Model

* **Purpose**:

  * Generates numerical representations (embeddings) of textual inputs for semantic search and retrieval tasks.

* **Implementation**:

  * Utilizes OpenAI's embedding API (`text-embedding-3-small`).

* **Usage Example**:

```python
from models.embedding_model import embedding_model_function

embeddings = embedding_model_function("Your text here")
```

### Generative Models

* **Purpose**:

  * Generates human-like textual responses based on retrieved contexts.

* **Available Generative Models**:

  * **Mistral-7B-Instruct-v0.3** (primary)
  * **Llama-3.1-8B-Instruct** (alternative)
  * **OpenAI GPT-3.5 Turbo** (alternative)

* **Implementation Details**:

  * Utilizes Hugging Face Endpoints and OpenAI API for model inference.
  * Each model instance is created as a singleton to optimize resource usage.

* **Usage Examples**:

```python
from models.generative_model import get_generative_model

model = get_generative_model()
response = model.invoke("Explain Retrieval-Augmented Generation briefly.")
```

### Configuration and Environment

* Models require API keys set in environment variables (`OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`).
* Ensure `.env` file is properly configured with these keys.

### Testing and Verification

Run a simple verification using:

```bash
python models/generative_model.py
```

### Best Practices

* Singleton pattern for model instantiation.
* Clear and explicit API key management via environment variables.
* Modular design to easily swap or update models.

This structured implementation of embedding and generative models ensures efficient, reliable, and scalable text processing for your chatbot application.

## Technical Documentation: Administrative Configuration

This document outlines the administrative configuration setup for the CBC Editorial Assistant Chatbot, detailing configuration management and API key handling.

### Configuration Files

* **Python Configuration** (`config/config.py`):

  * Dynamically loads and provides easy access to various configuration settings stored in YAML format.

* **YAML Configuration** (`config/config.yml`):

  * Stores structured project settings like paths, model names, and batch sizes.

* **Environment Variables** (`.env`):

  * Securely stores sensitive API keys for external services such as OpenAI, Hugging Face, and others.

### Usage

Configuration settings can be accessed through the `Config` class:

```python
from config.config import Config

config = Config("path/to/config.yml")
print(config.processed_news_path)
```

### API Keys

* Stored securely in `.env`.
* Ensure keys for OpenAI, Hugging Face, and other services are correctly set.

### Security Best Practices

* Never expose `.env` in public repositories.
* Regularly rotate API keys to maintain security.

### Running Instructions

Ensure `.env` and configuration files are properly set before running the project components.

This administrative setup supports maintainable, secure, and easily configurable project infrastructure.
