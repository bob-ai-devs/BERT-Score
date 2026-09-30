# 🧠 BERTScore Component Analyzer

A Streamlit-based interactive application for evaluating the semantic similarity between a generated response and a reference text using **BERTScore**.

The application provides detailed **Precision, Recall, and F1** scores along with tokenization details, raw tensor outputs, model configuration, and the original input texts.

---

## 📌 Overview

**BERTScore Component Analyzer** is designed to make BERTScore evaluation easier to inspect and understand.

Instead of displaying only a single BERTScore value, the application exposes the individual components:

* **Precision**
* **Recall**
* **F1 Score**
* Tokenized representation of both texts
* Raw BERTScore tensor outputs
* Model and language configuration
* Input comparison details

This makes the application useful for evaluating and debugging **LLM-generated responses**, NLP systems, question-answering systems, summarization systems, and other text-generation applications.

---

## ✨ Features

### 1. Generated Response vs Reference

The application accepts two text inputs:

* **Our Response** — the generated/candidate response
* **Reference** — the expected or reference response

Example:

```text
Our Response:
The cat is sitting on the mat.

Reference:
A cat is sitting on a mat.
```

---

### 2. BERTScore Evaluation

The application calculates three BERTScore components:

| Metric    | Description                                                                      |
| --------- | -------------------------------------------------------------------------------- |
| Precision | Measures how well the candidate/response tokens semantically match the reference |
| Recall    | Measures how well the reference information is captured by the candidate         |
| F1        | Harmonic mean of Precision and Recall                                            |

The scores are displayed with six decimal places.

---

### 3. Multiple Languages

The application provides language selection for:

* English (`en`)
* Hindi (`hi`)
* French (`fr`)
* German (`de`)
* Spanish (`es`)

The selected language is passed to the BERTScore calculation.

---

### 4. Multiple BERTScore Models

Users can select from the following configurations:

| Display Name            | Model                |
| ----------------------- | -------------------- |
| Default BERTScore Model | BERTScore default    |
| RoBERTa Large           | `roberta-large`      |
| RoBERTa Base            | `roberta-base`       |
| DistilRoBERTa Base      | `distilroberta-base` |

When **Default BERTScore Model** is selected, the application uses the BERTScore library's default model behavior.

---

### 5. Score Component Table

After calculation, the application displays a structured table:

| Component |    Score |
| --------- | -------: |
| Precision | 0.xxxxxx |
| Recall    | 0.xxxxxx |
| F1        | 0.xxxxxx |

This provides a convenient component-level view of the evaluation.

---

### 6. Tokenization Analysis

The application tokenizes both inputs using a Hugging Face tokenizer.

Two token tables are displayed:

#### Our Response Tokens

| Position | Token |
| -------: | ----- |
|        1 | The   |
|        2 | cat   |
|        3 | is    |
|      ... | ...   |

#### Reference Tokens

| Position | Token |
| -------: | ----- |
|        1 | A     |
|        2 | cat   |
|        3 | is    |
|      ... | ...   |

This allows users to inspect how the selected model tokenizer represents the input text.

---

### 7. Cached Tokenizer

The Hugging Face tokenizer is loaded using Streamlit's resource cache:

```python
@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer(model_type: str):
    return AutoTokenizer.from_pretrained(model_type)
```

This avoids repeatedly loading the tokenizer during Streamlit reruns.

---

### 8. Raw BERTScore Output

The application also exposes the original BERTScore tensor outputs.

Example:

| Metric    | Tensor        |    Value |
| --------- | ------------- | -------: |
| Precision | tensor([...]) | 0.xxxxxx |
| Recall    | tensor([...]) | 0.xxxxxx |
| F1        | tensor([...]) | 0.xxxxxx |

This is useful when debugging or validating the underlying BERTScore calculation.

---

### 9. Comparison Details

The application displays the configuration used for the calculation:

| Parameter             | Value                            |
| --------------------- | -------------------------------- |
| Language              | Selected language                |
| Model Selection       | Selected model                   |
| Actual Model Argument | Actual model passed to BERTScore |
| IDF Weighting         | No                               |
| Baseline Rescaling    | No                               |

This helps make individual evaluation runs reproducible and easier to inspect.

---

### 10. Original Input Display

The original **Our Response** and **Reference** texts are displayed after evaluation so that the calculated scores can be reviewed alongside the exact input text.

---

## 🧮 BERTScore

BERTScore evaluates text similarity using contextual embeddings rather than relying only on exact word overlap.

The application focuses on three primary components.

### Precision

Precision measures the semantic similarity of the candidate/response tokens against the reference.

Conceptually:

```text
Candidate → Reference
```

A higher Precision indicates that the generated response contains content that is semantically aligned with the reference.

---

### Recall

Recall measures how well the reference content is captured by the candidate response.

Conceptually:

```text
Reference → Candidate
```

A higher Recall indicates that more of the reference's semantic content is represented in the generated response.

---

### F1

F1 combines Precision and Recall using their harmonic mean:

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

F1 provides a combined measure of the two components.

---

## 🏗️ Application Flow

The application follows this workflow:

```text
User Input
    │
    ├── Our Response
    │
    └── Reference
          │
          ▼
   BERTScore Settings
          │
          ├── Language
          └── Model
          │
          ▼
    BERTScore Calculation
          │
          ├── Precision
          ├── Recall
          └── F1
          │
          ▼
   Detailed Analysis
          │
          ├── Score Components
          ├── Tokenization
          ├── Raw Tensor Output
          ├── Comparison Details
          └── Input Text
```

---

## 📁 Project Structure

A minimal project structure can be:

```text
bert-score-component-analyzer/
│
├── app.py
├── requirements.txt
└── README.md
```

Where:

* `app.py` — Streamlit application
* `requirements.txt` — Python dependencies
* `README.md` — Project documentation

---

## 📦 Requirements

The application requires Python packages including:

```text
streamlit
pandas
bert-score
transformers
```

A sample `requirements.txt` is:

```text
streamlit
pandas
bert-score
transformers
```

Depending on the environment and BERTScore backend, the required PyTorch dependencies may also be installed automatically or may need to be specified separately.

For a controlled deployment environment, it is recommended to pin package versions after validating the application.

---

## 🚀 Installation

### 1. Clone or create the project

```bash
git clone <repository-url>
cd bert-score-component-analyzer
```

Or place `app.py` in a new project directory.

---

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install the packages directly:

```bash
pip install streamlit pandas bert-score transformers
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in the browser, typically at:

```text
http://localhost:8501
```

---

## 🖥️ Using the Application

### Step 1 — Enter the generated response

Enter the output produced by your model/system in:

```text
Our Response
```

### Step 2 — Enter the reference

Enter the expected/reference text in:

```text
Reference
```

### Step 3 — Select language

Choose the appropriate language:

```text
English
Hindi
French
German
Spanish
```

### Step 4 — Select the BERTScore model

Choose either:

```text
Default BERTScore Model
RoBERTa Large
RoBERTa Base
DistilRoBERTa Base
```

### Step 5 — Calculate

Click:

```text
🚀 Calculate BERTScore
```

The application then generates the detailed evaluation.

---

## 📊 Output Sections

After successful calculation, the application displays:

```text
📊 BERTScore
```

Displays:

* Precision
* Recall
* F1

Then:

```text
🔍 Score Components
```

Displays the component-level score table.

Then:

```text
🔤 Tokenization
```

Displays tokens for both input texts.

Then:

```text
🧮 Raw BERTScore Output
```

Displays the underlying tensor outputs and scalar values.

Then:

```text
📝 Comparison Details
```

Displays the configuration used for the calculation.

Then:

```text
📄 Input Text
```

Displays both original inputs.

Finally:

```text
📐 BERTScore Components
```

Provides a short explanation of Precision, Recall, and F1.

---

## 🔬 Technical Implementation

### BERTScore

The application uses:

```python
from bert_score import score as bert_score
```

For the default model:

```python
P_o, R_o, F1_o = bert_score(
    [our_resp],
    [reference],
    lang=lang,
    verbose=False
)
```

When a specific model is selected:

```python
P_o, R_o, F1_o = bert_score(
    [our_resp],
    [reference],
    lang=lang,
    model_type=selected_model,
    verbose=False
)
```

The resulting tensors are converted into scalar values:

```python
precision = P_o.item()
recall = R_o.item()
f1 = F1_o.item()
```

---

## 🔤 Tokenization

The application uses Hugging Face Transformers:

```python
from transformers import AutoTokenizer
```

The tokenizer is loaded with:

```python
AutoTokenizer.from_pretrained(model_type)
```

For an explicitly selected model, the same model is used for tokenization.

When the default BERTScore model is selected, the application uses:

```text
roberta-large
```

for the tokenization display.

This tokenizer is used specifically to visualize tokens and should not be interpreted as changing the underlying BERTScore model selection.

---

## ⚙️ BERTScore Configuration

The current implementation uses:

```text
IDF Weighting: No
Baseline Rescaling: No
```

No custom IDF weighting or baseline rescaling is applied by the application.

---

## ⚠️ Important Considerations

### BERTScore Is a Semantic Metric

BERTScore evaluates semantic similarity using contextual representations.

Therefore, two sentences can receive a relatively high score even when their wording is different.

For example:

```text
The automobile is moving quickly.
```

and:

```text
The car is traveling fast.
```

may be considered semantically similar.

---

### Scores Should Not Be Interpreted in Isolation

A BERTScore value is dependent on factors such as:

* Input text
* Reference quality
* Language
* Model used
* Domain
* Text length
* Semantic similarity

Therefore, scores should generally be interpreted in the context of the evaluation task.

---

### Model Selection Can Affect Scores

Different underlying models may produce different BERTScore results.

For example:

```text
roberta-large
roberta-base
distilroberta-base
```

should not necessarily be expected to produce identical Precision, Recall, or F1 values.

For meaningful comparisons across experiments, keep the evaluation configuration consistent.

---

## 🛠️ Error Handling

The application includes validation for empty inputs.

If **Our Response** is empty:

```text
Please enter Our Response.
```

If **Reference** is empty:

```text
Please enter the Reference.
```

BERTScore calculation errors are displayed directly in the application.

Tokenizer-loading errors are handled separately so that a tokenization-display problem does not prevent the main BERTScore result from being shown.

---

## 🎯 Potential Use Cases

This application can be used for:

* LLM response evaluation
* NLP model evaluation
* Generative AI benchmarking
* Question-answering evaluation
* Summarization evaluation
* Chatbot response comparison
* RAG response evaluation
* Prompt engineering experiments
* Model comparison
* Semantic similarity analysis
* AI-generated text quality analysis

---

## 🔄 Example Evaluation

### Input

**Our Response**

```text
The cat is sitting on the mat.
```

**Reference**

```text
A cat is sitting on a mat.
```

### Configuration

```text
Language: en
Model: Default BERTScore Model
IDF Weighting: No
Baseline Rescaling: No
```

### Output

The application reports:

```text
Precision: <calculated score>
Recall:    <calculated score>
F1:        <calculated score>
```

The exact values depend on the installed BERTScore version, selected model, language configuration, and underlying model behavior.

---

## 🔐 Privacy

The application processes the entered text locally within the Python/Streamlit application environment.

No external application-specific storage or database is implemented in the provided code.

If deployed on a cloud platform, the privacy characteristics will additionally depend on the hosting environment and its configuration.

Avoid entering confidential, personal, financial, medical, or otherwise sensitive information unless the deployment environment has been approved for such data.

---

## 📈 Future Enhancements

Potential future enhancements include:

* Batch evaluation using CSV/Excel files
* Side-by-side model comparison
* Historical score tracking
* Score distribution charts
* Multiple reference responses
* CSV/Excel export
* PDF evaluation reports
* Sentence-level similarity visualization
* Token-level similarity visualization
* Evaluation dashboard
* Automated LLM evaluation pipeline
* Integration with RAG evaluation workflows

---

## 📄 License

Add the appropriate project license here, for example:

```text
MIT License
```

The licenses of third-party libraries used by this application remain applicable.

---

## 👨‍💻 Technology Stack

| Technology                | Purpose                            |
| ------------------------- | ---------------------------------- |
| Python                    | Application development            |
| Streamlit                 | Interactive web interface          |
| BERTScore                 | Semantic text evaluation           |
| Hugging Face Transformers | Tokenization                       |
| Pandas                    | Data tables and score presentation |
| PyTorch                   | Model/backend computation          |

---

## 🧠 Summary

**BERTScore Component Analyzer** provides an interactive way to inspect semantic similarity between a generated response and a reference.

Rather than exposing only a single evaluation score, it provides:

```text
Generated Response
        +
Reference
        │
        ▼
    BERTScore
        │
   ┌────┼────┐
   ▼    ▼    ▼
Precision Recall F1
   │    │    │
   └────┼────┘
        ▼
Detailed Analysis
        │
        ├── Score Components
        ├── Tokenization
        ├── Raw Tensor Output
        ├── Model Configuration
        └── Input Comparison

This makes the application suitable for **LLM/NLP evaluation, debugging, benchmarking, and semantic similarity analysis**.
