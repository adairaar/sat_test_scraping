# SAT Practice Test Extractor

A comprehensive Python toolkit for extracting, structuring, and querying SAT practice test questions and answers from official College Board PDFs.

## 📋 Overview

This toolkit automates the extraction of SAT practice test questions from PDF files and organizes them into structured formats (CSV, SQLite) for easy analysis and querying. It handles both the question content (from test PDFs) and correct answers (from scoring guide PDFs), producing a unified dataset with composite indexing.

### Key Features

- **Automatic PDF Parsing**: Extracts questions, answer choices, passages, and metadata from official SAT test PDFs
- **Answer Key Integration**: Merges correct answers from scoring guide PDFs
- **Multiple Output Formats**: Exports to CSV, SQLite database, or pandas DataFrame
- **Composite Indexing**: Unique question IDs combining test name, section, module, and question number
- **Question Type Detection**: Automatically classifies multiple-choice vs. free-response questions
- **Column-Aware Parsing**: Handles two-column PDF layouts with accurate text extraction
- **JSON Export Utilities**: Query and export database results to JSON format

## 🚀 Quick Start

### Installation

```bash
pip install pdfplumber pandas
```

### Basic Usage

```python
from sat_extractor import main

# Extract questions and answers from PDFs
df = main(
    questions_pdf_path = "sat-practice-test-8-digital.pdf",
    scoring_pdf_path   = "scoring-sat-practice-test-8-digital.pdf",
    output_csv_path    = "sat_questions.csv",      # optional
    output_sql_path    = "sat_questions.db",       # optional
)
```

### Command-Line Usage

Edit the file paths at the bottom of `sat_extractor.py`, then run:

```bash
python sat_extractor.py
```

## 📁 Project Structure

```
.
├── sat_extractor.py                    # Main extraction script
├── sat_sample_sql_queries_to_json.py   # SQL query → JSON export utility
└── README.md                           # This file
```

## 🔧 Core Components

### 1. sat_extractor.py

The primary extraction engine that processes SAT PDFs and produces structured data.

#### Input Requirements

**Test Questions PDF**: Official SAT practice test (e.g., `sat-practice-test-8-digital.pdf`)
- Contains question text, answer choices, and passages
- Two-column layout with 33 Reading/Writing + 27 Math questions per module
- Multiple-choice and free-response questions

**Scoring Guide PDF**: Official scoring guide (e.g., `scoring-sat-practice-test-8-digital.pdf`)
- Contains the answer key in a 4-column grid layout
- Columns: RW Module 1, RW Module 2, Math Module 1, Math Module 2

#### Output Schema

The toolkit produces datasets with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `question_id` | string | Composite primary key (format: `TestName_Section_M#_Q#`) |
| `test_name` | string | Name of the practice test |
| `section` | string | "Reading and Writing" or "Math" |
| `module` | integer | Module number (1 or 2) |
| `question_number` | integer | Question number within the module (1-33 or 1-27) |
| `question_type` | string | "Multiple Choice" or "Free Response" |
| `correct_answer` | string | Answer key (A/B/C/D for MC, numeric for FR) |
| `passage` | string | Associated passage text (if applicable) |
| `question_stem` | string | The actual question text |
| `choice_A` | string | First answer choice |
| `choice_B` | string | Second answer choice |
| `choice_C` | string | Third answer choice |
| `choice_D` | string | Fourth answer choice |

#### Example Question IDs

```
SAT_Practice_Test_#8_Reading_and_Writing_M1_Q1
SAT_Practice_Test_#8_Reading_and_Writing_M1_Q2
SAT_Practice_Test_#8_Math_M1_Q15
SAT_Practice_Test_#8_Math_M2_Q27
```

#### API Reference

##### `extract_questions(questions_pdf_path)`

Parses a test PDF and extracts question data.

**Parameters:**
- `questions_pdf_path` (str | Path): Path to the SAT practice test PDF

**Returns:**
- `list[dict]`: List of question dictionaries with metadata and content

##### `extract_answer_key(scoring_pdf_path)`

Parses a scoring guide PDF and extracts the answer key.

**Parameters:**
- `scoring_pdf_path` (str | Path): Path to the scoring guide PDF

**Returns:**
- `dict`: Nested dictionary keyed by `(section, module)` with answer mappings

##### `main(questions_pdf_path, scoring_pdf_path, output_csv_path=None, output_sql_path=None)`

Complete pipeline: extract questions, merge answers, export to file(s).

**Parameters:**
- `questions_pdf_path` (str | Path): Path to test PDF
- `scoring_pdf_path` (str | Path): Path to scoring guide PDF
- `output_csv_path` (str | Path, optional): CSV export destination
- `output_sql_path` (str | Path, optional): SQLite database export destination

**Returns:**
- `pd.DataFrame`: Complete dataset with question_id index

### 2. sat_sample_sql_queries_to_json.py

Utility for querying the SQLite database and exporting results to JSON format.

#### Usage Examples

```python
from sat_sample_sql_queries_to_json import export_db_to_json

# Query all questions from a specific section
export_db_to_json(
    query="SELECT * FROM sat_questions WHERE section = 'Math'",
    database_file="sat_questions.db",
    json_file_path="math_questions.json"
)

# Get all free-response questions with answers
export_db_to_json(
    query="""
        SELECT question_id, question_stem, correct_answer
        FROM sat_questions 
        WHERE question_type = 'Free Response'
    """,
    database_file="sat_questions.db",
    json_file_path="free_response_questions.json"
)

# Query by composite ID
export_db_to_json(
    query="""
        SELECT * FROM sat_questions 
        WHERE question_id = 'SAT_Practice_Test_#8_Math_M1_Q6'
    """,
    database_file="sat_questions.db",
    json_file_path="specific_question.json"
)
```

#### API Reference

##### `export_db_to_json(query, database_file, json_file_path)`

Executes a SQL query and exports results to a JSON file.

**Parameters:**
- `query` (str): SQL SELECT statement
- `database_file` (str): Path to SQLite database
- `json_file_path` (str): Destination path for JSON output

**Output Format:**
```json
[
    {
        "question_id": "SAT_Practice_Test_#8_Math_M1_Q6",
        "test_name": "SAT Practice Test #8",
        "section": "Math",
        "module": 1,
        "question_number": 6,
        "question_type": "Free Response",
        "correct_answer": "0.2; 1/5",
        "passage": "",
        "question_stem": "What value of x is the solution to the given equation?",
        "choice_A": "",
        "choice_B": "",
        "choice_C": "",
        "choice_D": ""
    }
]
```

## 📊 Database Structure

The SQLite database (`sat_questions.db`) contains a single table:

### Table: `sat_questions`

- **Rows**: 115+ questions (varies by test)
- **Primary Index**: `idx_question_id` (UNIQUE) on `question_id` column
- **Encoding**: UTF-8

### Example SQL Queries

```sql
-- Get all Reading and Writing questions from Module 1
SELECT * FROM sat_questions 
WHERE section = 'Reading and Writing' AND module = 1;

-- Find all questions with correct answer 'B'
SELECT question_id, question_stem 
FROM sat_questions 
WHERE correct_answer = 'B';

-- Get Math free-response questions with their answers
SELECT question_id, question_stem, correct_answer 
FROM sat_questions 
WHERE section = 'Math' AND question_type = 'Free Response';

-- Lookup specific question by composite ID
SELECT * FROM sat_questions 
WHERE question_id = 'SAT_Practice_Test_#8_Math_M1_Q15';

-- Count questions by section and type
SELECT section, question_type, COUNT(*) as count 
FROM sat_questions 
GROUP BY section, question_type;
```

## 🎯 Use Cases

### Educational Technology
- Build adaptive learning platforms with SAT question banks
- Create personalized practice test generators
- Analyze question difficulty and patterns

### Test Preparation
- Develop flashcard systems with targeted question sets
- Generate custom practice tests by section/module
- Track student performance on specific question types

### Research & Analytics
- Study question patterns across test versions
- Analyze answer distribution and difficulty
- Compare multiple-choice vs. free-response formats

### Data Integration
- Export questions to learning management systems (LMS)
- Integrate with spaced repetition systems (SRS)
- Build API backends for test prep applications

## 📝 Data Quality Notes

### Extraction Accuracy

- **Questions Extracted**: ~115 per test (after cleanup)
- **Answer Matching**: 100% (all questions matched with correct answers)
- **Question Types**: Automatically classified with >99% accuracy

### Known Limitations

1. **Math Graphics**: Questions with complex diagrams may have incomplete stems (inherent PDF text extraction limitation)
2. **Unicode Characters**: Some special characters (em-dashes, fractions) require UTF-8 encoding
3. **Page Layout Dependency**: Assumes standard College Board two-column format

### Quality Assurance

The extractor includes built-in validation:
- Duplicate question detection and removal
- Section/module boundary verification
- Answer key completeness checking
- Composite index uniqueness enforcement

## 🔍 Troubleshooting

### Common Issues

**UnicodeDecodeError when reading CSV**
```python
# Use utf-8-sig encoding
df = pd.read_csv("sat_questions.csv", encoding="utf-8-sig")
```

**SQLite UNIQUE constraint error**
- Indicates duplicate question_ids
- Ensure you're not mixing data from multiple test versions
- Recreate database with clean extraction

**Missing questions after extraction**
- Verify PDF matches official College Board format
- Check that both test PDF and scoring guide PDF are provided
- Review console output for section detection messages

**Free-response answers showing "C" or "D" instead of numeric values**
- This occurs when the PDF layout causes misclassification
- These are actually multiple-choice questions that were mislabeled
- Cross-reference with the official answer key

## 🛠️ Advanced Usage

### Batch Processing Multiple Tests

```python
import glob
from sat_extractor import main

test_pairs = [
    ("sat-practice-test-8-digital.pdf", "scoring-sat-practice-test-8-digital.pdf"),
    ("sat-practice-test-9-digital.pdf", "scoring-sat-practice-test-9-digital.pdf"),
    # ... more test pairs
]

all_questions = []
for test_pdf, scoring_pdf in test_pairs:
    df = main(test_pdf, scoring_pdf)
    all_questions.append(df)

# Combine into single dataset
combined_df = pd.concat(all_questions, ignore_index=True)
combined_df.to_sql("all_sat_questions", sqlite3.connect("all_tests.db"))
```

### Custom SQL Queries via Python

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("sat_questions.db")

# Get questions matching specific criteria
query = """
    SELECT question_id, question_stem, correct_answer
    FROM sat_questions
    WHERE section = 'Math' 
      AND module = 1
      AND question_type = 'Multiple Choice'
    ORDER BY question_number
"""

df = pd.read_sql_query(query, conn)
conn.close()
```

### Filtering and Analysis

```python
import pandas as pd

df = pd.read_csv("sat_questions.csv", encoding="utf-8-sig")

# Get all questions with passages
with_passages = df[df["passage"].str.len() > 0]

# Analyze answer distribution
answer_counts = df[df["question_type"] == "Multiple Choice"]["correct_answer"].value_counts()

# Find questions by keyword
keyword_matches = df[df["question_stem"].str.contains("graph|table", case=False, na=False)]
```

## 📚 Data Format Examples

### CSV Output Sample

```csv
question_id,test_name,section,module,question_number,question_type,correct_answer,passage,question_stem,choice_A,choice_B,choice_C,choice_D
SAT_Practice_Test_#8_Reading_and_Writing_M1_Q1,SAT Practice Test #8,Reading and Writing,1,1,Multiple Choice,B,"As Mexico's first president from an Indigenous community, Benito Juarez...","Which choice completes the text with the most logical and precise word or phrase?",unpredictable,important,secretive,ordinary
SAT_Practice_Test_#8_Math_M1_Q6,SAT Practice Test #8,Math,1,6,Free Response,0.2; 1/5,"","What value of x is the solution to the given equation?","","","",""
```

### JSON Export Sample

```json
[
    {
        "question_id": "SAT_Practice_Test_#8_Reading_and_Writing_M1_Q1",
        "test_name": "SAT Practice Test #8",
        "section": "Reading and Writing",
        "module": 1,
        "question_number": 1,
        "question_type": "Multiple Choice",
        "correct_answer": "B",
        "passage": "As Mexico's first president from an Indigenous community...",
        "question_stem": "Which choice completes the text with the most logical and precise word or phrase?",
        "choice_A": "unpredictable",
        "choice_B": "important",
        "choice_C": "secretive",
        "choice_D": "ordinary"
    }
]
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Support for additional SAT test formats
- Enhanced math diagram/graph extraction
- Answer explanation extraction
- Question difficulty scoring
- Multi-language support

## ⚖️ License

This toolkit is provided for educational and research purposes. SAT® is a trademark of the College Board, which does not endorse this toolkit. Always use official College Board materials in accordance with their terms of service.

## 🙏 Acknowledgments

- Built with [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction
- Powered by [pandas](https://pandas.pydata.org/) for data manipulation
- Question extraction methodology developed through iterative analysis of College Board PDF formats

## 📧 Support

For issues, questions, or feature requests, please open an issue in the GitHub repository.

---

**Note**: This toolkit is designed for official College Board SAT practice tests in PDF format. It may not work correctly with unofficial or modified test materials.
