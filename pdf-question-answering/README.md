# PDF Question Answering

Ask questions about a PDF document from the terminal.

## Features
- Extract text from PDF
- Split into chunks
- Retrieve most relevant chunks with TF-IDF similarity
- Answer with local heuristic or OpenAI model

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Local mode (default)

```bash
python main.py --pdf sample.pdf --question "What is the main topic?"
```

### OpenAI mode

```bash
set OPENAI_API_KEY=your_key_here
python main.py --mode openai --model gpt-4.1-mini --pdf sample.pdf --question "What is the main topic?"
```
