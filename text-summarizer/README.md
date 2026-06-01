# Text Summarizer

Generate a concise summary from long input text.

## Features
- `local` mode: extractive summarization (no API key required)
- `openai` mode: abstractive summarization with an LLM

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Local mode (default)

```bash
python main.py --text "<your long text>"
```

### OpenAI mode

```bash
set OPENAI_API_KEY=your_key_here
python main.py --mode openai --model gpt-4.1-mini --text "<your long text>"
```
