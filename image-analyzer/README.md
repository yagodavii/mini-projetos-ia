# Image Analyzer

Analyze an image and return a description or technical summary.

## Features
- `local` mode: image metadata + color statistics
- `openai` mode: multimodal image description with AI

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Local mode (default)

```bash
python main.py --image sample.jpg
```

### OpenAI mode

```bash
set OPENAI_API_KEY=your_key_here
python main.py --mode openai --model gpt-4.1-mini --image sample.jpg
```
