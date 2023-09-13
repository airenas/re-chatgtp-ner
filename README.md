# re-chatgtp-ner
Testing ChatGPT for NER task

The goal is to extract the required data into json.

## Preparation
Tested with Python 3.11

```bash
conda create -n openai2023 python=3.11
conda activate openai2023
pip install -r requirements.txt
```

## Testing latex file (1 page)

```bash
export OPENAI_API_KEY=<key>
python ner-authors.py <file>
## test for a valid json
python ner-authors.py <file> | jq .
```

