# Deploy Guide (Lightweight)

This app supports two modes:

- `USE_SEMANTIC_SEARCH=false` (default): lightweight, faster startup, smaller deploy size
- `USE_SEMANTIC_SEARCH=true`: full semantic matching (requires heavier model runtime)

## 1) Local lightweight run

```bash
pip install -r requirements.deploy.txt
export USE_SEMANTIC_SEARCH=false
python -m streamlit run app.py
```

## 2) Full mode (semantic)

```bash
pip install -r requirements.txt
export USE_SEMANTIC_SEARCH=true
python -m streamlit run app.py
```

## 3) Docker deploy (lightweight)

```bash
docker build -t askai-tutor .
docker run -p 8501:8501 -e USE_SEMANTIC_SEARCH=false askai-tutor
```

## Note

For smallest deploy package and fastest startup, keep `USE_SEMANTIC_SEARCH=false`.
