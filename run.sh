#!/usr/bin/env bash
# Full pipeline. Run from the repo root.
set -e

pip install -r requirements.txt

echo "=== Task 1: data ==="
python -m src.prepare_data --config configs/base.yaml

echo "=== Task 2: tokenizer ==="
python -m src.train_tokenizer --config configs/base.yaml

echo "=== sanity: overfit 32 pairs (loss must approach 0) ==="
python -m src.train --overfit 32 --epochs 30

echo "=== sanity: 10k subset, 1 epoch (loss must fall) ==="
python -m src.train --limit 10000 --epochs 1

echo "=== Task 3: full training ==="
python -m src.train --config configs/base.yaml

echo "=== Task 4: evaluation ==="
python -m src.evaluate --config configs/base.yaml
python -m src.analyze

echo "=== figures ==="
python -m src.viz --what all --index 0

echo "=== human eval sheets ==="
python -m src.human_eval --make

echo "Done. Now: rate the sheets, run 'python -m src.human_eval --score',"
echo "then 'streamlit run app/streamlit_app.py' and screenshot it."
