python sample.py \
    --input data/synthesis_planning/test_chembl.csv \
    --output results/test_split.csv \
    --model-path data/trained_weights/split.ckpt \
    --num-gpus -1 \
    --num-workers-per-gpu 2 \
    --exhaustiveness 512 \