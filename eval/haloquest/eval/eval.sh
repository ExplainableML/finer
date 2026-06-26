export GOOGLE_API_KEY="" # Enter your google api key

#Change --result-file to any response files inside /answers to compute the accuracy.

python3 eval_haloquest.py \
    --question-file ../haloquest-eval.jsonl \
    --result-file ../answers/internvl35_8b_finer.jsonl \
    --evaluation-result-file ../answers/internvl35_8b_finer.log