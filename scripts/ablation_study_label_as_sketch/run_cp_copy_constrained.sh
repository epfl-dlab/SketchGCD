# python scripts/run_experiment.py --model_name meta-llama/Llama-2-7b-hf --dataset ptb --input_column target --prompter PennTreeBank_copy  --n 144 --bs 72 --num_beams 1 --max_new_tokens 256
# python scripts/run_experiment.py --model_name meta-llama/Llama-2-13b-hf --dataset ptb --input_column target --prompter PennTreeBank_copy  --n 96 --bs 48 --num_beams 1 --max_new_tokens 256
# python scripts/run_experiment.py --model_name saibo/llama-33B --dataset ptb --input_column target --prompter PennTreeBank_copy            --n 64 --bs 32 --num_beams 1 --max_new_tokens 256
# python scripts/run_experiment.py --model_name meta-llama/Llama-2-70b-hf --dataset ptb --input_column target --prompter PennTreeBank_copy  --n 48 --bs 24 --num_beams 1 --max_new_tokens 256


python scripts/run_experiment.py --model_name meta-llama/Llama-2-70b-hf --dataset ptb --input_column target --prompter PennTreeBank_copy  --n 1000 --bs 12 --num_beams 1 --max_new_tokens 256  --use_wandb --wandb_group ptb_copy
python scripts/run_experiment.py --model_name saibo/llama-33B --dataset ptb --input_column target --prompter PennTreeBank_copy             --n 1000 --bs 24 --num_beams 1 --max_new_tokens 256  --use_wandb --wandb_group ptb_copy
python scripts/run_experiment.py --model_name meta-llama/Llama-2-7b-hf --dataset ptb --input_column target --prompter PennTreeBank_copy  --n 1000 --bs 24 --num_beams 1 --max_new_tokens 256  --use_wandb --wandb_group ptb_copy
python scripts/run_experiment.py --model_name meta-llama/Llama-2-13b-hf --dataset ptb --input_column target --prompter PennTreeBank_copy  --n 1000 --bs 24 --num_beams 1 --max_new_tokens 256  --use_wandb --wandb_group ptb_copy
