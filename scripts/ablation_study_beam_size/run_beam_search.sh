# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset wikinre --prompter wikinre_fe  --n 72 --bs 72 --num_beams 1
# # python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset wikinre --prompter wikinre_fe  --n 1000 --bs 48 --num_beams 2 --use_wandb
# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset wikinre --prompter wikinre_fe  --n 18 --bs 18 --num_beams 4
# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset wikinre --prompter wikinre_fe  --n 9 --bs 9 --num_beams 8

# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset wikinre --prompter wikinre_fe  --n 36 --bs 36 --num_beams 1
# # python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset wikinre --prompter wikinre_fe  --n 1000 --bs 24 --num_beams 2 --use_wandb
# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset wikinre --prompter wikinre_fe  --n 9 --bs 9 --num_beams 4
# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset wikinre --prompter wikinre_fe  --n 4 --bs 4 --num_beams 8



python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset wikinre --prompter wikinre_fe  --n 1000 --bs 72 --num_beams 1 --use_wandb --wandb_group beam_search
# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset wikinre --prompter wikinre_fe  --n 1000 --bs 48 --num_beams 2 --use_wandb --wandb_group beam_search
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset wikinre --prompter wikinre_fe  --n 1000 --bs 18 --num_beams 4 --use_wandb --wandb_group beam_search
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset wikinre --prompter wikinre_fe  --n 1000 --bs 9 --num_beams 8 --use_wandb --wandb_group beam_search


python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset wikinre --prompter wikinre_fe  --n 1000 --bs 36 --num_beams 1 --use_wandb --wandb_group beam_search
# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset wikinre --prompter wikinre_fe  --n 1000 --bs 24 --num_beams 2 --use_wandb --wandb_group beam_search
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset wikinre --prompter wikinre_fe  --n 1000 --bs 9 --num_beams 4 --use_wandb --wandb_group beam_search
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset wikinre --prompter wikinre_fe  --n 1000 --bs 4 --num_beams 8 --use_wandb --wandb_group beam_search
