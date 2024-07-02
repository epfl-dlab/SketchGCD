
# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset synthie --prompter synthie_fe_Q  --n 8  --sketcher GPT4 --bs 8
# python scripts/run_experiment.py --model_name=saibo/llama-33B  --dataset synthie --prompter synthie_fe_Q  --n 3  --sketcher GPT4 --bs 3
# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-70b-hf --dataset synthie --prompter synthie_fe_Q  --n 4  --sketcher GPT4 --bs 4




python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset synthie --prompter synthie_fe_Q  --n 1000  --sketcher GPT4 --bs 8 --use_wandb
python scripts/run_experiment.py --model_name=saibo/llama-33B  --dataset synthie --prompter synthie_fe_Q  --n 1000  --sketcher GPT4 --bs 3 --use_wandb
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-70b-hf --dataset synthie --prompter synthie_fe_Q  --n 1000  --sketcher GPT4 --bs 4 --use_wandb
