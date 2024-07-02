# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset synthie --prompter synthie_fe_A  --n 12  --sketcher GPT3.5 --bs 12
# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset synthie --prompter synthie_fe_A  --n 12  --sketcher GPT4 --bs 12


python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset synthie --prompter synthie_fe_A  --n 1000  --sketcher GPT3.5 --bs 12 --use_wandb
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset synthie --prompter synthie_fe_A  --n 1000  --sketcher GPT4 --bs 12 --use_wandb
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset synthie --prompter synthie_fe_A  --n 1000  --sketcher Claude --bs 12 --use_wandb
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset synthie --prompter synthie_fe_A  --n 1000  --sketcher Claude_Instant --bs 12 --use_wandb
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset synthie --prompter synthie_fe_A  --n 1000  --sketcher llama2_7B --bs 12 --use_wandb
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset synthie --prompter synthie_fe_A  --n 1000  --sketcher llama2_13B --bs 12 --use_wandb
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset synthie --prompter synthie_fe_A  --n 1000  --sketcher llama2_70B --bs 12 --use_wandb
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-7b-hf --dataset synthie --prompter synthie_fe_A  --n 1000  --sketcher llama_33B --bs 12 --use_wandb
