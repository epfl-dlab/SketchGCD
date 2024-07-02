# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset wikinre --prompter wikinre_fe_Q_few  --n 16  --sketcher GPT4 --bs 16
# python scripts/run_experiment.py --model_name=saibo/llama-33B  --dataset wikinre --prompter wikinre_fe_Q_few  --n 6  --sketcher GPT4 --bs 6
# python scripts/run_experiment.py --model_name=meta-llama/Llama-2-70b-hf --dataset wikinre --prompter wikinre_fe_Q_few  --n 10  --sketcher GPT4 --bs 10




python scripts/run_experiment.py --model_name=meta-llama/Llama-2-13b-hf --dataset wikinre --prompter wikinre_fe_Q_few  --n 1000  --sketcher GPT4 --bs 16 --use_wandb
python scripts/run_experiment.py --model_name=saibo/llama-33B  --dataset wikinre --prompter wikinre_fe_Q_few  --n 1000  --sketcher GPT4 --bs 6 --use_wandb
python scripts/run_experiment.py --model_name=meta-llama/Llama-2-70b-hf --dataset wikinre --prompter wikinre_fe_Q_few  --n 1000  --sketcher GPT4 --bs 10 --use_wandb
