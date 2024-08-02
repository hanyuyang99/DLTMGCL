# DLTMG_CL
the code for paper "A Brain-Inspired Distributed Long-Term Memory Guided Online Continual Learning Method" in HBAI2024 (IJCAI2024 Workshop)

## Run the code

pip install -r requirements.txt

python main.py --dataset seq-cifar100 --model dltmgcl --buffer_size 5000 --load_best_args
