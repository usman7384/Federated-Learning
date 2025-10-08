@echo off
echo => Train with different local epochs (Exercise 3)...

setlocal enabledelayedexpansion
set epochs_list=1 5 10 50 100

for %%E in (%epochs_list%) do (
  echo Running experiment with %%E local epochs...
  python train.py ^
    --experiment "mnist" ^
    --n_rounds 100 ^
    --local_steps %%E ^
    --local_optimizer sgd ^
    --local_lr 0.001 ^
    --server_optimizer sgd ^
    --server_lr 0.1 ^
    --bz 128 ^
    --device "cpu" ^
    --log_freq 1 ^
    --verbose 1 ^
    --logs_dir "logs/exercise3/local_epochs_%%E/" ^
    --seed 12
)

echo => Running FedSGD experiment (local_steps=1, batch_size=dataset_size)..
