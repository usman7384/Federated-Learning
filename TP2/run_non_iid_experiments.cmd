@echo off
echo => Generate non-IID data..

cd data || exit /b

:: Remove existing data
if exist mnist\all_data (
    rmdir /s /q mnist\all_data
)

:: Generate non-IID data with 2 classes per client
python generate_data.py --dataset_name mnist --n_clients 10 --non_iid --n_classes_per_client 2 --frac 0.2 --save_dir mnist --seed 1234

cd ..

echo => Running experiments with different local steps (epochs)..

:: Experiment 1: 1 local step
echo Running experiment with 1 local step...
python train.py --experiment "mnist" --n_rounds 100 --local_steps 1 --local_optimizer sgd --local_lr 0.001 --server_optimizer sgd --server_lr 0.1 --bz 128 --device "cpu" --log_freq 1 --verbose 1 --logs_dir "logs/mnist_non_iid_local_steps_1/" --seed 12

:: Experiment 2: 5 local steps
echo Running experiment with 5 local steps...
python train.py --experiment "mnist" --n_rounds 100 --local_steps 5 --local_optimizer sgd --local_lr 0.001 --server_optimizer sgd --server_lr 0.1 --bz 128 --device "cpu" --log_freq 1 --verbose 1 --logs_dir "logs/mnist_non_iid_local_steps_5/" --seed 12

:: Experiment 3: 10 local steps
echo Running experiment with 10 local steps...
python train.py --experiment "mnist" --n_rounds 100 --local_steps 10 --local_optimizer sgd --local_lr 0.001 --server_optimizer sgd --server_lr 0.1 --bz 128 --device "cpu" --log_freq 1 --verbose 1 --logs_dir "logs/mnist_non_iid_local_steps_10/" --seed 12

echo All experiments completed!
echo Results are saved in:
echo - logs\mnist_non_iid_local_steps_1\
echo - logs\mnist_non_iid_local_steps_5\
echo - logs\mnist_non_iid_local_steps_10\
pause
