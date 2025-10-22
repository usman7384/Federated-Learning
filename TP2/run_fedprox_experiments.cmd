@echo off
echo ================================================================
echo EXERCISE 6.1 - FedProx vs FedAvg on Non-IID Data
echo ================================================================
echo.

echo => Generating non-IID data if not exists...
cd data || exit /b

:: Check if data already exists
if not exist mnist\all_data (
    echo Removing old data...
    if exist mnist\all_data (
        rmdir /s /q mnist\all_data
    )
    
    echo Generating non-IID data with 2 classes per client...
    python generate_data.py --dataset_name mnist --n_clients 10 --non_iid --n_classes_per_client 2 --frac 0.2 --save_dir mnist --seed 1234
) else (
    echo Non-IID data already exists, skipping generation...
)

cd ..
echo.

echo ================================================================
echo FedProx Experiments with Different Local Steps
echo ================================================================
echo.

:: FedProx Experiment 1: 1 local step
echo [1/5] Running FedProx with 1 local step (mu=2)...
python train.py --experiment "mnist" --n_rounds 100 --local_steps 1 --local_optimizer prox_sgd --local_lr 0.001 --mu 2 --server_optimizer sgd --server_lr 0.1 --bz 128 --device "cpu" --log_freq 1 --verbose 1 --logs_dir "logs/fedprox_local_steps_1/" --seed 12
echo.

:: FedProx Experiment 2: 5 local steps
echo [2/5] Running FedProx with 5 local steps (mu=2)...
python train.py --experiment "mnist" --n_rounds 100 --local_steps 5 --local_optimizer prox_sgd --local_lr 0.001 --mu 2 --server_optimizer sgd --server_lr 0.1 --bz 128 --device "cpu" --log_freq 1 --verbose 1 --logs_dir "logs/fedprox_local_steps_5/" --seed 12
echo.

:: FedProx Experiment 3: 10 local steps
echo [3/5] Running FedProx with 10 local steps (mu=2)...
python train.py --experiment "mnist" --n_rounds 100 --local_steps 10 --local_optimizer prox_sgd --local_lr 0.001 --mu 2 --server_optimizer sgd --server_lr 0.1 --bz 128 --device "cpu" --log_freq 1 --verbose 1 --logs_dir "logs/fedprox_local_steps_10/" --seed 12
echo.

echo ================================================================
echo All FedProx experiments completed!
echo ================================================================
echo.
echo Results are saved in:
echo   - logs\fedprox_local_steps_1\
echo   - logs\fedprox_local_steps_5\
echo   - logs\fedprox_local_steps_10\
echo.
echo Compare these with FedAvg results from Exercise 4:
echo   - logs\mnist_non_iid_local_steps_1\
echo   - logs\mnist_non_iid_local_steps_5\
echo   - logs\mnist_non_iid_local_steps_10\
echo.
echo To visualize: tensorboard --logdir=logs
echo.
pause

