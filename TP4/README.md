# Federated Learning & Data Privacy, 2025-2026

## Forth Lab - 17 November 2025

Welcome to the forth lab session of the Federated Learning & Data Privacy course! Today's topic is on the **malicious attacks** in FL and the corresponding **defenses**. 


### Generation of the data
```
cd data/ || exit

rm -r mnist/all_data

python generate_data.py \
  --dataset mnist \
  --n_clients 10 \
  --iid \
  --frac 0.1 \
  --save_dir mnist \
  --seed 1234

```
### Ex1: Implementation of label-flipping attacks

- In `learner.py`, create a new class of learner with name `Label_Flipping_Learner`. Reimplement the training process of the learner that all labels will be shifted by one. 
- Add one argument `prop` in  `args.py` of folder `utils` to present the proportion of malicious clients in the system
- Modify the `init_client` function in `utils.py` of folder `utils`. When the client number is within [0, args.prop*10), the client will simulate a malicious one, i.e., the client will get the learner 
defined in the previous exercice `Label_Flipping_Learner`. 
- Run the experiments with `prop` equivalent to 0, 0.1, 0.3 and 0.5 when the data is iid. 
Remind 
```
cd ..

python train.py \
  --experiment "mnist" \
  --n_rounds 25 \
  --local_steps 1 \
  --local_optimizer sgd \
  --local_lr 0.001 \
  --server_optimizer sgd \
  --server_lr 0.1 \
  --bz 128 \
  --device "cpu" \
  --log_freq 1 \
  --verbose 1 \
  --logs_dir "logs/mnist/" \
  --prop 0
  --seed 12
```
Display a graph G1 where the x-axis represents the number of malicious clients and the y-axis represents the accuracy of the final model. What are your observations?


### Ex2: Implementation of Defenses

- In the file `aggregator.py`, include a median aggregator class named `MedianAggregator`. 
In the `torch_utils.py` file located in the `utils` folder, add a `median_models` function. This function should be callable by the `MedianAggregator` and should return a model where each coordinate corresponds to the median value of all the received clients' models.
- Add one option in `get_aggregator` function in `utils.py` of folder `utils`. When 'args.aggregator_type' is equivalent to `'median'`, 
the function will return the `MedianAggregator`.
- Run the experiments with `prop` equivalent to 0, 0.1, 0.3 and 0.5 and 'args.aggregator_type' is equivalent to `'median'`, when the data is iid. 
Display the results in the previous graph G1. What are your observations, especially compared with the scenario where no defense is implemented?

###Ex3: Simulation of non-iid case
- Regenerate the dataset with non-iid option activated
- Run the experiments with `prop` equivalent to 0, 0.1, 0.3 and 0.5. Display a graph G2 where the x-axis represents the number of malicious clients and the y-axis represents the accuracy of the final model. What are your observations?
- Run the experiments with `prop` equivalent to 0, 0.1, 0.3 and 0.5 and 'args.aggregator_type' is equivalent to `'median'`. 
Display the results in the previous graph G2. What are your observations, especially compared with the scenario where no defense is implemented?

**Send your answers to [francesco.diana@inria.fr](mailto:francesco.diana@inria.fr) by 01/12/2025**
