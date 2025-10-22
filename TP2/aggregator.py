import time
import random

from abc import ABC, abstractmethod

import numpy as np

from utils.torch_utils import *

from tqdm import tqdm


class Aggregator(ABC):
    r"""Base class for Aggregator.

    `Aggregator` dictates communications between clients

    Attributes
    ----------
    clients: Dict[int: Client]

    clients_weights: Dict[int: Client]

    global_learner: List[Learner]

    n_clients:

    model_dim: dimension if the used model

    c_round: index of the current communication round

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    logger: SummaryWriter

    rng: random number generator

    np_rng: numpy random number generator

    Methods
    ----------
    __init__

    mix

    update_clients

    write_logs

    save_state

    load_state

    """
    def __init__(
            self,
            clients,
            clients_weights,
            global_learner,
            logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            verbose=0,
            seed=None
    ):
        """

        Parameters
        ----------
        clients: Dict[int: Client]

        clients_weights: list[int: Client]

        global_learner: Learner

        logger: SummaryWriter

        sampling_rate: proportion of clients used at each round; default is `1.`

        sample_with_replacement: is True, client are sampled with replacement; default is False

        verbose: int

        seed: int

        """
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        self.clients = clients
        self.n_clients = len(clients)

        self.clients_weights = clients_weights

        self.global_learner = global_learner
        self.device = self.global_learner.device

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients_ids = list()
        self.sampled_clients = list()

        self.verbose = verbose
        self.logger = logger

        self.model_dim = self.global_learner.model_dim

        self.c_round = 0

    @abstractmethod
    def mix(self):
        """mix sampled clients according to weights

        Parameters
        ----------

        Returns
        -------
            None
        """
        pass

    def write_logs(self):
        global_train_loss = 0.
        global_train_metric = 0.
        global_test_loss = 0.
        global_test_metric = 0.

        for client_id, client in enumerate(self.clients):

            train_loss, train_metric, test_loss, test_metric = client.write_logs(counter=self.c_round)

            if self.verbose > 1:

                tqdm.write("*" * 30)
                tqdm.write(f"Client {client_id}..")

                tqdm.write(f"Train Loss: {train_loss:.3f} | Train Metric: {train_metric :.3f}|", end="")
                tqdm.write(f"Test Loss: {test_loss:.3f} | Test Metric: {test_metric:.3f} |")

                tqdm.write("*" * 30)

            global_train_loss += self.clients_weights[client_id] * train_loss
            global_train_metric += self.clients_weights[client_id] * train_metric
            global_test_loss += self.clients_weights[client_id] * test_loss
            global_test_metric += self.clients_weights[client_id] * test_metric

        if self.verbose > 0:

            tqdm.write("+" * 50)
            tqdm.write(f"Global | Round {self.c_round}..")
            tqdm.write(f"Train Loss: {global_train_loss:.3f} | Train Metric: {global_train_metric:.3f} |", end="")
            tqdm.write(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_metric:.3f} |")
            tqdm.write("+" * 50)

        self.logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
        self.logger.add_scalar("Train/Metric", global_train_metric, self.c_round)
        self.logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
        self.logger.add_scalar("Test/Metric", global_test_metric, self.c_round)
        self.logger.flush()

    def sample_clients(self):
        """
        sample a subset of clients

        Implements the sampling strategies in
        "On the Convergence of FedAvg on Non-IID Data"__(https://arxiv.org/abs/1907.02189)

        Attributes
        ----------
        self.sample_with_replacement (bool): determines the sampling strategy
            If `True`, clients are sampled with replacement according to `clients_weights`
            If `False`, clients are sampled without replacement

        self.n_clients (int): total number of clients available for sampling

        self.clients_weights (list of float): list of weights, for each client, used for weighted sampling.
            Only used if `sample_with_replacement` is `True`

        self.n_clients_per_round (int): number of clients to be sampled in each round

        self.clients (list): list of client objects available for training

        self.rng (random.Random()): instance of random number generator for sampling

    Returns:
        None: write the `self.sampled_clients_ids` and `self.sampled_clients` attributes of the class aggregator

        """

        # Exercise 5.1 & 5.2: Support both sampling with and without replacement
        if self.sample_with_replacement:
            # Exercise 5.2: Sampling with replacement based on client weights
            self.sampled_clients_ids = self.rng.choices(
                population=range(self.n_clients),
                weights=self.clients_weights,
                k=self.n_clients_per_round
            )
        else:
            # Exercise 5.1: Uniform sampling without replacement
            self.sampled_clients_ids = self.rng.sample(
                population=range(self.n_clients),
                k=self.n_clients_per_round
            )

        self.sampled_clients = [self.clients[id_] for id_ in self.sampled_clients_ids]


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self):

        for idx in range(self.n_clients):
            self.clients[idx].step()

        self.c_round += 1


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.

     Clients get fully synchronized with the average client.

    """
    def mix(self):

        # Sample clients for this round
        self.sample_clients()

        # Perform local training on sampled clients only
        for client in self.sampled_clients:
            client.step()

        # Gather learners from sampled clients
        learners = [client.learner for client in self.sampled_clients]

        # Adjust clients weights for sampled clients
        if self.sample_with_replacement:
            # Exercise 5.2: For sampling with replacement, use uniform weights (1/K for each)
            uniform_weight = 1.0 / self.n_clients_per_round
            sampled_weights = [uniform_weight] * self.n_clients_per_round
        else:
            # Exercise 5.1: For sampling without replacement, normalize original weights
            sampled_weights = [self.clients_weights[id_] for id_ in self.sampled_clients_ids]
            total_weight = sum(sampled_weights)
            sampled_weights = [w / total_weight for w in sampled_weights]
        
        clients_weights = torch.tensor(sampled_weights, dtype=torch.float32)

        # Aggregate learners params from sampled clients
        average_models(
            learners=learners,
            target_learner=self.global_learner,
            weights=clients_weights,
            average_params=True,
            average_gradients=False
        )

        # Assign the updated model to all clients
        for client in self.clients:
            copy_model(client.learner.model, self.global_learner.model)

        self.c_round += 1
