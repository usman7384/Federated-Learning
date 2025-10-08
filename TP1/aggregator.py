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
            verbose=0,
            seed=None
    ):
        """

        Parameters
        ----------
        clients: Dict[int: Client]

        clients_weights: Dict[int: Client]

        global_learner: Learner

        logger: SummaryWriter

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

    @abstractmethod
    def update_clients(self):
        """
        send the new global model to the clients
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


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self):

        for idx in range(self.n_clients):
            self.clients[idx].step()

        self.c_round += 1

    def update_clients(self):
        pass


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.

     Clients get fully synchronized with the average client.

    """
    def mix(self):

        clients_weights = torch.tensor(self.clients_weights, dtype=torch.float32)

        # Loop over clients and perform the local training steps calling client.step().
        for client in self.clients:
            client.step()

        # Collect the model from each client in a list after the local training steps.
        models = []
        for client in self.clients:
            models.append(client.learner)

        # Zero-out the gradients of the global learner calling optimizer.zero_grad()
        self.global_learner.optimizer.zero_grad()

        # Average the models calling average_models
        average_models(learners=models,target_learner=self.global_learner,weights=clients_weights)

        # Apply the global step calling global_learner.optimizer.step()
        self.global_learner.optimizer.step()

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

    def update_clients(self):
        for client in self.clients:

            copy_model(self.global_learner.model, client.learner.model)