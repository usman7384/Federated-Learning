import os
import time
import json

import numpy as np
from tqdm import tqdm

from torchvision.datasets import MNIST

from constants import *


def save_cfg(save_path, cfg):
    with open(save_path, "w") as f:
        json.dump(cfg, f)


def save_data(save_dir, train_data, train_targets, test_data, test_targets):
    """save data and targets as `.npy` files

    Parameters
    ----------
    save_dir: str
        directory to save data; it will be created it it does not exist

    train_data: numpy.array

    train_targets: numpy.array

    test_data: numpy.array

    test_targets: numpy.array

    """
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "train_data.npy"), "wb") as f:
        np.save(f, train_data)

    with open(os.path.join(save_dir, "train_targets.npy"), "wb") as f:
        np.save(f, train_targets)

    with open(os.path.join(save_dir, "test_data.npy"), "wb") as f:
        np.save(f, test_data)

    with open(os.path.join(save_dir, "test_targets.npy"), "wb") as f:
        np.save(f, test_targets)


def get_dataset(dataset_name, raw_data_path):

    if dataset_name == "mnist":

        dataset = MNIST(root=raw_data_path, download=True, train=True)
        test_dataset = MNIST(root=raw_data_path, download=True, train=False)

        dataset.data = np.concatenate((dataset.data, test_dataset.data))
        dataset.targets = np.concatenate((dataset.targets, test_dataset.targets))

    else:
        error_message = f"{dataset_name} is not available, possible datasets are:"
        for n in DATASETS:
            error_message += n + ",\t"

        raise NotImplementedError(error_message)

    return dataset


def iid_divide(l_, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py

    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups

    """
    num_elems = len(l_)
    group_size = int(len(l_) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l_[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l_[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l_, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l_[current_index: index])
        current_index = index

    return res


def iid_split(
        dataset,
        n_clients,
        frac,
        rng=None
):
    """
    split classification dataset among `n_clients` in an IID fashion. The dataset is split as follows:
        1) The dataset is shuffled and partitioned among n_clients

    Parameters
    ----------
    dataset: torch.utils.Dataset
        a classification dataset;
         expected to have attributes `data` and `targets` storing `numpy.array` objects

    n_clients: int
        number of clients

    frac: float
        fraction of dataset to be used

    rng: random number generator; default is None

    Returns
    -------
        * List[Dict]: list (size `n_clients`) of subgroups, each subgroup is a list of indices.

    """

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    n_samples = int(len(dataset) * frac)
    selected_indices = rng.choice(len(dataset), size=n_samples, replace=False)
    rng.shuffle(selected_indices)

    clients_indices = iid_divide(selected_indices, n_clients)

    return clients_indices


def pathological_non_iid_split(dataset, n_classes, n_clients, n_classes_per_client, frac=1, rng=None):
    """
    split classification dataset among `n_clients`. The dataset is split as follows:
        1) sort the data by label
        2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
        3) assign each of the `n_clients` with `n_classes_per_client` shards
    Inspired by the split in
     "Communication-Efficient Learning of Deep Networks from Decentralized Data"__(https://arxiv.org/abs/1602.05629)

    :param dataset:
        :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_classes_per_client:
    :param frac: fraction of dataset to use
    :param rng:random number generator; default is None
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.choice(list(range(len(dataset))), n_samples, replace=False).tolist()

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        label = dataset.targets[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_clients * n_classes_per_client
    shards = iid_divide(sorted_indices, n_shards)
    rng.shuffle(shards)
    tasks_shards = iid_divide(shards, n_clients)

    clients_indices = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            clients_indices[client_id] += shard

    return clients_indices


def generate_data(
        dataset,
        split_type,
        n_train_samples,
        n_clients,
        n_classes,
        n_classes_per_client,
        frac,
        save_dir,
        rng=None
):

    if split_type == "non_iid":
        print("==> Data are split non-iid")
        clients_indices = \
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=n_classes,
                n_clients=n_clients,
                n_classes_per_client=n_classes_per_client,
                frac=frac,
                rng=rng
            )
    elif split_type == "iid":
        print("==> Data are split iid")
        clients_indices = \
            iid_split(
                dataset=dataset,
                n_clients=n_clients,
                frac=frac,
                rng=rng
            )
    else:
        error_message = "Data split strategy not available !" \
                        "Please pass '--iid' or '--non_iid'"

        raise NotImplementedError(error_message)

    for client_id in tqdm(range(n_clients), total=n_clients, desc="Clients.."):

        client_indices = np.array(clients_indices[client_id])
        train_indices = client_indices[client_indices < n_train_samples]
        test_indices = client_indices[client_indices >= n_train_samples]

        train_data, train_targets = dataset.data[train_indices], dataset.targets[train_indices]
        test_data, test_targets = dataset.data[test_indices], dataset.targets[test_indices]

        client_dir = os.path.join(os.getcwd(), save_dir, f"client_{client_id}")

        save_data(
            save_dir=client_dir,
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets
        )
