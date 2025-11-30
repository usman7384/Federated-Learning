import torch

from copy import deepcopy


def average_models(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False
):
    r"""computes the average of learners and store it into target_learner

    Parameters
    ----------
    learners: List[Learner]

    target_learner: Learner

    weights: 1-D torch.tensor
        tensor of the same size as learners, having values between 0 and 1, and summing to 1,
        if not provided, uniform weights are used

    average_params: bool
        if set to true the parameters are averaged; default is True

    average_gradients: bool
        if set to true the gradient are averaged; default is False

    Returns
    -------
        None
    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    param_tensors = []
    grad_tensors = []

    for learner in learners:
        if average_params:
            param_tensors.append(deepcopy(learner.get_param_tensor()))

        if average_gradients:
            grad_tensors.append(deepcopy(learner.get_grad_tensor()))

    if average_params:
        param_tensors = torch.stack(param_tensors)
        average_params_tensor = weights @ param_tensors
        target_learner.set_param_tensor(average_params_tensor)

    if average_gradients:
        grad_tensors = torch.stack(grad_tensors)
        average_grads_tensor = weights @ grad_tensors
        target_learner.set_grad_tensor(average_grads_tensor)


def median_models(learners, target_learner):
    r"""computes the coordinate-wise median of learners and stores it into target_learner
    
    This function provides robustness against Byzantine attacks by using median aggregation
    instead of mean aggregation. The median is less sensitive to outliers and malicious updates.
    
    Parameters
    ----------
    learners: List[Learner]
        List of learners from which to compute the median
    
    target_learner: Learner
        Learner object where the median model will be stored
    
    Returns
    -------
        None
    """
    # Collect parameter tensors from all learners
    param_tensors = []
    
    for learner in learners:
        param_tensors.append(deepcopy(learner.get_param_tensor()))
    
    # Stack all parameter tensors: shape will be (n_learners, model_dim)
    param_tensors = torch.stack(param_tensors)
    
    # Compute coordinate-wise median
    # torch.median returns a tuple (values, indices) when dim is specified
    median_params_tensor = torch.median(param_tensors, dim=0).values
    
    # Set the median parameters to the target learner
    target_learner.set_param_tensor(median_params_tensor)


def copy_model(target, source):
    """copy learners_weights from target to source


    Parameters
    ----------
    target: torch.nn.Module

    source: torch.nn.Module


    Returns
    -------
        None

    """
    target.load_state_dict(source.state_dict())
