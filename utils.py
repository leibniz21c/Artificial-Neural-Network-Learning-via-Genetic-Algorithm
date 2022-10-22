import logging
from random import random

import gym
import torch

from models import NNModel as Model


def get_stream_logger(filename):
    # Set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("[%(asctime)s - %(levelname)s] : %(message)s")

    # handler
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@torch.no_grad()
def simulation(models, generation=None, n_valid=5, logger=None):
    num_models = len(models)
    envs = [gym.make("CartPole-v1") for i in range(num_models)]

    # Reset environments
    observations = torch.cat(
        [torch.FloatTensor(env.reset()).unsqueeze(dim=0) for env in envs], dim=0
    )
    dones = torch.tensor([False for i in range(num_models)])

    # Cv
    fitnesses = [torch.tensor([0.0 for i in range(num_models)]) for i in range(n_valid)]

    for k in range(n_valid):
        while not dones.all():
            for i, model in enumerate(models):
                if not dones[i]:
                    action = torch.argmax(model(observations[i])).item()
                    obs, reward, done, _ = envs[i].step(action)
                    observations[i], dones[i] = torch.FloatTensor(obs), done
                    fitnesses[k][i] += reward

    fitnesses = torch.cat([t.unsqueeze(dim=0) for t in fitnesses], dim=0)
    fitnesses = fitnesses.mean(dim=0)

    if logger is not None:
        logger.info(
            f"Generation {generation:7} : max fitness = {max(fitnesses):6.2f}, mean fitness = {fitnesses.mean().item():6.2f}"
        )

    return fitnesses


@torch.no_grad()
def selection(fitnesses, num_select, method="rank", p=0.95):
    if method not in ["rank", "wheel", "stochastic"]:
        raise NotImplementedError(f"{method} method is not implemented")

    if method == "stochastic":
        if random() < p:
            method = "rank"
        else:
            method = "wheel"

    if method == "rank":
        values, indices = torch.topk(fitnesses, num_select)
        return indices
    elif method == "wheel":
        sorted_norm_fitnesses = sorted(
            torch.nn.functional.normalize(fitnesses, dim=0, p=1), reverse=True
        )

        indices = []
        for k in range(num_select):
            p, q, i = random(), 0.0, 0
            while q < p:
                q += sorted_norm_fitnesses[i]
                i += 1
            i -= 1
            indices.append(i)
        return torch.tensor(indices)


@torch.no_grad()
def crossover(model1, model2, alpha=0.5):
    model = Model()
    model.weight = [
        alpha * model1.weight[i] + (1 - alpha) * model2.weight[i]
        for i in range(len(model1.linear_layers))
    ]
    return model


@torch.no_grad()
def mutation(model, p=0.01):
    weights = model.weight
    mutated_weight = [
        (torch.rand_like(weights[i]) < p) * 0.2 * torch.rand_like(weights[i]) - 0.1
        for i in range(len(weights))
    ]

    model.weight = [weights[i] + mutated_weight[i] for i in range(len(weights))]
