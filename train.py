import os
import argparse
from random import random
from copy import deepcopy
from time import localtime

from tqdm import tqdm
import torch

from models import NNModel as Model
from utils import (
    get_stream_logger,
    simulation,
    selection,
    crossover,
    mutation
)


@torch.no_grad()
def main(args):
    # Set arguments
    savedir = args.savedir
    num_populations = args.num_populations
    p_mutation = args.p_mutation
    max_generations = args.max_generations
    selection_method = args.selection_method
    num_validations = args.num_validations
    num_select = 2 # num crossover selection(default 2)
    save_indexes = [10, 100, 1000, 10000]

    # Set logging
    os.makedirs(savedir, exist_ok=True)

    # Logging experiments configs
    logger = get_stream_logger(f"{savedir}/evolution_log.log")
    logger.info(f"savedir == {savedir}"); print(f"savedir == {savedir}")
    logger.info(f"num_populations == {num_populations}"); print(f"num_populations == {num_populations}")
    logger.info(f"p_mutation == {p_mutation}"); print(f"p_mutation == {p_mutation}")
    logger.info(f"num_select == {num_select}"); print(f"num_select == {num_select}")
    logger.info(f"max_generations == {max_generations}"); print(f"max_generations == {max_generations}")
    logger.info(f"mode == {selection_method}"); print(f"mode == {selection_method}")
    logger.info(f"num_validations == {num_validations}"); print(f"num_validations == {num_validations}")

    # Initialize
    populations = [Model() for i in range(num_populations)]
    fitnesses = simulation(
        populations, 
        generation=0, 
        n_valid=num_validations,
        logger=logger
    )
    selected_idx = selection(
        fitnesses, 
        num_select, 
        method=selection_method
    )
    selected_model = [deepcopy(populations[i]) for i in selected_idx]
    torch.save(selected_model[0].state_dict(), f'{savedir}/generation0.pt')
    populations = [crossover(selected_model[0], selected_model[1], alpha=0.5) for i in range(num_populations - 2)] + [deepcopy(selected_model[0]), deepcopy(selected_model[1])]
    del selected_model

    # Evolutions
    best_fitness = 0
    for i in tqdm(range(max_generations), desc=f'Generation'):
        fitnesses = simulation(
            populations, 
            generation=i + 1, 
            n_valid=num_validations,
            logger=logger
        )
        selected_idx = selection(fitnesses, num_select, method=selection_method)

        # Selection
        selected_model = [deepcopy(populations[j]) for j in selected_idx]
        del populations

        if best_fitness < fitnesses[selected_idx[0]]:
            torch.save(selected_model[0].state_dict(), f'{savedir}/bestfit.pt')
            best_fitness = fitnesses[selected_idx[0]]

        if i + 1 in save_indexes:
            torch.save(selected_model[0].state_dict(), f'{savedir}/generation{i + 1}.pt')

        # Crossover(parent individuals added)
        populations = [crossover(selected_model[0], selected_model[1], alpha=0.5) for i in range(num_populations - 2)] + [deepcopy(selected_model[0]), deepcopy(selected_model[1])]
        del selected_model

        # Mutation
        for individual in populations:
            if random() < p_mutation:
                mutation(individual, p=p_mutation)

    logger.info(f'best fit : {best_fitness}'); print(f'best fit : {best_fitness}')
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    now = localtime()
    parser.add_argument(
        "--savedir", 
        type=str,
        default=f'saved/{now.tm_year}{now.tm_mon:02}{now.tm_mday:02}{now.tm_hour:02}{now.tm_min:02}{now.tm_sec:02}'
    )
    parser.add_argument("--num_populations", type=int, default=10)
    parser.add_argument("--p_mutation", type=float, default=0.05)
    parser.add_argument("--max_generations", type=int, default=10)
    parser.add_argument("--selection_method", type=str, default="rank")
    parser.add_argument("--num_validations", type=int, default=1)
    main(parser.parse_args())
