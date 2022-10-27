# Artificial Neural Network Learning via Genetic Algorithm

# Overview

This project simulates learning a neural network using a genetic algorithm rather than the conventional error backpropagation method. Experiment with this using the Cart Pole provided by the [open-ai gym library](https://github.com/openai/gym). We confirmed that simple fully connected networks evolve well enough in the environment of tournament selection, 1000 individuals per population, and 0.05 mutation probability. 

# Installation

```
$ git clone https://github.com/leibniz21c/Artificial-Neural-Network-Learning-via-Genetic-Algorithm
$ cd Artificial-Neural-Network-Learning-via-Genetic-Algorithm/
$ pip install -r requirements.txt
```

# Training

``` 
$ python train.py [OPTIONS]
```

|Argument                |Type       |Default         |Description                                                   |
|------------------------|-----------|----------------|--------------------------------------------------------------|
|```--num_populations``` |```int```  |10              |Number of individuals per population                          |
|```--p_mutation```      |```float```|0.05            |Probability of mutation for each individual                   |
|```--max_generations``` |```int```  |10              |Max iteration                                                 |
|```--selection_method```|```str```  |```tournament```|Selection method. (Available : ```wheel```, ```tournament```) |
|```--num_validations``` |```int```  |1               |Number of simulations per individual                          |

> ```tournament``` use every candidates.

# Test

``` 
$ python test.py [SAVED_MODEL_PATH]
```

> Default trained models are in ```saved/```

# Simulations

### 1. Best Model in Generation 1

<p align="center">
  <img src="https://user-images.githubusercontent.com/38760913/198166136-50c7c448-aae0-4d7a-b510-0168dcd1c76f.gif">
</p>

### 2. Best Model in Generation 10

<p align="center">
  <img src="https://user-images.githubusercontent.com/38760913/198166171-3a825758-0520-4a3c-9945-a6d07372b63e.gif">
</p>

### 3. Best Model in Generation 100

<p align="center">
  <img src="https://user-images.githubusercontent.com/38760913/198166192-2094ff85-e88c-4153-9670-8feecac4245f.gif">
</p>

### 4. Best Model

<p align="center">
  <img src="https://user-images.githubusercontent.com/38760913/198166227-9e047232-a6ed-48ee-9068-892ee6c140ac.gif">
</p>
