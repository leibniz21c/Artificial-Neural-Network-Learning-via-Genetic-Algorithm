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

### 2. Best Model in Generation 10

### 3. Best Model in Generation 100

### 4. Best Model

