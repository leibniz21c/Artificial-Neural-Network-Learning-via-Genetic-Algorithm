import sys
import gym
import torch

from models import NNModel as Model


if __name__ == "__main__":
    # Load model
    model = Model()
    model.load_state_dict(torch.load(sys.argv[1]))
    print(model)

    env = gym.make("CartPole-v1")
    obs = env.reset()

    for _ in range(500):
        action = torch.argmax(model(torch.FloatTensor(obs))).item()
        env.step(action)
        env.render()
