import argparse
import numpy as np

from es import EvolutionStrategy, Adam, SimpleEvaluator
from npnn import Module, Dense, LSTM

from cartpole_swingup import CartPoleSwingUpEnv


class Model(Module):
    def __init__(self, obs_dim: int, act_dim: int, hid_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim
        self.new_module("lstm", LSTM(obs_dim + act_dim, hid_dim))
        self.new_module("policy", Dense(obs_dim + hid_dim, act_dim))

    def reset(self) -> None:
        self.lstm.reset()
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = self.lstm(np.concatenate([obs, self.prev_action]))
        action = self.policy(np.concatenate([obs, h]))
        return action


class CartPoleSwingUpEvaluator(SimpleEvaluator):
    def evaluate_solution(self, seed: int, solution: np.ndarray) -> float:
        env = CartPoleSwingUpEnv(augmented=True)
        model = Model(5, 1, 16)
        model.transcribe(solution)

        env.seed(seed)
        fitness = 0
        for _ in range(5):
            obs = env.reset()
            model.reset()
            done = False
            while not done:
                action = model(obs)
                obs, reward, done, _ = env.step(action)
                fitness += reward
        fitness = fitness / 5
        return fitness


def main(args):
    np.random.seed(args.seed)

    params = np.zeros(len(Model(5, 1, 16)), dtype=np.float32)
    optimizer = Adam(params, lr=args.lr)
    es = EvolutionStrategy(
        optimizer,
        sigma=args.sigma,
        sigma_decay=0.001,
        lr_decay=args.lr_decay,
        antithetic=True,
    )

    evaluator = CartPoleSwingUpEvaluator(args.num_workers)
    global_best_fitness = -np.inf

    for gen in range(args.num_gen):
        solutions = es.sample(args.pop_size)
        fitness = evaluator.evaluate(solutions)
        es.update(fitness)

        best_fitness = np.max(fitness)
        if best_fitness > global_best_fitness:
            print(f"Improvement detected: {global_best_fitness} -> {best_fitness}")
            best = solutions[np.argmax(fitness)]
            np.save("model_final.npy", best)
            global_best_fitness = best_fitness

        stats = {
            "gen": gen,
            "fitness_mean": np.mean(fitness),
            "fitness_std": np.std(fitness),
            "fitness_max": np.max(fitness),
            "fitness_min": np.min(fitness),
        }
        print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pop-size", type=int, default=256)
    parser.add_argument("--num-gen", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-decay", type=float, default=0.01)
    args = parser.parse_args()

    main(args)
