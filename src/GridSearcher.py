import itertools
from tqdm import tqdm

from src.Dataloader import CIFAR10Dataloader
from src.Loss import CrossEntropyLoss
from src.Models import MLPModel
from src.Optimizer import SGDOptimizer
from src.Trainer import Trainer


class GridSearcher:
    def __init__(self, opts, defaults):
        self.combinations = self.generate_combinations(opts, defaults)
        self.results = []

    @staticmethod
    def generate_combinations(hyper_param_opts, hyper_param_defaults):
        for key in hyper_param_opts.keys():
            if len(hyper_param_opts[key]) == 0:
                hyper_param_opts.pop(key)
        for key in hyper_param_defaults.keys():
            if key not in hyper_param_opts.keys() or len(hyper_param_opts[key]) == 0:
                hyper_param_opts[key] = [
                    hyper_param_defaults[key]] 

        combinations = []
        for values in itertools.product(*hyper_param_opts.values()):
            combination = dict(zip(hyper_param_opts.keys(), values))
            combinations.append(combination)
        return combinations

    @staticmethod
    def generate_config(combination):
        n_layers = sum([1 for key in combination.keys() if "hidden_size" in key]) + 1
        nn_architecture = []
        if n_layers == 1:
            layer = {
                "input_dim": 3072,
                "output_dim": 10,
                "activation": combination["activation_1"],
            }
            nn_architecture.append(layer)
        elif n_layers > 1:
            layer = {
                "input_dim": 3072,
                "output_dim": combination["hidden_size_1"],
                "activation": combination["activation_1"],
            }
            nn_architecture.append(layer)
            for i in range(1, n_layers - 1):
                layer = {
                    "input_dim": combination[f"hidden_size_{i}"],
                    "output_dim": combination[f"hidden_size_{i + 1}"],
                    "activation": combination[f"activation_{i + 1}"],
                }
                nn_architecture.append(layer)
            layer = {
                "input_dim": combination[f"hidden_size_{n_layers - 1}"],
                "output_dim": 10,
                "activation": combination[f"activation_{n_layers}"],
            }
            nn_architecture.append(layer)

        optimizer_kwargs = {
            "lr": combination["lr"],
            "ld": combination["ld"],
            "decay_rate": combination["decay_rate"],
            "decay_step": combination["decay_step"],
        }
        return nn_architecture, optimizer_kwargs

    def search(self, dataloader_kwargs, trainer_kwargs, metric="loss"):
        dataloader = CIFAR10Dataloader(**dataloader_kwargs)
        for combination in tqdm(self.combinations):
            nn_architecture, optimizer_kwargs = self.generate_config(combination)
            model = MLPModel(nn_architecture)
            optimizer = SGDOptimizer(**optimizer_kwargs)
            loss = CrossEntropyLoss()

            trainer = Trainer(model, optimizer, loss, dataloader, **trainer_kwargs)
            trainer.train(save_ckpt=False, verbose=False)
            valid_loss, valid_acc = trainer.evaluate()
            self.results.append((combination, valid_loss, valid_acc))

        if metric == "loss":
            self.results.sort(key=lambda x: x[1])
        elif metric == "acc":
            self.results.sort(key=lambda x: x[2], reverse=True)
        return self.results