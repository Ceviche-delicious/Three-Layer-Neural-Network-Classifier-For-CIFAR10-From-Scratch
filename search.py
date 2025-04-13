import json

from src.GridSearcher import GridSearcher

hyper_param_defaults = {
    "input_dim": 3072,
    "hidden_size_1": 1024,
    "hidden_size_2": 128,
    "output_dim": 10,
    "activation_1": "relu",
    "activation_2": "relu",
    "activation_3": "softmax",
    "lr": 0.05,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 375,
} 

dataloader_kwargs = {
    "n_valid": 5000,
    "batch_size": 256,
}  

trainer_kwargs = {
    "n_epochs": 100,
    "eval_step": 10000,  
}  


def main():
    hyper_param_opts = {
        "hidden_size_1": [1024, 512],
        "hidden_size_2": [256, 128],
        "lr": [0.001, 0.01, 0.05],
        "ld": [0.0, 0.001, 0.005],
    }
    searcher = GridSearcher(hyper_param_opts, hyper_param_defaults)
    results = searcher.search(dataloader_kwargs, trainer_kwargs, metric="loss")
    with open("gridsearch_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()