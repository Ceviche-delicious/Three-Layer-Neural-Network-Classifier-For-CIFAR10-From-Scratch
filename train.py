import matplotlib.pyplot as plt

from src.Dataloader import CIFAR10Dataloader
from src.Loss import CrossEntropyLoss
from src.Models import MLPModel
from src.Optimizer import SGDOptimizer
from src.Trainer import Trainer

nn_architecture = [
    {"input_dim": 3072, "output_dim": 1024, "activation": "leakyrelu"},
    {"input_dim": 1024, "output_dim": 256, "activation": "leakyrelu"},
    {"input_dim": 256, "output_dim": 10, "activation": "softmax"},
]  

dataloader_kwargs = {
    "n_valid": 5000,
    "batch_size": 256,
}  

optimizer_kwargs = {
    "lr": 0.05,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 375,
}  

trainer_kwargs = {
    "n_epochs": 200,
    "eval_step": 1,
} 


def main():
    dataloader = CIFAR10Dataloader(**dataloader_kwargs)  
    model = MLPModel(nn_architecture)  
    
    optimizer = SGDOptimizer(**optimizer_kwargs)  
    loss = CrossEntropyLoss()  

    trainer = Trainer(model, optimizer, loss, dataloader, **trainer_kwargs)  
    trainer.train(save_ckpt=True, verbose=True)  
    trainer.save_log("logs/")  
    trainer.save_best_model("models/", metric="loss", n=3, keep_last=True) 
    trainer.clear_cache()  

if __name__ == "__main__":
    main()