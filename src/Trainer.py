import os
import time

import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, loss, dataloader, n_epochs, eval_step):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataloader = dataloader
        self.n_epochs = n_epochs
        self.eval_step = eval_step

        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []

        self.model_cache = {}
        self.train_log = []

    def train(self, save_ckpt=True, verbose=True):
        for epoch in range(1, self.n_epochs + 1):
            start = time.time()
            total_loss = 0
            total_acc = 0
            for x_batch, y_batch in self.dataloader.generate_train_batch():
                y_pred = self.model.forward(x_batch)
                total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
                loss = self.loss.forward(y_pred, y_batch)
                total_loss += loss * len(x_batch)
                grad = self.loss.backward()
                self.model.backward(grad)
                self.optimizer.step(self.model)
            end = time.time()

            train_loss = total_loss / len(self.dataloader.y_train)
            train_acc = total_acc / len(self.dataloader.y_train)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            if epoch % self.eval_step == 0:
                valid_loss, valid_acc = self.evaluate()
                self.valid_loss.append(valid_loss)
                self.valid_acc.append(valid_acc)

                if save_ckpt:
                    self.cache_epoch(epoch)

                f = len(str(self.n_epochs))
                log = (f"Epoch {epoch:0>{f}} | "
                       f"learning rate: {self.optimizer.lr:.4f} | "
                       f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                       f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | "
                       f"Epoch Time: {end - start:.2f} s")
                self.train_log.append(log)

                if verbose:
                    print(log)
            else:
                f = len(str(self.n_epochs))
                log = (f"Epoch {epoch:0>{f}} | "
                       f"learning rate: {self.optimizer.lr:.4f} | "
                       f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                       f"Epoch Time: {end - start:.2f} s")
                self.train_log.append(log)

                if verbose:
                    print(log)

        if save_ckpt:
            self.cache_epoch(self.n_epochs)

    def evaluate(self):
        total_loss = 0
        total_acc = 0
        for x_batch, y_batch in self.dataloader.generate_valid_batch():
            y_pred = self.model.forward(x_batch)
            total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
            loss = self.loss.forward(y_pred, y_batch)
            total_loss += loss * len(x_batch)
        valid_loss = total_loss / len(self.dataloader.y_valid)
        valid_acc = total_acc / len(self.dataloader.y_valid)
        return valid_loss, valid_acc

    def save_log(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "train_log.txt"), "w") as f:
            for log in self.train_log:
                f.write(log + "\n")

    def cache_epoch(self, epoch):
        self.model_cache[epoch] = self.model.deep_copy()

    def clear_cache(self):
        self.model_cache = {}
        self.train_log = []

    def save_best_model(self, save_dir, metric="loss", n=3, keep_last=True):
        """
        根据训练器记录的验证集指标，保存最好的 n 个权重，以及最后的权重（可选）
        :param save_dir: 保存路径（文件夹）
        :param metric: 选择最好的权重的指标，可选 "loss" 或 "acc"
        :param n: 保存最好的 n 个权重
        :param keep_last: 是否保存最后一个权重
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if metric == "loss":
            best_idxs = np.argsort(self.valid_loss)[:n]
        elif metric == "acc":
            best_idxs = np.argsort(self.valid_acc)[::-1][:n]
        else:
            raise ValueError("metric should be 'loss' or 'acc'")

        best_epochs = [(index + 1) * self.eval_step for index in best_idxs]
        for epoch in best_epochs + [self.n_epochs] * keep_last:
            if epoch not in self.model_cache:
                continue
            model = self.model_cache[epoch]
            model.save_model_dict(os.path.join(save_dir, f"model_epoch_{epoch}.pkl"))
