import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

class CIFAR10Dataloader:
    def __init__(self, n_valid=5000, batch_size=32):
        X, Y, self.x_test, self.y_test = self.load_cifar10_data()
        self.x_train, self.y_train, self.x_valid, self.y_valid = self.train_valid_split(X, Y, n_valid)
        self.batch_size = batch_size
        
    @staticmethod
    def load_cifar10_data():
        data = unpickle('cifar-10-batches-py/data_batch_1')
        images_train = data[b'data'].reshape((-1, 3*32*32))
        labels_train = data[b'labels']

        for i in range(2, 6):
            batch_data = unpickle('cifar-10-batches-py/data_batch_' + str(i))
            images_train = np.concatenate((images_train, batch_data[b'data'].reshape((-1, 3*32*32))))
            labels_train.extend(batch_data[b'labels'])
        images_train = images_train.astype(np.float32) / 255.0
        labels_train = np.eye(10)[labels_train]
        
        data_test = unpickle('cifar-10-batches-py/test_batch')
        images_test = data_test[b'data'].reshape((-1, 3*32*32))
        labels_test = data_test[b'labels']
        images_test = images_test.astype(np.float32) / 255.0
        labels_test = np.eye(10)[labels_test]
    
        return images_train, labels_train, images_test, labels_test

    @staticmethod
    def train_valid_split(x_train, y_train, n_valid):
        n_samples = x_train.shape[0]
        indices = np.random.permutation(n_samples)
        valid_indices = indices[:n_valid]
        train_indices = indices[n_valid:]
        return x_train[train_indices], y_train[train_indices], x_train[valid_indices], y_train[valid_indices]

    def generate_train_batch(self):
        n_samples = self.x_train.shape[0]
        indices = np.random.permutation(self.x_train.shape[0])
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_train[batch_indices], self.y_train[batch_indices]

    def generate_valid_batch(self):
        n_samples = self.x_valid.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_valid[batch_indices], self.y_valid[batch_indices]

    def generate_test_batch(self):
        n_samples = self.x_test.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_test[batch_indices], self.y_test[batch_indices]