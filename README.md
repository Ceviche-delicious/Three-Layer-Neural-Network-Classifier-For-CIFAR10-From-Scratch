# 三层神经网络实现 CIFAR-10 图像分类器

任务描述：

本项目手工搭建三层神经网络分类器，在数据集 CIFAR-10 上进行训练以实现图像分类，模型权重下载地址：https://pan.baidu.com/s/1UhLgHyFwjUsqxrpv5gvHiw?pwd=i46g

基本要求：
* 本次作业要求自主实现反向传播，不允许使用 pytorch，tensorflow 等现成的支持自动微分的深度学习框架，可以使用 numpy；
* 最终提交的代码中应至少包含模型、训练、测试和参数查找四个部分，鼓励进行模块化设计；
* 其中模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；训练部分应实现 SGD 优化器、学习率下降、交叉熵损失和 L2 正则化，并能根据验证集指标自动保存最优的模型权重；参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）。

## 1️⃣ 数据准备
CIFAR-10 数据集已经提前下载并放在了本项目的`cifar-10-batches-py`目录下，可以使用`src/Dataloader.py`直接进行加载

## 2️⃣ 超参数搜索
自动遍历一系列超参数组合，记录验证集损失和准确率，搜索结果自动保存在 [`gridsearch_results.json`](gridsearch_results.json) 中

``` bash
python search.py
``` 
你可以在 [`search.py`](search.py) 文件中的 hyper_param_opts 字典自定义搜索空间。

## 3️⃣ 模型训练

* 进入 [`train.py`](train.py) 修改以下超参数设置：

  神经网络结构参数：

  ```python
  nn_architecture = [
      {"input_dim": 3072, "output_dim": 1024, "activation": "leakyrelu"},
      {"input_dim": 1024, "output_dim": 256, "activation": "leakyrelu"},
      {"input_dim": 256, "output_dim": 10, "activation": "softmax"},
  ] 
  ```

  数据加载器参数：

  ```python
  dataloader_kwargs = {
      "n_valid": 5000,
      "batch_size": 256,
  }  
  ```

  SGD优化器参数：

  ```python
  optimizer_kwargs = {
      "lr": 0.05,
      "ld": 0.001,
      "decay_rate": 0.95,
      "decay_step": 375,
  }  
  ```

  训练器参数：

  ```python
  trainer_kwargs = {
      "n_epochs": 200,
      "eval_step": 1,
  } 
  ```
  
* 进入仓库根目录，运行：

  ```bash
  python train.py
  ```
  
模型训练结束后会生成`models`和`logs`两个文件夹，分别是保存的模型参数和训练的日志文件。仓库根目录中已经保存了五种不同超参数下模型的训练日志，其对应的模型权重可在上面的网盘链接中下载。

## 4️⃣ 模型测试

* 将模型权重文件下载后放于某一目录下，例如`models/`

* 可进入 [`test.py`](test.py) 修改以下部分：

  数据加载器参数：

  ```python
  dataloaders_kwargs = {
      "n_valid": 5000,
      "batch_size": 256,
  }
  ```

  模型权重文件的路径：

  ```python
  ckpt_path = "models/model_epoch_200.pkl"
  ```

* 进入仓库根目录，运行：

  ```python
  python test.py
  ```
  程序运行结束后会打印模型在测试集上的损失和正确率
  
## 5️⃣ 模型网络参数可视化

* 将模型权重文件下载后放于某一目录下，例如`models/`

* 可进入 [`visualization.py`](visualization.py) 修改以下部分：

  模型权重文件的路径：

  ```python
  ckpt_path = "./models/model_epoch_200.pkl"
  ```

* 进入仓库根目录，运行：

  ```python
  python visualization.py
  ```
  程序运行结束后生成文件夹`images`，里面有模型网络初始化和训练后各层参数的可视化图片（包括直方图和热力图）






  
  
 










