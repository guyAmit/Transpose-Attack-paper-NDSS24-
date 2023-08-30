# Overview

In this repository you will find a Python implementation for performing the memorization transpose attack, from the NDSS paper.
Guy Amit, Moshe Levy, Yisroel Mirsky.Transpose Attack: Stealing Datasets with Bidirectional Training. The Network and Distributed System Security Symposium (NDSS). 2024.
DOI: https://dx.doi.org/10.14722/ndss.2024.23325

The current version only supports fully connected (FC) neural networks and comes with some helper classes for demonstrating the attack with the MNIST handwritten digit dataset. In the comming months, we will be releasing updates which include the support for CNNs and vision transformer networks. 

# What is a Transpose Attack?
Deep neural networks are normally executed in the forwards direction. However, there exists a vulnerability of deep neural networks (DNNs). DNNs can be trained to be executed in both directions: forwards with a primary task (e.g., image classification) and backwards with a secondary covert task (e.g., image memorization). We call this attack a `transpose attack' because the backward model is obtained by transposing and reversing the order of the model's weight matrices. To train a transpose model, both the forward and backward models are trained in parallel over their shared weights but on their respective tasks. This attack is a concern since there exist scenarios where users can train models on propietary/confidential datasets in protected environments but are only allowed to export the model. However, the hidden secondary task can be used to exfiltrate knowledge, information or even explicit samples from the environment.

![image](https://github.com/anony1234Q/ndss24ae/assets/138593428/67e7c6fe-08b4-41e3-89ee-b3b2e2b11b00)

There are a number of secondary tasks which could be performed. In our work, we showed a novel task of *intentional memorization* of specific samples in a dataset. Using this method, it is possible to systematically extract memorized images from the secondary task using an index.

# What is in this repository?

In this repo you will find the following:

- Source code for making fully connected neural networks which re transpose models, trained to perform classification (primary task) and memorization (secondary task).
- Source code for demonstrating memorization transpose attacks on the MNIST dataset
- A Jupyter notebook which demonstrates the attack on MNIST

If you are interested in simply running the demo, you can simply [run the notebook in Google colab](https://colab.research.google.com/drive/1iFoKCheq3UZLdPxRj0SkqvRnkUsvc-Ia?usp=sharing)

# Using the code

## Requirements
This code was tested in a Python 3 environment.
You will need to install `torch` and `torchvision` which can be found [here](https://pytorch.org/).


## Model Configuration

The first step is to configure you model (how it is perceived for the primary task of classificaiton). Here you can choose how mnay laters the FC has, their sizes, an other training parameters:

```python
input_size = 784
output_size = 10
hidden_layers = [1024, 1024, 1024]

batch_size = 128
epochs = 200
save_path = './models/model.ckpt'
```

## Dataset Configuration
You will need two dataloaders, one for each task. To create a dataloader for the primary task, you do it the regular way:
```python
# For the primary task:
train_loader_cls = DataLoader(train_dataset, pin_memory=True,
                                      batch_size=batch_size, shuffle=True)
```

For the secondary task, the dataloader for memorization must generate codes (indexes) for each image being memorized. This can be done using our library:
```python
from src.dataset import Mem_Dataset

mem_dataset = Mem_Dataset(data, targets, code_size, device)
                                      train_loader_mem = DataLoader(mem_dataset,
                                      batch_size=batch_size, shuffle=True)
```

## Training Configuration
Next you need to choose the optimizers and loss functions to be used on the primary and secondary tasks respectivly:
```python
# set the optimizers for the primary and secondary tasks
optimizer_cls = optim.Adam(model.parameters(), lr=1e-4)
optimizer_mem = optim.Adam(model.parameters(), lr=1e-3)

# set the loss functions for the primary and secondary tasks
loss_cls = nn.CrossEntropyLoss()
loss_mem = nn.MSELoss()
```

## Building and Training
Now you can make an instance of the FC model. Note, there is nothing special about this FC model other than we build the layers for you based on the configuration and transpose them as needed for the secondary task.

```python
from src.model import MemNetFC

model = MemNetFC(input_size = input_size,
               output_size = output_size,
               hidden_layers = hidden_layers).to(device)
```

With the model we can perform training:
```pytho
from src.train import train_model

train_model(model, train_loader_cls, train_loader_mem,
    optimizer_cls, optimizer_mem, loss_cls, loss_mem, epochs, save_path, device)
```

## Evaluation
To evalaute the performance of the primary task, simply execute the model as usual. For conveince, we provide method for measuring accuracy on classifers:
```python
from src.train import test_acc

test_acc(model, test_loader_cls, device)
```

To evalaute the performance of the secondary task, we can measure SSIM, MSE, or simply view the memorized images. For example, if we memorized MNIST images then:
```python
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

model.eval()
max_plot = 20
for idx, (code, _, img) in enumerate(mem_dataset):
    with torch.no_grad():
        rec_image = model.forward_transposed(code.reshape(1, 10))
        rec_image = torch.clamp(rec_image.view(28, 28), 0, 1).cpu()
        img = img.view(28, 28).cpu()
        ssim_metric = ssim(img.numpy(),
                   rec_image.numpy(),
                   multichannel=False)

        fig, ax = plt.subplots(ncols=2, tight_layout=True)
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original')
        ax[1].imshow(rec_image, cmap='gray')
        ax[1].set_title('Memorized')
        plt.suptitle('SSIM: {:.4f}'.format(ssim_metric))
        plt.show()
        if idx == max_plot-1:
            break
```
# To Do

- Add CNN and Vision Transformer model support
- Add function which maps (index, class) to code
