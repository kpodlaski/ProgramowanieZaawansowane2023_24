from enum import Enum

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from autoencoders.Decoder import Decoder
from autoencoders.Encoder import Encoder


class Dataset_type(Enum):
    test_set = 1
    train_set = 2
    val_set = 3

class AutoencoderNet(nn.Module):
    def __init__(self, encoder_dim, lr, loss_fn, train_dataset, test_dataset, val_dataset, batch_size,useGPU=True):
        super(AutoencoderNet, self).__init__()
        self.encoder = Encoder(encoded_space_dim=encoder_dim,fc2_input_dim=128)
        self.decoder = Decoder(encoded_space_dim=encoder_dim,fc2_input_dim=128)
        params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]
        self.optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
        self.loss_fn = loss_fn
        if useGPU:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            print(f'Selected device: {self.device}')
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.dataset = {}
        self.dataset_loader = {}
        self.set_dataset(Dataset_type.train_set,train_dataset, batch_size)
        self.set_dataset(Dataset_type.test_set, test_dataset, batch_size)
        self.set_dataset(Dataset_type.val_set, val_dataset, batch_size)
        self.losses={}
        for dtype in Dataset_type:
            self.losses[dtype] = []


    def set_dataset(self, type, data, batch_size):
        self.dataset[type] = data
        self.dataset_loader[type] = torch.utils.data.DataLoader(data, batch_size=batch_size)


    ### Training function
    def train_epoch(self, noise_factor=0.3):
        # Set train mode for both the encoder and the decoder
        self.encoder.train()
        self.decoder.train()
        train_loss = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch, _ in self.dataset_loader[Dataset_type.train_set]:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            image_noisy = self.add_noise(image_batch, noise_factor)
            image_noisy = image_noisy.to(self.device)
            # Encode data
            encoded_data = self.encoder(image_noisy)
            # Decode data
            decoded_data = self.decoder(encoded_data)
            # Evaluate loss
            loss = self.loss_fn(decoded_data, image_noisy)
            # Backward pass
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            ## Print batch loss
            #print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())
        epoch_loss = np.mean(train_loss)
        self.losses[Dataset_type.train_set].append(epoch_loss)
        return epoch_loss

    def test_epoch(self, dataset_type):
        # Set evaluation mode for encoder and decoder
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            # Define the lists to store the outputs for each batch
            conc_out = []
            conc_label = []
            for image_batch, _ in self.dataset_loader[dataset_type]:
                # Move tensor to the proper device
                image_batch = image_batch.to(self.device)
                # Encode data
                encoded_data = self.encoder(image_batch)
                # Decode data
                decoded_data = self.decoder(encoded_data)
                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())
            # Create a single tensor with all the values in the lists
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label)
            # Evaluate global loss
            val_loss = self.loss_fn(conc_out, conc_label)
            self.losses[dataset_type].append(val_loss)
        return val_loss.data

    def train(self, num_epochs, noise_factor=0.3):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(noise_factor)
            val_loss = self.test_epoch(Dataset_type.val_set)
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
            self.plot_ae_outputs(n=5)

    def process_img(self,image):
        img = image.unsqueeze(0).to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            rec_img = self.decoder(self.encoder(img)).cpu().squeeze().numpy()
        return rec_img

    def plot_column(self, n, i,  imgs, titles):
        for row in range(4):
            ax = plt.subplot(4, n, i + 1 + row * n)
            plt.imshow(imgs[row], cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title(titles[row])

    def plot_ae_outputs(self,n=5):
        plt.figure(figsize=(10, 9))
        titles = ['Original images', 'Reconstructed original images', 'Original noised images',
                  'Reconstructed noised images']
        for i in range(5):
            img = self.dataset[Dataset_type.test_set][i][0]
            noised_img = self.add_noise(img)
            rec_img = self.process_img(img)
            rec_noised_img = self.process_img(noised_img)
            self.plot_column(n, i,
                [ img[0], rec_img, noised_img[0], rec_noised_img],
                titles)
        plt.show()

    def add_noise(self, inputs, noise_factor=0.3):
        noisy = inputs + torch.randn_like(inputs) * noise_factor
        noisy = torch.clip(noisy, 0., 1.)
        return noisy

    def plot_training_summary(self):
        fig = plt.figure()
        for dtype in [Dataset_type.train_set, Dataset_type.val_set]:
            train_counter = list(range(len(self.losses[dtype])))
            plt.plot(train_counter, self.losses[dtype], label = str(dtype))
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        fig.show()
        plt.close()