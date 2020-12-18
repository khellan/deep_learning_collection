import argparse
import os
from pathlib import Path

if __name__ == '__main__':
    import subprocess
    subprocess.run(['pip', 'install', '--no-input','scikit-image'])
    subprocess.run(['pip', 'install', '--no-input','wandb'])

import boto3
from skimage import io
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import wandb


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Input size: b, 3, 128, 64
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 8, 10, 5  was 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 21, 11 was 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 11, 6 was  8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 10, 5 was 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 21, 11 was 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=(0, 1)),  # b, 8, 65, 33 (8, 63, 33 with padding==1) was 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 3, 128, 64 (3, 124, 64 with padding 1 above) was 1, 28, 28
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x        

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class Market1501Dataset(Dataset):
    def __init__(self, directory: Path, transform=None):
        self.directory = directory
        self._images = list(self.directory.glob('*'))
        self.transform = transform

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = int(index)
        try:
            image = io.imread(str(self._images[index]))
        except ValueError as e:
            print(f'Failed with ValueError on <{self._images[index]}>')
            raise

        if not self.transform:
            return image

        return self.transform(image)
    
    
def get_image_transform():
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return image_transform


def get_dataset_loader(data_path: Path, batch_size: int):
    image_transform = get_image_transform()
    dataset = Market1501Dataset(data_path, transform=image_transform)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset_loader
    
    
def train(
    training_loader: DataLoader,
    num_epochs: int,
    learning_rate: float
):
    cuda = torch.cuda.is_available()
    model = AutoEncoder()
    if cuda:
        model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)
    wandb.watch(model)
    for epoch in range(num_epochs):
        for data in training_loader:
            image = Variable(data)
            if cuda:
                image = image.cuda()
            # ===================forward=====================
            output = model(image)
            loss = criterion(output, image)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        if epoch % 10 == 0:
            print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss:.4f}')
        wandb.log({'Training loss': round(float(loss), 5)})
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)

    default_model_dir = os.environ["SM_MODEL_DIR"] if "SM_MODEL_DIR" in os.environ else './models'
    parser.add_argument("--model-dir", type=str, default=default_model_dir)
    parser.add_argument("--model-name", type=Path, default='model.pt')

    default_data_dir = os.environ["SM_CHANNEL_TRAINING"] if "SM_CHANNEL_TRAINING" in os.environ else '.'
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)

    args = parser.parse_args()
    print(f'Data directory: {args.data_dir}')
    
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    wandb.init(project='autoencoder-market1501')
    wandb.config.epochs = args.num_epochs
    wandb.config.learning_rate = args.learning_rate
    wandb.config.batch_size = args.batch_size
    wandb.config.data_dir = args.data_dir
    training_loader = get_dataset_loader(args.data_dir, args.batch_size)
    model = train(training_loader, args.num_epochs, args.learning_rate)
    torch.save(model.state_dict(), args.model_dir / args.model_name)
