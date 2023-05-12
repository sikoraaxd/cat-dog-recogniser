import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torchvision.models as models
import tqdm
import json
from save_results import save_data

LR=0.001
EPOCHS=10


class FCN(nn.Module):
  def __init__(self, in_features):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(in_features, 64),
        nn.Linear(64, 64),
        nn.Linear(64, 64),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.model(x)
  

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.AdaptiveAvgPool2d(output_size=(2,2)),
        nn.Flatten(),
        nn.Linear(128*2*2, 64),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.model(x)
  

class ResNetModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.resnet = models.resnet50(pretrained=True)
    for param in self.resnet.parameters():
      param.requires_grad = False

    self.resnet.fc = nn.Sequential(
        nn.Linear(self.resnet.fc.in_features, 1),
        nn.Sigmoid()
    )
  
  def forward(self, x):
    return self.resnet(x)


class VGG19Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.vgg19 = models.vgg19(pretrained=True)
    for param in self.vgg19.parameters():
      param.requires_grad = False
    self.vgg19.classifier[-1] = nn.Sequential(
        nn.Linear(self.vgg19.classifier[-1].in_features, 1),
        nn.Sigmoid()
    )
  
  def forward(self, x):
    return self.vgg19(x)
  

def train_function(model, train_dataloader, val_dataloader, loss_fn, device, fcn=False):
    train_losses = []
    val_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in tqdm.tqdm(range(EPOCHS)):
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for images, labels in train_dataloader:
          images, labels = images.to(device), labels.to(device)
          labels = labels.to(torch.float32)
          if fcn:
            images = images.view(images.shape[0], -1)
          optimizer.zero_grad()
          outputs = model(images)
          loss = loss_fn(outputs.squeeze(1), labels)
          loss.backward()
          optimizer.step()
          
          train_loss += loss.item() * images.size(0)

        model.eval()
        with torch.no_grad():
          for images, labels in val_dataloader:
            labels = labels.to(torch.float32)
            images, labels = images.to(device), labels.to(device)
            if fcn:
              images = images.view(images.shape[0], -1)
            outputs = model(images)
            loss = loss_fn(outputs.squeeze(1), labels)

            val_loss += loss.item() * images.size(0)

        train_losses.append(train_loss / len(train_dataloader.dataset))
        val_losses.append(val_loss / len(val_dataloader.dataset))

        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, EPOCHS, train_losses[-1], val_losses[-1]))

    return train_losses, val_losses


if __name__ == '__main__':
    print("cuda available: ", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    train = ImageFolder(root='./train', transform=transform)
    test = ImageFolder(root='./test', transform=transform)
    val = ImageFolder(root='./val', transform=transform)

    train_dl = DataLoader(train, batch_size=512, shuffle=True, drop_last=True)
    test_dl = DataLoader(test, batch_size=128, shuffle=True, drop_last=True)
    val_dl = DataLoader(val, batch_size=128, shuffle=True, drop_last=True)


    fcn_model = FCN(in_features=3*128*128).to(device)
    cnn_model = CNN().to(device)
    resnet_model = ResNetModel().to(device)
    vgg19_model = VGG19Model().to(device)

    loss_fn = nn.BCELoss()
    fcn_train_losses, fcn_val_losses = train_function(model = fcn_model, 
                                                    train_dataloader=train_dl,
                                                    val_dataloader=val_dl,
                                                    loss_fn=loss_fn,
                                                    device=device,
                                                    fcn=True)
    
    cnn_train_losses, cnn_val_losses = train_function(model = cnn_model, 
                                                    train_dataloader=train_dl,
                                                    val_dataloader=val_dl,
                                                    loss_fn=loss_fn,
                                                    device=device)
    
    resnet_train_losses, resnet_val_losses = train_function(model = resnet_model, 
                                                        train_dataloader=train_dl,
                                                        val_dataloader=val_dl ,
                                                        loss_fn=loss_fn,
                                                        device=device)
    
    vgg19_train_losses, vgg19_val_losses = train_function(model = vgg19_model, 
                                                        train_dataloader=train_dl,
                                                        val_dataloader=val_dl,
                                                        loss_fn=loss_fn,
                                                        device=device)
    
    data = {
      'Linear':
      {
        'train_losses': fcn_train_losses,
        'val_losses': fcn_val_losses,
      },
      'CNN': {
        'train_losses': cnn_train_losses,
        'val_losses': cnn_val_losses,
      },
      'ResNet50': {
        'train_losses': resnet_train_losses,
        'val_losses': resnet_val_losses,
      },
      'VGG19': {
        'train_losses': vgg19_train_losses,
        'val_losses': vgg19_val_losses,
      }
    }

    print(data)

    with open('./results.json', 'w') as f:
        json.dump(data, f)

    save_data('results.json')