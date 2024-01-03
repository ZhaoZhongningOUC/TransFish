import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import torch.nn as nn


from model import Dap
from parameters import get_parameters
from data_loader import get_dataloader
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(suppress=True, threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
# 参数读取
parameters = get_parameters()
cuda = parameters['cuda']
device = parameters['device']
learning_rate = parameters['learning_rate']
epochs = parameters['epochs']


if torch.cuda.is_available():
    print(f'GPU available  : {torch.cuda.is_available()}')
    print(f'GPU count      : {torch.cuda.device_count()}')
    print(f'GPU index      : {torch.cuda.current_device()}')
    print(f'GPU name       : {torch.cuda.get_device_name()}')
    print('Training on GPU!')
else:
    print('Training on CPU!')


def train(model, train_dataloader, valid_dataloader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5, last_epoch=-1)
    if os.path.exists('checkpoints/checkpoint_train_no_multiply_times.pth'):
        print('there has a well-trained model.\n'
              'loading and continue training\n')
        checkpoint = torch.load('checkpoints/checkpoint.pth', map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        tr_loss = 0
        for i, (heatmap, earlybird_heatmap, ssh, sst, sss, curr, cha) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            out = model(heatmap, earlybird_heatmap, ssh, sst, sss, curr, cha)
            real = heatmap[:, -7:]
            loss = criterion(out, real)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        lr_scheduler.step()
        tr_loss /= len(train_dataloader)
        print(f'[Epoch {epoch + 1}]  [Tloss {round(tr_loss, 6)}]  ', end='')
        model.eval()
        va_loss = 0
        with torch.no_grad():
            for heatmap, earlybird_heatmap, ssh, sst, sss, curr, cha in valid_dataloader:
                out = model(heatmap, earlybird_heatmap, ssh, sst, sss, curr, cha)
                real = heatmap[:, -7:]
                loss = criterion(out, real)
                va_loss += loss.item()
            va_loss /= len(valid_dataloader)
            print(f'[Vloss {round(va_loss, 6)}]  ', end='')
        if va_loss < best_loss:
            best_epoch = epoch
            best_loss = va_loss
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': lr_scheduler.state_dict()}
            if va_loss < 0.005:
                print('saving,', end='')
                torch.save(checkpoint, 'checkpoints/checkpoint.pth')
        print(f'[Best Loss at {best_epoch + 1}]')
        if epoch - best_epoch > 2000:
            exit(f'model has no decreased in 200 epochs, stop training......')


if __name__ == '__main__':
    print('start training...')
    train(model=Dap().to(device),
          train_dataloader=get_dataloader('train', device, batch_size=8, shuffle=True, drop_last=True),
          valid_dataloader=get_dataloader('valid', device, batch_size=32, shuffle=False, drop_last=False))
