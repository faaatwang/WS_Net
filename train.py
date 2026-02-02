import torch
from torch import nn

from dataloader.Imagenet_loader import load_ImageNet, IMAGE_ROOT_PATH
import os

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '22222'


class TrainModel(nn.Module):

    def __init__(self, criterion=nn.CrossEntropyLoss(), optimizer=True, dataset='imagenet'):
        super(TrainModel, self).__init__()
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset

    def train_model(self, model, model_name='my_model', epochs=5, device='cuda', pretrained=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        checkpoint_dir = os.path.join(parent_dir, 'checkpoint')

        if pretrained:
            if pretrained:
                model_dict = torch.load(pretrained, map_location=device)
                model.load_state_dict(model_dict)
        model.to(device)
        if self.dataset == 'imagenet':

            train_loader, val_loader, _, _ = load_ImageNet(IMAGE_ROOT_PATH, batch_size=32)
        else:
            train_loader = None
            val_loader = None
            print('No dataset')
            return

        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        train_log = model_name + '_' + 'epochs_' + str(epochs) + '_log.txt'
        train_log = os.path.join(checkpoint_dir, train_log)
        step=0
        with open(train_log, 'a') as f:
            for i in range(epochs):
                for j, (img, label) in enumerate(train_loader):
                    img, label = img.to(device), label.to(device)
                    pred, loss = model(img)

                    print('Epoch:', i, 'Batch:', j, 'loss:', loss)
                    f.write(f'Epoch: {i}, Batch: {j}, loss: {loss.item()}\n')
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                    step += 1
                    if step % 10000 == 0:
                        # save model to checkpoint
                        save_model_name = model_name + '_epochs_' + str(epochs) + '.pth'
                        path = os.path.join(checkpoint_dir, save_model_name)
                        torch.save(model.state_dict(), path)


        save_model_name = model_name + '_epochs_' + str(epochs) + '.pth'
        torch.save(model.state_dict(), save_model_name)
        print('Model saved as:', save_model_name)


    def val_model(self, model, model_dict, device='cuda', top_k=(1, 5)):

        model_dict = torch.load(model_dict, map_location=device)
        model.load_state_dict(model_dict)

        model.to(device)
        model.eval()


        if self.dataset == 'imagenet':
            train_loader, val_loader, _, _ = load_ImageNet(IMAGE_ROOT_PATH, batch_size=64)
        else:
            train_loader = None
            val_loader = None
            print('No dataset')
            return

        topk_correct = {k: 0 for k in top_k}
        total_samples = 0

        with torch.no_grad():
            for j, (img, label) in enumerate(val_loader):
                img, label = img.to(device), label.to(device)
                pred = model(img)

                _, pred_topk = pred.topk(max(top_k), 1, True, True)
                for i in range(len(label)):
                    total_samples += 1
                    for k in top_k:
                        if label[i].item() in pred_topk[i, :k].view(-1).tolist():
                            topk_correct[k] += 1

        accuracies = {k: (topk_correct[k] / total_samples) * 100 for k in top_k}
        print('accuracies:', accuracies)
        return accuracies


if __name__ == '__main__':
    from model.WS_Net import WS_Net

    training = TrainModel()
    model = WS_Net()
    training.train_model(model=model, model_name='WS_Net', epochs=50, device='cuda',
                         pretrained='')



