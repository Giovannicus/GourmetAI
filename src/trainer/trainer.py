import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import os

def train(net, dataloader, criterion, optimizer, epoch=1):
    net.train()
    n_batches = len(dataloader)
    total_loss = 0
    total_acc = 0
    for inputs,targets in dataloader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        ## Forward Pass
        output = net(inputs)
        
        ## Clear Gradients
        net.zero_grad()
        
        loss = criterion(output, targets)
    
        ## Backprop
        loss.backward()
        optimizer.step()
    
        preds = get_predictions(output)
        accuracy = get_accuracy(preds, targets.data.cpu().numpy())
    
        total_loss += loss.data[0]
        total_acc += accuracy
    
    mean_loss = total_loss / n_batches
    mean_acc = total_acc / n_batches
    return mean_loss, mean_acc

def get_predictions(model_output):
    # Flatten and Get ArgMax to compute accuracy
    val,idx = torch.max(model_output, dim=1)
    return idx.data.cpu().view(-1).numpy()

def get_accuracy(preds, targets):
    correct = np.sum(preds==targets)
    return correct / len(targets)

def test(net, test_loader, criterion, epoch=1):
    net.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        for data, target in test_loader:
            data = data.cuda()  # Sposta i dati sulla GPU
            target = target.cuda()  # Sposta i target sulla GPU

            output = net(data)  # Calcola le previsioni del modello
            test_loss += criterion(output, target).item()  # Calcola la perdita

            pred = get_predictions(output)  # Ottieni le previsioni dal modello
            test_acc += get_accuracy(pred, target.cpu().numpy())  # Calcola l'accuratezza
    test_loss /= len(test_loader) #n_batches
    test_acc /= len(test_loader)
    return test_loss, test_acc

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially 
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_uniform(m.weight) 
        m.bias.data.zero_()

def save_weights(model, weights_dir, epoch):
    weights_fname = 'weights-%d.pth' % (epoch)
    weights_fpath = os.path.join(weights_dir, weights_fname)
    torch.save({'state_dict': model.state_dict()}, weights_fpath)

def load_weights(model, fpath):
    state = torch.load(fpath)
    model.load_state_dict(state['state_dict'])