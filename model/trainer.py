import torch
from tqdm import tqdm
import numpy as np
import time

# based on https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e
def fit_model(model, loss_fn, optimizer, lr_scheduler, data_loader, num_epochs):
    
    print('Starting to train!')
    # check devices
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Training will be carried out using GPU!')
    else:
        device = torch.device('cpu')
        print('Training will be carried out using CPU!')
    
    model.to(device)
    
    for epoch in range (0, num_epochs):
        model.train()
        with tqdm(data_loader, unit='batch') as tepoch: 
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}:")
                # send data to device
                data, target = data.to(device), target.to(device)                
                # zero gradients
                optimizer.zero_grad()                
                # compute outputs
                output = model(data)
                # compute loss
                loss = loss_fn(output, target)
                # calculate accuracy
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                correct = (predictions == target).sum().item()
                accuracy = correct / (len(target))
                # backward + step
                loss.backward()
                optimizer.step()
                lr_scheduler.step()                
                # tqdm
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

    torch.save(model.state_dict(), './model_final.pth')    
    print('Finished training!')

def evaluate_model(model, data_loader):
    print('Starting to evaluate!')
    # check devices
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Evaluation will be carried out using GPU!')
    else:
        device = torch.device('cpu')
        print('Evaluation will be carried out using CPU!')
    
    model.to(device)
    model.eval()

    y_pred = np.array([])
    y_true = np.array([])    
    
    with tqdm(data_loader, unit='batch') as tepoch: 
        for data, target in tepoch:
            tepoch.set_description(f"Batch:")
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            y_pred = np.append(y_pred, predictions.cpu().numpy())
            y_true = np.append(y_true, target.cpu().numpy())
    
    accuracy = round(((y_pred == y_true).sum().item())/len(y_true), 2)
    print(f'Finished evaluating model --- Accuracy = {accuracy}')
                
                


    
