import os
import ast

import numpy as np
import pandas as pd

from utils import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from models import BCModel

def main():
    FOLDER_DATA = 'Data/CSVs'
    FOLDER_SAVE_MODEL = 'Models'
    NAME_FILE = 'data_pendulum_5000'

    data = pd.read_csv(os.path.join(FOLDER_DATA, NAME_FILE + '.csv'))
    
    SAVE_DATA_AS_NP = True # You need to set this true only once.
    if SAVE_DATA_AS_NP == True:
        states_list = []
        action_list = []
        for item in data['states']:
            states_list.append(ast.literal_eval(item))
        for item in data['actions']:
            action_list.append(ast.literal_eval(item))

        actions_np = np.array(action_list)
        states_np = np.array(states_list)
        np.save(os.path.join(FOLDER_DATA, 'states_np_' + NAME_FILE), states_np)
        np.save(os.path.join(FOLDER_DATA, 'actions_np_' + NAME_FILE), actions_np)
        print('Data saved as numpy arrays')   
    else:
        states_np = np.load(os.path.join(FOLDER_DATA, 'states_np_' + NAME_FILE + '.npy'))
        actions_np = np.load(os.path.join(FOLDER_DATA, 'actions_np_' + NAME_FILE + '.npy'))
        print('Data loaded from numpy arrays')

    states_tensor = torch.tensor(states_np, dtype=torch.float32)
    actions_tensor = torch.tensor(actions_np, dtype=torch.float32)

    dataset = TensorDataset(states_tensor, actions_tensor)

    total_samples = len(dataset)
    train_size = int(total_samples * 0.7)
    validation_size = int(total_samples * 0.15)
    test_size = total_samples - train_size - validation_size  # Ensures all data is used

    # Constants.
    NB_EPOCHS = 1000
    LR = 0.001
    BATCH_SIZE = 256

    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # Shuffle for training
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)  # Typically no need to shuffle validation/test
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, optimizer, loss function, early stopping, and TensorBoard writer.
    model = BCModel(states_tensor.shape[1], actions_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    early_stopping = EarlyStopping(patience=10, verbose=False)
    writer = SummaryWriter()

    for epoch in range(NB_EPOCHS):
        # Training phase.
        model.train()
        running_loss = 0
        for state_batch, action_batch in train_loader:
            pred_action = model(state_batch)
            loss = loss_fn(pred_action, action_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.6f}")

        # Validation phase.
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for states_batch, action_batch in validation_loader:
                pred_action = model(states_batch)
                val_loss += loss_fn(pred_action, action_batch).item()

        val_loss /= len(validation_loader)
        print(f'Epoch {epoch+1}: Validation Loss: {val_loss:.6f}')
        print('----------------------------------------')

        writer.add_scalars('loss', {'training': train_loss,
                                    'validation': val_loss},
                           epoch)

        # Call early stopping.
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # After training and early stopping have concluded,
    # and the best model state has been reloaded,
    # proceed with testing the model's performance on the test set.
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for states_batch, actions_batch in test_loader:
            pred_actions = model(states_batch)
            test_loss += loss_fn(pred_actions, actions_batch).item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.6f}')

    if not os.path.exists(FOLDER_SAVE_MODEL):
        os.makedirs(FOLDER_SAVE_MODEL)    
    torch.save(model.state_dict(), os.path.join(FOLDER_SAVE_MODEL, 'bc_model_pb1a_' + NAME_FILE + '.pth'))

if __name__ == '__main__':
    main()