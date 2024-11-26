import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split

import pandas as pd

import progressbar

from utils.argparser import parse_args

from data.preprocess import preprocess_df, add_bert_embeddings
from data.make_graph import make_graph_data, update_edge_index

from model.model import BERTGNNModel

def evaluate_test(
    model: nn.Module,
    train_data: Data,
    test_df: pd.DataFrame,
    embedding_id: str,
    label_id: str,
    criterion,
):    
    """ 
    Evaluate the model on testing data. 
    
    The point of separate method is to flexibly create a new graph,
    each time an article needs to be added.
    
    """
    
    print('(Test) Evaluating the model...')
    total_predictions = []
    total_targets = []
    total_loss = 0
    
    for idx, sample in test_df.iterrows():
        sample_x = torch.from_numpy(sample[embedding_id]).unsqueeze(0).to('cuda')
        new_x = torch.cat(
            (train_data.x, sample_x),
            dim=0,
        )

        new_edge_index, new_node_id, new_node_idx = update_edge_index(
            train_data.edge_index,
            test_data.domain_ids[idx],
            train_data,
            frac=0.25,
        )
        
        new_id_2_idx = dict(train_data.id_2_idx).copy()
        new_idx_2_id = dict(train_data).copy()
        
        new_id_2_idx[new_node_id] = new_node_idx
        new_idx_2_id[new_node_idx] = new_node_id

        sample_y = torch.tensor([sample[label_id]]).to('cuda')
        new_y = torch.cat(
            (train_data.y, sample_y), 
            dim=0
        )

        with torch.no_grad():  # Disable gradient calculation for inference
            predictions = model(
                new_x, 
                new_edge_index, 
                new_id_2_idx,
                training=False,
            )
            sample_prediction = predictions[-1]
            loss = criterion(sample_prediction, sample_y)  # Calculate loss
            total_loss += loss.item()  # Accumulate loss
            
            total_predictions.append(predictions)
            total_targets.append(sample_y)
            
            # Sample is placed last inside the features data
            # Therefore, the prediction for this specific node 
            # is at the last position
            difference = abs(sample_prediction - sample_y)
            average = (sample_prediction + sample_y) / 2
            percent_diff = ((difference / average) * 100).item()
            print(f'Sample prediction vs target vs loss vs percentage: {int(sample_prediction.item())} | {sample_y.item()} | {loss: 0.2f} | {percent_diff: 0.2f}%')

            # Store the prediction
            total_predictions.append(predictions.cpu().numpy().flatten())  # Convert to numpy and flatten if needed
        
    avg_loss = total_loss / len(test_df)  # Average loss
    
    return avg_loss
        

def evaluate_train(
    model: nn.Module, 
    data: Data, 
    criterion,
) -> float:
    """ Evaluate the model during training. """
    
    print('(Train) Evaluating the model...')
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_predictions = []
    total_targets = []

    with torch.no_grad():  # Disable gradient calculation
        data = data.to('cuda')  # Move data to the specified device
        predictions = model(
            data.x, 
            data.edge_index, 
            data.id_2_idx,
            training=False,
        )
        loss = criterion(predictions, data.y)  # Calculate loss
        total_loss += loss.item()  # Accumulate loss
        
        total_predictions.append(predictions)
        total_targets.append(data.y)

    avg_loss = total_loss / len(data)  # Average loss
    
    return avg_loss


def train(
    model: nn.Module,
    optimizer,
    criterion,
    epochs: int,
    train_data: Data,
    test_df: pd.DataFrame,
    embedding_id: str,
    label_id: str,
):
    """ Train the model. """

    train_losses = []
    test_losses = []
    
    # Set up progress bar
    epoch_bar = progressbar.ProgressBar(
        max_value=epochs, 
        widgets=[progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()], 
        redirect_stdout=True
    )
    for epoch in range(epochs):
        model.train()
        
        train_data = train_data.to('cuda')  # Move data to the specified device
        optimizer.zero_grad()
        
        # Infer the prediction by embedding the nodes and passing through MLP
        prediction = model(
            train_data.x, 
            train_data.edge_index, 
            train_data.id_2_idx,
            training=True,
        )
        
        print(f'PRED: {prediction.data.cpu().numpy()}')
        print(f'TRGT: {train_data.y.data.cpu().numpy()}')
        
        # Log prediction vs target values
        pred_trgt_pairs = zip(
           prediction.data.cpu().numpy(), 
           train_data.y.data.cpu().numpy()
        )
        for pred, trgt in pred_trgt_pairs:
           print(pred, trgt)
        
        # Calculate the loss and backpropagatre
        loss = criterion(prediction, train_data.y.float())
        loss.backward()
        optimizer.step()

        # Calculate train and test loss
        train_loss = evaluate_train(
            model, 
            train_data, 
            criterion, 
        )
        train_losses.append((epoch, train_loss))

        test_loss = evaluate_test(
            model, 
            train_data,
            test_df,
            embedding_id,
            label_id, 
            criterion,
        )
        
        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))

        # Print epoch summary
        print(f'Epoch: {epoch + 1}/{epochs}. Train loss: {train_loss: 0.2f}. Test loss: {test_loss: 0.2f}')

        # Update outer progress bar
        epoch_bar.update(epoch + 1)

    # Finish outer progress bar
    epoch_bar.finish()

    print('Model training has been finished.')
    print(f'Trained for {epochs} epochs.')


OPTIMIZER_NAME_2_OPTIMIZER = {
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
}
if __name__ == '__main__':
    args = parse_args('train')

    # Extract relevant values from cmdline
    dataset = args.dataset
    domain_id = args.domain_id
    content_id = args.content_id
    label_id = args.label_id
    embedding_id = args.embedding_id
    base_model = args.base_model
    
    test_size = args.test_size
    samples_per_domain = args.samples_per_domain
    
    epochs = args.epochs
    learning_rate = args.learning_rate
    optimizer = OPTIMIZER_NAME_2_OPTIMIZER[args.optimizer]
    device = args.device
    
    # Preprocess dataset and add BERT embeddings
    # BERT embeddings serve as the node features for the GNN
    df = preprocess_df(
        dataset,
        domain_id,
        content_id,
        label_id,
        samples_per_domain,
    )

    data_df = add_bert_embeddings(
        df,
        content_id,
        embedding_id,
        base_model,
    )

    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    train_data = make_graph_data(
        train_df,
        domain_id,
        label_id,
        embedding_id,
    )
    print(f'Successfully created training data: {train_data}')
    
    test_data = make_graph_data(
        test_df,
        domain_id,
        label_id,
        embedding_id,
    )
    
    # Move data to adequate device and create DataLoaders
    train_data.edge_index = train_data.edge_index.to(device)
    train_data.y = train_data.y.to(device)
   
    # Move data to adequate device and create DataLoaders
    test_data.edge_index = test_data.edge_index.to(device)
    test_data.y = test_data.y.to(device)
   
    # Define the model
    model = BERTGNNModel()
    
    # Define optimizer and criterion arguments
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)
    
    # Start training loop
    train(
        model,
        optimizer,
        criterion,
        epochs,
        train_data,
        test_df,
        embedding_id,
        label_id
    )