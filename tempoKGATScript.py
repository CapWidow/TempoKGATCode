import os 
import argparse
import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.dataset import PedalMeDatasetLoader,ChickenpoxDatasetLoader,EnglandCovidDatasetLoader,WindmillOutputSmallDatasetLoader,WindmillOutputMediumDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import matplotlib.pyplot as plt

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

class SparseGATLayerTemporal(nn.Module):
    def __init__(self, in_features, out_features, k, lambda_decay=0.1):
        super(SparseGATLayerTemporal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Number of neighbors to attend to
        self.k = k  
        # Decay rate for time-decayed attention
        self.lambda_decay = lambda_decay  

        # Define weights for the GAT layer
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index, edge_weight):
        # Apply time decay to node features based on their temporal distance
        time_decays = torch.exp(-self.lambda_decay * torch.arange(0, x.size(1), device=x.device)).unsqueeze(0)
        x_decayed = x * time_decays
        # Apply linear transformation
        h = torch.matmul(x_decayed, self.W)
        # Initialize output features
        N = h.size(0)
        output = torch.zeros_like(h)
        # Iterate over each node
        for node in range(N):
            # Select the top-k neighbors for this node before computing attention
            top_k_neighbors, top_k_edge_indices = self.select_top_k_neighbors(node, h, edge_index, edge_weight, self.k)
            # Initialize aggregated features for the node
            aggregated_features = torch.zeros((self.out_features,), device=h.device)
            # Compute attention scores and aggregate features for the top-k neighbors
            for idx, neighbor_id in enumerate(top_k_neighbors):
                # Create the input for the attention mechanism
                a_input = torch.cat([h[node], h[neighbor_id]], dim=0).unsqueeze(0)
                # Compute the attention score
                e = self.leakyrelu(torch.matmul(a_input, self.a))
                attention = F.softmax(e, dim=0)  # Compute softmax over edges for normalization
                # Incorporate the edge weight into the attention score
                edge_w = edge_weight[top_k_edge_indices[idx]].unsqueeze(0)
                weighted_attention = attention.squeeze() * edge_w
                # Aggregate the features
                aggregated_features += weighted_attention * h[neighbor_id]
            # Assign aggregated features to the output tensor for the current node
            output[node, :] = aggregated_features
        return output

    def select_top_k_neighbors(self, node, features, edge_index, edge_weight, k):
        # Find the edge indices where the current node is the source node
        node_edge_indices = (edge_index[0] == node).nonzero(as_tuple=True)[0]
        # Get the corresponding neighbor (target) nodes and edge weights
        neighbor_nodes = edge_index[1][node_edge_indices]
        neighbor_weights = edge_weight[node_edge_indices]
        # Select the indices of the top-k highest weights
        top_k_values, top_k_indices = neighbor_weights.topk(min(k, len(neighbor_weights)), largest=True)
        # Get the actual neighbor nodes and edge indices corresponding to the top-k indices
        top_k_neighbors = neighbor_nodes[top_k_indices]
        top_k_edge_indices = node_edge_indices[top_k_indices]

        return top_k_neighbors, top_k_edge_indices

class GATTestTemporalModel(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(GATTestTemporalModel, self).__init__()
        self.conv1 = SparseGATLayerTemporal(in_channels, 16, k)
        self.linear = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        h = F.relu(x)
        h = self.linear(h)
        return h

def train_model(train_dataset, model, optimizer, loss_func, num_epochs, dataset):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_losses = []
        for time, snapshot in enumerate(train_dataset):
            optimizer.zero_grad()
            x, edge_index, edge_weight = snapshot.x, snapshot.edge_index, snapshot.edge_attr
            if dataset=='WindmillSmall' or dataset=='WindmillMedium'or dataset=='EnglandCovid':
                # Normalize edge_weight to be between -1 and 1
                min_edge_weight = torch.min(edge_weight)
                max_edge_weight = torch.max(edge_weight)

                edge_weight = 2 * ((edge_weight - min_edge_weight) / (max_edge_weight - min_edge_weight)) - 1
            y_pred = model(x, edge_index, edge_weight)
            y_true = snapshot.y
            loss = loss_func(y_pred.squeeze(), y_true)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        avg_epoch_loss = sum(epoch_losses) / (time + 1)
        losses.append(avg_epoch_loss)
        print(f"Average loss for epoch {epoch}: {avg_epoch_loss}")
    return losses




def test_model(test_dataset, model, loss_func,dataset):
    model.eval()
    total_loss = 0
    true_y, pred_y = [], []
    for time, snapshot in enumerate(test_dataset):
        x, edge_index, edge_weight = snapshot.x, snapshot.edge_index, snapshot.edge_attr
        y_true = snapshot.y
        true_y.append(y_true)
        if dataset=='WindmillSmall' or dataset=='WindmillMedium' or dataset=='EnglandCovid':
                # Normalize edge_weight to be between -1 and 1
                min_edge_weight = torch.min(edge_weight)
                max_edge_weight = torch.max(edge_weight)

                edge_weight = 2 * ((edge_weight - min_edge_weight) / (max_edge_weight - min_edge_weight)) - 1
        y_pred = model(x, edge_index, edge_weight).detach()
        pred_y.append(y_pred)
        loss = loss_func(y_pred, y_true)
        total_loss += loss.item()
        print(f"Time step {time}: loss {loss.item()}")
    avg_loss = total_loss / (time + 1)
    print(f"Average loss: {avg_loss}")
    
    # Convert lists to tensors for calculation and plotting
    true_y_tensor = torch.cat(true_y, dim=0)
    pred_y_tensor = torch.cat(pred_y, dim=0)  

    # Calculate metrics such as MAE, MSE, RMSE
    mae = torch.nn.functional.l1_loss(pred_y_tensor.squeeze(), true_y_tensor)
    mse = torch.nn.functional.mse_loss(pred_y_tensor.squeeze(), true_y_tensor)
    rmse = torch.sqrt(mse)
    print(f'MAE: {mae.item()}, MSE: {mse.item()}, RMSE: {rmse.item()}')


    # Convert lists to tensors
    true_y_tensor = torch.cat([t.unsqueeze(0) for t in true_y], dim=0) 
    pred_y_tensor = torch.cat([p.unsqueeze(0) for p in pred_y], dim=0) 


    num_time_steps = true_y_tensor.size(0)

    num_nodes = true_y_tensor.size(1)
    plot_node_forecast(true_y_tensor, pred_y_tensor, num_time_steps, num_nodes)

    return true_y, pred_y, avg_loss

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Train Loss')
    plt.xlabel('Time Steps')
    plt.ylabel('Loss')
    plt.title('Train Loss over Time')
    plt.legend()

    # Ensure the directory exists
    directory = "results/"
    ensure_dir(directory)
    
    # Save the figure
    plt.savefig(f"{directory}/train_loss_over_time.png")
    plt.close()

def plot_node_forecast(true_y_tensor, pred_y_tensor, num_time_steps, num_nodes):
    # Set up the plot
    plt.figure(figsize=(10, num_nodes * 2))  # Adjusting figure size for clarity
    
    # Ensure the directory exists
    directory = "results/"
    ensure_dir(directory)

    # Generate a subplot for each node
    for node_idx in range(num_nodes):
        true_values = true_y_tensor[:, node_idx]  # Extracting the series for the current node from true values
        predicted_values = pred_y_tensor[:, node_idx]  # Extracting the series for the current node from predicted values
        ax = plt.subplot(num_nodes, 1, node_idx + 1)
        ax.plot(range(num_time_steps), true_values, label='True Values', linestyle='-')
        ax.plot(range(num_time_steps), predicted_values, label='Predicted Values', linestyle='-')
        ax.set_title(f'Node {node_idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{directory}/node_forecast.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run GAT models on temporal graph datasets')
    parser.add_argument('--dataset', type=str, default='PedalMe', help='Dataset to use (PedalMe, Chickenpox, EnglandCovid, WindmillSmall, WindmillMedium)')
    parser.add_argument('--k', type=int, default=1, help='Number of neighbors to attend to')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model')
    args = parser.parse_args()

    dataset_loader = {
        'PedalMe': PedalMeDatasetLoader,
        'Chickenpox': ChickenpoxDatasetLoader,
        'EnglandCovid': EnglandCovidDatasetLoader,
        'WindmillSmall': WindmillOutputSmallDatasetLoader,
        'WindmillMedium': WindmillOutputMediumDatasetLoader
    }

    if args.dataset in dataset_loader:
        ssl._create_default_https_context = ssl._create_unverified_context
        loader = dataset_loader[args.dataset]()
    else:
        raise ValueError("Invalid dataset specified")

    dataset = loader.get_dataset()
    if args.dataset == 'WindmillSmall' or args.dataset == 'WindmillMedium':
        dataset = dataset[:500]
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

    model = GATTestTemporalModel(in_channels=dataset[0].x.size(1), out_channels=dataset[0].x.size(0), k=args.k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()

    losses = train_model(train_dataset, model, optimizer, loss_func, args.epochs,args.dataset)
    plot_loss(losses)
    true_y, pred_y, avg_loss = test_model(test_dataset, model, loss_func,args.dataset)


if __name__ == '__main__':
    main()
