import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, DynamicEdgeConv, MLP

def collate_fn_gnn(batch):
    """
    Custom function that defines how batches are formed.

    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list = []
    labels = []

    for b in batch:
        # loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the feautures are (time, x, y)
        tensordata = torch.from_numpy(b["data"].to_numpy()).T
        # single precision (float32) to save memory and computation time
        tensordata = tensordata.to(dtype=torch.float32)

        # Geometric Data object for each event
        this_graph_item = Data(x=tensordata)
        data_list.append(this_graph_item)

        # labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    labels = torch.cat(labels, dim=0) # convert the list of tensors to a single tensor
    packed_data = Batch.from_data_list(data_list) # convert the list of Data objects to a single Batch object
    return packed_data, labels


class GNNEncoder(nn.Module):
    def __init__(self, 
                in_channels: int = 3,
                hidden_channels: int = 64, # hidden layer dimension of the MLP
                out_channels: int = 2, # 2 floats (xpos, ypos)
                k: int = 12, # number of nearest neighbors to consider
                n_layers: int = 4 # number of DynamicEdgeConv layers to use
                ):
    
        super(GNNEncoder, self).__init__()
        
        # list of layers
        self.layer_list = nn.ModuleList()

        # add intermediate layers
        for i in range(n_layers):
            dim_in = 2 * (in_channels if i == 0 else hidden_channels) # input dimension is twice the number of features
            self.layer_list.append(
                DynamicEdgeConv(
                        MLP([dim_in, hidden_channels, hidden_channels], batch_norm=True),
                        aggr='mean', k=k
                )
            )

        # output layer
        self.final_mlp = MLP([hidden_channels, hidden_channels//2, out_channels])
    
    def forward(self, data):
        # data is a batch graph item. it contains a list of tensors (x) and how the batch is structured along this list (batch)
        x = data.x
        batch = data.batch

        # loop over the DynamicEdgeConv layers:
        for layer in self.layer_list:
            x = layer(x, batch) 

        # combine all node feauters into single prediction
        x = global_mean_pool(x, batch) # -> (n_batch, hidden_channels)
        # map to two output labels (xpos, ypos)
        x = self.final_mlp(x) # -> (n_batch, out_channels)

        return x