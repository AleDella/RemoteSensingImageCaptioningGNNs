import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

# Simple convo GNN
class CustomGNNModule(nn.Module):
    """Graph convolution module used by the GraphSAGE model with edge features.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(CustomGNNModule, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        with g.local_scope():
            g.ndata['h'] = h
            # update_all is a message passing API.
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)

# Simple GNN model
class GNN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GNN, self).__init__()
        self.conv1 = dgl.nn.GINConv(torch.nn.Linear(in_feats, h_feats), 'mean')
        self.conv2 = dgl.nn.GINConv(torch.nn.Linear(h_feats, h_feats), 'mean')

    def forward(self, g, in_feat):
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, in_feat))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return hg



# Testing
if __name__ == '__main__':
    node_feats = torch.rand((5, 300))
    model = Model(300, 300, 8)
    g = dgl.rand_graph(5, 3)
    output = model(g, node_feats)