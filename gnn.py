import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

# Simple convo GNN
class CGNN(nn.Module):
    """Graph convolution module used by the GraphSAGE model with edge features.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(CGNN, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h, e):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        e : Tensor
            The edge features.
        """
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['e'] = e
            g.update_all(message_func=fn.u_mul_e('h', 'e', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)

# Simple GNN model
class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = CGNN(in_feats, h_feats)
        self.conv2 = CGNN(h_feats, num_classes)

    def forward(self, g, in_feat, edge_feat):
        h = self.conv1(g, in_feat, edge_feat)
        h = F.relu(h)
        h = self.conv2(g, h, edge_feat)
        return h

# Testing
if __name__ == '__main__':
    node_feats = torch.rand((5, 300))
    rel_feats = torch.rand((3, 300))
    model = Model(300, 300, 8)
    g = dgl.rand_graph(5, 3)
    output = model(g, node_feats, rel_feats)