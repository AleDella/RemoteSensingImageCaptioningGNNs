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


##################################
# Taken from MLAP code (STILL TO ADAPT)
def _encode_seq_to_arr(seq: list[str], vocab2idx: dict[str, int], max_seq_len: int) -> torch.Tensor:
    seq = seq[:max_seq_len] + ["__EOS__"] * max(0, max_seq_len - len(seq))
    return torch.tensor([vocab2idx[w] if w in vocab2idx else vocab2idx["__UNK__"] for w in seq], dtype=torch.int64)

class LSTMDecoder(nn.Module):
    def __init__(self, dim_feat: int, max_seq_len: int, vocab2idx: dict[str, int]):
        super().__init__()
        self._max_seq_len = max_seq_len
        self._vocab2idx = vocab2idx

        self.lstm = nn.LSTMCell(dim_feat, dim_feat)
        self.w_hc = nn.Linear(dim_feat * 2, dim_feat)
        self.layernorm = nn.LayerNorm(dim_feat)
        self.vocab_encoder = nn.Embedding(len(vocab2idx), dim_feat)
        self.vocab_bias = nn.Parameter(torch.zeros(len(vocab2idx)))

    def forward(self, graph: dgl.DGLGraph, feats: torch.Tensor, labels: any):
        if self.training:
            # teacher forcing
            batched_label = torch.vstack([_encode_seq_to_arr(label, self._vocab2idx, self._max_seq_len - 1) for label in labels])
            batched_label = torch.hstack((torch.zeros((graph.batch_size, 1), dtype=torch.int64), batched_label))
            true_emb = self.vocab_encoder(batched_label.to(device=graph.device))
        # h_t, c_t = feats[-1].clone(), feats[-1].clone()
        h_t, c_t = feats.clone(), feats.clone()
        # feats = feats.transpose(0, 1)  # (batch_size, L + 1, dim_feat)
        out = []
        pred_emb = self.vocab_encoder(torch.zeros((graph.batch_size), dtype=torch.int64, device=graph.device))

        vocab_mat = self.vocab_encoder(torch.arange(len(self._vocab2idx), dtype=torch.int64, device=graph.device))

        for i in range(self._max_seq_len):
            if self.training:
                _in = true_emb[:, i]
            else:
                _in = pred_emb
            h_t, c_t = self.lstm(_in, (h_t, c_t))
            a = F.softmax(torch.bmm(feats.unsqueeze(-1), h_t.unsqueeze(0)).squeeze(-1), dim=1)  # (batch_size, L + 1)
            context = torch.bmm(a, feats.unsqueeze(-1)).squeeze(1)
            # print("HT: {}\tContext: {}\n".format(h_t.shape, context.shape))
            pred_emb = torch.tanh(self.layernorm(self.w_hc(torch.hstack((h_t, context.squeeze(-1))))))  # (batch_size, dim_feat)

            out.append(torch.matmul(pred_emb, vocab_mat.T) + self.vocab_bias.unsqueeze(0))

        return out
###########################

# Testing
if __name__ == '__main__':
    node_feats = torch.rand((5, 300))
    model = GNN(300, 300)
    g = dgl.rand_graph(5, 3)
    output = model(g, node_feats)