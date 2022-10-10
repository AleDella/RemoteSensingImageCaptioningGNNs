import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.ops import edge_softmax
from torch.nn.utils.rnn import pack_padded_sequence

# GNN modules ####################################################
class GATLayer(nn.Module):
    '''
    Graph Attention Layer
    '''
    def __init__(self, dim_feat: int, num_heads: int):
        super().__init__()
        self.fc_n = nn.Linear(dim_feat, dim_feat * num_heads, bias=False)
        self.attention_ns = nn.Parameter(torch.FloatTensor(size=(1, num_heads, dim_feat)))
        self.attention_nd = nn.Parameter(torch.FloatTensor(size=(1, num_heads, dim_feat)))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.trans = nn.Linear(dim_feat * num_heads, dim_feat)
        self._dim_feat = dim_feat
        self._num_heads = num_heads
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_n.weight, gain=gain)
        nn.init.xavier_normal_(self.attention_ns, gain=gain)
        nn.init.xavier_normal_(self.attention_nd, gain=gain)

    def forward(self, graph, feat):
        feat = self.fc_n(feat).view(-1, self._num_heads, self._dim_feat)

        att_ns = (feat * self.attention_ns).sum(dim=-1).unsqueeze(-1)
        att_nd = (feat * self.attention_nd).sum(dim=-1).unsqueeze(-1)

        graph.srcdata.update({"ft": feat, "att_ns": att_ns})
        graph.dstdata.update({"att_nd": att_nd})
        graph.apply_edges(fn.u_add_v("att_ns", "att_nd", "att"))
        att = self.leaky_relu(graph.edata.pop("att"))
        graph.edata["a"] = edge_softmax(graph, att)
        graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
        return self.trans(graph.dstdata["ft"].view(-1, self._num_heads * self._dim_feat))

# Simple GNN model
class GNN(nn.Module):
    '''
    Graph neural network class
    '''
    def __init__(self, in_feats, gnn):
        super(GNN, self).__init__()
        self.gnn = gnn
        if gnn == 'gat':
            self.conv1 = GATLayer(in_feats, 8)
            self.conv2 = GATLayer(in_feats, 8)
            self.pooling = dgl.nn.GlobalAttentionPooling(torch.nn.Linear(in_feats, 1))
        elif gnn == 'gcn':
            self.conv1 = dgl.nn.pytorch.conv.GraphConv(in_feats, in_feats)
            self.conv2 = dgl.nn.pytorch.conv.GraphConv(in_feats, in_feats)
            self.pooling = dgl.nn.GlobalAttentionPooling(torch.nn.Linear(in_feats, 8))
        
    def forward(self, g, in_feat):
        # Perform graph convolution and activation function.
        if self.gnn == 'gcn':
            # Add self loops for gcn
            g = dgl.add_self_loop(g)
        h = F.relu(self.conv1(g, in_feat))
        h = F.relu(self.conv2(g, h))
        return self.pooling(g, h)

class MLAPModel(nn.Module):
    def __init__(
            self,
            res: bool,
            vir: bool,
            dim_feat: int,
            depth: int,
            dropout: bool=True,
            parallels: int = 3
    ):
        super().__init__()

        self._dim_feat = dim_feat
        self._depth = depth
        self._res = res
        self._vir = vir
        self._dropout = dropout
        self.parallels = parallels

        self.layers = nn.ModuleList([GATLayer(dim_feat, 3) for _ in range(depth)])
        
        if vir:
            self.vnode_emb = nn.Embedding(1, dim_feat)
            nn.init.constant_(self.vnode_emb.weight.data, 0)

            self.vnode_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim_feat, 2 * dim_feat),
                    nn.BatchNorm1d(2 * dim_feat),
                    nn.ReLU(),
                    nn.Linear(2 * dim_feat, dim_feat),
                    nn.BatchNorm1d(dim_feat),
                    nn.ReLU(),
                )
                for _ in range(depth - 1)
            ])
        self.Tflag=0
        # Default pooling method (Global attention)
        self.poolings = nn.ModuleList(
            [dgl.nn.GlobalAttentionPooling(nn.Sequential(
                nn.Linear(self._dim_feat, 2 * self._dim_feat),
                nn.ReLU(),
                nn.Linear(2 * self._dim_feat, 1),
            )) for _ in range(self._depth)]
        )
        
                

    def forward(self, graph, feat):
        self._graph_embs = []

        if self._vir:
            vnode_emb = self.vnode_emb(torch.zeros(graph.batch_size, dtype=torch.int64).to(feat.device))

        for d in range(self._depth):
            if self._vir:
                feat = feat + vnode_emb[torch.repeat_interleave(graph.batch_num_nodes())]

            feat_in = feat
            feat = self.layers[d](graph, feat)
            if d < self._depth - 1:
                feat = F.relu(feat)
            if self._dropout:
                feat = F.dropout(feat, training=self.training)
            if self._res:
                feat = feat + feat_in

            if self.Tflag:
                feat_in = feat
                feat = self.poolings[d](graph, feat)
                self._graph_embs.append(feat_in + feat)
            else:    
                self._graph_embs.append(self.poolings[d](graph, feat))

            if self._vir and d < self._depth - 1:
                vnode_emb_tmp = dgl.nn.SumPooling()(graph, feat) + vnode_emb
                vnode_emb_tmp = F.dropout(self.vnode_mlp[d](vnode_emb_tmp), training=self.training)
                if self._res:
                    vnode_emb = vnode_emb + vnode_emb_tmp
                else:
                    vnode_emb = vnode_emb_tmp
        
        return self._aggregate()

    def _aggregate(self):
        return torch.stack(self._graph_embs, dim=0).mean(dim=0)

    def get_emb(self, graph, feat):
        out = self.forward(graph, feat)
        self._graph_embs.append(out)
        return torch.stack(self._graph_embs, dim=0)

# Util function
def _encode_seq_to_arr(seq, vocab2idx, max_seq_len) -> torch.Tensor:
    '''
    Function that encodes a sequence in the lstm model (used internally)
    '''
    seq = seq.tolist()
    seq = seq[:max_seq_len] + ["<pad>"] * max(0, max_seq_len - len(seq))
    return torch.tensor([vocab2idx[w] if w in vocab2idx else vocab2idx["<unk>"] for w in seq], dtype=torch.int64)

def encode_seq_to_arr_loss(sequence, vocab2idx, max_seq_len) -> torch.Tensor:
    '''
    Function that encodes a sequence in the model loss(used internally)
    '''
    seq = [seq[:max_seq_len] + ["<pad>"] * max(0, max_seq_len - len(seq)) for seq in sequence]
    return torch.tensor([vocab2idx[w] if w in vocab2idx else vocab2idx["<unk>"] for x in seq for w in x], dtype=torch.int64)

def fixed_seq_to_arr(sequence, vocab2idx, max_seq_len):
    seq = [seq[:max_seq_len] + ["<pad>"] * max(0, max_seq_len - len(seq)) for seq in sequence]
    # print("\nSequence: ", len(seq), len(seq[0]), len(seq[0][0]))
    res = torch.tensor([[vocab2idx[x] if x in vocab2idx else vocab2idx["<unk>"] for x in s ] for s in seq], dtype=torch.int64)
    # print(res.shape)
    return res

# Decoders ###########################################################à
class LSTMDecoder(nn.Module):
    '''
    LSTM decoder for graph features
    '''
    def __init__(self, dim_feat, max_seq_len, vocab2idx):
        super().__init__()
        self._max_seq_len = max_seq_len
        self._vocab2idx = vocab2idx

        self.lstm = nn.LSTMCell(dim_feat, dim_feat)
        self.w_hc = nn.Linear(dim_feat * 2, dim_feat)
        self.layernorm = nn.LayerNorm(dim_feat)
        self.vocab_encoder = nn.Embedding(len(vocab2idx), dim_feat, padding_idx=0)
        self.vocab_bias = nn.Parameter(torch.zeros(len(vocab2idx)))

    def forward(self, graph: dgl.DGLGraph, feats: torch.Tensor, labels: any, training: bool):
        self.training = training
        # if self.training:
        #     # teacher forcing
        #     batched_label = torch.vstack([_encode_seq_to_arr(label, self._vocab2idx, self._max_seq_len - 1) for label in labels])
        #     batched_label = torch.hstack((torch.zeros((graph.batch_size, 1), dtype=torch.int64), batched_label))
        #     true_emb = self.vocab_encoder(batched_label.to(device=graph.device))
        feats = feats.unsqueeze(0)
        # h_t, c_t = feats.clone(), feats.clone()
        h_t, c_t = feats[-1].clone(), feats[-1].clone()
        # Try for batch_first = True
        # h_t, c_t = feats.clone(), feats.clone()
        # print("\nFeats: ", feats.shape)
        feats = feats.transpose(0,1) # (batch_size, L+1, dim_feat)
        # feats = feats.unsqueeze(0)
        out = []
        pred_emb = self.vocab_encoder(torch.zeros((graph.batch_size), dtype=torch.int64, device=graph.device))

        vocab_mat = self.vocab_encoder(torch.arange(len(self._vocab2idx), dtype=torch.int64, device=graph.device))
        # print("Pred Emb: {}\nVocab mat: {}\tHT: {}\t CT: {}\t Feats: {}\n".format(pred_emb.shape, vocab_mat.shape, h_t.shape, c_t.shape, feats.shape))
        # if self.training:
        #     max = true_emb.size(1)
        # else:
        #     max = pred_emb.size(1)
        # for i in range(max-1):
        # print("Pred_emb: ", pred_emb.shape)
        
        for i in range(self._max_seq_len):
            _in = pred_emb
            # print("Max seq len: ", self._max_seq_len)
            # print(true_emb.shape)
            # if self.training:
            #     _in = true_emb[:, i]
            # else:
            #    _in = pred_emb
            h_t, c_t = self.lstm(_in, (h_t, c_t))
            # a = F.softmax(torch.bmm(feats.unsqueeze(-1), h_t.unsqueeze(1)), dim=1)  # (batch_size, L + 1)
            # context = torch.bmm(a, feats.unsqueeze(-1)).squeeze(1)
            # pred_emb = torch.tanh(self.layernorm(self.w_hc(torch.hstack((h_t, context.squeeze(-1))))))  # (batch_size, dim_feat)
            # print("Feats size: {}\tHT: {}\n".format(feats.shape, h_t.shape))
            # a = torch.bmm(feats, h_t.unsqueeze(-1))
            a = torch.bmm(feats, h_t.unsqueeze(-1))
            # print(a.shape)
            a = F.softmax(a.squeeze(-1), dim=1)# (batch_size, L + 1)
            # print(a.shape)
            # context = torch.bmm(a, feats).squeeze(0)
            context = torch.bmm(a.unsqueeze(1), feats).squeeze(1)
            # print(context.shape)
            # Try to fix spaghetti
            # context = context.transpose(0,1)
            # print(h_t.shape)
            pred_emb = torch.hstack((h_t, context))
            # print("Pred: ", pred_emb.shape)
            pred_emb = self.w_hc(pred_emb)
            
            # print("Pred: ", pred_emb.shape)
            pred_emb = self.layernorm(pred_emb)
            # print("Pred: ", pred_emb.shape)
            pred_emb = torch.tanh(pred_emb)
            # print("Pred: ", pred_emb.shape)
            
            # print("Output: ", pred_emb.shape)
            # exit(0)
            out.append(torch.matmul(pred_emb, vocab_mat.T) + self.vocab_bias.unsqueeze(0))
        
        return out


class decoderRNN(nn.Module):
    '''
    RNN decoder for graph features
    '''
    def __init__(self, embed_size,vocab_size, hidden_size, num_layers, max_len):
        super(decoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax()
        self.max_len = max_len
    
    def forward(self, features, encoded_captions, lengths):
        # Feature size: [batch_size, BERT_feat_dim*2]
        # Produce the initial embeddings with <sos>
        # embeddings = self.embedding(torch.ones((features.shape[0], self.max_len), dtype=torch.int64, device=features.device)) # (batch_size, max_len, BERT_feat_dim*2)
        embeddings = self.embedding(encoded_captions.to(features.device))
        packed_embs = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed_embs, (features.clone().unsqueeze(0), features.clone().unsqueeze(0)))
        outputs = self.linear(hiddens[0]) # (batch_size, max_len, num_vocabs)
        return outputs

# Testing
if __name__ == '__main__':
    node_feats = torch.rand((5, 300))
    model = GNN(300, 300)
    g = dgl.rand_graph(5, 3)
    output = model(g, node_feats)