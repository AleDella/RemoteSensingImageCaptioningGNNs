from turtle import forward
import torch
import torch.nn as nn
from torchvision.models import resnet152
from gnn import GNN, LSTMDecoder, _encode_seq_to_arr, decoderRNN, MLAPModel

class TripletClassifier(nn.Module):
    '''
    Model which takes as input an image and predict the corresponding triplets that are in that image. 
    It will be based on resnet-152 for the extraction of the features, so it will be a finetuning on the target dataset.
    '''
    def __init__(self, input_size, num_classes):
        super(TripletClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = resnet152(weights="IMAGENET1K_V2")
        # Replace the last layer with a new layer for classification
        self.model.fc = nn.Linear(in_features=2048,out_features=num_classes)
        
        # Freeze all the layers except the fully connected
        for name, parameter in self.model.named_parameters():
            if(not 'fc' in name):
                parameter.requires_grad = False
        
        # Just for testing 
        # for name, parameter in self.model.named_parameters():
        #     print(name,parameter.requires_grad)
    
    def forward(self, x):
        '''
        x -> shape (batch_size, channels, width, height)
        '''
        assert x.shape[2]==self.input_size
        assert x.shape[3]==self.input_size
        
        return self.model(x)

def load_model(path):
    return torch.load(path)
    
class CaptionGenerator(nn.Module):
    '''
    Caption generation network (encoder-decoder)

    Args:
        feats_dim: dimension of the features
        max_seq_len: maximum tokens in a caption
        vocab2idx: dictionary for one hot encoding of tokens
        decoder: type of decoder (linear, lstm or rnn)
    '''
    def __init__(self, feats_dim, max_seq_len, vocab2idx, gnn='gat', vir=True, depth=1, decoder='lstm') -> None:
        super(CaptionGenerator, self).__init__()
        if gnn == 'gat':
            self.encoder = GNN(feats_dim)
        elif gnn == 'mlap':
            self.encoder = MLAPModel(True, vir, feats_dim, depth)
        self.decoder_type = decoder
        if self.decoder_type == 'linear':
            self.decoder = nn.ModuleList([nn.Linear(feats_dim, len(vocab2idx)) for _ in range(max_seq_len)])
        if self.decoder_type == 'lstm':
            self.decoder = LSTMDecoder(feats_dim, max_seq_len, vocab2idx)
        if self.decoder_type == 'rnn':
            self.decoder = decoderRNN(feats_dim, len(vocab2idx), feats_dim, 3)
        self.dropout = nn.Dropout(p=0.3)
        self.vocab2idx = vocab2idx
        self.idx2vocab = {v: k for k, v in vocab2idx.items()}

    def forward(self, g, feats, labels):
        graph_feats = self.dropout(self.encoder(g, feats))
        
        if self.decoder_type == 'linear':
            decoded_out = [d(graph_feats) for d in self.decoder]
        if self.decoder_type == 'lstm':
            decoded_out = self.decoder(g, graph_feats, labels)
        if self.decoder_type == 'rnn':
            decoded_out = self.decoder(graph_feats, labels)
        return decoded_out

    def _loss(self, out, labels, vocab2idx, max_seq_len, device) -> torch.Tensor:
        batched_label = torch.vstack([_encode_seq_to_arr(label, vocab2idx, max_seq_len) for label in labels])
        return sum([nn.CrossEntropyLoss()(out[i], batched_label[:, i].to(device=device)) for i in range(max_seq_len)])/max_seq_len



class ImprovedCaptionGenerator(nn.Module):
    '''
    Caption generation network (encoder-decoder)

    Args:
        feats_dim: dimension of the features
        max_seq_len: maximum tokens in a caption
        vocab2idx: dictionary for one hot encoding of tokens
        decoder: type of decoder (linear, lstm or rnn)
    '''
    def __init__(self, img_encoder, feats_dim, max_seq_len, vocab2idx, gnn='gat', vir=True, depth=1, decoder='lstm') -> None:
        super(ImprovedCaptionGenerator, self).__init__()
        if gnn == 'gat':
            self.encoder = GNN(feats_dim)
        elif gnn == 'mlap':
            self.encoder = MLAPModel(True, vir, feats_dim, depth)
        
        # Incorporate image in the pipeline
        self.img_encoder = img_encoder
        self.img_encoder.model.fc = nn.Linear(2048, feats_dim)
        # Freeze all the layers except the fully connected
        for name, parameter in self.img_encoder.named_parameters():
            if(not 'fc' in name):
                parameter.requires_grad = False

        # Initialize the weight at a random value
        self.img_weight = torch.nn.parameter.Parameter(torch.randn(1, requires_grad=True))
        
        self.decoder_type = decoder
        if self.decoder_type == 'linear':
            self.decoder = nn.ModuleList([nn.Linear(feats_dim, len(vocab2idx)) for _ in range(max_seq_len)])
        if self.decoder_type == 'lstm':
            self.decoder = LSTMDecoder(feats_dim, max_seq_len, vocab2idx)
        if self.decoder_type == 'rnn':
            self.decoder = decoderRNN(feats_dim, len(vocab2idx), feats_dim, 3)
        self.dropout = nn.Dropout(p=0.3)
        self.vocab2idx = vocab2idx
        self.idx2vocab = {v: k for k, v in vocab2idx.items()}

    def forward(self, g, g_feats, img, labels):
        i_feats = self.img_encoder(img)
        
        graph_feats = self.dropout(self.encoder(g, g_feats))
        # Weighted sum 
        mod_feats = graph_feats + (i_feats * self.img_weight)
        
        if self.decoder_type == 'linear':
            decoded_out = [d(mod_feats) for d in self.decoder]
        if self.decoder_type == 'lstm':
            decoded_out = self.decoder(g, mod_feats, labels)
        if self.decoder_type == 'rnn':
            decoded_out = self.decoder(mod_feats, labels)
        return decoded_out

    def _loss(self, out, labels, vocab2idx, max_seq_len, device) -> torch.Tensor:
        batched_label = torch.vstack([_encode_seq_to_arr(label, vocab2idx, max_seq_len) for label in labels])
        return sum([nn.CrossEntropyLoss()(out[i], batched_label[:, i].to(device=device)) for i in range(max_seq_len)])/max_seq_len




if __name__=="__main__":
    model = TripletClassifier(224,10)
    dummy_img = torch.randn((5,3,224,224))
    out = model(dummy_img)