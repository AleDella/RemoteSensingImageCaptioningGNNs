from turtle import forward
import torch
import torch.nn as nn
from torchvision.models import resnet152
from gnn import GNN, LSTMDecoder, _encode_seq_to_arr

class TripletClassifier(nn.Module):
    '''
    Model which takes as input an image and predict the corresponding triplets that are in that image. 
    It will be based on resnet-152 for the extraction of the features, so it will be a finetuning on the target dataset.
    '''
    def __init__(self, input_size, num_classes):
        super(TripletClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = resnet152(pretrained=True)
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
    '''
    def __init__(self, feats_dim, max_seq_len, vocab2idx) -> None:
        super(CaptionGenerator, self).__init__()
        self.encoder = GNN(feats_dim, feats_dim)
        self.decoder = LSTMDecoder(feats_dim, max_seq_len, vocab2idx)
        self.max_seq_len = max_seq_len
        self.vocab2idx = vocab2idx
        self.idx2vocab = {v: k for k, v in vocab2idx.items()}

    def forward(self, g, feats, labels):
        graph_feats = self.encoder(g, feats)
        decoded_out = self.decoder(g, graph_feats, labels)
        return decoded_out

    def _loss(self, out: torch.Tensor, labels: list[list[str]], vocab2idx, max_seq_len, device) -> torch.Tensor:
        batched_label = torch.vstack([_encode_seq_to_arr(label, vocab2idx, max_seq_len) for label in labels])
        return sum([nn.CrossEntropyLoss()(out[i], batched_label[:, i].to(device=device)) for i in range(max_seq_len)])/max_seq_len

if __name__=="__main__":
    model = TripletClassifier(224,10)
    dummy_img = torch.randn((5,3,224,224))
    out = model(dummy_img)