from cProfile import label
from turtle import forward
import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights, VGG16_Weights, vgg16
from torch.nn.utils.rnn import pack_padded_sequence
from gnn import GNN, LSTMDecoder, encode_seq_to_arr_loss, decoderRNN, MLAPModel, fixed_seq_to_arr
from transformers import BertModel, BertTokenizer
from graph_utils import tripl2graph

class TripletClassifier(nn.Module):
    '''
    Model which takes as input an image and predict the corresponding tripts that are in that image. 
    It will be based on resnet-152 for the extraction of the features, so it will be a finetuning on the target dataset.
    '''
    def __init__(self, input_size, num_classes, pil=False):
        super(TripletClassifier, self).__init__()
        self.pil = pil
        self.input_size = input_size
        self.num_classes = num_classes
        weights = ResNet152_Weights.DEFAULT
        self.model = resnet152(weights=weights)
        self.preprocess = weights.transforms()
        # Replace the last layer with a new layer for classification
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=2048,out_features=1024),
            nn.Dropout(0.3),
            nn.Linear(in_features=1024,out_features=num_classes)
            )
        
        # Freeze all the layers except the fully connected
        for name, parameter in self.model.named_parameters():
            if(not 'fc' in name):
                parameter.requires_grad = False
    
    def forward(self, x):
        '''
        x -> shape (batch_size, channels, width, height)
        '''
        assert x.shape[2]==self.input_size
        assert x.shape[3]==self.input_size
        
        return self.model(self.preprocess(x))

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
    def __init__(self, feats_dim, max_seq_len, vocab2idx, gnn='gat', vir=False, res=False, depth=1, decoder='lstm') -> None:
        super(CaptionGenerator, self).__init__()
        self.max_seq_len = max_seq_len
        if gnn == 'gat' or gnn == 'gcn':
            self.encoder = GNN(feats_dim, gnn)
        elif gnn == 'mlap':
            self.encoder = MLAPModel(res, vir, feats_dim, depth)
        self.decoder_type = decoder
        if self.decoder_type == 'linear':
            self.decoder = nn.ModuleList([nn.Linear(feats_dim, len(vocab2idx)) for _ in range(max_seq_len)])
        if self.decoder_type == 'lstm':
            self.decoder = LSTMDecoder(feats_dim, max_seq_len, vocab2idx)
        if self.decoder_type == 'rnn':
            self.decoder = decoderRNN(feats_dim, vocab2idx, feats_dim, 1, max_seq_len)
        self.dropout = nn.Dropout(p=0.3)
        self.vocab2idx = vocab2idx
        self.idx2vocab = {v: k for k, v in vocab2idx.items()}

    def forward(self, g, feats, labels, lengths, training):
        graph_feats = self.dropout(self.encoder(g, feats))
        
        if self.decoder_type == 'linear':
            decoded_out = [d(graph_feats) for d in self.decoder]
        if self.decoder_type == 'lstm':
            decoded_out = self.decoder(g, graph_feats, labels, training)
        if self.decoder_type == 'rnn':
            decoded_out = self.decoder(graph_feats, labels, lengths)
        return decoded_out

    def _loss(self, out, labels, lenghts, vocab2idx, max_seq_len, device) -> torch.Tensor:
        if self.decoder_type == 'lstm' or self.decoder_type == 'rnn':
            new_labels = [label[1:] for label in labels]
            # batched_label = torch.vstack([encode_seq_to_arr_loss(label, vocab2idx, max_seq_len) for label in new_labels])
            batched_label = fixed_seq_to_arr(new_labels, vocab2idx, max_seq_len)# (batch_size, num_captions, tokens)
            targets = pack_padded_sequence(batched_label, lenghts, batch_first=True)[0]
            # batched_label = batched_label.transpose(0,1) # (num_captions, batch_size, tokens)
            # batched_label = batched_label.flatten(1,2)# (num_captions, batch_szie*tokens)
            batched_label = batched_label.flatten(0,1)
            #out = out.flatten(0, 1)# (batch_szie*tokens, vocab_len)
            
            # c_loss = 0.0
            # for gt in batched_label:
            #     c_loss += nn.CrossEntropyLoss(ignore_index=vocab2idx['<pad>'])(out, gt.to(device=device))
            c_loss = nn.CrossEntropyLoss(ignore_index=vocab2idx['<pad>'])(out, targets.to(device=device))
            # return c_loss/batched_label.shape[0]# Mean of the losses for each of the 5 captions
            return c_loss
            
        else:
            batched_label = torch.vstack([encode_seq_to_arr_loss(label, vocab2idx, max_seq_len) for label in labels])
            return sum([nn.CrossEntropyLoss(ignore_index=vocab2idx['<pad>'])(out[i], batched_label[:, i].to(device=device)) for i in range(max_seq_len)])/max_seq_len
    
    def sample(self, graph, graph_feats):
        self.eval()
        with torch.no_grad():
            graph_feats = self.dropout(self.encoder(graph, graph_feats))
            # Mod feats with concatenation
            caption = self.decoder.sample(graph_feats)
            return caption



class AugmentedCaptionGenerator(nn.Module):
    '''
    Caption generation network (encoder-decoder)

    Args:
        feats_dim: dimension of the features
        max_seq_len: maximum tokens in a caption
        vocab2idx: dictionary for one hot encoding of tokens
        decoder: type of decoder (linear, lstm or rnn)
    '''
    def __init__(self, img_encoder, feats_dim, max_seq_len, vocab2idx, gnn='gat', res=True, vir=True, depth=1, decoder='lstm') -> None:
        super(AugmentedCaptionGenerator, self).__init__()
        if gnn == 'gat' or gnn == 'gcn':
            self.encoder = GNN(feats_dim, gnn)
        elif gnn == 'mlap':
            self.encoder = MLAPModel(res, vir, feats_dim, depth)
        
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

    def forward(self, g, g_feats, img, labels=None):
        i_feats = self.img_encoder(img)
        
        graph_feats = self.dropout(self.encoder(g, g_feats))
        mod_feats = graph_feats + (i_feats * self.img_weight)
        
        if self.decoder_type == 'linear':
            decoded_out = [d(mod_feats) for d in self.decoder]
        if self.decoder_type == 'lstm':
            decoded_out = self.decoder(g, mod_feats, labels)
        if self.decoder_type == 'rnn':
            decoded_out = self.decoder(mod_feats, labels)
        return decoded_out

    def _loss(self, out, labels, vocab2idx, max_seq_len, device) -> torch.Tensor:
        batched_label = torch.vstack([encode_seq_to_arr_loss(label, vocab2idx, max_seq_len) for label in labels])
        return sum([nn.CrossEntropyLoss()(out[i], batched_label[:, i].to(device=device)) for i in range(max_seq_len)])/max_seq_len
    


class MultiHead(torch.nn.Module):
    '''
    Class for the multihead classifier for the triplet prediction
    
    Args:
        backbone (torch.nn.Module): backbone for the images
        heads List[torch.nn.Module]: list of heads for the tasks
    '''
    def __init__(self, backbone, heads):
        super().__init__()
        self.backbone = backbone
        # Initializing all the heads as part of a ModuleList
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, x):
        common_features = self.backbone(x)  # compute the shared features
        outputs = [head(common_features) for head in self.heads]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class MultiHeadClassifier(nn.Module):
    
    def __init__(self, input_size, dict_size, pil=False):
        super(MultiHeadClassifier, self).__init__()
        self.pil=pil
        self.input_size = input_size
        weights = ResNet152_Weights.DEFAULT
        self.backbone = resnet152(weights=weights)
        self.preprocess = weights.transforms()
        classifiers = [torch.nn.Linear(2048, 2) for _ in range(dict_size)]
        self.backbone.fc = MultiHead(torch.nn.Identity(), classifiers)
        # Freeze all the layers except the fully connected
        for name, parameter in self.backbone.named_parameters():
            if(not 'fc' in name):
                parameter.requires_grad = False
        
    def forward(self, img):
        
        assert img.shape[2]==self.input_size
        assert img.shape[3]==self.input_size
        
        features = self.backbone(self.preprocess(img))
        
        return features
    
class FinalModel(nn.Module):
    '''
    Caption generation network (encoder-decoder)

    Args:
        feats_dim: dimension of the features
        max_seq_len: maximum tokens in a caption
        vocab2idx: dictionary for one hot encoding of tokens
        decoder: type of decoder (linear, lstm or rnn)
    '''
    def __init__(self, img_encoder, feats_dim, max_seq_len, vocab2idx, img_dim, tripl2idx, gnn='gat', res=False, vir=True, depth=1, decoder='lstm', pil=False) -> None:
        super(FinalModel, self).__init__()
        self.max_seq_len = max_seq_len
        if gnn == 'gat' or gnn == 'gcn':
            self.graph_encoder = GNN(feats_dim, gnn)
        elif gnn == 'mlap':
            self.graph_encoder = MLAPModel(res, vir, feats_dim, depth)
        self.tripl_classifier = MultiHeadClassifier(img_dim, len(tripl2idx),pil)
        # self.tripl_classifier = TripletClassifier(img_dim, len(tripl2idx), pil)
        self.sigmoid = nn.Sigmoid()
        self.idx2tripl = {v: k for k, v in tripl2idx.items()}
        self.feature_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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
            # self.decoder = nn.ModuleList([nn.Linear(feats_dim, len(vocab2idx)) for _ in range(max_seq_len)])
            # For modified concatenation
            self.decoder = nn.ModuleList([nn.Linear(feats_dim*2, len(vocab2idx)) for _ in range(self.max_seq_len)])
        if self.decoder_type == 'lstm':
            self.decoder = LSTMDecoder(feats_dim*2, self.max_seq_len, vocab2idx)
        if self.decoder_type == 'rnn':
            self.decoder = decoderRNN(feats_dim*2, vocab2idx, feats_dim*2, 1, self.max_seq_len)
        self.dropout = nn.Dropout(p=0.3)
        self.vocab2idx = vocab2idx
        self.idx2vocab = {v: k for k, v in vocab2idx.items()}

    def forward(self, img, captions, labels=None, lengths=None, training=False):
        # Triplet classification
        triplets = self.sigmoid(self.tripl_classifier(img))
        # For normal classifier
        # class_out = triplets
        # # For multihead classifier
        triplets = triplets.reshape((triplets.shape[0], int(triplets.shape[1]/2), 2))
        class_out = triplets
        triplets = [[torch.argmax(logits).item() for logits in img] for img in triplets]
        # # Changed for BCE loss
        # Extract indeces greater or equal than the threshold
        threshold = 0.5
        indeces = [[ i for i, d in enumerate(s) if d >= threshold] for s in triplets ]
        # Extract the triplets
        triplets = [[self.idx2tripl[i] for i in s] for s in indeces]
        # Add "proxy" triplets due to the fact that the network can't process void triplets
        for s in triplets:
            if s == []:
                s.append("('There', 'is', 'no triplet')")
        
        # Retrieve the graph and graph features
        graph, graph_feats = tripl2graph(triplets, self.feature_encoder, self.tokenizer)
        i_feats = self.img_encoder(img)
        graph, graph_feats = graph.to(img.device), graph_feats.to(img.device)
        graph_feats = self.dropout(self.graph_encoder(graph, graph_feats))
        # Mod feats with concatenation
        mod_feats = torch.cat([graph_feats, i_feats], dim=1)
        # Mod feats with weighted sum and main graph
        # mod_feats = graph_feats + ( i_feats * self.img_weight)
        # Mod feats with weighted sum and main Image
        # mod_feats = i_feats + ( graph_feats * self.img_weight)
        if self.decoder_type == 'linear':
            decoded_out = [d(mod_feats) for d in self.decoder]
        # Need to solve the problem with lstm and rnn for the labels
        if self.decoder_type == 'lstm':
            decoded_out = self.decoder(graph, mod_feats, labels, training)
        if self.decoder_type == 'rnn':
            decoded_out = self.decoder(mod_feats, labels, lengths)
        
        return decoded_out, class_out
    
    def sample(self, img):
        self.eval()
        with torch.no_grad():
            triplets = self.sigmoid(self.tripl_classifier(img))
            triplets = triplets.reshape((triplets.shape[0], int(triplets.shape[1]/2), 2))
            triplets = [[torch.argmax(logits).item() for logits in img] for img in triplets]
            # # Changed for BCE loss
            # Extract indeces greater or equal than the threshold
            threshold = 0.5
            indeces = [[ i for i, d in enumerate(s) if d >= threshold] for s in triplets ]
            # Extract the triplets
            triplets = [[self.idx2tripl[i] for i in s] for s in indeces]
            # Add "proxy" triplets due to the fact that the network can't process void triplets
            for s in triplets:
                if s == []:
                    s.append("('There', 'is', 'no triplet')")
            # Retrieve the graph and graph features
            graph, graph_feats = tripl2graph(triplets, self.feature_encoder, self.tokenizer)
            i_feats = self.img_encoder(img)
            graph, graph_feats = graph.to(img.device), graph_feats.to(img.device)
            graph_feats = self.dropout(self.graph_encoder(graph, graph_feats))
            # Mod feats with concatenation
            mod_feats = torch.cat([graph_feats, i_feats], dim=1)
            caption = self.decoder.sample(mod_feats)
            return caption
        

    def _loss(self, out, labels, lenghts, vocab2idx, max_seq_len, device) -> torch.Tensor:
        
        if self.decoder_type == 'lstm' or self.decoder_type == 'rnn':
            new_labels = [label[1:] for label in labels]
            # batched_label = torch.vstack([encode_seq_to_arr_loss(label, vocab2idx, max_seq_len) for label in new_labels])
            batched_label = fixed_seq_to_arr(new_labels, vocab2idx, max_seq_len)# (batch_size, num_captions, tokens)
            targets = pack_padded_sequence(batched_label, lenghts, batch_first=True)[0]
            # batched_label = batched_label.transpose(0,1) # (num_captions, batch_size, tokens)
            # batched_label = batched_label.flatten(1,2)# (num_captions, batch_szie*tokens)
            batched_label = batched_label.flatten(0,1)
            #out = out.flatten(0, 1)# (batch_szie*tokens, vocab_len)
            
            # c_loss = 0.0
            # for gt in batched_label:
            #     c_loss += nn.CrossEntropyLoss(ignore_index=vocab2idx['<pad>'])(out, gt.to(device=device))
            c_loss = nn.CrossEntropyLoss(ignore_index=vocab2idx['<pad>'])(out, targets.to(device=device))
            # return c_loss/batched_label.shape[0]# Mean of the losses for each of the 5 captions
            return c_loss
            
        else:
            batched_label = torch.vstack([encode_seq_to_arr_loss(label, vocab2idx, max_seq_len) for label in labels])
            return sum([nn.CrossEntropyLoss(ignore_index=vocab2idx['<pad>'])(out[i], batched_label[:, i].to(device=device)) for i in range(max_seq_len)])/max_seq_len
        
        print("\n",batched_label.shape)
        print(out.shape)
        
        print(batched_label.shape)
        print(out.shape)
        exit(0)
        # return sum([nn.CrossEntropyLoss(ignore_index=vocab2idx['<pad>'])(out[i], batched_label[:, i].to(device=device)) for i in range(max_seq_len)])/max_seq_len
        # batched_label = batched_label.flatten()
    


class FinetunedModel(nn.Module):
    '''
    Caption generation network (encoder-decoder)

    Args:
        feats_dim: dimension of the features
        max_seq_len: maximum tokens in a caption
        vocab2idx: dictionary for one hot encoding of tokens
        decoder: type of decoder (linear, lstm or rnn)
    '''
    def __init__(self, vocab2idx, img_dim, tripl2idx, decoder) -> None:
        super(FinetunedModel, self).__init__()
        self.decoder = torch.load(decoder)
        self.tripl_classifier = MultiHeadClassifier(img_dim, len(tripl2idx))
        self.idx2tripl = {v: k for k, v in tripl2idx.items()}
        self.feature_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vocab2idx = vocab2idx
        self.idx2vocab = {v: k for k, v in vocab2idx.items()}

    def forward(self, img):
        # Triplet classification
        triplets = self.tripl_classifier(img)
        # For multihead classifier
        triplets = triplets.reshape((triplets.shape[0], int(triplets.shape[1]/2), 2))
        class_out = triplets
        triplets = [[torch.argmax(logits).item() for logits in img] for img in triplets]
        # Extract indeces greater or equal than the threshold
        threshold = 0.5
        indeces = [[ i for i, d in enumerate(s) if d >= threshold] for s in triplets ]
        # Extract the triplets
        triplets = [[self.idx2tripl[i] for i in s] for s in indeces]
        # Add "proxy" triplets due to the fact that the network can't process void triplets
        for s in triplets:
            if s == []:
                s.append("('There', 'is', 'no triplet')")
        
        # Retrieve the graph and graph features
        graph, graph_feats = tripl2graph(triplets, self.feature_encoder, self.tokenizer)
        graph, graph_feats = graph.to(img.device), graph_feats.to(img.device)
        
        
        
        decoded_out = self.decoder(graph, graph_feats, img)
        
        
        return decoded_out, class_out

    def _loss(self, out, labels, vocab2idx, max_seq_len, device) -> torch.Tensor:
        batched_label = torch.vstack([encode_seq_to_arr_loss(label, vocab2idx, max_seq_len) for label in labels])
        return sum([nn.CrossEntropyLoss()(out[i], batched_label[:, i].to(device=device)) for i in range(max_seq_len)])/max_seq_len


# Normal caption generator implemented by Riccardo
class TextGenerator(nn.ModuleList):
    '''
    This model is a caption generator for the UCM dataset.
    '''
    def __init__(self, vocab_size, hidden_dim, type='gru', backbone = 'resnet152', pretrained_back = True, trainable=True):
        super(TextGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        
        if(backbone=='resnet152'):
            self.weights = ResNet152_Weights.DEFAULT
            self.preprocess = self.weights.transforms()
            
            if(pretrained_back):
                self.backbone = resnet152(weights=self.weights)
            else:
                self.backbone = resnet152()
        elif(backbone=='vgg16'):
            self.weights = VGG16_Weights.DEFAULT
            self.preprocess = self.weights.transforms()
            
            if(pretrained_back):
                self.backbone = vgg16(weights=self.weights)
            else:
                self.backbone = vgg16()
        else:
            raise RuntimeError('Backbone not found!')
        
        if not trainable: 
            for _ , parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)
        
        if (backbone=='resnet152'):   
            img_feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif(backbone=='vgg16'):
            img_feat_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Sequential(self.backbone.classifier[:-1])
                
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim,padding_idx=0) 
        self.dropout = nn.Dropout(0.5)
        # LSTM
        if(type=='gru'):
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, num_layers=1) 
        elif(type=='lstm'):
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=1) 
        else:
            print('Select RNN type')
            
        # Feature processing
        self.process_features = nn.Sequential(
            nn.Linear(img_feat_dim, self.hidden_dim),
        )
        # Linear layer
        self.linear1 = nn.Linear(hidden_dim, vocab_size)
		
    def forward(self, x, img, lengths):
        # Process the image
        img = self.preprocess(img)
        img_feat = self.backbone(img)
        img_feat = self.process_features(img_feat)
        
        # From idx to embedding
        x = x.long()
        x = self.dropout(self.embedding(x))

        embeddings = torch.cat((img_feat.unsqueeze(1), x), 1)
        
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        out,_ = self.rnn(packed)
        
        out = self.linear1(out[0])
        
        return out
    
    def sample(self, img, max_seq_len, endseq_index, k, device):
        """Generate captions for given image features using beam search."""
        # The algorithm must implement beam search to generate captions 
        # Beam search works like that
        # At the first iteration all the k prediction starts with the most probable word (usually start of seq!)
        # Other iterations build up on the same start of sequence, but the next tokens are sampled in a mutually exclusive way, so that different captions are created. 
        # For each caption, the k most probable words are selected. Then based on the sum of the probabilities up until that point, k sentences are kept, and so on until all reach end of sequence. 

        with torch.no_grad():
            sampled_ids = torch.zeros((k,max_seq_len+1),dtype=torch.long).to(device) # To store the sampled ids
            img = img.to(device)
            img = self.preprocess(img)
            img_feat = self.backbone(img.unsqueeze(0))
            inputs = self.process_features(img_feat)
            
            states = None
            first = True
            sentences = []
            sum_probs_list = []
            for i in range(max_seq_len+1):
                hiddens, states = self.rnn(inputs, states)
                out = self.linear1(hiddens)
                out = out.sort(descending=True)
                probs = out[0].squeeze(1)
                predicted = out[1].squeeze(1)
                if(i==0):
                    # Append the first prediction
                    sampled_ids[:,i] = torch.tile(predicted[:,0],(k,))
                    inputs = self.embedding(sampled_ids[:,i]).unsqueeze(1)
                    states = torch.tile(states,(1,k,1))
                else:
                    probs = torch.softmax(probs,dim=1)
                    if(first):
                        sampled_ids[:,i] = predicted[0,:k]
                        sum_probs = probs[0,:k]
                        first = False
                    else:
                        # CHECK AND REMOVE FINISHED SENTENCES
                        mask = torch.all(sampled_ids!=endseq_index,dim=1)
                        idx_mask = (mask==True).nonzero()
                        
                        for j in range(mask.shape[0]):
                            if(not mask[j]):
                                sentences.append(sampled_ids[j,:].tolist())
                                sum_probs_list.append((sum_probs[j]).item())
                                
                        sampled_ids = sampled_ids[idx_mask,:].squeeze(1)
                        # Check to break the run
                        if(sampled_ids.shape[0]==0):
                            indexsorted = sorted(range(len(sum_probs_list)), key=lambda k: sum_probs_list[k],reverse=True)
                            sentences = [sentences[i] for i in indexsorted]
                            
                            break
                        
                        k = sampled_ids.shape[0]
                        # Get predictions and probabilities
                        predicted = predicted[idx_mask,:k].flatten()
                        probs = probs[idx_mask,:k].flatten()
                        sum_probs = sum_probs[idx_mask].squeeze(1)
                        # Get candidates 
                        candidates = torch.repeat_interleave(sampled_ids,k,dim=0)
                        candidates[:,i] = predicted
                        states = torch.repeat_interleave(states,k,dim=1)

                        sum_probs = torch.repeat_interleave(sum_probs,k,dim=0)
                        sum_probs = torch.mul(sum_probs,probs).sort(descending=True)
                        indices = sum_probs[1]
                        
                        sampled_ids=candidates[indices[:k],:]
                        
                        states = states[:,indices[:k],:]
                        sum_probs = sum_probs[0][:k]
                    
                    inputs = self.embedding(sampled_ids[:,i].unsqueeze(1))
                    
        return sentences



if __name__=="__main__":
    model = TripletClassifier(224,10)
    dummy_img = torch.randn((5,3,224,224))
    out = model(dummy_img)