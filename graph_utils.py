from numpy import argmax
import sng_parser
from gensim.models import Word2Vec
import torch
import json
import dgl
import matplotlib.pyplot as plt

def extract_encoding(sentences):
    '''
    Function that extracts the one-hot encoding from the labels of the nodes and edges
    of the graph for all the sentences.

    Input:
        sentences: list of strings of the sentences

    Return:
        word2idx: dictionary word->idx for one-hot encoding
        idx2word: dictionary idx->word for one-hot encoding
    '''
    word2idx = {}
    idx2word = {}
    id = 0
    for sentence in sentences:
        # Create the graph in order to obtain labels for edges and nodes
        g = sng_parser.parse(sentence)
        for rel in g['relations']:
            if rel['relation'] not in word2idx.keys():
                word2idx[rel['relation']] = id
                idx2word[id] = rel['relation']
                id += 1
            if g['entities'][rel['subject']]['head'] not in word2idx.keys():
                word2idx[g['entities'][rel['subject']]['head']] = id
                idx2word[id] = g['entities'][rel['subject']]['head']
                id += 1
            if g['entities'][rel['object']]['head'] not in word2idx.keys():
                word2idx[g['entities'][rel['object']]['head']] = id
                idx2word[id] = g['entities'][rel['object']]['head']
                id += 1
    
    return word2idx, idx2word



def create_feats(sentences, save_model=False, loaded_model=None, tokenize=False, attributes=False):
    '''
    Extract the features for the relevant words in the caption. 
    NB: "relevant" means that are the labels of nodes and edges of the scene graphs

    Input:
        sentences: list of sentences
        save_feats: if True, saves the features for the words in a json file (Default False)
        save_model: if True, saves the word2vec model in the same folder (Default False)
        loaded_model: if not None, load the word2vec model indicated in the specified path (Default None)
        tokenize: if True, returns the tokenized sentences using entities of the graph
        attributes: if True, each node of the graph is formed by the head with attributes
    
    Return:
        model: the word2vec model
        final_input: tokenized sentences
    '''

    final_input = []
    for sentence in sentences:
        g = sng_parser.parse(sentence)
        # Get the tokenization like in the graph
        sentence = []
        for rel in g['relations']:
            if attributes:
                sentence.append(g['entities'][rel['subject']]['lemma_span'])
                sentence.append(rel['relation'])
                sentence.append(g['entities'][rel['object']]['lemma_span'])
            else:
                sentence.append(g['entities'][rel['subject']]['head'])
                sentence.append(rel['relation'])
                sentence.append(g['entities'][rel['object']]['head'])
        final_input.append(sentence)
    print("Final input: ", final_input)
    # Train word2vec on the sentences for the embeddings
    if loaded_model is None:
        model = Word2Vec(final_input, min_count=1)
    else:
        # Fine tune the model
        model = Word2Vec.load(loaded_model)
    
    

    if save_model:
        print("Saving word2vec model...")
        model.save('word2vecUAV.bin')

    if tokenize:
        return final_input
    else:
        return model

def get_node_features(features, num_nodes):
    '''
    Given the padded node features, extract the original ones

    Args:
        features: tensor with the padded features (batch_size, max_num_nodes, feature_size)
        num_nodes: total number of nodes of the unified graph
    
    Return:
        new_feats: tensor with the features for each node (total_num_nodes, feat_size)
    '''
    new_feats = torch.zeros((num_nodes, features.size(-1)))
    checkpoint = 0
    for sample in features:
        for feat in sample:
            if sum(feat) == 0.0:
                continue
            else:
                new_feats[checkpoint] = feat
                checkpoint+=1
    
    return new_feats
            
            
def decode_output(out, idx2word):
    '''
    Function that decodes the network's output into the actual captions
    '''
    sentences = [[] for _ in range(out[0].size(0))]
    for toks in out:
        for i, sample in enumerate(toks):
            sentences[i].append(argmax(sample.cpu().detach().numpy()))
    
    for j, sent in enumerate(sentences):
        for i, id in enumerate(sent):
            sentences[j][i] = idx2word[id]
    try:
        sentences = [sent[:sent.index("<eos>")+1] for sent in sentences]
    except:
        try:
            sentences = [sent[:sent.index("<pad>")] for sent in sentences]
        except:
            print(sentences)
        
    
    return sentences

def encode_caption(caption, word2idx):
    '''
    Function that encodes the captions and return a tensor
    '''
    sentences = []
    for captions in caption:
        tmp = []
        for sent in captions:
            tmp.append([word2idx[t] for t in sent])
        sentences.append(tmp)

    return torch.Tensor(sentences)




def polish_triplets(triplets):
    '''
    Function that deletes double triplets and eliminates sentence division
    '''
    new_tripl = {}
    discarded_ids = []
    for id in triplets:
        final_tripl = []
        for sentence in triplets[id]:
            if sentence == []:
                continue
            else:
                for tripl in sentence:
                    if tripl not in final_tripl:
                        final_tripl.append(tripl)
        if final_tripl == []:
            discarded_ids.append(id)
        else:
            new_tripl[id] = final_tripl
    
    return new_tripl, discarded_ids


def arrange_triplet_file(json_name):
    '''
    Function that creates the Triplet_to_idx and discarded image sections; in addition, keeps only unique triplets.
    (Mainly used for UCM)
    '''
    triplets = load_json(json_name)
    new_triplets = {}
    disc_ids = []
    for split in list(triplets.keys()):
        if str(split) != 'Triplet_to_idx':
            tmp, dsc = polish_triplets(triplets[split])
            new_triplets[split] = tmp
            disc_ids.append(dsc)
    disc_ids = [id for s in disc_ids for id in s]
    new_triplets['discarded_images'] = disc_ids
    new_triplets['Triplet_to_idx'] = triplets['Triplet_to_idx']
    with open(json_name, 'w') as f:
        json.dump(new_triplets, f)


def tripl2list(tripl):
    '''
    Support function for tripl2graph()
    '''
    tripl = tripl.replace('(', '')
    tripl = tripl.replace(')', '')
    tripl = tripl.replace("'", '')
    tripl = tripl.split(',')
    tripl = [t.strip() for t in tripl]
    return tripl





def tripl2graph(triplets, model, tokenizer):
    '''
    Function that creates and extracts the graph from the triplets
    
    Args:
        triplets List[List]: list of lists of triplets
        model (torch.nn.Module): model used for extracting the features from the nodes
        
    Return:
        graph List[dgl.DGLGraph]: list of graphs
        graph_features List[torch.Tensor]: list of features for each graph
    '''
    feats = []
    graphs = []
    for sample in triplets:
        tmp_dict = {}
        tmp_id = 0
        tmp_src_ids = []
        tmp_dst_ids = []
        tmp_node_feats = []
        # Extract features from triplets
        for _, tripl in enumerate(sample):
            encoded_input = tokenizer(tripl2list(tripl), return_tensors='pt', add_special_tokens=False, padding=True)
            output = model(**encoded_input.to('cuda:0'))
            if tripl[0] not in list(tmp_dict.keys()):
                tmp_dict[tripl[0]]=tmp_id
                tmp_id+=1
                tmp_node_feats.append(list(output.pooler_output[0]))
            if tripl[1] not in list(tmp_dict.keys()):
                tmp_dict[tripl[1]]=tmp_id
                tmp_id+=1
                tmp_node_feats.append(list(output.pooler_output[1]))
            if tripl[2] not in list(tmp_dict.keys()):
                tmp_dict[tripl[2]]=tmp_id
                tmp_id+=1
                tmp_node_feats.append(list(output.pooler_output[2]))
            
            # Create source and destination lists
            tmp_src_ids.append(tmp_dict[tripl[0]])
            tmp_dst_ids.append(tmp_dict[tripl[1]])
            tmp_src_ids.append(tmp_dict[tripl[1]])
            tmp_dst_ids.append(tmp_dict[tripl[2]])
        
        g = dgl.graph((tmp_src_ids, tmp_dst_ids))
        f = torch.Tensor(tmp_node_feats)
        graphs.append(g)
        feats.append(f)
    
    g = dgl.batch(graphs)
    new_feats = torch.zeros((g.num_nodes(), feats[0].size(1)))
    i = 0
    for ft in feats:
        for f in ft:
            new_feats[i] = f
            i+=1
        
    return g, new_feats
    

def pad_encodings(captions, pad_id, training=True) -> torch.Tensor:
    '''
    Function that pads the sequences of ids using pytorch pad functions
    
    Args:
        captions List[List[int]]: list of id-coded captions
        pad_id int: id corresponding to the pad token
        
    Return:
        res torch.Tensor: padded sequences 
    '''
    res = []
    for sample in captions:
        if training:
            index = torch.randperm(len(sample))[:1]
            res.append(torch.tensor(sample[index]))
        else:
            tmp = []
            for cap in sample:
                cap = torch.tensor(cap)
                index = torch.randperm(len(sample))[:1]
                tmp.append(torch.tensor(cap[:, index]).reshape((sample.size(0))))
                
            res.append(torch.nn.utils.rnn.pad_sequence(tmp, padding_value=pad_id)) # (max_len, number_captions)

    return torch.nn.utils.rnn.pad_sequence(res, batch_first=True, padding_value=pad_id) # (batch_size, max_len, number_captions) if training; else (batch_size, max_len)

def load_json(path):
    '''
    Simple function to load a json
    
    Args:
        path: path to the file
    Return:
        data: data contained in the file
    '''
    f = open(path, 'r')
    data = json.load(f)
    f.close()
    return data


def bleuFormat(filename):
    '''
    Function that converts the output json produced by the testing; into an
    approriate format for BLEU scoring
    
    Args:
        filename (str): name of the json
    '''
    ucm = load_json(filename)
    def renew(data):
        new_data = {} 
        for k,v in data.items():
            caption = v['caption ']
            try:
                while True:
                    caption.remove('<sos>')
            except ValueError:
                pass
            try:
                caption.remove('<eos>')
            except:
                pass
            new_data[k] = caption
        return new_data
    ucm = renew(ucm)
    with open(filename, 'w') as f:
        json.dump(ucm, f)


def load_graph_data(graph_path, split):
    '''
    Function to load all the graph data taken from json format
    
    Args:
        graph_path: path to the folder containing graph data
        split: string indicating the split we want
    Return:
        dst_ids: list of destination nodes for constructing the graph (DGL library)
        src_ids: list of source nodes for constructing the graph (DGL Library)
        node_feats: list of node features for each graph
        num_nodes: list of total number of nodes for each graph
    '''
    return load_json(graph_path+'/'+'dst_ids_'+str(split)+'.json'), load_json(graph_path+'/'+'src_ids_'+str(split)+'.json'), load_json(graph_path+'/'+'node_feats_'+str(split)+'.json'), load_json(graph_path+'/'+'num_nodes_'+str(split)+'.json')
    
def save_plots(train_losses, val_losses, epochs, combo, gnn):
    '''
    Function to save the images of the plots of the training losses
    '''
    plt.plot([i+1 for i in range(epochs)], train_losses, label='Train loss')
    plt.plot([i+1 for i in range(epochs)], val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if combo:
        loss = 'CombinedLoss'
        l = 'cl'
    else:
        loss = 'UniqueLoss'
        l = 'ul'
    plt.title('Training Loss '+gnn.upper()+' + MultiHead + '+loss)
    plt.legend()
    plt.savefig('loss_images/'+str(gnn).lower()+'_mh_'+l+'_'+str(epochs)+'.png')