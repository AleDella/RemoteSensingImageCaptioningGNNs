from numpy import argmax
import sng_parser
from gensim.models import Word2Vec
import torch

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



def create_feats(sentences, save_feats=False, save_model=False, loaded_model=None, tokenize=False):
    '''
    Extract the features for the relevant words in the caption. 
    NB: "relevant" means that are the labels of nodes and edges of the scene graphs

    Input:
        sentences: list of sentences
        save_feats: if True, saves the features for the words in a json file (Default False)
        save_model: if True, saves the word2vec model in the same folder (Default False)
        loaded_model: if not None, load the word2vec model indicated in the specified path (Default None)
        tokenize: if True, returns the tokenized sentences using entities of the graph
    
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
    
    if save_feats:
        print("Saving features of the words in the captions...")
        # TBI

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
    ids = []
    sentences = [[] for _ in range(out[0].size(0))]
    for tok in out:
        ids.append([argmax(emb.detach().numpy()) for emb in tok])
    for tok in ids:
        for i, id in enumerate(tok):
            sentences[i].append(idx2word[id])
            
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