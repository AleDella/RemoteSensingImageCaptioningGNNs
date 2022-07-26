import sng_parser
from gensim.models import Word2Vec


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



def create_feats(sentences, save_feats=False, save_model=False, loaded_model=None):
    '''
    Extract the features for the relevant words in the caption. 
    NB: "relevant" means that are the labels of nodes and edges of the scene graphs

    Input:
        sentences: list of sentences
        save_feats: if True, saves the features for the words in a json file (Default False)
        save_model: if True, saves the word2vec model in the same folder (Default False)
        loaded_model: if not None, load the word2vec model indicated in the specified path (Default None)
    
    Return:
        None
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
        model = Word2Vec.load(loaded_model)
    
    if save_model:
        print("Saving word2vec model...")
        model.save('word2vecUAV.bin')
    
    if save_feats:
        print("Saving features of the words in the captions...")
        # TBI

    return model