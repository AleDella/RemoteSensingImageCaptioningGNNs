import sng_parser


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