import json
import sng_parser

def extract_ent(sentences):
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
    for sent in sentences:
        g = sng_parser.parse(sent['raw'])
        for rel in g['relations']:
            final_input.append((g['entities'][rel['subject']]['head'], rel['relation'], g['entities'][rel['object']]['head']))
    return final_input






anno_path = "dataset/dataset_rsicd.json"
f = open(anno_path)
anno = json.load(f)
f.close()
anno = anno['images']
final_file = {'attributes': []}
number = 0
for img in anno:
    tripl = list(set(extract_ent(img['sentences'])))
    img_dict = {'filename': img['filename'], 'imgid': img['imgid'], 'triplets': tripl}
    final_file['attributes'].append(img_dict)
    number+=len(tripl)
    

with open("triplets.json", "w") as outfile:
    json.dump(final_file, outfile)

print("Total number of triplets found: ", number)


