import torch
from torch.utils.data import Dataset
import dgl
import json
from transformers import BertTokenizer, BertModel

# Util function cpied by extract_triplets
def readfile(path):
    with open(path,'r') as file:
        text = file.readlines()
    return text

class UCMTriplets(Dataset):
    '''
    Class for transforming triplets in graphs for the UCM dataset
    '''
    def __init__(self, triplets_path, caption_path, model, tokenizer, split=None):
        '''
        Args:
            triplets_path: path to the JSON containing the triplets
            caption_path: path to the TXT containing the captions
            model: model used to produce the features for the tokens
            tokenizer: tokenizer used to extract ids from the tokens
        '''
        # JSON read
        f = open(triplets_path)
        full_data = json.load(f)
        f.close()
        if split is None:
            self.triplets = full_data
        else:
            self.triplets = full_data[split]
        self.captions = {}
        for anno in readfile(caption_path):
            id = int(anno.split(" ")[:1][0])
            sentence = anno.replace(' \n', '').split(" ")[1:]
            try:
                self.captions[id].append(sentence)
            except:
                self.captions[id] = [sentence]
        f_split = {}
        for id in full_data[split]:
            f_tripl = []
            # Extract features from triplets
            for tripl in full_data[split][id]:
                encoded_input = tokenizer(tripl, return_tensors='pt', add_special_tokens=False, padding=True)
                output = model(**encoded_input)
                f_tripl.append(output.pooler_output)
            f_split[id] = f_tripl
        self.features = f_split
    
    def __len__(self):
        # Number of samples
        return len(self.triplets)

    def __getitem__(self, index):
        if torch.is_tensor(index):
          index = index.tolist()
        # Get the image ID
        id = list(self.triplets.keys())[index]
        
        sample = {'imgid': id, 'triplets': self.triplets[id], 'captions': self.captions[int(id)]}
        # Graph creation
        # 1: get a list of source node and destination node
        tmp_dict = {}
        tmp_id = 0
        added_rel = []
        src_ids = []
        dst_ids = []
        # Now must find a way to add the right features for the nodes and edges
        node_feats = []
        rel_feats = []
        for i, tripl in enumerate(self.triplets[id]):
            # Add the nodes
            if tripl[0] not in list(tmp_dict.keys()):
                tmp_dict[tripl[0]]=tmp_id
                tmp_id+=1
                node_feats.append(self.features[id][i][0])
            if tripl[2] not in list(tmp_dict.keys()):
                tmp_dict[tripl[2]]=tmp_id
                tmp_id+=1
                node_feats.append(self.features[id][i][2])
            # Add the relation
            rel_feats.append(self.features[id][i][1])
            added_rel.append(tripl[1])
            # Create source and destination lists
            src_ids.append(tmp_dict[tripl[0]])
            dst_ids.append(tmp_dict[tripl[2]])
        # Create the graph
        g = dgl.graph((src_ids, dst_ids))
        # Add the graph infos to the sample
        sample['graph'] = g
        sample['node_feats'] = node_feats
        sample['rel_feats'] = rel_feats
        
        return sample




# Test code
if __name__== "__main__":
    import time
    ini = time.time()
    tripl_path = 'example_tripl.json'
    anno_path = 'example_anno.txt'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    dataset = UCMTriplets(tripl_path, anno_path, model, tokenizer, split='train')
    # example of dataset sample
    print(dataset[0].keys())
    print(str(time.time()-ini))