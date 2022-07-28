import torch
from torch.utils.data import Dataset
import dgl
import json
from transformers import BertTokenizer, BertModel
from PIL import Image
import numpy as np

# Util function cpied by extract_triplets
def readfile(path):
    with open(path,'r') as file:
        text = file.readlines()
    return text

class UCMTriplets(Dataset):
    '''
    Class for transforming triplets in graphs for the UCM dataset
    '''
    def __init__(self, image_folder, image_filenames, triplets_path, caption_path, model, tokenizer, split=None):
        '''
        Args:
            triplets_path: path to the JSON containing the triplets
            caption_path: path to the TXT containing the captions
            model: model used to produce the features for the tokens
            tokenizer: tokenizer used to extract ids from the tokens
        '''
        # IMG read for CV part
        files = readfile(image_filenames)
        self.images = {}
        for file in files:
            id = int(file.split('.')[0])
            path = image_folder + file.replace('\n', '')
            img = Image.open(path)
            self.images[id] = torch.from_numpy(np.array(img))
        
        # JSON read for NLP part
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
        self.node_feats = {}
        self.rel_feats = {}
        self.graphs = {}
        f_split = {}
        for id in full_data[split]:
            f_tripl = []
            tmp_dict = {}
            tmp_id = 0
            added_rel = []
            src_ids = []
            dst_ids = []
            node_feats = []
            rel_feats = []
            # Extract features from triplets
            for i, tripl in enumerate(full_data[split][id]):
                encoded_input = tokenizer(tripl, return_tensors='pt', add_special_tokens=False, padding=True)
                output = model(**encoded_input)
                f_tripl.append(output.pooler_output)
                if tripl[0] not in list(tmp_dict.keys()):
                    tmp_dict[tripl[0]]=tmp_id
                    tmp_id+=1
                    node_feats.append(output.pooler_output[0])
                if tripl[2] not in list(tmp_dict.keys()):
                    tmp_dict[tripl[2]]=tmp_id
                    tmp_id+=1
                    node_feats.append(output.pooler_output[2])
                # Add the relation
                rel_feats.append(output.pooler_output[1])
                added_rel.append(tripl[1])
                # Create source and destination lists
                src_ids.append(tmp_dict[tripl[0]])
                dst_ids.append(tmp_dict[tripl[2]])
            self.node_feats[id] = node_feats
            self.rel_feats[id] = rel_feats
            self.graphs[id] = dgl.graph((src_ids, dst_ids))
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
        
        sample = {'image': self.images[id], 'imgid': id, 'triplets': self.triplets[id], 'captions': self.captions[int(id)], 'graphs':self.graphs[id], 'node_feats': self.node_feats[id], 'rel_feats':self.rel_feats[id]}
        return sample




# Test code
if __name__== "__main__":
    import time
    ini = time.time()
    filenames = 'filenames_train.txt'
    img_path = 'test_images/'
    tripl_path = 'example_tripl.json'
    anno_path = 'example_anno.txt'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    
    dataset = UCMTriplets(img_path, filenames, tripl_path, anno_path, model, tokenizer, split='train')
    # example of dataset sample
    print(dataset[0].keys())
    print(str(time.time()-ini))