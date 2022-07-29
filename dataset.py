import torch
from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertTokenizer, BertModel
import cv2
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
    def __init__(self, image_folder, image_filenames, triplets_path, caption_path, model, tokenizer, return_keys, split=None):
        '''
        Args:
            triplets_path: path to the JSON containing the triplets
            caption_path: path to the TXT containing the captions
            model: model used to produce the features for the tokens
            tokenizer: tokenizer used to extract ids from the tokens
        '''
        # Save return keys
        self.return_keys = return_keys
        # IMG read for CV part
        files = readfile(image_filenames)
        self.images = {}
        for file in files:
            id = int(file.split('.')[0])
            path = image_folder + file.replace('\n', '')
            img = cv2.imread(path)[:,:,::-1] # CV2 reads images in BGR, so convert to RGB for the networks 
            self.images[id] = torch.from_numpy(img.copy())
        # JSON read for NLP part
        f = open(triplets_path)
        full_data = json.load(f)
        f.close()
        
        # FOR TRIPLET CLASSIFICATION 
        self.unique_triplets = len(full_data['Triplet_to_idx'])
        self.triplet_to_idx = full_data['Triplet_to_idx']
        
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
        self.src_ids = {}
        self.dst_ids = {}
        f_split = {}
        
        # Here to check what happen when split is not passed
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
            self.src_ids[id] = src_ids
            self.dst_ids[id] = dst_ids
            self.node_feats[id] = node_feats
            self.rel_feats[id] = rel_feats
            f_split[id] = f_tripl
        self.features = f_split
        
    
    def __len__(self):
        # Number of samples
        return len(self.images)  # CHANGED TO IMAGES FOR THE FIRST PART OF THE NETWORK!

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # Get the image ID
        id = list(self.triplets.keys())[index]
        
        sample = {'image': self.images[int(id)], 'imgid': id, 'triplets': self.triplets[id], 'captions': self.captions[int(id)], 'src_ids':self.src_ids[id], 'dst_ids':self.dst_ids[id], 'node_feats': self.node_feats[id], 'rel_feats':self.rel_feats[id]}
        
        # Filter only what is needed 
        out = { your_key: sample[your_key] for your_key in self.return_keys}
        
        return out
    

def collate_fn_classifier(data, triplet_to_idx):
    '''
    Collate function to train the triplet classifier 
    '''
    images = [d['image'] for d in data]
    triplets = [d['triplets'] for d in data]
    captions = [d['captions'] for d in data]
        
    images = torch.stack(images, 0)
    images = images.permute(0,3,1,2)
    
    print(triplets[3])
    print(captions[3])
    
    #print(data[0]['triplets'].shape)
    
    return None

# Test code
if __name__== "__main__":
    filenames = 'D:/Alessio/Provone/dataset/UCM_dataset/filenames/filenames_test.txt'
    img_path = 'D:/Alessio/Provone/dataset/UCM_dataset/images/'
    tripl_path = 'triplets.json'
    anno_path = 'D:/Alessio/Provone/dataset/UCM_dataset/filenames/descriptions_UCM.txt'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    
    dataset = UCMTriplets(img_path, filenames, tripl_path, anno_path, model, tokenizer, split='test')
    # # example of dataset sample