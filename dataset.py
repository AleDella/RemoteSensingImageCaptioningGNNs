import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer, BertModel
import cv2
from graph_utils import pad_encodings, load_graph_data, load_json


# Util function cpied by extract_triplets
def readfile(path):
    with open(path,'r') as file:
        text = file.readlines()
    return text

class UCMTriplets(Dataset):
    '''
    Class for transforming triplets in graphs for the UCM dataset
    '''
    def __init__(self, image_folder, image_filenames, graph_path, triplets_path, polished_tripl_path, caption_path, word2idx_path, return_keys, split='train'):
        '''
        Args:
            image_folder: path to the folder with all the images
            image_filenames: path to the file with all the filenames
            triplets_path: path to the JSON containing the triplets
            polished_triplets_path: path to the JSON containing only unique triplets
            caption_path: path to the TXT containing the captions
            word2idx_path: path to the JSON containing the word2idx dictionary
            model: model used to produce the features for the tokens
            tokenizer: tokenizer used to extract ids from the tokens
            return_keys: list of keys to return in the sample
            split: define if its train, val or test split
        '''
        # Polished triplets parts
        f = open(polished_tripl_path)
        polished_data = json.load(f)
        f.close()
        # Added tripl filtering so no split needed
        _, discarded_ids = polished_data['tripl'], polished_data['discarded_ids']
        # Save return keys
        self.return_keys = return_keys
        # IMG read for CV part
        files = readfile(image_filenames)
        self.images = {}
        for file in files:
            id = int(file.split('.')[0])
            if str(id) not in discarded_ids:
                path = image_folder + file.replace('\n', '')
                img = cv2.imread(path)[:,:,::-1] # CV2 reads images in BGR, so convert to RGB for the networks 
                self.images[id] = torch.from_numpy(img.copy())
        
        # Part using the full triplet file
        f = open(triplets_path)
        unpolished_data = json.load(f)
        f.close()
        # FOR TRIPLET CLASSIFICATION
        self.triplet_to_idx = unpolished_data['Triplet_to_idx']
        if split is None:
            self.triplets = {id:value for (id,value) in unpolished_data.items() if id not in discarded_ids}
        else:
            self.triplets = {id:value for (id,value) in unpolished_data[split].items() if id not in discarded_ids}

        self.captions = {}
        # Upload the word2idx file
        f = open(word2idx_path)
        self.word2idx=json.load(f)
        f.close()
        
        self.max_capt_length = 0
        for anno in readfile(caption_path):
            id = int(anno.split(" ")[:1][0])
            if str(id) not in discarded_ids:
                sentence = anno.replace(' \n', '').split(" ")[1:]
                # Add to the captions starting and ending tokens
                sentence.insert(0, "<sos>")
                sentence.append("<eos>")
                tmp = []
                for tok in sentence:
                    tmp.append(self.word2idx[tok])
                if len(sentence)>self.max_capt_length:
                        self.max_capt_length = len(sentence)
                try:
                    self.captions[id].append(sentence)
                except:
                    self.captions[id] = [sentence]
                

        # # Graph data part
        self.dst_ids, self.src_ids, self.node_feats, self.num_nodes = load_graph_data(graph_path=graph_path, split=split)  
        
    
    def __len__(self):
        # Number of samples
        return len(self.images)  # CHANGED TO IMAGES FOR THE FIRST PART OF THE NETWORK!

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # Get the image ID
        id = list(self.triplets.keys())[index]
        try:
            sample = {'image': self.images[int(id)], 'imgid': id, 'triplets': self.triplets[id], 'captions': self.captions[int(id)], 'src_ids':self.src_ids[id], 'dst_ids':self.dst_ids[id], 'node_feats': self.node_feats[id], 'num_nodes': self.num_nodes[id]}
        except:
            sample = {'image': self.images[id], 'imgid': id, 'triplets': self.triplets[id], 'captions': self.captions[int(id)], 'src_ids':self.src_ids[id], 'dst_ids':self.dst_ids[id], 'node_feats': self.node_feats[id], 'num_nodes': self.num_nodes[id]}
        # Filter only what is needed 
        out = { your_key: sample[your_key] for your_key in self.return_keys}
        
        return out
    

def collate_fn_classifier(data, triplet_to_idx):
    '''
    Collate function to train the triplet classifier 
    '''
    images = [d['image'] for d in data]
    triplets = [d['triplets'] for d in data]
        
    images = torch.stack(images, 0)
    images = images.permute(0,3,1,2)
    # Between 0 and 1 for pytorch
    images = images/255
    
    triplets_tensor = torch.zeros((len(triplets),len(triplet_to_idx))) # To store one hot encodings
    
    i=0
    for image_triplet in triplets[0]:
        for triplet in image_triplet:
            triplets_tensor[i,triplet_to_idx[str(tuple(triplet))]] = 1
    
    return images, triplets_tensor


def collate_fn_captions(data, word2idx, training):
    '''
    Collate function for the graph to caption
    '''
    src_ids = [d['src_ids'] for d in data]
    dst_ids = [d['dst_ids'] for d in data]
    node_feats = [torch.tensor(d['node_feats'])  if type(d['node_feats']) != torch.Tensor else d['node_feats'] for d in data]
    num_nodes = [d['num_nodes'] for d in data]
    max_rel = len(max(src_ids, key=len))
    max_nodes = max(num_nodes)
    # ID
    for i, elem in enumerate(src_ids):
        # Add self loops to fix length
        while len(elem)<max_rel:
            src_ids[i].append(0)
            dst_ids[i].append(0)
        # Add random node feats to fix lengths
        if node_feats[i].size(0)<max_nodes:
            try:
                new_node_feats = torch.zeros((node_feats[i].size(0)+(max_nodes-node_feats[i].size(0)), node_feats[i].size(1)))
            except:
                print("Len: {}\tSize: {}\t ID: {}\n".format(len(node_feats), node_feats[i].size(), node_feats))
                exit(0)
            for j in range(node_feats[i].size(0)):
                new_node_feats[j] = node_feats[i][j]
            node_feats[i] = new_node_feats
    # Create the final Tensor
    new_feats = torch.zeros((len(node_feats), node_feats[0].size(0), node_feats[0].size(1)))
    for i, elem in enumerate(node_feats):
        new_feats[i] = elem
    
    # Create the captions tensor
    new_cap_ids = []
    for d in data:
        smth = []
        for cap in d['captions']:
            tmp = [word2idx[word] if word in word2idx else word2idx['<unk>'] for word in cap]
            smth.append(tmp)
        new_cap_ids.append(smth)

    return [d['imgid'] for d in data], [d['captions'] for d in data], pad_encodings(new_cap_ids, word2idx['<pad>'], training=training), src_ids, dst_ids, new_feats, num_nodes
            
### WIP
# Class for the RSICD dataset
class RSICDDataset(Dataset):
    
    def __init__(self, image_folder, graph_path, polished_tripl_path, annotation_path, word2idx_path, return_keys, split=None) -> None:
        # Polished triplets parts
        self.triplets = load_json(polished_tripl_path)[split]
        # Save the selected keys
        self.return_keys = return_keys
        # Save annotations
        self.annotations = load_json(annotation_path)['images']
        # IMG read for CV part
        self.images = {}
        for anno in self.annotations:
            # Check for images that have triplets and are of the desired split
            if anno['split']==split:
                path = image_folder +"/" + anno['filename']
                # print("Path: ", path)
                img = cv2.imread(path)[:,:,::-1] # CV2 reads images in BGR, so convert to RGB for the networks 
                self.images[anno['imgid']] = torch.from_numpy(img.copy())
        # Upload the word2idx file
        self.word2idx=load_json(word2idx_path)
        # Captions part
        self.captions = {}
        self.max_capt_length = 0
        for anno in self.annotations:
            if anno['split']==split:
                for sent in anno['sentences']:
                    # Add to the captions starting and ending tokens
                    sentence = sent['tokens']
                    sentence.insert(0, "<sos>")
                    sentence.append("<eos>")
                    tmp = []
                    for tok in sentence:
                        tmp.append(self.word2idx[tok])
                    if len(sentence)>self.max_capt_length:
                            self.max_capt_length = len(sentence)
                    try:
                        self.captions[anno['imgid']].append(sentence)
                    except:
                        self.captions[anno['imgid']] = [sentence]
        # # Graph data part
        self.dst_ids, self.src_ids, self.node_feats, self.num_nodes = load_graph_data(graph_path=graph_path, split=split)


    def __len__(self):
        # Number of samples
        return len(self.images)  # CHANGED TO IMAGES FOR THE FIRST PART OF THE NETWORK!


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # Get the image ID
        id = list(self.triplets.keys())[index]
        try:
            sample = {'image': self.images[int(id)], 'imgid': id, 'triplets': self.triplets[id], 'captions': self.captions[int(id)], 'src_ids':self.src_ids[id], 'dst_ids':self.dst_ids[id], 'node_feats': torch.Tensor(self.node_feats[id]), 'num_nodes': self.num_nodes[id]}
        except:
            sample = {'image': self.images[id], 'imgid': id, 'triplets': self.triplets[id], 'captions': self.captions[int(id)], 'src_ids':self.src_ids[id], 'dst_ids':self.dst_ids[id], 'node_feats': torch.Tensor(self.node_feats[id]), 'num_nodes': self.num_nodes[id]}
        # Filter only what is needed 
        out = { your_key: sample[your_key] for your_key in self.return_keys}
        
        return out

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