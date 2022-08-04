import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer, BertModel
import cv2
from graph_utils import polish_triplets


# Util function cpied by extract_triplets
def readfile(path):
    with open(path,'r') as file:
        text = file.readlines()
    return text

class UCMTriplets(Dataset):
    '''
    Class for transforming triplets in graphs for the UCM dataset
    '''
    def __init__(self, image_folder, image_filenames, triplets_path, caption_path, model, tokenizer, return_keys, split=None, classification=False):
        '''
        Args:
            image_folder: path to the folder with all the images
            image_filenames: path to the file with all the filenames
            triplets_path: path to the JSON containing the triplets
            caption_path: path to the TXT containing the captions
            model: model used to produce the features for the tokens
            tokenizer: tokenizer used to extract ids from the tokens
            return_keys: list of keys to return in the sample
            split: define if its train, val or test split
            classification: if True, use triplet2idx and idx2triplet
        '''
        # JSON read for NLP part
        f = open(triplets_path)
        full_data = json.load(f)
        f.close()
        if split is None:
            # Discard IDs that have no scene graph for caption
            caption_tripl, discarded_ids = polish_triplets(full_data)
        else:
            # Discard IDs that have no scene graph for caption
            caption_tripl, discarded_ids = polish_triplets(full_data[split])
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
        
        # Here must add the ID filtering
        
        # FOR TRIPLET CLASSIFICATION
        if classification: 
            self.unique_triplets = len(full_data['Triplet_to_idx'])
            self.triplet_to_idx = full_data['Triplet_to_idx']
        
        if split is None:
            self.triplets = {id:value for (id,value) in full_data.items() if id not in discarded_ids}
        else:
            self.triplets = {id:value for (id,value) in full_data[split].items() if id not in discarded_ids}

        self.captions = {}
        # GLOBAL word2idx for the LSTM decoder
        self.word2idx = {}
        self.max_capt_length = 0
        self.word2idx["__UNK__"] = 0
        self.word2idx["__EOS__"] = 1
        word_id = 2
        for anno in readfile(caption_path):
            id = int(anno.split(" ")[:1][0])
            if str(id) not in discarded_ids:
                sentence = anno.replace(' \n', '').split(" ")[1:]
                if len(sentence)>self.max_capt_length:
                    self.max_capt_length = len(sentence)
                # Word2idx part
                for token in sentence:
                    if token not in list(self.word2idx.keys()):
                        self.word2idx[token] = word_id
                        word_id+=1
        for anno in readfile(caption_path):
            id = int(anno.split(" ")[:1][0])
            if str(id) not in discarded_ids:
                sentence = anno.replace(' \n', '').split(" ")[1:]
                while len(sentence)<self.max_capt_length:
                    sentence.append("__EOS__")
                try:
                    self.captions[id].append(sentence)
                except:
                    self.captions[id] = [sentence]
                


        self.node_feats = {}
        self.num_nodes = {}
        self.src_ids = {}
        self.dst_ids = {}
        f_split = {}
        # Discard IDs that have no scene graph for caption
        caption_tripl, discarded_ids = polish_triplets(self.triplets)
        # Here to check what happen when split is not passed
        for id in caption_tripl:
            if id not in discarded_ids:
                f_tripl = []
                tmp_dict = {}
                tmp_id = 0
                src_ids = []
                dst_ids = []
                node_feats = []
                # Extract features from triplets
                for i, tripl in enumerate(caption_tripl[id]):
                    encoded_input = tokenizer(tripl, return_tensors='pt', add_special_tokens=False, padding=True)
                    output = model(**encoded_input)
                    f_tripl.append(output.pooler_output)
                    if tripl[0] not in list(tmp_dict.keys()):
                        tmp_dict[tripl[0]]=tmp_id
                        tmp_id+=1
                        node_feats.append(list(output.pooler_output[0]))
                    if tripl[1] not in list(tmp_dict.keys()):
                        tmp_dict[tripl[1]]=tmp_id
                        tmp_id+=1
                        node_feats.append(list(output.pooler_output[1]))
                    if tripl[2] not in list(tmp_dict.keys()):
                        tmp_dict[tripl[2]]=tmp_id
                        tmp_id+=1
                        node_feats.append(list(output.pooler_output[2]))
                    
                    # Create source and destination lists
                    src_ids.append(tmp_dict[tripl[0]])
                    dst_ids.append(tmp_dict[tripl[1]])
                    src_ids.append(tmp_dict[tripl[1]])
                    dst_ids.append(tmp_dict[tripl[2]])
                self.src_ids[id] = src_ids
                self.dst_ids[id] = dst_ids
                self.node_feats[id] = torch.Tensor(node_feats)
                self.num_nodes[id] = len(node_feats)
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
        try:
            sample = {'image': self.images[int(id)], 'imgid': id, 'triplets': self.triplets[id], 'captions': self.captions[int(id)], 'src_ids':self.src_ids[id], 'dst_ids':self.dst_ids[id], 'node_feats': self.node_feats[id], 'num_nodes': self.num_nodes[id]}
        except:
            sample = {'image': self.images[id], 'imgid': id, 'triplets': self.triplets[id], 'captions': self.captions[int(id)], 'src_ids':self.src_ids[id], 'dst_ids':self.dst_ids[id], 'node_feats': self.node_feats[id], 'num_nodes': self.num_nodes[id]}
        #     print("\nSmth wrong")
        #     print("Problematic ID: ", id)
        #     exit(0)
        # sample = {'image': self.images[int(id)], 'imgid': id, 'triplets': self.triplets[id], 'captions': self.captions[int(id)]}
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

    #     indices = torch.nonzero(triplets_tensor[0,:])
    #     print(indices)
    #     for index in indices:
    #         print([key for key in triplet_to_idx.keys() if triplet_to_idx[key]==index])
        
    # print(triplets[0])
    
    return images, triplets_tensor


def collate_fn_captions(data):
    '''
    Collate function for the graph to caption
    '''
    src_ids = [d['src_ids'] for d in data]
    dst_ids = [d['dst_ids'] for d in data]
    node_feats = [d['node_feats'] for d in data]
    num_nodes = [d['num_nodes'] for d in data]
    ids = [d['imgid'] for d in data]
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
    
    return ids, [d['captions'] for d in data], src_ids, dst_ids, new_feats, num_nodes
            





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