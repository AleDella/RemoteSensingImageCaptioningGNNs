from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import dgl
from graph_utils import decode_output, get_node_features
import json
from functools import partial
from dataset import collate_fn_captions, collate_fn_classifier, augmented_collate_fn, collate_fn_full
from numpy import argmax



def eval_captions(dataset, model, filename):
    '''
    Function that tests a model
    
    Args:
        dataset (torch.utils.data.Dataset): dataset to use for testing.
        model (torch.nn.Module): model to test on the dataset
        filename (str): name of the file in which the captions are saved
    
    Return:
        None
    '''
    testloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=partial(collate_fn_captions, word2idx=dataset.word2idx, training=True))
    # Set the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Create the conversion id -> token
    idx2word = {v: k for k, v in dataset.word2idx.items()}
    with torch.no_grad():
        model.eval()
        result = {}
        for _, data in enumerate(tqdm(testloader)):
            ids, _, encoded_captions, src_ids, dst_ids, node_feats, num_nodes = data
            graphs = dgl.batch([dgl.graph((src_id, dst_id)) for src_id, dst_id in zip(src_ids, dst_ids)]).to(device)
            feats = get_node_features(node_feats, sum(num_nodes)).to(device)
            outputs = model(graphs, feats, encoded_captions)
            decoded_outputs = decode_output(outputs, idx2word)
            for i, id in enumerate(ids):
                result[id] = {"caption length": len(decoded_outputs[i]),"caption ": decoded_outputs[i]}
            
    with open(filename, "w") as outfile:
        json.dump(result, outfile)
        

def augmented_eval_captions(dataset, model, filename):
    '''
    Function that tests a model
    
    Args:
        dataset (torch.utils.data.Dataset): dataset to use for testing.
        model (torch.nn.Module): model to test on the dataset
        filename (str): name of the file in which the captions are saved
    
    Return:
        None
    '''
    testloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=partial(augmented_collate_fn, word2idx=dataset.word2idx, training=True))
    # Set the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Create the conversion id -> token
    idx2word = {v: k for k, v in dataset.word2idx.items()}
    with torch.no_grad():
        model.eval()
        result = {}
        for _, data in enumerate(tqdm(testloader)):
            ids, images, _, encoded_captions, src_ids, dst_ids, node_feats, num_nodes = data
            graphs = dgl.batch([dgl.graph((src_id, dst_id)) for src_id, dst_id in zip(src_ids, dst_ids)]).to(device)
            feats = get_node_features(node_feats, sum(num_nodes)).to(device)
            img = images.to(device)
            outputs = model(graphs, feats, img, encoded_captions)
            decoded_outputs = decode_output(outputs, idx2word)
            for i, id in enumerate(ids):
                result[id] = {"caption length": len(decoded_outputs[i]),"caption ": decoded_outputs[i]}
            
    with open(filename, "w") as outfile:
        json.dump(result, outfile)


def eval_classification(dataset, model, filename):
    '''
    Function that tests a model
    
    Args:
        dataset (torch.utils.data.Dataset): dataset to use for testing.
        model (torch.nn.Module): model to test on the dataset
        filename (str): name of the file in which the captions are saved
    
    Return:
        None
    '''
    # Create the dataloader
    testloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=partial(collate_fn_classifier, triplet_to_idx=dataset.triplet_to_idx))
    # Set the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Create the conversion id -> triplet
    idx2triplet = {v: k for k, v in dataset.triplet_to_idx.items()}
    with torch.no_grad():
        model.eval()
        accuracy_test = 0 
        for i, data in enumerate(tqdm(testloader)):
            images, triplets = data
            images = images.to(device)
            triplets = triplets.to(device)
            outputs = model(images)
            outputs = outputs.reshape((outputs.shape[0], int(outputs.shape[1]/2), 2))
            outputs = torch.sigmoid(outputs)
            # outputs[outputs>=0.5] = 1
            # outputs[outputs<0.5] = 0
            outputs = torch.tensor([[torch.argmax(task).item() for task in sample ] for sample in outputs]).to(outputs.device)
            accuracy = torch.sum(outputs==triplets).item()/(triplets.shape[0]*triplets.shape[1])
            outputs = outputs.squeeze()
            outputs = outputs.nonzero().squeeze()
            triplets = triplets.squeeze()
            triplets = triplets.nonzero().squeeze()
            print('True triplets')
            for i in range(triplets.shape[0]):
                print(idx2triplet[triplets[i].item()])
            print('Predicted triplets')
            for i in range(outputs.shape[0]):
                print(idx2triplet[outputs[i].item()])
            break
            accuracy_test += accuracy
    
    print('Test accuracy: {:.3f}'.format(accuracy_test/i))
        
        
        
# def eval_pipeline(dataset, model, filename):
#     '''
#     Function that tests a model
    
#     Args:
#         dataset (torch.utils.data.Dataset): dataset to use for testing.
#         model (torch.nn.Module): model to test on the dataset
#         filename (str): name of the file in which the captions are saved
    
#     Return:
#         None
#     '''
#     testloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=partial(collate_fn_full, word2idx=dataset.word2idx, training=True))
#     # Set the correct device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     # Create the conversion id -> token
#     idx2word = {v: k for k, v in dataset.word2idx.items()}
#     with torch.no_grad():
#         model.eval()
#         result = {}
#         for _, data in enumerate(tqdm(testloader)):
#             ids, images, triplet_tensor, captions, encoded_captions, src_ids, dst_ids, node_feats, num_nodes = data
#             graphs = dgl.batch([dgl.graph((src_id, dst_id)) for src_id, dst_id in zip(src_ids, dst_ids)]).to(device)
#             feats = get_node_features(node_feats, sum(num_nodes)).to(device)
#             img = images.to(device)
#             outputs = model(graphs, feats, img, encoded_captions)
#             decoded_outputs = decode_output(outputs, idx2word)
#             for i, id in enumerate(ids):
#                 result[id] = {"caption length": len(decoded_outputs[i]),"caption ": decoded_outputs[i]}
            
#     with open(filename, "w") as outfile:
#         json.dump(result, outfile)