from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import dgl
from graph_utils import decode_output, get_node_features
import json
from functools import partial
from dataset import collate_fn_captions, collate_fn_classifier
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


# Thinking if makes sense to add this one (need to know how to extract more than one class from the classifier)
# def eval_classification(dataset, model, filename):
#     '''
#     Function that tests a model
    
#     Args:
#         dataset (torch.utils.data.Dataset): dataset to use for testing.
#         model (torch.nn.Module): model to test on the dataset
#         filename (str): name of the file in which the captions are saved
    
#     Return:
#         None
#     '''
#     # Create the dataloader
#     testloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=partial(collate_fn_classifier, triplet_to_idx=dataset.triplet_to_idx))
#     # Set the correct device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     # Create the conversion id -> triplet
#     idx2triplet = {v: k for k, v in dataset.triplet_to_idx.items()}
#     with torch.no_grad():
#         model.eval()
#         result = {}
#         for _, data in enumerate(tqdm(testloader)):
#             images, triplets = data
#             images = images.to(device)
#             triplets = triplets.to(device)
#             outputs = model(images)
#             # print("Out info: {} {}\n".format(len(outputs), outputs[0].shape))
#             # print("Dictionary length: ", len(dataset.triplet_to_idx))
#             # print(images)
#             print(argmax(triplets.cpu().detach().numpy()))
#             print(argmax(outputs.cpu().detach().numpy()))
#             exit(0)