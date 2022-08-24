from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import dgl
from graph_utils import decode_output, get_node_features
import json
from functools import partial
from dataset import collate_fn_captions, collate_fn_classifier


def eval_captions(dataset, model, batch_size, filename):
    '''
    Function that tests a model
    
    Args:
        dataset (torch.utils.data.Dataset): dataset to use for testing.
        model (torch.nn.Module): model to test on the dataset
        batch_size (int): size of the batch
        filename (str): name of the file in which the captions are saved
    
    Return:
        None
    '''
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn_captions, word2idx=dataset.word2idx, training=True))
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
            graphs = dgl.batch([dgl.graph((src_id, dst_id)) for src_id, dst_id in zip(src_ids, dst_ids)]).to('cuda')
            feats = get_node_features(node_feats, sum(num_nodes)).to('cuda')
            outputs = model(graphs, feats, encoded_captions)
            decoded_outputs = decode_output(outputs, idx2word)
            for i, id in enumerate(ids):
                result[id] = {"caption length": len(decoded_outputs[i]),"caption ": decoded_outputs[i]}
            
    with open(filename, "w") as outfile:
        json.dump(result, outfile)
