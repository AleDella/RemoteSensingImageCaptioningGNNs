from dataset import UCMDataset, RSICDDataset, collate_fn_captions
# from transformers import BertTokenizer, BertModel
from models import CaptionGenerator
from train import caption_trainer
import torch

def train_gnn(dataset, task, epochs, lr, batch_size, decoder, network_name):
    '''
    Function that initialize the training for the gnn depending on the task and dataset
    '''
    if task == "tripl2caption":
        # Dataset definition
        if dataset == 'ucm':
            train_filenames = 'dataset/UCM_dataset/filenames/filenames_train.txt'
            val_filenames = 'dataset/UCM_dataset/filenames/filenames_val.txt'
            img_path = 'dataset/UCM_dataset/images/'
            tripl_path = 'dataset/UCM_dataset/triplets.json'
            polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm.json'
            anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
            word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
            graph_path = 'dataset/UCM_dataset/Graph_data'
            return_k = ['imgid', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            train_dataset = UCMDataset(img_path, train_filenames, graph_path, tripl_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='train')
            val_dataset = UCMDataset(img_path, val_filenames, graph_path, tripl_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='val')
        
        if dataset == 'rsicd':
            graph_path = 'dataset/RSICD_dataset/Graph_data'
            word2idx_path = 'dataset/RSICD_dataset/caption_dict_RSICD.json'
            anno_path = 'dataset/RSICD_dataset/polished_dataset.json'
            img_path = 'dataset/RSICD_dataset/RSICD_images'
            tripl_path = 'dataset/RSICD_dataset/triplets_rsicd.json'
            return_k = ['imgid', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            train_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='train')
            val_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='val')
        
        # Network training part
        feats_n = torch.Tensor(train_dataset.node_feats['1'])[0].size(0)
        max = train_dataset.max_capt_length
        if val_dataset.max_capt_length>max:
            max = val_dataset.max_capt_length
        model = CaptionGenerator(feats_n, max, train_dataset.word2idx, decoder=decoder)
        trainer = caption_trainer(model,train_dataset,val_dataset,collate_fn_captions, train_dataset.word2idx, max, network_name)
        trainer.fit(epochs, lr, batch_size, model._loss, early_stopping=True, tol_threshold=1)
    else:
        print("Task not yet implemented.")

# # # Load old model
# test_dataset = UCMTriplets(img_path, test_filenames, tripl_path, tripl_path_test, anno_path, word2idx_path, model, tokenizer, return_keys=return_k, split='test')
# feats_n = test_dataset.node_feats[list(test_dataset.node_feats.keys())[0]][0].size(0)
# max = test_dataset.max_capt_length
# model = CaptionGenerator(feats_n, max, test_dataset.word2idx)
# model = load_model('GNN.pth')
# # Printing predictions to file
# from torch.utils.data import DataLoader
# import torch
# from tqdm import tqdm
# import dgl
# from graph_utils import decode_output, get_node_features
# import json
# testloader = DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=partial(collate_fn_captions, word2idx=test_dataset.word2idx, training=True))
# model = model.to('cuda')
# idx2word = {v: k for k, v in test_dataset.word2idx.items()}
# with torch.no_grad():
#     model.eval()
#     result = {}
#     for _, data in enumerate(tqdm(testloader)):
#         ids, captions, encoded_captions, src_ids, dst_ids, node_feats, num_nodes = data
#         graphs = dgl.batch([dgl.graph((src_id, dst_id)) for src_id, dst_id in zip(src_ids, dst_ids)]).to('cuda')
#         feats = get_node_features(node_feats, sum(num_nodes)).to('cuda')
#         outputs = model(graphs, feats, encoded_captions)
#         # print("Out info: {} {}\n".format(len(outputs), outputs[0].shape))
#         # print("Dictionary length: ", len(test_dataset.word2idx))
#         decoded_outputs = decode_output(outputs, idx2word, [])
#         for i, id in enumerate(ids):
#             result[id] = {"caption length": len(decoded_outputs[i]),"caption ": decoded_outputs[i]}
            
# with open("captions_linear.json", "w") as outfile:
#     json.dump(result, outfile)


