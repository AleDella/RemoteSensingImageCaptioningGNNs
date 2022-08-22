from dataset import UCMDataset, collate_fn_captions
# from transformers import BertTokenizer, BertModel
from models import CaptionGenerator, load_model
from train import caption_trainer
from functools import partial

train_filenames = 'dataset/UCM_dataset/filenames/filenames_train.txt'
val_filenames = 'dataset/UCM_dataset/filenames/filenames_val.txt'
test_filenames = 'dataset/UCM_dataset/filenames/filenames_test.txt'
img_path = 'dataset/UCM_dataset/images/'
tripl_path = 'dataset/UCM_dataset/triplets.json'
tripl_path_train = 'dataset/UCM_dataset/polished_triplets_train.json'
tripl_path_val = 'dataset/UCM_dataset/polished_triplets_val.json'
tripl_path_test = 'dataset/UCM_dataset/polished_triplets_test.json'
anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
graph_path = 'dataset/UCM_dataset/Graph_data'
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# return_k = ['src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
return_k = ['imgid', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']

train_dataset = UCMDataset(img_path, train_filenames, graph_path, tripl_path, tripl_path_train, anno_path, word2idx_path, return_keys=return_k, split='train')
val_dataset = UCMDataset(img_path, val_filenames, tripl_path, tripl_path_val, anno_path, word2idx_path, return_keys=return_k, split='val')
feats_n = train_dataset.node_feats['1'][0].size(0)
max = train_dataset.max_capt_length
if val_dataset.max_capt_length>max:
    max = val_dataset.max_capt_length


model = CaptionGenerator(feats_n, max, train_dataset.word2idx, decoder='linear')
trainer = caption_trainer(model,train_dataset,val_dataset,collate_fn_captions, train_dataset.word2idx, max, 'GNN.pth')
trainer.fit(10, 0.0001, 8, model._loss, early_stopping=True, tol_threshold=1)

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


