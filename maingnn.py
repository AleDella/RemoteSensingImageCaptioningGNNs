from dataset import UCMTriplets, collate_fn_captions
from transformers import BertTokenizer, BertModel
from models import CaptionGenerator
from train import caption_trainer

filenames = 'D:/Alessio/Provone/dataset/UCM_dataset/filenames/filenames_train.txt'
img_path = 'D:/Alessio/Provone/dataset/UCM_dataset/images/'
tripl_path = 'triplets.json'
anno_path = 'D:/Alessio/Provone/dataset/UCM_dataset/filenames/descriptions_UCM.txt'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
return_k = ['src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']

train_dataset = UCMTriplets(img_path, filenames, tripl_path, anno_path, model, tokenizer, return_keys=return_k, split='train')
feats_n = train_dataset.node_feats['1'][0].size(0)
model = CaptionGenerator(feats_n, train_dataset.max_capt_length, train_dataset.word2idx)
trainer = caption_trainer(model,train_dataset,'',collate_fn_captions, train_dataset.word2idx, train_dataset.max_capt_length, 'prova.pth', use_cuda=False, device='cpu')
trainer.fit(5, 0.01, 2, model._loss)