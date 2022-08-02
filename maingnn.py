from dataset import UCMTriplets, collate_fn_captions
from transformers import BertTokenizer, BertModel
from models import CaptionGenerator
from train import caption_trainer

filenames = 'filenames_train.txt'
img_path = 'test_images/'
tripl_path = 'example_tripl.json'
anno_path = 'example_anno.txt'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
return_k = ['src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']

dataset = UCMTriplets(img_path, filenames, tripl_path, anno_path, model, tokenizer, return_keys=return_k, split='train')
feats_n = dataset.node_feats['1'][0].size(0)
model = CaptionGenerator(feats_n, dataset.max_capt_length, dataset.word2idx)
trainer = caption_trainer(model,dataset,'',collate_fn_captions, dataset.word2idx, dataset.max_capt_length, 'prova.pth', use_cuda=False, device='cpu')
trainer.fit(5, 0.01, 2, model._loss)