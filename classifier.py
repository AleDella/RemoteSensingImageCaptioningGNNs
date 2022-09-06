from dataset import UCMDataset, collate_fn_classifier
from models import TripletClassifier, load_model 
from train import classifier_trainer
from transformers import BertTokenizer, BertModel
import torch

filenames_train = 'D:/Alessio/Provone/dataset/UCM_dataset/filenames/filenames_train.txt'
filenames_val = 'D:/Alessio/Provone/dataset/UCM_dataset/filenames/filenames_val.txt'

img_path = 'D:/Alessio/Provone/dataset/UCM_dataset/images/'
tripl_path = 'triplets.json'
anno_path = 'D:/Alessio/Provone/dataset/UCM_dataset/filenames/descriptions_UCM.txt'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

return_keys = ['image','triplets']

trainset = UCMDataset(img_path, filenames_train, tripl_path, anno_path, model, tokenizer, return_keys, split='train')
valset = UCMDataset(img_path, filenames_val, tripl_path, anno_path, model, tokenizer, return_keys, split='val')

model = TripletClassifier(256,trainset.unique_triplets)

trainer = classifier_trainer(model,trainset,valset,collate_fn_classifier,'model_finetuned.pth')

#trainer.fit(50,0.001,8)

# Load old model
model_path = 'model.pth'

model = load_model(model_path)

trainer.finetune(model,10,0.0001,8)