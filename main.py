from dataset import UCMTriplets, collate_fn_classifier
from models import TripletClassifier
from train import classifier_trainer
from transformers import BertTokenizer, BertModel

filenames = 'D:/Alessio/Provone/dataset/UCM_dataset/filenames/filenames_test.txt'
img_path = 'D:/Alessio/Provone/dataset/UCM_dataset/images/'
tripl_path = 'triplets.json'
anno_path = 'D:/Alessio/Provone/dataset/UCM_dataset/filenames/descriptions_UCM.txt'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

return_keys = ['image','triplets','captions']

dataset = UCMTriplets(img_path, filenames, tripl_path, anno_path, model, tokenizer, return_keys, split='test')

model = TripletClassifier(224,dataset.unique_triplets)

trainer = classifier_trainer(model,dataset,collate_fn_classifier,'model.pth')

trainer.fit(10,0.1,10)