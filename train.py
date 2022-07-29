from torch.utils.data import DataLoader
from functools import partial

class classifier_trainer():
    '''
    Class to train the classifier of triplets on a dataset.
    The dataset should return the image as well as the ground truth triplets which are in the image
    '''
    def __init__(self, model, dataset, collate_fn, save_path):
        self.model = model
        self.dataset = dataset
        self.save_path = save_path
        self.collate_fn = collate_fn
    
    def fit(self, epochs, learning_rate, batch_size):
        # Define dataloader
        dataloader = DataLoader(self.dataset,batch_size=batch_size,shuffle=True,collate_fn=partial(self.collate_fn,triplet_to_idx=self.dataset.triplet_to_idx))
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                print(data.keys())
                break
            break