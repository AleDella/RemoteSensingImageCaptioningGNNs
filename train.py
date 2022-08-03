from torch.utils.data import DataLoader
from functools import partial
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch
import dgl
from graph_utils import get_node_features

class classifier_trainer():
    '''
    Class to train the classifier of triplets on a dataset.
    The dataset should return the image as well as the ground truth triplets which are in the image
    '''
    def __init__(self, model, dataset_train, dataset_val, collate_fn, save_path, use_cuda=True, device = None):
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.save_path = save_path
        self.collate_fn = collate_fn
        self.use_cuda = use_cuda
        if device is None:
            # Take default cuda:0 
            device = torch.device("cuda:0")
        self.device = device
    
    def fit(self, epochs, learning_rate, batch_size):
        # Define dataloader
        trainloader = DataLoader(self.dataset_train,batch_size=batch_size,shuffle=True,collate_fn=partial(self.collate_fn,triplet_to_idx=self.dataset_train.triplet_to_idx))
        valloader = DataLoader(self.dataset_val,batch_size=1,shuffle=False,collate_fn=partial(self.collate_fn,triplet_to_idx=self.dataset_train.triplet_to_idx))
        # Define the criterion
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        # Define the optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        if(self.use_cuda):
            self.model = self.model.to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss_train = 0
            epoch_loss_val = 0
            print('Epoch: '+str(epoch))
            for i, data in enumerate(tqdm(trainloader)):
                images, triplets = data
                images = images.to(self.device)
                triplets = triplets.to(self.device)
                outputs = self.model(images)
                optimizer.zero_grad()
                loss = criterion(outputs,triplets)
                loss.backward()
                optimizer.step()
                epoch_loss_train+=loss.item()
            
            with torch.no_grad():
                self.model.eval()
                for j, data in enumerate(tqdm(valloader)):
                    images, triplets = data
                    images = images.to(self.device)
                    triplets = triplets.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs,triplets)
                    epoch_loss_val+=loss.item()
                    
            
            print('Training loss: {:.3f}'.format(epoch_loss_train/i))
            print('Validation loss: {:.3f}'.format(epoch_loss_val/j))
        
        torch.save(self.model,self.save_path)
    
    def finetune(self, model, epochs, learning_rate, batch_size):
        self.model = model
        # Set all the parameters to trainable 
        for param in self.model.parameters():
            param.requires_grad = True
            
        trainloader = DataLoader(self.dataset_train,batch_size=batch_size,shuffle=True,collate_fn=partial(self.collate_fn,triplet_to_idx=self.dataset_train.triplet_to_idx))
        valloader = DataLoader(self.dataset_val,batch_size=1,shuffle=False,collate_fn=partial(self.collate_fn,triplet_to_idx=self.dataset_train.triplet_to_idx))
        # Define the criterion
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        # Define the optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        if(self.use_cuda):
            self.model = self.model.to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss_train = 0
            epoch_loss_val = 0
            print('Epoch: '+str(epoch))
            for i, data in enumerate(tqdm(trainloader)):
                images, triplets = data
                images = images.to(self.device)
                triplets = triplets.to(self.device)
                outputs = self.model(images)
                optimizer.zero_grad()
                loss = criterion(outputs,triplets)
                loss.backward()
                optimizer.step()
                epoch_loss_train+=loss.item()
            
            with torch.no_grad():
                self.model.eval()
                for j, data in enumerate(tqdm(valloader)):
                    images, triplets = data
                    images = images.to(self.device)
                    triplets = triplets.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs,triplets)
                    epoch_loss_val+=loss.item()
                    
            
            print('Training loss: {:.3f}'.format(epoch_loss_train/i))
            print('Validation loss: {:.3f}'.format(epoch_loss_val/j))
        
        torch.save(self.model,self.save_path)




class caption_trainer():
    '''
    Class to train the caption inference on a dataset.
    The dataset should return the graphs, captions, max_sequence_length and word2idx dictionary
    '''
    def __init__(self, model, dataset_train, dataset_val, collate_fn, word2idx, max_capt_length, save_path, use_cuda=True, device = None):
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.save_path = save_path
        self.collate_fn = collate_fn
        self.use_cuda = use_cuda
        self.word2idx = word2idx
        self.max_capt_length = max_capt_length
        if device is None:
            # Take default cuda:0 
            device = torch.device("cuda:0")
        self.device = device
    
    def fit(self, epochs, learning_rate, batch_size, criterion):
        # Define dataloader
        trainloader = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        if self.dataset_val!='':
            valloader = DataLoader(self.dataset_val,batch_size=1,shuffle=False,collate_fn=self.collate_fn)
        # Define the optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        if(self.use_cuda):
            self.model = self.model.to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss_train = 0
            epoch_loss_val = 0
            print('Epoch: '+str(epoch))
            for i, data in enumerate(tqdm(trainloader)):
                captions, src_ids, dst_ids, node_feats, num_nodes = data
                graphs = dgl.batch([dgl.graph((src_id, dst_id)) for src_id, dst_id in zip(src_ids, dst_ids)]).to(self.device)
                feats = get_node_features(node_feats, sum(num_nodes)).to(self.device)
                outputs = self.model(graphs, feats, captions)
                optimizer.zero_grad()
                loss = criterion(outputs, captions, self.word2idx, self.max_capt_length, self.device)
                loss.backward()
                optimizer.step()
                epoch_loss_train+=loss.item()
            if self.dataset_val!='':
                with torch.no_grad():
                    self.model.eval()
                    for j, data in enumerate(tqdm(valloader)):
                        captions, src_ids, dst_ids, node_feats, num_nodes = data
                        graphs = dgl.batch([dgl.graph((src_id, dst_id)) for src_id, dst_id in zip(src_ids, dst_ids)]).to(self.device)
                        feats = get_node_features(node_feats, sum(num_nodes)).to(self.device)
                        outputs = self.model(graphs, feats, captions)
                        loss = criterion(outputs, captions, self.word2idx, self.max_capt_length, self.device)
                        epoch_loss_val+=loss.item()
                    
            print('Training loss: {:.3f}'.format(epoch_loss_train/i))
            if self.dataset_val!='':
                print('Validation loss: {:.3f}'.format(epoch_loss_val/j))
        
        torch.save(self.model,self.save_path)
    
    # def finetune(self, model, epochs, learning_rate, batch_size):
    #     self.model = model
    #     # Set all the parameters to trainable 
    #     for param in self.model.parameters():
    #         param.requires_grad = True
            
    #     trainloader = DataLoader(self.dataset_train,batch_size=batch_size,shuffle=True,collate_fn=partial(self.collate_fn,triplet_to_idx=self.dataset_train.triplet_to_idx))
    #     valloader = DataLoader(self.dataset_val,batch_size=1,shuffle=False,collate_fn=partial(self.collate_fn,triplet_to_idx=self.dataset_train.triplet_to_idx))
    #     # Define the criterion
    #     criterion = nn.BCEWithLogitsLoss(reduction='mean')
    #     # Define the optimizer
    #     optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
    #     if(self.use_cuda):
    #         self.model = self.model.to(self.device)
        
    #     for epoch in range(epochs):
    #         self.model.train()
    #         epoch_loss_train = 0
    #         epoch_loss_val = 0
    #         print('Epoch: '+str(epoch))
    #         for i, data in enumerate(tqdm(trainloader)):
    #             images, triplets = data
    #             images = images.to(self.device)
    #             triplets = triplets.to(self.device)
    #             outputs = self.model(images)
    #             optimizer.zero_grad()
    #             loss = criterion(outputs,triplets)
    #             loss.backward()
    #             optimizer.step()
    #             epoch_loss_train+=loss.item()
            
    #         with torch.no_grad():
    #             self.model.eval()
    #             for j, data in enumerate(tqdm(valloader)):
    #                 images, triplets = data
    #                 images = images.to(self.device)
    #                 triplets = triplets.to(self.device)
    #                 outputs = self.model(images)
    #                 loss = criterion(outputs,triplets)
    #                 epoch_loss_val+=loss.item()
                    
            
    #         print('Training loss: {:.3f}'.format(epoch_loss_train/i))
    #         print('Validation loss: {:.3f}'.format(epoch_loss_val/j))
        
    #     torch.save(self.model,self.save_path)
        