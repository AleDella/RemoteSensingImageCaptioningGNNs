from torch.utils.data import DataLoader
from torchmetrics.functional import f1_score
from functools import partial
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch
import dgl
from graph_utils import get_node_features, tripl2graph, tripl2graphw
from transformers import BertModel, BertTokenizer


def multitask_loss(criterion, outputs, targets):
    '''
    Function that computes the loss of a multi-head classification problem
    
    Args:
        criterion (torch.nn.Module): loss criterion
        outputs (torch.Tensor): outputs of the network (batch_size, number_of_tasks, labels_tasks)
        targets (torch.Tensor): targtet labels for each task (batch_szie, number_of_tasks)
        
    Return:
        losses (torch.Tensor): losses for each task (1, batch_size)
    '''
    
    losses = torch.hstack([
      criterion(t_input, t_target.long().flatten().clone().detach())
      for t_input, t_target in zip(outputs, targets)
    ])
    
    return losses


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
        criterion = nn.CrossEntropyLoss()
        # Define the optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        if(self.use_cuda):
            self.model = self.model.to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss_train = 0
            accuracy_train = 0
            epoch_loss_val = 0
            accuracy_val = 0 
            print('Epoch: '+str(epoch))
            for i, data in enumerate(tqdm(trainloader)):
                images, triplets = data
                images = images.to(self.device)
                triplets = triplets.to(self.device)
                outputs = self.model(images)
                # Reshape with the right size
                outputs = outputs.reshape((outputs.shape[0], int(outputs.shape[1]/2), 2))
                # Now must create the loss function
                loss = multitask_loss(criterion, outputs, triplets).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Calculate accuracy on training
                outputs = torch.sigmoid(outputs)
                outputs = torch.tensor([[torch.argmax(task).item() for task in sample ] for sample in outputs]).to(outputs.device)
                accuracy = f1_score(outputs, triplets.long(), num_classes=2, mdmc_average='global')
                accuracy_train += accuracy
                epoch_loss_train+=loss.item()
            
            with torch.no_grad():
                self.model.eval()
                for j, data in enumerate(tqdm(valloader)):
                    images, triplets = data
                    images = images.to(self.device)
                    triplets = triplets.to(self.device)
                    outputs = self.model(images)
                    # Reshape with the right size
                    outputs = outputs.reshape((outputs.shape[0], int(outputs.shape[1]/2), 2))
                    # Now must create the loss function
                    loss = multitask_loss(criterion, outputs, triplets).mean()
                    # Calculate accuracy on training
                    outputs = torch.sigmoid(outputs)
                    outputs = torch.tensor([[torch.argmax(task).item() for task in sample ] for sample in outputs]).to(outputs.device)
                    
                    accuracy = f1_score(outputs, triplets.long(), num_classes=2, mdmc_average='global')
                    accuracy_val += accuracy
                    epoch_loss_val+=loss.item()
                    
            
            print('Training loss: {:.3f}'.format(epoch_loss_train/i))
            print('Validation loss: {:.3f}'.format(epoch_loss_val/j))
            print('Training accuracy: {:.3f}'.format(accuracy_train/i))
            print('Validation accuracy: {:.3f}'.format(accuracy_val/j))
        
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
    
    def fit(self, epochs, learning_rate, batch_size, criterion, early_stopping=False, tol_threshold=5):
        # Define dataloader
        trainloader = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, collate_fn=partial(self.collate_fn, word2idx=self.word2idx, training=True))
        if self.dataset_val!='':
            valloader = DataLoader(self.dataset_val,batch_size=1,shuffle=False,collate_fn=partial(self.collate_fn, word2idx=self.word2idx, training=True))
        # Define the optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        if(self.use_cuda):
            self.model = self.model.to(self.device)
        if early_stopping:
            val_max = float('inf')
            train_max = float('inf')
            tollerance = 0
        for epoch in range(epochs):
            self.model.train()
            epoch_loss_train = 0
            epoch_loss_val = 0
            print('Epoch: '+str(epoch))
            for i, data in enumerate(tqdm(trainloader)):
                _, captions, encoded_captions, src_ids, dst_ids, node_feats, num_nodes = data
                graphs = dgl.batch([dgl.graph((src_id, dst_id)) for src_id, dst_id in zip(src_ids, dst_ids)]).to(self.device)
                feats = get_node_features(node_feats, sum(num_nodes)).to(self.device)
                outputs = self.model(graphs, feats, encoded_captions)
                optimizer.zero_grad()
                loss = criterion(outputs, captions, self.word2idx, encoded_captions.size(1), self.device)
                loss.backward()
                optimizer.step()
                epoch_loss_train+=loss.item()
            if self.dataset_val!='':
                with torch.no_grad():
                    self.model.eval()
                    for j, data in enumerate(tqdm(valloader)):
                        _, captions, encoded_captions, src_ids, dst_ids, node_feats, num_nodes = data
                        graphs = dgl.batch([dgl.graph((src_id, dst_id)) for src_id, dst_id in zip(src_ids, dst_ids)]).to(self.device)
                        feats = get_node_features(node_feats, sum(num_nodes)).to(self.device)
                        outputs = self.model(graphs, feats, encoded_captions)
                        loss = criterion(outputs, captions, self.word2idx, encoded_captions.size(1), self.device)
                        epoch_loss_val+=loss.item()
                    
            print('Training loss: {:.3f}'.format(epoch_loss_train/i))
            if self.dataset_val!='':
                print('Validation loss: {:.3f}'.format(epoch_loss_val/j))
            if early_stopping:
                if ((epoch_loss_val/j) < val_max) and ((epoch_loss_train/i < train_max)) and tollerance<tol_threshold :
                    val_max = epoch_loss_val/j
                    train_max = epoch_loss_train/i
                    best_model=self.model
                else:
                    tollerance+=1
                    if tollerance>tol_threshold:
                        print("Stopped training due to overfit")
                        break
                    # Restart from the best checkpoint
                    self.model = best_model
        
        if early_stopping:
            torch.save(best_model,self.save_path)
        else:
            torch.save(self.model,self.save_path)
            
            
class augmented_caption_trainer():
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
    
    def fit(self, epochs, learning_rate, batch_size, criterion, early_stopping=False, tol_threshold=5):
        # Define dataloader
        trainloader = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, collate_fn=partial(self.collate_fn, word2idx=self.word2idx, training=True))
        if self.dataset_val!='':
            valloader = DataLoader(self.dataset_val,batch_size=1,shuffle=False,collate_fn=partial(self.collate_fn, word2idx=self.word2idx, training=True))
        # Define the optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        if(self.use_cuda):
            self.model = self.model.to(self.device)
        if early_stopping:
            val_max = float('inf')
            train_max = float('inf')
            tollerance = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss_train = 0
            epoch_loss_val = 0
            print('Epoch: '+str(epoch))
            for i, data in enumerate(tqdm(trainloader)):
                _, img, captions, encoded_captions, src_ids, dst_ids, node_feats, num_nodes = data
                graphs = dgl.batch([dgl.graph((src_id, dst_id)) for src_id, dst_id in zip(src_ids, dst_ids)]).to(self.device)
                feats = get_node_features(node_feats, sum(num_nodes)).to(self.device)
                img = img.to(self.device)
                outputs = self.model(graphs, feats, img, encoded_captions)
                optimizer.zero_grad()
                loss = criterion(outputs, captions, self.word2idx, encoded_captions.size(1), self.device)
                loss.backward()
                optimizer.step()
                epoch_loss_train+=loss.item()
            if self.dataset_val!='':
                with torch.no_grad():
                    self.model.eval()
                    for j, data in enumerate(tqdm(valloader)):
                        _, img, captions, encoded_captions, src_ids, dst_ids, node_feats, num_nodes = data
                        graphs = dgl.batch([dgl.graph((src_id, dst_id)) for src_id, dst_id in zip(src_ids, dst_ids)]).to(self.device)
                        feats = get_node_features(node_feats, sum(num_nodes)).to(self.device)
                        img = img.to(self.device)
                        outputs = self.model(graphs, feats, img, encoded_captions)
                        loss = criterion(outputs, captions, self.word2idx, encoded_captions.size(1), self.device)
                        epoch_loss_val+=loss.item()
                    
            print('Training loss: {:.3f}'.format(epoch_loss_train/i))
            if self.dataset_val!='':
                print('Validation loss: {:.3f}'.format(epoch_loss_val/j))
            if early_stopping:
                if ((epoch_loss_val/j) < val_max) and ((epoch_loss_train/i < train_max)) and tollerance<tol_threshold :
                    val_max = epoch_loss_val/j
                    train_max = epoch_loss_train/i
                    best_model=self.model
                else:
                    tollerance+=1
                    if tollerance>tol_threshold:
                        print("Stopped training due to overfit")
                        break
                    # Restart from the best checkpoint
                    self.model = best_model
        
        if early_stopping:
            torch.save(best_model,self.save_path)
        else:
            torch.save(self.model,self.save_path)
            

class full_pipeline_trainer():
    '''
    Class to train full pipeline on a dataset.
    The dataset should return the image as well as the ground truth triplets which are in the image
    '''
    def __init__(self, model, dataset_train, dataset_val, collate_fn, word2idx, max_capt_length, save_path, use_cuda=True, device = None, pil=False):
        self.pil = pil
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.save_path = save_path
        self.collate_fn = collate_fn
        self.word2idx = word2idx
        self.max_capt_length = max_capt_length
        self.use_cuda = use_cuda
        if device is None:
            # Take default cuda:0 
            device = torch.device("cuda:0")
        self.device = device
    
    def fit(self, epochs, learning_rate, batch_size, criterion, early_stopping=False, tol_threshold=5, plot=False, combo=False):
        # For plotting purposes
        if plot:
            train_losses = []
            val_losses = []
        # Define dataloader
        trainloader = DataLoader(self.dataset_train,batch_size=batch_size,shuffle=True,collate_fn=partial(self.collate_fn,triplet_to_idx=self.dataset_train.triplet_to_idx, word2idx=self.word2idx, training=True, pil=self.pil))
        valloader = DataLoader(self.dataset_val,batch_size=1,shuffle=False,collate_fn=partial(self.collate_fn,triplet_to_idx=self.dataset_train.triplet_to_idx, word2idx=self.word2idx, training=True, pil=self.pil))
        # Define the optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate) 
        if(self.use_cuda):
            self.model = self.model.to(self.device)
        # Early stopping
        if early_stopping:
            val_max = float('inf')
            train_max = float('inf')
            tollerance = 0
        
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss_train = 0
            epoch_loss_val = 0
            print('Epoch: '+str(epoch))
            for i, data in enumerate(tqdm(trainloader)):
                _, images, triplets, captions, encoded_captions, lengths, _, _, _, _ = data
                images = images.to(self.device)
                triplets = triplets.to(self.device)
                inputs = encoded_captions[:,:-1]
                lengths = [lengths-1 for lengths in lengths]
                if combo:
                    # Combined Loss
                    cap_outputs, class_outputs = self.model(images, inputs, lengths, True)
                    class_loss = multitask_loss(nn.CrossEntropyLoss(), class_outputs, triplets).mean()
                    cap_loss = criterion(cap_outputs, captions, lengths, self.word2idx, encoded_captions.shape[1] , self.device)
                    loss = 0.5*cap_loss + 0.5*class_loss
                else:
                    # Unique Loss
                    cap_outputs, _ = self.model(images, captions, inputs, lengths, True)
                    cap_loss = criterion(cap_outputs, captions, lengths, self.word2idx, encoded_captions.shape[1], self.device)
                    loss = cap_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss_train+=loss.item()
                # print("\nFirst cicle done!")
                # exit(0)
            
            with torch.no_grad():
                self.model.eval()
                for j, data in enumerate(tqdm(valloader)):
                    _, images, triplets, captions, encoded_captions, lengths, _, _, _, _ = data
                    images = images.to(self.device)
                    triplets = triplets.to(self.device)
                    if combo:
                        # Combined Loss
                        try:
                            cap_outputs, class_outputs = self.model(images)
                        except:
                            cap_outputs, class_outputs = self.model(images, captions, encoded_captions, lengths, training=False)
                        class_loss = multitask_loss(nn.CrossEntropyLoss(), class_outputs, triplets).mean()
                        cap_loss = criterion(cap_outputs, captions, lengths, self.word2idx, encoded_captions.shape[1] , self.device)
                        loss = 0.5*cap_loss + 0.5*class_loss
                    else:
                        # Unified Loss
                        try:
                            cap_outputs, _ = self.model(images)
                        except:
                            cap_outputs, _ = self.model(images, captions, encoded_captions, lengths, training=False)
                        cap_loss = criterion(cap_outputs, captions, lengths, self.word2idx, encoded_captions.shape[1] , self.device)
                        loss = cap_loss
                    epoch_loss_val+=loss.item()
                    
            
            print('Training loss: {:.3f}'.format(epoch_loss_train/i))
            print('Validation loss: {:.3f}'.format(epoch_loss_val/j))
            
            # Saving the losses for plotting purposes
            if plot:
                train_losses.append(epoch_loss_train/i)
                val_losses.append(epoch_loss_val/j)
            # Early stopping algorithm
            if early_stopping:
                if ((epoch_loss_val/j) < val_max) and ((epoch_loss_train/i < train_max)) and tollerance<tol_threshold :
                    val_max = epoch_loss_val/j
                    train_max = epoch_loss_train/i
                    best_model=self.model
                else:
                    tollerance+=1
                    if tollerance>tol_threshold:
                        print("Stopped training due to overfit")
                        break
                    # Restart from the best checkpoint
                    self.model = best_model
        
        if early_stopping:
            torch.save(best_model,self.save_path)
            if plot:
                return train_losses, val_losses
        else:
            torch.save(self.model,self.save_path)
            if plot:
                return train_losses, val_losses
            
            
class enc_finetuning():
    '''
    Class to train full pipeline on a dataset.
    The dataset should return the image as well as the ground truth triplets which are in the image
    '''
    def __init__(self, model, dataset_train, dataset_val, collate_fn, word2idx, max_capt_length, save_path, use_cuda=True, device = None):
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.save_path = save_path
        self.collate_fn = collate_fn
        self.word2idx = word2idx
        self.max_capt_length = max_capt_length
        self.use_cuda = use_cuda
        if device is None:
            # Take default cuda:0 
            device = torch.device("cuda:0")
        self.device = device
    
    def fit(self, epochs, learning_rate, batch_size, criterion, early_stopping=False, tol_threshold=5):
        # Define dataloader
        trainloader = DataLoader(self.dataset_train,batch_size=batch_size,shuffle=True,collate_fn=partial(self.collate_fn,triplet_to_idx=self.dataset_train.triplet_to_idx, word2idx=self.word2idx, training=True))
        valloader = DataLoader(self.dataset_val,batch_size=1,shuffle=False,collate_fn=partial(self.collate_fn,triplet_to_idx=self.dataset_train.triplet_to_idx, word2idx=self.word2idx, training=True))
        # Define the optimizer
        final_layer_weights = []
        rest_of_the_net_weights = []
        # final_names = []
        # rest_names = []
        # we will iterate through the layers of the network
        for name, param in self.model.named_parameters():
            if not name.startswith('feature_encoder'):
                if name.startswith('tripl_classifier') and 'fc' in name:
                    final_layer_weights.append(param)
                    # final_names.append(name)
                else:
                    rest_of_the_net_weights.append(param)
                    # rest_names.append(name)
        
        # so now we have divided the network weights into two groups.
        # We will train the final_layer_weights with learning_rate = lr
        # and rest_of_the_net_weights with learning_rate = lr / 10
        
        optimizer = torch.optim.AdamW([
            {'params': rest_of_the_net_weights},
            {'params': final_layer_weights, 'lr': learning_rate}
        ], lr=learning_rate / 10)
        
        if(self.use_cuda):
            self.model = self.model.to(self.device)
        # Early stopping
        if early_stopping:
            val_max = float('inf')
            train_max = float('inf')
            tollerance = 0
        
        # crit = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss_train = 0
            epoch_loss_val = 0
            print('Epoch: '+str(epoch))
            for i, data in enumerate(tqdm(trainloader)):
                _, images, triplets, captions, encoded_captions, _, _, _, _ = data
                images = images.to(self.device)
                cap_outputs, class_outputs = self.model(images)
                # Loss for both tasks
                triplets = triplets.to(self.device)
                class_loss = multitask_loss(nn.CrossEntropyLoss(), class_outputs, triplets).mean()
                # class_loss = crit(class_outputs, triplets)
                cap_loss = criterion(cap_outputs, captions, self.word2idx, encoded_captions.size(1), self.device)
                loss = 0.5*cap_loss + 0.5*class_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss_train+=loss.item()
            
            with torch.no_grad():
                self.model.eval()
                for j, data in enumerate(tqdm(valloader)):
                    _, images, triplets, captions, encoded_captions, _, _, _, _ = data
                    images = images.to(self.device)
                    cap_outputs, class_outputs = self.model(images)
                    triplets = triplets.to(self.device)
                    class_loss = multitask_loss(nn.CrossEntropyLoss(), class_outputs, triplets).mean()
                    # class_loss = crit(class_outputs, triplets)
                    cap_loss = criterion(cap_outputs, captions, self.word2idx, encoded_captions.size(1), self.device)
                    loss = 0.5*cap_loss + 0.5*class_loss
                    epoch_loss_val+=loss.item()
                    
            
            print('Training loss: {:.3f}'.format(epoch_loss_train/i))
            print('Validation loss: {:.3f}'.format(epoch_loss_val/j))
            if early_stopping:
                if ((epoch_loss_val/j) < val_max) and ((epoch_loss_train/i < train_max)) and tollerance<tol_threshold :
                    val_max = epoch_loss_val/j
                    train_max = epoch_loss_train/i
                    best_model=self.model
                else:
                    tollerance+=1
                    if tollerance>tol_threshold:
                        print("Stopped training due to overfit")
                        break
                    # Restart from the best checkpoint
                    self.model = best_model
        
        if early_stopping:
            torch.save(best_model,self.save_path)
        else:
            torch.save(self.model,self.save_path)
            
            

class waterfall_trainer():
    '''
    Class to train the Waterfall pipeline using a caption generator pre-trained on UCM.
    The dataset should return the graphs, captions, max_sequence_length and word2idx dictionary
    '''
    def __init__(self, model, dataset_train, dataset_val, collate_fn, word2idx, max_capt_length, save_path, use_cuda=True, device = None, pil=False):
        self.pil = pil
        self.model = model
        self.feature_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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
    
    def fit(self, epochs, learning_rate, batch_size, criterion, early_stopping=False, tol_threshold=5, plot=False):
        # For plotting purposes
        if plot:
            train_losses = []
            val_losses = []
        # Define dataloader
        trainloader = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, collate_fn=partial(self.collate_fn, word2idx=self.word2idx, training=True, pil=self.pil))
        valloader = DataLoader(self.dataset_val,batch_size=1,shuffle=False,collate_fn=partial(self.collate_fn, word2idx=self.word2idx, training=True, pil=self.pil))
        # Define the optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        if(self.use_cuda):
            self.model = self.model.to(self.device)
        if early_stopping:
            val_max = float('inf')
            train_max = float('inf')
            tollerance = 0
        for epoch in range(epochs):
            self.model.train()
            epoch_loss_train = 0
            epoch_loss_val = 0
            print('Epoch: '+str(epoch))
            for i, data in enumerate(tqdm(trainloader)):
                imgid, img, triplets, captions, encoded_captions, lengths = data
                graphs, graph_feats = tripl2graphw(triplets, self.feature_encoder, self.tokenizer)
                graphs, graph_feats = graphs.to(self.device), graph_feats.to(self.device)
                outputs = self.model(graphs, graph_feats, encoded_captions, lengths, training=True)
                optimizer.zero_grad()
                loss = criterion(outputs, captions, lengths, self.word2idx, encoded_captions.size(1), self.device)
                loss.backward()
                optimizer.step()
                epoch_loss_train+=loss.item()
            if self.dataset_val!='':
                with torch.no_grad():
                    self.model.eval()
                    for j, data in enumerate(tqdm(valloader)):
                        imgid, img, triplets, captions, encoded_captions, lengths = data
                        graphs, graph_feats = tripl2graphw(triplets, self.feature_encoder, self.tokenizer)
                        graphs, graph_feats = graphs.to(self.device), graph_feats.to(self.device)
                        # img = img.to(self.device)
                        outputs = self.model(graphs, graph_feats, encoded_captions, lengths, training=False)
                        loss = criterion(outputs, captions, lengths, self.word2idx, encoded_captions.size(1), self.device)
                        epoch_loss_val+=loss.item()
                    
            print('Training loss: {:.3f}'.format(epoch_loss_train/i))
            print('Validation loss: {:.3f}'.format(epoch_loss_val/j))
            # Saving the losses for plotting purposes
            if plot:
                train_losses.append(epoch_loss_train/i)
                val_losses.append(epoch_loss_val/j)
            if early_stopping:
                if ((epoch_loss_val/j) < val_max) and ((epoch_loss_train/i < train_max)) and tollerance<tol_threshold :
                    val_max = epoch_loss_val/j
                    train_max = epoch_loss_train/i
                    best_model=self.model
                else:
                    tollerance+=1
                    if tollerance>tol_threshold:
                        print("Stopped training due to overfit")
                        break
                    # Restart from the best checkpoint
                    self.model = best_model
        
        if early_stopping:
            torch.save(best_model,self.save_path)
        else:
            torch.save(self.model,self.save_path)
        if plot:
                return train_losses, val_losses