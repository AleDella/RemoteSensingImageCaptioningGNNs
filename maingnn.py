from dataset import UCMDataset, RSICDDataset, collate_fn_captions, collate_fn_classifier, collate_fn_full, augmented_collate_fn
from models import CaptionGenerator, TripletClassifier, AugmentedCaptionGenerator, FinalModel, MultiHeadClassifier, FinetunedModel
from train import caption_trainer, classifier_trainer, augmented_caption_trainer, full_pipeline_trainer, enc_finetuning
from eval import eval_captions, augmented_eval_captions, eval_classification, eval_pipeline
from graph_utils import save_plots
import torch
import os


def train_gnn(dataset, task, epochs, lr, batch_size, decoder, network_name, early_stopping, threshold, gnn, vir, depth, attributes, plot, combo):
    '''
    Function that initialize the training for the gnn depending on the task and dataset
    
    Args:
        dataset (str): dataset used for training
        task (str): type of desired task
        epochs (int): number of training epochs
        lr (float): learning rate to be used
        batch_size (int): batch size used for training
        decoder (str): decoder used for training
        network_name (str): name of the file to which the network will be saved
        early_stopping (bool): True if allow the use of early stopping; False otherwise
        threshold (int): number of epochs after which early stopping activates
    
    Return:
        None
    '''
    
    
    if task == "tripl2caption":
        # Dataset definition
        if dataset == 'ucm':
            train_filenames = 'dataset/UCM_dataset/filenames/filenames_train.txt'
            val_filenames = 'dataset/UCM_dataset/filenames/filenames_val.txt'
            img_path = 'dataset/UCM_dataset/images/'
            if attributes:
                polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm_attributes.json'
            else:
                polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm.json'
            anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
            word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
            graph_path = 'dataset/UCM_dataset/Graph_data'
            return_k = ['imgid', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            train_dataset = UCMDataset(img_path, train_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='train')
            val_dataset = UCMDataset(img_path, val_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='val')
        
        if dataset == 'rsicd':
            graph_path = 'dataset/RSICD_dataset/Graph_data'
            word2idx_path = 'dataset/RSICD_dataset/caption_dict_RSICD.json'
            anno_path = 'dataset/RSICD_dataset/polished_dataset.json'
            img_path = 'dataset/RSICD_dataset/RSICD_images'
            if attributes:
                tripl_path = 'dataset/RSICD_dataset/triplets_rsicd_attributes.json'
            else:
                tripl_path = 'dataset/RSICD_dataset/triplets_rsicd.json'
            return_k = ['imgid', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            train_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='train')
            val_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='val')
        
        # Network training part
        feats_n = torch.Tensor(train_dataset.node_feats[list(train_dataset.node_feats.keys())[0]])[0].size(0)
        max = train_dataset.max_capt_length
        if val_dataset.max_capt_length>max:
            max = val_dataset.max_capt_length
        model = CaptionGenerator(feats_n, max, train_dataset.word2idx, gnn=gnn, vir=vir, depth=depth, decoder=decoder)
        trainer = caption_trainer(model,train_dataset,val_dataset,collate_fn_captions, train_dataset.word2idx, max, network_name)
        trainer.fit(epochs, lr, batch_size, model._loss, early_stopping=early_stopping, tol_threshold=threshold)
    elif task == "img2tripl":
        if dataset == 'ucm':
            train_filenames = 'dataset/UCM_dataset/filenames/filenames_train.txt'
            val_filenames = 'dataset/UCM_dataset/filenames/filenames_val.txt'
            img_path = 'dataset/UCM_dataset/images/'
            if attributes:
                polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm_attributes.json'
            else:
                polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm.json'
            anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
            word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
            graph_path = 'dataset/UCM_dataset/Graph_data'
            return_k = ['image','triplets']
            img_dim = 256
            train_dataset = UCMDataset(img_path, train_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='train')
            val_dataset = UCMDataset(img_path, val_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='val')
        if dataset == 'rsicd':
            graph_path = 'dataset/RSICD_dataset/Graph_data'
            word2idx_path = 'dataset/RSICD_dataset/caption_dict_RSICD.json'
            anno_path = 'dataset/RSICD_dataset/polished_dataset.json'
            img_path = 'dataset/RSICD_dataset/RSICD_images'
            if attributes:
                tripl_path = 'dataset/RSICD_dataset/triplets_rsicd_attributes.json'
            else:
                tripl_path = 'dataset/RSICD_dataset/triplets_rsicd.json'
            return_k = ['image','triplets']
            img_dim = 224
            train_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='train')
            val_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='val')
        
        # copied from main.py
        model = MultiHeadClassifier(img_dim, len(train_dataset.triplet_to_idx))
        trainer = classifier_trainer(model,train_dataset,val_dataset,collate_fn_classifier, network_name)
        trainer.fit(epochs, lr, batch_size)
    elif task == "augmented_tripl2caption":
        if dataset == 'ucm':
            train_filenames = 'dataset/UCM_dataset/filenames/filenames_train.txt'
            val_filenames = 'dataset/UCM_dataset/filenames/filenames_val.txt'
            img_path = 'dataset/UCM_dataset/images/'
            if attributes:
                polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm_attributes.json'
            else:
                polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm.json'
            anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
            word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
            graph_path = 'dataset/UCM_dataset/Graph_data'
            return_k = ['imgid', 'image', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            img_dim = 256
            train_dataset = UCMDataset(img_path, train_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='train')
            val_dataset = UCMDataset(img_path, val_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='val')
        
        if dataset == 'rsicd':
            graph_path = 'dataset/RSICD_dataset/Graph_data'
            word2idx_path = 'dataset/RSICD_dataset/caption_dict_RSICD.json'
            anno_path = 'dataset/RSICD_dataset/polished_dataset.json'
            img_path = 'dataset/RSICD_dataset/RSICD_images'
            if attributes:
                tripl_path = 'dataset/RSICD_dataset/triplets_rsicd_attributes.json'
            else:
                tripl_path = 'dataset/RSICD_dataset/triplets_rsicd.json'
            return_k = ['imgid', 'image', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            img_dim = 224
            train_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='train')
            val_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='val')
        
        
        feats_n = torch.Tensor(train_dataset.node_feats[list(train_dataset.node_feats.keys())[0]])[0].size(0)
        max = train_dataset.max_capt_length
        if val_dataset.max_capt_length>max:
            max = val_dataset.max_capt_length
        img_encoder = TripletClassifier(img_dim,len(train_dataset.triplet_to_idx))
        model = AugmentedCaptionGenerator(img_encoder, feats_n, max, train_dataset.word2idx, gnn=gnn, vir=vir, depth=depth, decoder=decoder)
        trainer = augmented_caption_trainer(model,train_dataset,val_dataset, augmented_collate_fn, train_dataset.word2idx, max, network_name)
        trainer.fit(epochs, lr, batch_size, model._loss, early_stopping=early_stopping, tol_threshold=threshold)
    elif task == "img2caption":
        if dataset == 'ucm':
            train_filenames = 'dataset/UCM_dataset/filenames/filenames_train.txt'
            val_filenames = 'dataset/UCM_dataset/filenames/filenames_val.txt'
            img_path = 'dataset/UCM_dataset/images/'
            if attributes:
                polished_tripl_path = 'dataset/UCM_dataset/filtered_triplets_ucm_attributes.json'
            else:    
                polished_tripl_path = 'dataset/UCM_dataset/filtered_triplets_ucm.json'
            anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
            word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
            graph_path = 'dataset/UCM_dataset/Graph_data'
            return_k = ['imgid', 'image', 'triplets', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            img_dim = 256
            train_dataset = UCMDataset(img_path, train_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='train')
            val_dataset = UCMDataset(img_path, val_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='val')
        
        if dataset == 'rsicd':
            graph_path = 'dataset/RSICD_dataset/Graph_data'
            word2idx_path = 'dataset/RSICD_dataset/caption_dict_RSICD.json'
            anno_path = 'dataset/RSICD_dataset/polished_dataset.json'
            img_path = 'dataset/RSICD_dataset/RSICD_images'
            if attributes:
                tripl_path = 'dataset/RSICD_dataset/triplets_rsicd_attributes.json'
            else:
                tripl_path = 'dataset/RSICD_dataset/triplets_rsicd.json'
            return_k = ['imgid', 'image', 'triplets', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            img_dim = 224
            train_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='train')
            val_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='val')
        
        feats_n = torch.Tensor(train_dataset.node_feats[list(train_dataset.node_feats.keys())[0]])[0].size(0)
        max = train_dataset.max_capt_length
        if val_dataset.max_capt_length>max:
            max = val_dataset.max_capt_length
        img_encoder = TripletClassifier(img_dim,len(train_dataset.triplet_to_idx))
        model = FinalModel(img_encoder, feats_n, max, train_dataset.word2idx, img_dim, train_dataset.triplet_to_idx, gnn=gnn, vir=vir, depth=depth, decoder=decoder)
        trainer = full_pipeline_trainer(model,train_dataset,val_dataset, collate_fn_full, train_dataset.word2idx, max, network_name)
        if not plot:
            trainer.fit(epochs, lr, batch_size, model._loss, early_stopping=early_stopping, tol_threshold=threshold, plot=False, combo=combo)
        else:
            train_losses, val_losses = trainer.fit(epochs, lr, batch_size, model._loss, early_stopping=early_stopping, tol_threshold=threshold, plot=True, combo=combo)
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            save_plots(train_losses, val_losses, epochs, combo, gnn)
    # WIP
    elif task == 'finetune': 
        if dataset == 'ucm':
            train_filenames = 'dataset/UCM_dataset/filenames/filenames_train.txt'
            val_filenames = 'dataset/UCM_dataset/filenames/filenames_val.txt'
            img_path = 'dataset/UCM_dataset/images/'
            if attributes:
                polished_tripl_path = 'dataset/UCM_dataset/filtered_triplets_ucm_attributes.json'
            else:    
                polished_tripl_path = 'dataset/UCM_dataset/filtered_triplets_ucm.json'
            anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
            word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
            graph_path = 'dataset/UCM_dataset/Graph_data'
            return_k = ['imgid', 'image', 'triplets', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            img_dim = 256
            train_dataset = UCMDataset(img_path, train_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='train')
            val_dataset = UCMDataset(img_path, val_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='val')
        
        
        feats_n = torch.Tensor(train_dataset.node_feats[list(train_dataset.node_feats.keys())[0]])[0].size(0)
        max = train_dataset.max_capt_length
        if val_dataset.max_capt_length>max:
            max = val_dataset.max_capt_length
        model = FinetunedModel(train_dataset.word2idx, img_dim, train_dataset.triplet_to_idx, 'decoder.pth')
        trainer = enc_finetuning(model,train_dataset,val_dataset, collate_fn_full, train_dataset.word2idx, max, network_name)
        trainer.fit(epochs, lr, batch_size, model._loss, early_stopping=early_stopping, tol_threshold=threshold)
    else:
        print("Task not yet implemented.")
        
        
        
def test_gnn(dataset, task, decoder, network_name, filename, gnn, vir, depth, attributes):
    '''
    Function that initialize the training for the gnn depending on the task and dataset
    
    Args:
        dataset (str): dataset used for training
        task (str): type of desired task
        epochs (int): number of training epochs
        lr (float): learning rate to be used
        batch_size (int): batch size used for training
        decoder (str): decoder used for training
        network_name (str): name of the file to which the network will be saved
        early_stopping (bool): True if allow the use of early stopping; False otherwise
        threshold (int): number of epochs after which early stopping activates
    
    Return:
        None
    '''
    if task == "tripl2caption":
        # Dataset definition
        if dataset == 'ucm':
            test_filenames = 'dataset/UCM_dataset/filenames/filenames_test.txt'
            img_path = 'dataset/UCM_dataset/images/'
            polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm.json'
            anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
            word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
            graph_path = 'dataset/UCM_dataset/Graph_data'
            return_k = ['imgid', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            test_dataset = UCMDataset(img_path, test_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='test')
            
        if dataset == 'rsicd':
            graph_path = 'dataset/RSICD_dataset/Graph_data'
            word2idx_path = 'dataset/RSICD_dataset/caption_dict_RSICD.json'
            anno_path = 'dataset/RSICD_dataset/polished_dataset.json'
            img_path = 'dataset/RSICD_dataset/RSICD_images'
            tripl_path = 'dataset/RSICD_dataset/triplets_rsicd.json'
            return_k = ['imgid', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            test_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='test')
            
        # Network training part
        feats_n = torch.Tensor(test_dataset.node_feats[list(test_dataset.node_feats.keys())[0]])[0].size(0)
        max = test_dataset.max_capt_length
        model = CaptionGenerator(feats_n, max, test_dataset.word2idx, gnn=gnn, vir=vir, depth=depth, decoder=decoder)
        model = torch.load(network_name)
        eval_captions(test_dataset, model, filename)
    elif task == "augmented_tripl2caption":
        if dataset == 'ucm':
            test_filenames = 'dataset/UCM_dataset/filenames/filenames_test.txt'
            img_path = 'dataset/UCM_dataset/images/'
            polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm.json'
            anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
            word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
            graph_path = 'dataset/UCM_dataset/Graph_data'
            return_k = ['imgid', 'image', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            img_dim = 256
            test_dataset = UCMDataset(img_path, test_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='test')
            
        if dataset == 'rsicd':
            graph_path = 'dataset/RSICD_dataset/Graph_data'
            word2idx_path = 'dataset/RSICD_dataset/caption_dict_RSICD.json'
            anno_path = 'dataset/RSICD_dataset/polished_dataset.json'
            img_path = 'dataset/RSICD_dataset/RSICD_images'
            tripl_path = 'dataset/RSICD_dataset/triplets_rsicd.json'
            return_k = ['imgid', 'image', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            img_dim = 224
            test_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='test')
        
        feats_n = torch.Tensor(test_dataset.node_feats[list(test_dataset.node_feats.keys())[0]])[0].size(0)
        max = test_dataset.max_capt_length
        img_encoder = TripletClassifier(img_dim,len(test_dataset.triplet_to_idx))
        model = AugmentedCaptionGenerator(img_encoder, feats_n, max, test_dataset.word2idx, gnn=gnn, vir=vir, depth=depth, decoder=decoder)
        model = torch.load(network_name)
        augmented_eval_captions(test_dataset, model, filename)

    elif task == "img2tripl":
        # Dataset definition
        if dataset == 'ucm':
            test_filenames = 'dataset/UCM_dataset/filenames/filenames_test.txt'
            img_path = 'dataset/UCM_dataset/images/'
            polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm.json'
            anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
            word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
            graph_path = 'dataset/UCM_dataset/Graph_data'
            return_k = ['image','triplets']
            img_dim = 256
            test_dataset = UCMDataset(img_path, test_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='test')
            
        if dataset == 'rsicd':
            graph_path = 'dataset/RSICD_dataset/Graph_data'
            word2idx_path = 'dataset/RSICD_dataset/caption_dict_RSICD.json'
            anno_path = 'dataset/RSICD_dataset/polished_dataset.json'
            img_path = 'dataset/RSICD_dataset/RSICD_images'
            tripl_path = 'dataset/RSICD_dataset/triplets_rsicd.json'
            return_k = ['image','triplets']
            img_dim = 224
            test_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='test')
        
        model = TripletClassifier(img_dim,len(test_dataset.triplet_to_idx))
        model = torch.load(network_name)
        eval_classification(test_dataset, model, filename)
    elif task == "img2caption":
        if dataset == 'ucm':
            test_filenames = 'dataset/UCM_dataset/filenames/filenames_test.txt'
            img_path = 'dataset/UCM_dataset/images/'
            if attributes:
                polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm_attributes.json'
            else:
                polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm.json'
            anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
            word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
            graph_path = 'dataset/UCM_dataset/Graph_data'
            return_k = ['imgid', 'image', 'triplets', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            img_dim = 256
            test_dataset = UCMDataset(img_path, test_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='test')
        
        if dataset == 'rsicd':
            graph_path = 'dataset/RSICD_dataset/Graph_data'
            word2idx_path = 'dataset/RSICD_dataset/caption_dict_RSICD.json'
            anno_path = 'dataset/RSICD_dataset/polished_dataset.json'
            img_path = 'dataset/RSICD_dataset/RSICD_images'
            if attributes:
                tripl_path = 'dataset/RSICD_dataset/triplets_rsicd_attributes.json'
            else:
                tripl_path = 'dataset/RSICD_dataset/triplets_rsicd.json'
            return_k = ['imgid', 'image', 'triplets', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            img_dim = 224
            test_dataset = RSICDDataset(img_path, graph_path, tripl_path, anno_path, word2idx_path, return_k, split='test')
        
        feats_n = torch.Tensor(test_dataset.node_feats[list(test_dataset.node_feats.keys())[0]])[0].size(0)
        max = test_dataset.max_capt_length
        img_encoder = TripletClassifier(img_dim,len(test_dataset.triplet_to_idx))
        model = FinalModel(img_encoder, feats_n, max, test_dataset.word2idx, img_dim, test_dataset.triplet_to_idx, gnn=gnn, vir=vir, depth=depth, decoder=decoder)
        model = torch.load(network_name)
        eval_pipeline(test_dataset, model, filename)

    elif task == 'finetune': 
        if dataset == 'ucm':
            test_filenames = 'dataset/UCM_dataset/filenames/filenames_test.txt'
            img_path = 'dataset/UCM_dataset/images/'
            if attributes:
                polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm_attributes.json'
            else:
                polished_tripl_path = 'dataset/UCM_dataset/triplets_ucm.json'
            anno_path = 'dataset/UCM_dataset/filenames/descriptions_UCM.txt'
            word2idx_path = 'dataset/UCM_dataset/caption_dict_UCM.json'
            graph_path = 'dataset/UCM_dataset/Graph_data'
            return_k = ['imgid', 'image', 'triplets', 'src_ids', 'dst_ids', 'node_feats', 'captions', 'num_nodes']
            img_dim = 256
            test_dataset = UCMDataset(img_path, test_filenames, graph_path, polished_tripl_path, anno_path, word2idx_path, return_keys=return_k, split='test')
        
        
        
        feats_n = torch.Tensor(test_dataset.node_feats[list(test_dataset.node_feats.keys())[0]])[0].size(0)
        max = test_dataset.max_capt_length
        model = FinetunedModel(test_dataset.word2idx, img_dim, test_dataset.triplet_to_idx, 'decoder.pth')
        model = torch.load(network_name)
        # try:
        #     model = torch.load(network_name)
        # except:
        #     print("Model name not found.")
        #     exit(0)
        eval_pipeline(test_dataset, model, filename)
    else:
        print("Task not yet implemented.")