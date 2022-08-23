import argparse

from maingnn import train_gnn

parser = argparse.ArgumentParser(description="Extract triplets from dataset's captions.")
parser.add_argument('--dataset', type=str,   default='ucm',           required=False, help='Name of the dataset of which you want to create the triplets.')
parser.add_argument('--task',    type=str,   default='tripl2caption', required=True,  help='Name of the selected task. Possible ones: "tripl2caption", "img2tripl", "img2caption"') 
parser.add_argument('--e',       type=int,   default=10,              required=False, help='Number of Epochs.')
parser.add_argument('--lr',      type=float, default=0.0001,          required=False, help='Learning rate.')
parser.add_argument('--bs',      type=int,   default=8,               required=False, help='Batch size.')
parser.add_argument('--decoder', type=str,   default="linear",        required=False, help='Type of decoder used for gnn.')
parser.add_argument('--name',    type=str,   default="GNN.pth",       required=False, help='Name for the network.')
parser.add_argument('--es',      action='store_true',                 required=False, help='Usage of early stopping.')
parser.add_argument('--thresh',  type=int,   default=1,               required=False, help='Number of epochs before early stopping.')



if __name__=="__main__":
    args = parser.parse_args()
    train_gnn(args.dataset,
              args.task,
              args.e,
              args.lr,
              args.bs,
              args.decoder,
              args.name,
              args.es,
              args.thresh
              )