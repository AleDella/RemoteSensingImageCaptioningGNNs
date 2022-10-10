import argparse

from maingnn import train_gnn, test_gnn

parser = argparse.ArgumentParser(description="Extract triplets from dataset's captions.")
parser.add_argument('--dataset', type=str,            default='ucm',           required=False, help='Name of the dataset of which you want to create the triplets.')
parser.add_argument('--task',    type=str,            default='tripl2caption', required=True,  help='Name of the selected task. Possible ones: "tripl2caption", "img2tripl", "img2caption", "augmented_tripl2caption"') 
parser.add_argument('--e',       type=int,            default=10,              required=False, help='Number of Epochs.')
parser.add_argument('--lr',      type=float,          default=0.0001,          required=False, help='Learning rate.')
parser.add_argument('--bs',      type=int,            default=8,               required=False, help='Batch size.')
parser.add_argument('--decoder', type=str,            default="linear",        required=False, help='Type of decoder used for gnn.')
parser.add_argument('--name',    type=str,            default="GNN.pth",       required=False, help='Name for the network.')
parser.add_argument('--es',      action='store_true',                          required=False, help='Usage of early stopping.')
parser.add_argument('--thresh',  type=int,            default=1,               required=False, help='Number of epochs before early stopping.')
parser.add_argument('--test',    action='store_true',                          required=False, help='Use network in test mode.')
parser.add_argument('--o',       type=str,            default="captions.json", required=False, help='Name of the file for captions.')
parser.add_argument('--gnn',     type=str,            default="gat",           required=False, help='Type of graph neural network to use.')
parser.add_argument('--vir',     action='store_true',                          required=False, help='If True, use virtual node.')
parser.add_argument('--depth',   type=int,            default=1,               required=False, help='Depth of the GNN.')
parser.add_argument('--attr',    action='store_true',                          required=False, help='If True, use virtual node.')
parser.add_argument('--plt',     action='store_true',                          required=False, help='If True, save the plots.')
parser.add_argument('--combo',   action='store_true',                          required=False, help='If True, use the combined loss during training; use the unique one otherwise.')
parser.add_argument('--pil',     action='store_true',                          required=False, help='If True, use PIL module for images; use OpenCV module otherwise.')


if __name__=="__main__":
    args = parser.parse_args()
    if args.test:
        test_gnn(args.dataset,
                args.task,
                args.decoder,
                args.name,
                args.o,
                args.gnn,
                args.vir,
                args.depth,
                args.attr,
                args.pil
                )
    else:
        train_gnn(args.dataset,
                args.task,
                args.e,
                args.lr,
                args.bs,
                args.decoder,
                args.name,
                args.es,
                args.thresh,
                args.gnn,
                args.vir,
                args.depth,
                args.attr,
                args.plt,
                args.combo,
                args.pil
                )