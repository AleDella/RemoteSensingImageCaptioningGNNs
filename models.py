import torch
import torch.nn as nn

class TripletClassifier(nn.Module):
    def __init__(self):
        super(TripletClassifier, self).__init__()