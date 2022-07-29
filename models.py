import torch
import torch.nn as nn
from torchvision.models import resnet152

class TripletClassifier(nn.Module):
    '''
    Model which takes as input an image and predict the corresponding triplets that are in that image. 
    It will be based on resnet-152 for the extraction of the features, so it will be a finetuning on the target dataset.
    '''
    def __init__(self, input_size, num_classes):
        super(TripletClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = resnet152(pretrained=True)
        # Replace the last layer with a new layer for classification
        self.model.fc = nn.Linear(in_features=2048,out_features=num_classes)
        
        # Freeze all the layers except the fully connected
        for name, parameter in self.model.named_parameters():
            if(not 'fc' in name):
                parameter.requires_grad = False
        
        # Just for testing 
        # for name, parameter in self.model.named_parameters():
        #     print(name,parameter.requires_grad)
    
    def forward(self, x):
        '''
        x -> shape (batch_size, channels, width, height)
        '''
        assert x.shape[2]==self.input_size
        assert x.shape[3]==self.input_size
        
        return self.model(x)
    

if __name__=="__main__":
    model = TripletClassifier(224,10)
    dummy_img = torch.randn((5,3,224,224))
    out = model(dummy_img)