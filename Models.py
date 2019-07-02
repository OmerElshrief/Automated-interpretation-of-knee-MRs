# Building a Model based on the Pre-processed Data
### For later uses, to enhance the results of the base mode

## Pre-trained models
model_alexNet  = models.alexnet(pretrained=True)
model_Resnet   = models.resnet18(pretrained=True)
model_denseNet = models.densenet121(pretrained=True)

## ===========================================================================================

## Helper functions 
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
      for param in model.parameters():
            param.requires_grad = True
      
    
      
def create_optimizer(model , r = 0.0001):
  params_to_update = model.parameters()
  print("Params to learn:")
  
  params_to_update = []
  for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
        
  return  optim.Adam(params_to_update, lr=r)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
       
      
      
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
      
class Transform(nn.Module):
  def __init__(self):
    super(Transform, self).__init__()
  
  def forward(self,x):
    return transforms(x)
        
## ==============================================================================

### DensetNet


class DenseNet_Model (nn.Module):
  def __init__(self):
    
        super().__init__()
        
        
        self.dense = model_denseNet.features
        set_parameter_requires_grad(self.dense,True)
        set_parameter_requires_grad(self.dense[10].denselayer15,False)
        set_parameter_requires_grad(self.dense[10].denselayer16,False)
        #num_ftrs = model_denseNet.classifier.in_features
       
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fla = Flatten()
        self.bn0 = nn.BatchNorm1d(num_ftrs*2 ,eps=1e-05, momentum=0.1, affine=True)
        self.dropout0 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(num_ftrs*2 , 256)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256,eps=1e-05, momentum=0.1, affine=True)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        self.out = nn.Sigmoid()
        

  def forward(self, x):
       
        #print(x.shape)
        x = self.dense(x)
        #print(x.shape)
        ap = self.avgpool(x)
        mp = self.maxpool(x)
        x = torch.cat((ap,mp),dim=1)
        x = self.fla(x)
        
        x = self.bn0(x)
        x = self.dropout0(x)
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)         
        x = self.fc2(x)
        
        
        return self.out(x)

  def __call__(self, x): return self.forward(x)
  
feature_extraction_model = DenseNet_Model()
feature_extraction_model.to(device)
#print(feature_extraction_model)
summary(feature_extraction_model,(3,224,224))

## =================================================================================

### AlexNet
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MRModel (nn.Module):
  def __init__(self):
    
        super().__init__()
        
        self.features_extractor = model_alexNet.features
        #set_parameter_requires_grad(self.features_extractor,True)
        self.features_extractor[10] = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_extractor[8] = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fla = Flatten()
        self.bn0 = nn.BatchNorm1d(512 ,eps=1e-05, momentum=0.1, affine=True)
        self.dropout0 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512 , 256)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256,eps=1e-05, momentum=0.1, affine=True)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        self.out = nn.Sigmoid()
        

  def forward(self, x):
        
        x = self.features_extractor(x)
       
        ap = self.avgpool(x)
        mp = self.maxpool(x)
        x = torch.cat((ap,mp),dim=1)
        x = self.fla(x)
        
        x = self.bn0(x)
        x = self.dropout0(x)
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)         
        x = self.fc2(x)
        
        
        return self.out(x)

  def __call__(self, x): return self.forward(x)
  
feature_extraction_model = MRModel()
feature_extraction_model.to(device)
print(feature_extraction_model)
summary(feature_extraction_model,(3,224,224))

## ===================================================================

## ReseNet

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Flatten(nn.Module):
  def forward(self, input):
      return input.view(input.size(0), -1)


      
class Resnet_Model (nn.Module):
  def __init__(self):
    
        super().__init__()
        set_parameter_requires_grad(model_Resnet, True)
        set_parameter_requires_grad(model_Resnet.layer4, False)
        
        #num_ftrs = model_Resnet.fc.in_features
        model_Resnet.fc = Identity()
        model_Resnet.layer4[1].conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         
        self.Resnet = model_Resnet
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fla = Flatten()
        self.bn0 = nn.BatchNorm1d(num_ftrs,eps=1e-05, momentum=0.1, affine=True)
        self.dropout0 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(num_ftrs, 1024)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024,eps=1e-05, momentum=0.1, affine=True)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1)
        self.out = nn.Sigmoid()
        

  def forward(self, x):
        
        x = self.Resnet(x)
        
        #x = x.reshape(-1,1000,1,1)
        #print(x.shape)
        mp = self.maxpool(x)
        ap = self.avgpool(x)
        
        x = torch.cat((ap,mp),dim=1)
        x = self.flat(x)
        
        x = self.bn0(x)
        x = self.dropout0(x)
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)         
        x = self.fc2(x)
        
        
        return self.out(x)

  def __call__(self, x): return self.forward(x)
  
feature_extraction_model = Resnet_Model()
feature_extraction_model.to(device)

summary(feature_extraction_model,(3,224,224))

## ====================================================================

# Base Model

class MyModel (nn.Module):
  def __init__(self):
    
        super().__init__()
        
        self.features_extractor = model_alexNet.features
        set_parameter_requires_grad(self.features_extractor,True)
        self.features_extractor[10] = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_extractor[8] = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
        self.Avg3dPool = nn.AvgPool3d(kernel_size = (1,6,6), stride = 1, padding = 0)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1,256))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size = (1,256))
        self.fla = Flatten()
        self.bn0 = nn.BatchNorm1d(256,eps=1e-05, momentum=0.1, affine=True)
        
        
        self.classifier =nn.Sequential( 
                                      # nn.Dropout(0.5),
                                       nn.Linear(256*2, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 128),
                                       nn.ReLU(),
                                       #nn.BatchNorm1d(1024,eps=1e-05, momentum=0.1, affine=True),
                                       #nn.Dropout(0.5),
                                       nn.Linear(128,1),
                                       nn.Sigmoid())
        

  def forward(self, x):
        
        x = self.features_extractor(x)
        x = self.Avg3dPool(x)
        
        x = x.view(1,-1,256)
        #print(x.shape)
        ap = self.avgpool(x)
        mp = self.maxpool(x)
        #print(x.shape)
        x = torch.cat((ap,mp),dim=1)
        x = self.fla(x)
        #print(x.shape)
        
       
        
        return self.classifier(x)

  def __call__(self, x): return self.forward(x)
  
feature_extraction_model = MyModel()
feature_extraction_model.to(device)
print(summary(feature_extraction_model,(3,256,256)))