
## Data preparation

# First transform the Data 
preprocessor = Preprocessor()
transformed_train_data = preprocessor.fit_transform(train_data)
transformed_valid_data = preprocessor.fit_transform(valid_data)


## Building Dataloaders
import torch.utils.data as utils
from torch.utils.data.dataset import random_split
tensor_y = torch.stack([torch.Tensor(i) for i in np.array(labels['Meniscus']).reshape(-1,1)])

my_dataset = utils.TensorDataset((transformed_train_data),tensor_y) # create your datset

train_size = int(0.8 * len(my_dataset))
test_size = len(my_dataset) - train_size
train_dataset, validate_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size])

train_dataloader = utils.DataLoader(train_dataset,batch_size=100, shuffle=True) # create your dataloader

validate_dataloader =   utils.DataLoader(validate_dataset,batch_size=20, shuffle=False) # create your dataloader


tensor_y = torch.stack([torch.Tensor(i) for i in np.array(valid_labels['Meniscus']).reshape(-1,1)])

my_dataset = utils.TensorDataset((transformed_valid_data),tensor_y) # create your datset
test_dataloader = utils.DataLoader(my_dataset,batch_size=10, shuffle=False) # create your dataloader


## =================================================================

# Training functions

def train_model(model  ,train_dataloader,valid_dataloader,NUM_EPOCHS = 100,path='drive/MyDrive/Pattern Assignments/MRI-Ass4/model.h5',lr=0.00007 ):

  classification_model = model
  classification_model.to(device)
  criterion = nn.BCELoss()
  optimizer = create_optimizer(classification_model,r =lr)
  
  j=0
  running_loss,train_correct,val_acc = 0,0,-9999
  losses = []
  accu = []
  history = []
  #print(classification_model)

  for i in range(NUM_EPOCHS):

    print('Training Epoch {} ...'.format(i+1))
    running_loss,train_correct,val_correct,val_loss = 0,0,0,0
    
    # Training for eboch
    for batch,y_true in train_dataloader:
      y_true= y_true.type(torch.FloatTensor)
      y_true = y_true.cuda()
      #print(batch.shape)
      batch = torch.stack(data_augmentation_transform(np.array(batch.data.numpy(),dtype = np.uint8)))
      batch = (batch.cuda())
      # Backward and optimize
      optimizer.zero_grad()
      y_pred = classification_model(batch)
      y_pred = y_pred.to(device)
      train_correct +=(y_true.cpu().numpy() == np.around(y_pred.cpu().detach().numpy())).sum()

#       y_true= y_true.type(torch.FloatTensor)
#       y_true = y_true.cuda()
#       y_pred = y_pred.cuda()
      loss = criterion(y_pred, y_true)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

      
    ## Validaition Step for each Eboch

    print('Validating Epoch {} ...'.format(i+1)) 
    
    for batch , y_true in valid_dataloader:
        batch = torch.stack(data_augmentation_transform(np.array(batch.data.numpy(),dtype = np.uint8),validate = True))
     
        batch = batch.cuda()
        y_true= y_true.type(torch.FloatTensor)
        y_true = y_true.cuda()
       

        # Backward and optimize
        optimizer.zero_grad()
        y_pred = classification_model(batch)
        y_pred = y_pred.to(device)
       
        val_correct +=(y_true.cpu().numpy() == np.around(y_pred.cpu().detach().numpy())).sum()
        
        y_true= y_true.type(torch.FloatTensor)
        y_true = y_true.cuda()
        y_pred = y_pred.cuda()
        loss = criterion(y_pred, y_true)
        val_loss += loss.item()
       



    test_accu =  int(train_correct) / train_size
    valid_accu = int(val_correct) / test_size

    print('Train correct: ',train_correct )
    print('Tets correct: ', val_correct, 'out of: ',len(valid_dataloader))
    print ('Epoch [{}/{}], -T-Loss : {:.6f} , Train_accuracy: {:.4f} ,  Val_loss: {:.6f}  -Val_acc: {:.4f}'.format(i+1, NUM_EPOCHS,running_loss,test_accu ,val_loss, valid_accu))

    if val_acc < valid_accu:
     # torch.save(classification_model.state_dict(), path)
      print('val_acc has improved from {:.4f} to {:.4f}, model is saved at {}'.format(val_acc , valid_accu,path))
      val_acc = valid_accu
    else:
      print('val_acc has not Improved from {:.4f}'.format(val_acc))    
    history.append([i,running_loss,val_acc,val_loss])

    #accu.append(100 * int(test_correct) / len(X_test))

  return history
    
## ===================================================================

# Training 
MyModel = MRModel()   ## You may change the Model to anything you want
MyModel.to(device)
history = train_model(MyModel ,train_dataloader,validate_dataloader,NUM_EPOCHS=20, lr = 5e-3)


## Validation step

x = torch.stack(data_augmentation_transform(np.array(transformed_valid_data.data.numpy(),dtype = np.uint8)))
test_Y_i_hat =  MyModel((x).cuda())
test_Y_i = tensor_y
#print(test_Y_i_hat)
val_correct =(test_Y_i.cpu().numpy() == np.around(test_Y_i_hat.cpu().detach().numpy())).sum()

print("Test Accuracy : {:.4f} %".format(100 * (val_correct / 120)))
conf = np.zeros([2,2])
confnorm = np.zeros([2,2])
for i in range(0,120):
    j = int(np.around(test_Y_i[i].cpu().detach().numpy())[0])
    k = int(np.around(test_Y_i_hat[i].cpu().detach().numpy())[0])
   
    conf[j,k] = conf[j,k] + 1
for i in range(0,2):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i])
plt.figure()
plot_confusion_matrix(confnorm, labels=['1','0'], title="Validation Set")
