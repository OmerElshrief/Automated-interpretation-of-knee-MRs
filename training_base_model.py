


def train (model, data, labels, no_epochs, valid_data, valid_labels, disease, lr = 0.001):
  
  model.to(device)
  
  criterion = nn.BCELoss()
  optimizer = create_optimizer(model,r = lr)
   
  losses = []
  accu = []
  history = []
  data_size = len(data)
  valid_size = len(valid_data)
  print(model)
  valid_acc_heighst = -999
  PATH = "drive/My Drive/Pattern Assignments/MRI-Ass4/"+disease+"_base_model2.h5"
  
  labels = [i for i in labels[disease]]
  valid_labels = [i for i in valid_labels[disease]]
  for e in range(no_epochs):
      running_loss,valid_loss = 0,0
      valid_correct,train_correct = 0,0
      
      
      for i in range(data_size):
        y_true = torch.tensor(labels[i])
        y_true= y_true.type(torch.FloatTensor)
        y_true = y_true.cuda()
        
       
        test_tensor = data_augmentation_transform(data[i])
       # print(test_tensor.shape)
        
        # Convert to FloatTensor first
        test_tensor = test_tensor.type(torch.FloatTensor)
        test_tensor = test_tensor.cuda()
        # Backward and optimize
        optimizer.zero_grad()
        y_pred = model(test_tensor)
        y_pred = y_pred.to(device)
        train_correct +=(y_true.cpu().numpy() == np.around(y_pred.cpu().detach().numpy())).sum()


        loss = criterion(y_pred, y_true)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        
        
        #print(y_pred)
        
     
      
      ## Validaition Step for each Eboch
      print('Validating Epoch {} ...'.format(e+1)) 
      for v in range(valid_size):
        y_true = torch.tensor(valid_labels[v])
        y_true= y_true.type(torch.FloatTensor)
        y_true = y_true.cuda()

        #x = np.stack((valid_data[v],valid_data[v],valid_data[v]), 1)
        
        test_tensor = data_augmentation_transform(valid_data[v],validate = True)

        # Convert to FloatTensor first
        test_tensor = test_tensor.type(torch.FloatTensor)
        test_tensor = test_tensor.cuda()
       
      
        y_pred = model(test_tensor)
        #print(y_pred)
        #print(y_true)

        y_pred = y_pred.to(device)
        valid_correct +=(y_true.cpu().numpy() == np.around(y_pred.cpu().detach().numpy())).sum()
        loss = criterion(y_pred, y_true)
        valid_loss += loss.item()
        
      test_accu =  int(train_correct) / data_size
      valid_accu = int(valid_correct) / valid_size
      #print(y_pred)
      print ('Epoch [{}/{}], -T-Loss : {:.6f} , Train_accuracy: {:.4f} ,  Val_loss: {:.6f}  -Val_acc: {:.4f}'.format(e+1, no_epochs,running_loss,test_accu ,valid_loss, valid_accu))
      history.append([e,running_loss,valid_accu,valid_loss])
      if valid_acc_heighst < valid_accu:
        torch.save(model.state_dict(), PATH)
        print('valid_acc has improved from {:.4f} to {:.4f}, model is saved at {}'.format(valid_acc_heighst , valid_accu,PATH))
        valid_acc_heighst = valid_accu
      else:
        print('valid_acc has not Improved from {:.4f}'.format(valid_acc_heighst))    


  return history


from sklearn.metrics import classification_report
def validate(valid_data , valid_labels,model , disease , history ):
  history = np.array(history)
  plt.figure()
  plt.title('Training performance')
  plt.plot(history[:,0], history[:,1], label='train loss+error')
  plt.plot(history[:,0], history[:,3], label='val_error')
  plt.legend()
  
  val_correct = 0;
  test_Y_i_hat = []
  test_Y_i = []
  for i in range(len(valid_data)):
    x = data_augmentation_transform(valid_data[i],validate = True)
    y_hat = model((x).cuda())
    test_Y_i_hat.append(np.around(y_hat.cpu().detach().numpy())[0])
    #print(y_hat)
    test_Y_i.append(valid_labels[disease][i])
    
    val_correct +=(valid_labels[disease][i] == np.around(y_hat.cpu().detach().numpy())).sum()

  print("Valid Accuracy : {:.4f} %".format(100 * (val_correct / len(valid_data))))
  
  print(classification_report(test_Y_i,test_Y_i_hat))
  conf = np.zeros([2,2])
  confnorm = np.zeros([2,2])
  for i in range(0,120):
      j = int(np.around(test_Y_i[i]))
      k = int(np.around(test_Y_i_hat[i]))

      conf[j,k] = conf[j,k] + 1
  for i in range(0,2):
      confnorm[i,:] = conf[i,:] / np.sum(conf[i])
  plt.figure()
  plot_confusion_matrix(confnorm, labels=['1','0'], title="Validation Set")
  





  ## Splitting training data
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(train_data,labels,test_size=0.2, random_state=42)

## Training 
model = MyModel()
model.to(device)

history = train(model,X_train,y_train,10,X_test, y_test, disease = "ACL",lr = 0.0005)

## Load model state

## model.load_state_dict(torch.load('drive/My Drive/Pattern Assignments/MRI-Ass4/ACL_base_model.h5'))
## model.eval()

## Plotting training results
history = np.array(history)
plt.figure()
plt.title('Training performance')
plt.plot(history[:,0], history[:,1], label='train loss+error')
plt.plot(history[:,0], history[:,3], label='val_error')
plt.legend()

## plotting validation/Test results
validate(valid_data , valid_labels , model , "ACL" )
 # You may change ACL to the desired TEST