
class Preprocessor:
  
    
    
  def get_mean_threshold(self,stack):
    out_stack = []
    for img in stack:
      #img = cv2.fastNlMeansDenoising()

      img = cv2.medianBlur(img,5)

      img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
              cv2.THRESH_BINARY,11,2)
      out_stack.append(img)
    return np.array(out_stack, dtype= np.float32)

  def get_canny_edges(self,stack):
    out_stack = []
    for img in stack:
     # img = cv2.fastNlMeansDenoising(img,None,10)
      img = cv2.GaussianBlur(img,(5,5),0)

      img = cv2.Canny(img,100,200)
      img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
      out_stack.append(img)
    return np.array(out_stack, dtype= np.float32) 

  def get_lablacian_edge(self,stack):
    out_stack = []
    for img in stack:
      #mg = cv2.fastNlMeansDenoising(img,None,10)
      img = cv2.GaussianBlur(img,(5,5),0)
      img = cv2.Laplacian(img,cv2.CV_8U)
      img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      out_stack.append(img)
    return np.array(out_stack, dtype= np.float32) 



  def get_adabtive_avg_pooling(self,stack , plot = False):
    x = torch.from_numpy(stack)
    globalMaxPool = nn.AdaptiveAvgPool3d(output_size = (1,256,256))
    x=x.reshape(1,-1,256,256)
    out = globalMaxPool(x)
    out = out.reshape(256,256)
    if plot == True:
      plt.imshow(out)
    return out


  def get_3ch(self,stck):
    ch1 = self.get_adabtive_avg_pooling( self.get_canny_edges(stck) )
    ch1 = cv2.normalize(ch1.data.numpy(), None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 
    ch2 = self.get_adabtive_avg_pooling( self.get_lablacian_edge(stck))
    ch1 = cv2.normalize(stck[int(len(stck)/2)+1],None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ch2 = cv2.normalize(stck[int(len(stck)/2)],None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ch3 = cv2.normalize(stck[int(len(stck)/2)-1] , None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ch3 = self.get_adabtive_avg_pooling(np.array((stck),dtype = np.float32))
    ch3 = cv2.normalize(ch3.data.numpy(), None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #ch3 = cv2.normalize(ch3.data.numpy(), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                              
    return ch3
    #return np.stack([ch1,ch2,ch3], axis = 0)

  def fit_transform(self,data):
    
    transformed_data = []
    for stck in data:
      
      transformed_data.append(self.get_3ch(stck)) 
    return torch.tensor(transformed_data)


### Transformation Functions
# For data Augmentation
transforms = torchvision.transforms.Compose([
    
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    torchvision.transforms.ToTensor()
])

validation_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor()
])


def ToRGB(x):
  x.unsqueeze_(0)
  x = x.repeat(3, 1, 1)
  return x

def data_augmentation_transform(train_data,validate = False):
  transformed_augmanted_train_data = []
  for img in train_data:
    
    if validate:
      x = validation_transform(img).reshape(224,224)
    else:
      x = transforms(img).reshape(224,224)
        
    x = cv2.normalize(x.data.numpy(), None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    x = ToRGB(torch.from_numpy(x))
    transformed_augmanted_train_data.append(torch.tensor(x))
  return torch.stack(transformed_augmanted_train_data)


  ## ------------------------------------------------------------
  # Preprocessing Functions 
  def get_mean_threshold(stack):
  out_stack = []
  for img in stack:
    #img = cv2.fastNlMeansDenoising()

    img = cv2.medianBlur(img,5)

    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    out_stack.append(img)
  return np.array(out_stack, dtype= np.float32)

def get_canny_edges(stack):
  out_stack = []
  for img in stack:
    img = cv2.fastNlMeansDenoising(img,None,10)
    img = cv2.GaussianBlur(img,(5,5),0)

    img = cv2.Canny(img,100,200)
    out_stack.append(img)
  return np.array(out_stack, dtype= np.float32) 

def get_lablacian_edge(stack):
  out_stack = []
  for img in stack:
    img = cv2.fastNlMeansDenoising(img,None,10)
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.Laplacian(img,cv2.CV_8U)
    out_stack.append(img)
  return np.array(out_stack, dtype= np.float32) 



def get_adabtive_avg_pooling(stack , plot = False):
  x = torch.from_numpy(stack)
  globalMaxPool = nn.AdaptiveAvgPool3d(output_size = (1,256,256))
  x=x.reshape(1,-1,256,256)
  out = globalMaxPool(x)
  out = out.reshape(256,256)
  if plot == True:
    plt.imshow(out)
  return out


def get_3ch(stck):
  ch1 = get_adabtive_avg_pooling( get_canny_edges(stck) )
  ch2 = get_adabtive_avg_pooling( get_lablacian_edge(stck))
  ch3 = cv2.medianBlur(stck[int(len(stck)/2)],5)

  return np.stack([ch1,ch2,ch3], axis = 0)

def fit_transform(self,data):

  transformed_data = []
  for stck in data:

    transformed_data.append(get_3ch(stck)) 
  return torch.tensor(transformed_data)




# 
# 
#x = x.reshape(224,224)
# x = transforms(transformed_train_data)
# transformed_train_data.shape