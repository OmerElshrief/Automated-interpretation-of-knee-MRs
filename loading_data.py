
# interactive viewer
class KneePlot():
    def __init__(self, x, figsize=(10, 10)):
        self.x = x
        self.planes = list(x.keys())
        self.slice_nums = {plane: self.x[plane].shape[0] for plane in self.planes}
        self.figsize = figsize
    
    def _plot_slices(self, plane, im_slice): 
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.imshow(self.x[plane][im_slice, :, :])
        plt.show()
    
    def draw(self):
        planes_widget = Dropdown(options=self.planes)
        plane_init = self.planes[0]
        slice_init = self.slice_nums[plane_init] - 1
        slices_widget = IntSlider(min=0, max=slice_init, value=slice_init//2)
        def update_slices_widget(*args):
            slices_widget.max = self.slice_nums[planes_widget.value] - 1
            slices_widget.value = slices_widget.max // 2
        planes_widget.observe(update_slices_widget, 'value')
        interact(self._plot_slices, plane=planes_widget, im_slice=slices_widget)
    
    def resize(self, figsize): self.figsize = figsize

# example usage
#plot = KneePlot(x, figsize=(8, 8))
#plot.draw()


## ----------------------------------------------------------------------
# Define path for each plane

data_path = 'drive/My Drive/Pattern Assignments/MRI-Ass4/MRNet-v1.0'
train_data_path = 'drive/My Drive/Pattern Assignments/MRI-Ass4/MRNet-v1.0/train'
dev_data_path = 'drive/My Drive/Pattern Assignments/MRI-Ass4/MRNet-v1.0/valid'
#views = {'axial': np.array(), 
#         'coronal': np.array(), 
#         'sagittal': np.array()}

train_abnormal = 'drive/My Drive/Pattern Assignments/MRI-Ass4/MRNet-v1.0/train-abnormal.csv'
train_acl = 'drive/My Drive/Pattern Assignments/MRI-Ass4/MRNet-v1.0/train-acl.csv'
train_meniscus = 'drive/My Drive/Pattern Assignments/MRI-Ass4/MRNet-v1.0/train-meniscus.csv'

valid_abnormal = 'drive/My Drive/Pattern Assignments/MRI-Ass4/MRNet-v1.0/valid-abnormal.csv'
valid_acl = 'drive/My Drive/Pattern Assignments/MRI-Ass4/MRNet-v1.0/valid-acl.csv'
valid_meniscus = 'drive/My Drive/Pattern Assignments/MRI-Ass4/MRNet-v1.0/valid-meniscus.csv'

# data loading functions
def load_one_stack(case, data_path=train_data_path, plane='coronal'):
    fpath = data_path+'/'+plane+'/'+'{}.npy'.format(case)
    #print(fpath)
    return np.load(fpath)

def load_stacks(case, data_path=train_data_path):
    x = {}
    # We may just use 1 plane for each CNN because The Data is consuming all the Ram
    planes = ['coronal', 'sagittal', 'axial']
    ## You may not be able to load all the data at once, but you can load one plane at a time
    for i, plane in enumerate(planes):
        x[plane] = load_one_stack(case, plane=plane,data_path=data_path)
    return x
  
## --------------------------------------------------------------------------
## --------------------------------------------------------------------------
  # Loading the Data
cases_size = 1130
train_data = []

for i in range(0,cases_size):
  case = train_abnl_labels.Case[i]
  stacks = load_stacks(case)
  train_data.append((stacks))
  
print("Train Data loaded")

 # Loading labels 
train_abnl_labels = pd.read_csv(train_abnormal, header=None,
                       names=['Case', 'Abnormal'], 
                       dtype={'Case': str, 'Abnormal': np.int64})

train_acl_labels = pd.read_csv(train_acl, header=None,
                       names=['Case_acl', 'ACL'], 
                       dtype={'Case_acl': str, 'ACL': np.int64})

train_meniscus_labels = pd.read_csv(train_meniscus, header=None,
                       names=['Case_men', 'Meniscus'], 
                       dtype={'Case_men': str, 'Meniscus': np.int64})


labels = [train_abnl_labels, train_acl_labels, train_meniscus_labels]
labels= pd.concat(labels, axis = 1)
labels.drop(columns=['Case_acl','Case_men'],inplace = True)
print("train_Labels loaded")


 # Load validation Data
cases_size = 120
valid_data = []

for i in range(0,cases_size):
  case = valid_abnl_labels.Case[i]
  stacks = load_stacks(case,data_path =dev_data_path )
  
  valid_data.append((stacks['sagittal']))
valid_data = np.array(valid_data)
print('validation_data Loaded')

 # Load validation data labels
valid_abnl_labels = pd.read_csv(valid_abnormal, header=None,
                       names=['Case', 'Abnormal'], 
                       dtype={'Case': str, 'Abnormal': np.int64})

valid_acl_labels = pd.read_csv(valid_acl, header=None,
                       names=['Case_acl', 'ACL'], 
                       dtype={'Case_acl': str, 'ACL': np.int64})

valid_meniscus_labels = pd.read_csv(valid_meniscus, header=None,
                       names=['Case_men', 'Meniscus'], 
                       dtype={'Case_men': str, 'Meniscus': np.int64})


valid_labels = [valid_abnl_labels, valid_acl_labels, valid_meniscus_labels]
valid_labels= pd.concat(valid_labels, axis = 1)
valid_labels.drop(columns=['Case_acl','Case_men'],inplace = True)
print("valid_labels loaded")