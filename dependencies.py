import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from ipywidgets import interact, Dropdown, IntSlider
%matplotlib inline
plt.style.use('grayscale')
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

import torchvision.models as models
from torchsummary import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import PIL
from PIL import Image
