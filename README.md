# Automated-interpretation-of-knee-MRs
Models for medical imaging examinations
The Stanford ML Group recently released their third public data set of medical imaging examinations, called MRNet, which can be found here. From the website:

The MRNet dataset consists of 1,370 knee MRI exams performed at Stanford University Medical Center. The dataset contains 1,104 (80.6%) abnormal exams, with 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears; labels were obtained through manual extraction from clinical reports.

The data set accompanies the publication of the Stanford ML Group’s work, which can be found here: https://stanfordmlgroup.github.io/projects/mrnet/ 
Once again, they are hosting a competition to drive innovation in automated analysis of medical imaging.

### Base Model
The primary building block of our prediction system is MRNet, a convolutional neural network (CNN) mapping a 3-dimensional MRI series to a probability [15] (Fig 2). The input to MRNet has dimensions s × 3 × 256 × 256, where s is the number of images in the MRI series (3 is the number of color channels). First, each 2-dimensional MRI image slice was passed through a feature extractor based on AlexNet to obtain a s × 256 × 7 × 7 tensor containing features for each slice. A global average pooling layer was then applied to reduce these features to s × 256. We then applied max pooling across slices to obtain a 256-dimensional vector, which was passed to a fully connected layer and sigmoid activation function to obtain a prediction in the 0 to 1 range. We optimized the model using binary cross-entropy loss. To account for imbalanced class sizes on all tasks, the loss for an example was scaled inversely proportionally to the prevalence of that example’s class in the dataset.

![base_model](https://github.com/OmerElshrief/Automated-interpretation-of-knee-MRs/blob/master/base_model.PNG)

## Other models
Beside the base model, we bult other models using different feature extractors and trained them all then tunned the best results models
