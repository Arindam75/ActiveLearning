# ActiveLearning
An uncertainty based sampling to develop a model using Active Learning
This repository demosntrates Active Learning by training a model based on uncertainty based sampling.

### Files
There are three jupyter notbooks in this repository
- 01_Training_Full.ipynb
- 02_Training_AL.ipynb
- 03_Evaluation.ipynb

### Pre-Requisites

__System Requirements__<br>
__Packages__<br>
__Dataset__<br>

Clone the repository and download the dataset dogs-vs-cats.zip from __[here](https://www.kaggle.com/competitions/dogs-vs-cats/data)__ and extract train.zip into the project folder. The code in the 1st notebook creates a folder __train__ and extracts all the image files into this.<br>
Run the 1st notebook, that creates a baseline model_baseline.h5 in the folder model<br>
The 2nd notebook builds model_al.h5 based on active learning and compares it with the baseline<br>

Note that , the 2nd notebook will fail to run if it doesn't find the __train__ folder. If the baseline model is not found , then also 2nd notebook will throw a FileNotFoundError in the section __Baseline Model Evaluation on Test Dataset__ and the rest of the notebook will fail to run.<br>

After running both the notebooks , the requires CSV files would have been created in the __logger__ folder. These are used in the 3rd notebook to create the charts. 


