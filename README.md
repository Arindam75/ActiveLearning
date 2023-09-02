# ActiveLearning
<br><br>
This repository demonstrates Active Learning by training a model based on uncertainty based sampling.<br> This is the companion repository to the Medium article __[A Gentle Introduction to Active Learning](https://arindam-dey.medium.com/a-gentle-introduction-to-active-learning-e983b9d175cb)__

### Files
There are three jupyter notbooks in this repository
- 01_Training_Full.ipynb
- 02_Training_AL.ipynb
- 03_Evaluation.ipynb

### Pre-Requisites

__System Requirements__<br>

I ran the experiemnets on a acer Nitro-5 laptop with NVIDIA GeForce RTX 3060 GPU.<br><br>
I haven't tried it on google colab, but it should be straightforward with some tweaks. In case you want to run the experiment on google colab, move all the contents of the repository into a folder in your google drive. Get the train.zip ( see Dataset and Notebooks ) below. folder also in the same folder. Then in colab , mount your google drive and use this folder as your project folder.

__Packages__<br>

tensorflow == 2.10.1

__Dataset and Notebooks__<br>

Clone the repository and download the dataset dogs-vs-cats.zip from __[here](https://www.kaggle.com/competitions/dogs-vs-cats/data)__ and extract train.zip into the project folder. The code in the 1st notebook creates a folder __train__ and extracts all the image files into this.<br><br>
Run the 1st notebook, that creates a baseline model_baseline.h5 in the folder model<br>
The 2nd notebook builds model_al.h5 based on active learning and compares it with the baseline<br>

Note that , the 2nd notebook will fail to run if it doesn't find the __train__ folder. If the baseline model is not found , then also 2nd notebook will throw a FileNotFoundError in the section __Baseline Model Evaluation on Test Dataset__ and the rest of the notebook will fail to run.<br><br>

After running both the notebooks , the required CSV files would have been created in the __logger__ folder. These are used in the 3rd notebook to create the charts.
