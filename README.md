# Histovision-AI

Welcome to my image classification model that models and predicts histological images of breast tissue that may or may not have breast cancer. I decided that I really wanted to step into the world of AI/ML, and this would be my first step. So here goes!

Dataset is sourced from Kaggle at this URL: https://www.kaggle.com/datasets/jocelyndumlao/biglycan-breast-cancer?select=Biglycan+breast+cancer+dataset

To run in your own IDEs/Notebooks, you must download the datasets using these commands in your own notebook:
```
! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle // Make sure to go to Kaggle's website and go to your profile, and download your API key as a kaggle.json file. An option will pop up to upload a file when you run this command. Select your kaggle.json file.
!mv ./kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d jocelyndumlao/biglycan-breast-cancer // This link is specifically for the breast cancer histological image dataset I am working with. This will change with every dataset!
import zipfile
zip_ref = zipfile.ZipFile('biglycan-breast-cancer.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()
```
If you are using a virtual notebook such as on Google Colab or Kaggle, you can delete these commands from your code after it is executed.
Check out this Kaggle forum post for more information: https://www.kaggle.com/discussions/general/74235
