# Histovision-AI

Welcome to my image classification model that models and predicts histological images of skin tissue that may or may not have melanoma. I decided that I really wanted to step into the world of AI/ML, and this would be my first step. So here goes!

Dataset is sourced from Kaggle at this URL: https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset

To run in your own IDEs/Notebooks, you must download the datasets using these commands in your own notebook:
```
! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle // Make sure to go to Kaggle's website and go to your profile, and download your API key as a kaggle.json file. An option will pop up to upload a file when you run this command. Select your kaggle.json file.
!mv ./kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d bhaveshmittal/melanoma-cancer-dataset // This link is specifically for the melanoma cancer histological image dataset I am working with. This will change with every dataset!
import zipfile
zip_ref = zipfile.ZipFile('melanoma-cancer-dataset.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()
```
If you are using a virtual notebook such as on Google Colab or Kaggle, you can delete these commands from your code after it is executed.
Check out this Kaggle forum post for more information: https://www.kaggle.com/discussions/general/74235

To run the hypertuner with Keras Tuner, you must run this command before running the program:
```! pip install -q -U keras-tuner```

Utilized Keras Tuner to find the optimal number of units in the first densely-connected layer is 96 and the optimal learning rate for the optimizer is 0.001. The best number of epochs to run to get the most accurate results is 38 epochs.
My use of the Keras Tuner can be found in the "Histovision AI Hypertuner" Python File.

Hypertuning Results:
![Screen Shot 2025-01-10 at 4 19 31 AM](https://github.com/user-attachments/assets/a3ed5294-4fe7-42a7-ba5a-2b36f907bb1c)

Incorporated Prefetching, Normalization/Standardization, Data Augmentation, and Dropout Regularization to finetune my model and achieve a validation accuracy rate around 80-90%. My CNN model without transfer learning can be found in the "Histovision AI" Python file.

Training Results Before Transfer Learning:

<img width="680" alt="Screen Shot 2025-01-10 at 5 22 21 PM" src="https://github.com/user-attachments/assets/092311a7-4a85-4abe-930b-6551404f3037" />
<img width="1173" alt="Screen Shot 2025-01-10 at 5 23 39 PM" src="https://github.com/user-attachments/assets/f5650a7d-45b4-4c6a-bd74-7f38ddbda1b3" />

Also finetuned and experimented with other convolutional neural network models best known for cancer detection, such as AlexNet, MobilenetV2, and Resnet 50, to incorporate transfer learning and improve validation accuracy results.
My incorporation of transfer learning from a combination hybrid model of the 3 aforementioned CNN models can be found in the "Histovision AI Transfer learning" Python file.
