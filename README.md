Here is the code used for various Kaggle competitions. These competitions are ML and data science challenges based around real-life use cases.

### PII Data Detection (NLP - Token classification)

The goal of this competition is to provide a model capable of detecting and labelling student data in essays. I have first created a baseline using a pre-trained model (Presidio by Microsoft). Then, I used different models like Bert to improve efficiency. These were the models used:

1. [Presidio baseline](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/pii_data_detection/1-pii-data-detection-exploration-baseline.ipynb)
2. [NER fine-tuned Deberta](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/pii_data_detection/2-pii-data-detection-second-model.ipynb) 
3. Deberta ([training](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/pii_data_detection/3-deberta-fine-tuned-training.ipynb), [inference](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/pii_data_detection/3-deberta-fine-tuned-inference.ipynb))

### Digit Recognition (Computer Vision - Image classification)

Based around the famous MNIST dataset, the goal of this competition is to build models able to correctly classify handwritten digits. It is a simple playground to experiment using a lot of models, from sklearn to handmade CNNs and ensembles of more complex models such as ResNet and DenseNet. I explored the following models:

1.  Scikit-learn [base models](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/digit_recognition/01_baseline_models.ipynb) (logistic regression, random forest, XGboost)
2. [Logistic regression](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/digit_recognition/02_logit_from_scratch.ipynb) from scratch
3. Custom [MLP](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/digit_recognition/03_mlp_with_pytorch_opti.ipynb)
4. Custom [CNN](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/digit_recognition/04_cnn_pytorch_opti.ipynb) + various regularization techniques
5. [ResNet](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/digit_recognition/05_resnet.ipynb)
6. [DenseNet](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/digit_recognition/07_densenet.ipynb)
7. CNNs [Ensemble](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/digit_recognition/06_classifiers%2C%20ensemble.ipynb)
8. Vision Transformer ([training](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/digit_recognition/08A_vision_transformer_train.ipynb), [inference](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/digit_recognition/08B_vision_transformer_infere.ipynb))
9. Multi-modal transformer ([Phi3](https://github.com/GaetanHHHHHH/kaggle-competitions/blob/main/digit_recognition/09_phi3.ipynb))
