# Codes for brain age prediction
Overview of the prediction scripts:
* [GPR.py](GPR.py): Fits a GPR model on raw features
* [GPR_similarity.py](GPR_similarity.py): Fits a GPR model on the similarity metric calculated with respect to the subjects in training dataset
* [PCA_GPR.py](PCA_GPR.py): Fits a GPR model on the (top) principal components of the feature matrix
* [CCA_GPR.py](CCA_GPR.py): Fits a GPR model on the canonical correlation component obtained from the covariance of features and age
* [CCA_BS.py](CA_BS.py): Retrieves the bootstrapped ratio of loadings corresponding to each feature in the CCA model
