# IsoFrog
## 1. Description
IsoFrog is a feature selection-based computational approach for isoform function prediction. It employs a Reversible Jump Monte Carlo Markov Chain (RJMCMC) feature selection framework to assess the feature importance to the function. Then a sequential feature selection procedure is applied to extract a subset of function-relevant features. The selected features are input into MdiPLS. In such a way, the features of importance to isoform function can be singled out and leveraged for isoform function prediction.


## 2. Input data
Both gene- and isoform-level data are required as input for IsoFrog. The demo input data are provided in the folder 'data', which includes training data for building models and test data for evaluating the performance of IsoFrog.


## 3. Implementation
IsoFrog is implemented in Python. It is tested on both MacOS and Linux operating system. They are freely available for non-commercial use.


## 4. Usage
We provide a demo script to show how to run IsoFrog. To test IsoFrog on an independent test dataset, run the following command from the command line:

```bash
python run_IsoFrog_traintest.py
```

With this command, we will first build a model on the training data and then make predictions on the test data.


## 5. Contact
If any questions, please do not hesitate to contact us at:
<br>
Yiwei Liu, `ywlmicky@csu.edu.cn`
<br>
Jianxin Wang, `jxwang@mail.csu.edu.cn`
