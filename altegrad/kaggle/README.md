## Altegrad Final Kaggle Challenge, Link Prediction in a Citation Network.

For details of my implementation please see the final_report.pdf. To run the codes you will need to first download the datasets 'edgelist.txt, abstracts.txt, authors.txt and test.txt' and place them in the same directory as the python files.

### Feature extraction

The file 'features.py' details the computation of the features used to create our dataset. The file 'build_features.py' uses the functions defined in 'features.py' to build the dataset and then saves it to the folder ./save/train and ./save/test.

### Classifiers

#### Random Forest

The file 'RandomForestClassifier.py' runs the Random Forest classifier on as many cores as your CPU has and saves the predicted probabilities for the edges in 'test.txt' in the file 'submission.csv'. Slight modification of this file can be done to retrieve the pie chart figure given in the report.

#### MLP

The file 'MLPModel.py' details the network architecture of the Multi-Layer Perceptron as is described in the final report. In order to find the correct set of hyper-parameters one can first run the file 'train.py' in order to explore the hyper-parameter space as specified in the main function of that file. The best hyper-parameters will be displayed at the end and can be copied in the 'MLPClassifier.py' file in order to train for longer epochs and save the predicted probablities of the edges in 'test.txt' in the file 'submission.csv'.

To train/test run: `python MLPClassifier.py`

#### GCN

The implementation of the graph convolutional network using the StellarGraph library is available as both a Jupyter file ('GCN.ipynb') and a python file ('GCN.py'). 

### Compare Classifiers

The file 'classifier_comparer.py' allows to compare easily the results of different classifiers.
