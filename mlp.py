#ELECTROENCEPHALOGRAM OF TWO HUMAN STATES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
import itertools
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
###################################################
plt.rcParams['figure.figsize'] = (15, 5) 
###################################################
#Read 'Human_calm_state.xlsx' file
human_calm_state = pd.read_excel('D:/Positive_influence_state.xlsx', encoding='latin1')
############################
#Plot of electroencephalogram of calm state of human
human_calm_state.plot()
plt.grid()
plt.title('Electroencephalogram of calm state of human')
###################################################
#Graph of electrical voltage everyone electrode of electroencephalogram
#for calm state of human
plt.figure()

plt.subplot(121)
plt.plot(human_calm_state.iloc[:,0].values)
plt.title('Fp1 electrode (calm state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_calm_state.iloc[:,1].values)
plt.title('Fp2 electrode (calm state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_calm_state.iloc[:,2].values)
plt.title('F3 electrode (calm state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_calm_state.iloc[:,3].values)
plt.title('F4 electrode (calm state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_calm_state.iloc[:,4].values)
plt.title('F7 electrode (calm state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_calm_state.iloc[:,5].values)
plt.title('F8 electrode (calm state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_calm_state.iloc[:,6].values)
plt.title('T3 electrode (calm state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_calm_state.iloc[:,7].values)
plt.title('T4 electrode (calm state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_calm_state.iloc[:,8].values)
plt.title('C3 electrode (calm state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_calm_state.iloc[:,9].values)
plt.title('C4 electrode (calm state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_calm_state.iloc[:,10].values)
plt.title('T5 electrode (calm state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_calm_state.iloc[:,11].values)
plt.title('T6 electrode (calm state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_calm_state.iloc[:,12].values)
plt.title('P3 electrode (calm state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_calm_state.iloc[:,13].values)
plt.title('P4 electrode (calm state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_calm_state.iloc[:,14].values)
plt.title('O1 electrode (calm state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_calm_state.iloc[:,15].values)
plt.title('O2 electrode (calm state of human)')
plt.grid()

plt.show()
###################################################
#Read 'Human_active_state.xlsx' file
human_active_state = pd.read_excel('D:/Negative_influence_state.xlsx', encoding='latin1')
############################
#Plot of electroencephalogram of active state of human
human_active_state.plot()
plt.grid()
plt.title('Electroencephalogram of active state of human')
###################################################
#Graph of electrical voltage everyone electrode of electroencephalogram
#for active state of human
plt.figure()

plt.subplot(121)
plt.plot(human_active_state.iloc[:,0].values, color='red')
plt.title('Fp1 electrode (active state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_active_state.iloc[:,1].values, color='red')
plt.title('Fp2 electrode (active state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_active_state.iloc[:,2].values, color='red')
plt.title('F3 electrode (active state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_active_state.iloc[:,3].values, color='red')
plt.title('F4 electrode (active state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_active_state.iloc[:,4].values, color='red')
plt.title('F7 electrode (active state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_active_state.iloc[:,5].values, color='red')
plt.title('F8 electrode (active state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_active_state.iloc[:,6].values, color='red')
plt.title('T3 electrode (active state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_active_state.iloc[:,7].values, color='red')
plt.title('T4 electrode (active state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_active_state.iloc[:,8].values, color='red')
plt.title('C3 electrode (active state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_active_state.iloc[:,9].values, color='red')
plt.title('C4 electrode (active state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_active_state.iloc[:,10].values, color='red')
plt.title('T5 electrode (active state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_active_state.iloc[:,11].values, color='red')
plt.title('T6 electrode (active state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_active_state.iloc[:,12].values, color='red')
plt.title('P3 electrode (active state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_active_state.iloc[:,13].values, color='red')
plt.title('P4 electrode (active state of human)')
plt.grid()

plt.show()

plt.figure()

plt.subplot(121)
plt.plot(human_active_state.iloc[:,14].values, color='red')
plt.title('O1 electrode (active state of human)')
plt.grid()

plt.subplot(122)
plt.plot(human_active_state.iloc[:,15].values, color='red')
plt.title('O2 electrode (active state of human)')
plt.grid()

plt.show()
###################################################
#Concatenate DataFrame 'human_calm_state' and 'human_active_state'
frame = [human_calm_state, human_active_state]
data = pd.concat(frame)
###################################################
#Remove missing data
data.dropna(axis=0, how='any',inplace=True)
###################################################
#Converting float type to DataFrame type
df = pd.DataFrame(data=data)
###################################################
#Separation of DataFrame 'df' on independent features and target feature
independent_features = df.iloc[:,0:15]
target_feature = df.iloc[:,16]
###################################################
#Feature Selection
F_test_Y = f_classif(independent_features, target_feature)
column_names = independent_features.columns
Set_Y = {'F_statistic': F_test_Y[0], 'p_value': F_test_Y[1]}
F_test_X_Y = pd.DataFrame(data = Set_Y, index = column_names)
print(F_test_X_Y)
plt.figure()
F_test_X_Y.iloc[:,0].plot(kind='bar')
plt.ylabel('F-statistic')
plt.xlabel('features')
plt.title('Feature Selection')
plt.grid()
plt.show()
###################################################
#Standardization of features
independent_features = preprocessing.scale(independent_features)
###################################################
###################################################
###################################################
#Split of df on train (80%) and test (20%) samples
train_df, test_df = train_test_split(df, test_size = 0.2)
###################################################
#Create train and test samples of independent and dependent variables
train_X = train_df.iloc[:,0:15]
train_Y = train_df.iloc[:,16]
test_X = test_df.iloc[:,0:15]
test_Y = test_df.iloc[:,16]
###################################################
#Standardization of features
train_X = preprocessing.scale(train_X)
test_X = preprocessing.scale(test_X)
###################################################
#Initialization of 'MLPClassifier' object
Clf = MLPClassifier()
###################################################
#Train a artificial neural networks on train sample
Clf.fit(train_X, train_Y)
###################################################
#Predict on test sample
predict_test_Y = Clf.predict(test_X)
###################################################
#Creating predict_and_target_test_Y_df DataFrames
Set = {'target': test_Y, 'predict': predict_test_Y}
predict_and_target_test_Y_df = pd.DataFrame(data = Set)
###################################################
#Return unique values for dependent variable 
class_test_Y = pd.Series.unique(test_Y)
###################################################
#Visualization of the Confusion (Classification) Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round_(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2, out=None) 
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('target')
    plt.xlabel('predict')
#Compute confusion matrix
cnf_matrix = confusion_matrix(predict_and_target_test_Y_df['target'], predict_and_target_test_Y_df['predict'], labels=class_test_Y)
np.set_printoptions(precision=2)
#Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_test_Y,
                      title='Confusion matrix, without normalization')
#Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_test_Y, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
#################################################
#################################################
#################################################
#Evaluate a accuracy
accuracy = accuracy_score(test_Y, predict_test_Y)
#Print accuracy
print('*********************************************')
print('Accuracy =', accuracy)
#########################
#Evaluate a recall
recall = recall_score(test_Y, predict_test_Y)
#Print recall
print('*********************************************')
print('Recall =', recall)
#########################
#Evaluate a precision
precision = precision_score(test_Y, predict_test_Y)
#Print precision
print('*********************************************')
print('Precision =', precision)
#########################
#Evaluate a F-measure (macro-average approach)
F_measure = f1_score(test_Y, predict_test_Y, average='macro')
#Print F-measure
print('*********************************************')
print('F-measure =', F_measure)
#########################
#Evaluate a AUC-ROC
AUC_ROC = roc_auc_score(test_Y, predict_test_Y)
#Print AUC-ROC
print('*********************************************')
print('AUC-ROC =', AUC_ROC)
print('*********************************************',
      '*********************************************')
#################################################
#Evaluate a accuracy by cross-validation
cross_val_accuracy = cross_val_score(Clf, train_X, train_Y, cv=3, scoring='accuracy')
mean_accuracy = cross_val_accuracy.mean()
#Print cross-validation accuracy
print('*********************************************')
print('Cross-validation accuracy =', mean_accuracy)
#########################
#Evaluate a recall by cross-validation
cross_val_recall = cross_val_score(Clf, train_X, train_Y, cv=3, scoring='recall')
mean_recall = cross_val_recall.mean()
#Print cross-validation recall
print('*********************************************')
print('Cross-validation recall =', mean_recall)
#########################
#Evaluate a precision by cross-validation
cross_val_precision = cross_val_score(Clf, train_X, train_Y, cv=3, scoring='precision')
mean_precision = cross_val_precision.mean()
#Print cross-validation precision
print('*********************************************')
print('Cross-validation precision =', mean_precision)
#########################
#Evaluate a F-measure by cross-validation (macro-average approach)
cross_val_F_measure = cross_val_score(Clf, train_X, train_Y, cv=3, scoring='f1_macro')
mean_F_measure = cross_val_F_measure.mean()
#Print cross-validation F-measure
print('*********************************************')
print('Cross-validation F-measure =', mean_F_measure)
#########################
#Evaluate a AUC-ROC by cross-validation
cross_val_AUC_ROC = cross_val_score(Clf, train_X, train_Y, cv=3, scoring='roc_auc')
mean_AUC_ROC = cross_val_AUC_ROC.mean()
#Print cross-validation AUC-ROC
print('*********************************************')
print('Cross-validation AUC-ROC =', mean_AUC_ROC)
#########################
#Build a text report showing the main classification metrics
report = classification_report(test_Y, predict_test_Y)
print('*********************************************')
print('Text report showing the main classification metrics')
print(report)
print('*********************************************')
#################################################
#################################################
#################################################
#Saving trained model
joblib.dump(Clf, 'D:/Clf.pkl')
#########################
#Loading trained model
Clf_decision_making = joblib.load('D:/Clf.pkl')
#########################
#Decision making by classifier
decision_making = Clf_decision_making.predict(train_X)
#################################################
#################################################
#################################################






















