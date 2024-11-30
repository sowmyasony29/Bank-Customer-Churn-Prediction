import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

a=pd.read_csv("/content/drive/MyDrive/Bank+Customer+Churn+Prediction.csv")

#a.head()

print(a)


'''b=a.fillna(0)
print(b.iloc[0:120,5:8])'''


target=a['churn']
print("target variable is:")
print("",target)
features=a.drop(columns = 'churn')
print("the features are:")
print("",features)

x=a['gender']
print(x)
y=a['churn']
print(y)


import matplotlib.pyplot as plt

#(b.head())

#print(b.tail())

plt.scatter(x,y)



plt.bar(x,20)

plt.hist(x,30)

a['gender'].replace({'Female': 1,'Male':0}, inplace=True)
#a['gender'].replace({'Male': 0}, inplace=False)
a['country'].replace({'France':0,'Spain':1,'Germany':2}, inplace=True)
print(a)

x2 = a['age']
y = a['tenure']
import matplotlib .pyplot as plt
plt.xlabel('age')
plt.ylabel('tenure')
plt.plot(x2,y)
plt.show()

#normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(a)
nd = scaler.transform(a)
#print(nd)
a.loc[:, :] = nd
a.head()

import matplotlib.pyplot as plt
plt.hist ( a [ a [ 'churn' ] == 1 ] [ 'gender' ] , color = '#7A6174' , bins = 10 , label = 'Churned'  );
plt.hist ( a [  a[ 'churn' ] == 0 ] [ 'gender' ] , color = '#DB5375' , bins = 10 , label = 'Not Churned' , alpha = 0.5 ) ;
plt.xlabel ( 'Major Axis of an Ellipse in millimeters(scaled to 0-1)' )
plt.legend ( )
plt.show ( )

cols = ['customer_id',	'credit_score',	'country',	'gender',	'age',	'tenure','balance',	'products_number',	'credit_card',	'active_member',	'estimated_salary']

for label in cols[:-1]:
    plt.hist(a[a['churn'] == 1][label],color = 'red',label = 'churned',alpha = 0.5,density = True)
    plt.hist(a[a['churn'] == 0][label],color = 'blue',label = 'not churned',alpha = 0.5,density = True)
    plt.title(label)
    plt.ylabel("probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

data_variables = a.drop(columns = 'churn',axis = 1)
a_var= pd.DataFrame(data_variables, columns = ['customer_id',	'credit_score',	'country',	'gender',	'age',	'tenure','balance',	'products_number',	'credit_card',	'active_member',	'estimated_salary'])
correlation_matrix = a_var.corr()
# Set up the figure and axis
plt.figure(figsize=(8, 6))
ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Customize the plot
plt.title("Correlation Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Show the plot
plt.show()

import numpy as np
target = a['churn']
y = np.array(target)
print(y.shape)
features = a.drop(columns = ['credit_score','churn'],axis = 1)
print(features)
X = np.array(features)
print(X.shape)
print(y)
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(len(X_test))
print(len(y_test))

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

x2 = a['age']
y = a['churn']
import matplotlib .pyplot as plt
plt.xlabel('age')
plt.ylabel('churn')
plt.plot(x2,y)
plt.show()

x2 = a['gender']
y = a['churn']
import matplotlib .pyplot as plt
plt.xlabel('gender')
plt.ylabel('churn')
plt.plot(x2,y)
plt.show()

x2 = a['age']
y = a['churn']
import matplotlib .pyplot as plt
plt.xlabel('age')
plt.ylabel('churn')
plt.plot(x2,y)
plt.show()

from sklearn.model_selection import train_test_split
import numpy as np
yp=[]
 #implementing perceptron model
def perceptron(w,x1,b):
  q=0
  for i in range(len(x1)):
    for j in range(len(w)):
      s=(x1[i][j]*w[j])+b
      s2=1/(1+np.exp(-s))
      if(s2>=0.5):
        yp.append(1)
      else:
        yp.append(0)
    q=q+1
  print(q)
  return yp
w=[]
#print("enter weights w : ")
w=[0.3,0.2,0.6,0.1,0.2,0.8,0.2,0.8,0.9,0.3]
b= float(input("enter b:"))
perceptron(w,X_train,b)







# Assuming you have the true labels for your data
# Assuming you have the true labels for your data
#true_labels = [1, 0, 1]  # Replace with your actual true labels

# Calculate accuracy
predictions = sum(1 for y, yp in zip(y, yp) if y == yp)
total_predictions = len(y)
accuracy = predictions / total_predictions

print("Accuracy:" ,accuracy)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
yp=[]
 #implementing perceptron model

def perceptron(w,x1,b):
    q=0
    for i in range(len(x1)):
       for j in range(len(w)):
          s=(x1[i][j]*w[j])+b
          s2=1/(1+np.exp(-s))
          if(s2>=0.5):
             yp.append(1)
          else:
             yp.append(0)
    q=q+1
    return yp
accuracy_list = []
#print("enter weights w : ")
num_iter = 8
import random
for i in range(num_iter):
    w = []
    yp=[]
    for j in range(10):
        val = random.uniform(0.0,1.0)
        w.append(val)
    #print(w)
    b = random.uniform(0.0,1.0)
    yp = perceptron(w,X_train,b)
    print(yp)
    predictions = sum(1 for y, yp in zip(y, yp) if y == yp)
    total_predictions = len(y)
    accuracy = predictions / total_predictions
    print(accuracy)
    accuracy_list.append(accuracy)
    print(accuracy_list)


num_iterations= [1,2,3,4,5,6,7,8]
plt.figure( figsize = ( 10 , 5 ) )
plt.xlabel("number of iterations")
plt.ylabel("accuracy")
# Accuracy scores plot obtained from above iterations
plt.plot(num_iterations, accuracy_list, label = 'Accuracy', marker = 'o', color = '#42253B')

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Create a Perceptron instance and fit the training data
clf = Perceptron(random_state=42, max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)
print(y_pred)
# Calculate the accuracy of the model
accuracy_PM = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_PM)
print(classification_report(y_test, y_pred))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



import numpy as np
costs_list = []
num_iter = []
# implementation of logistic regression from scratch without any libraries
class Logisticregression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y):
        m = X.shape[0]
        h = self.sigmoid(np.dot(X, self.weights) + self.bias)
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.num_iterations):
            h = self.sigmoid(np.dot(X, self.weights) + self.bias)
            dw = (1 / m) * np.dot(X.T, (h - y))
            db = (1 / m) * np.sum(h - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            cost = self.cost_function(X, y)
            self.cost_history.append(cost)
            if i % 100 == 0:
                cost = self.cost_function(X, y)
                print(f"Cost after iteration {i}: {cost}")

        print("",self.cost_history)


    def predict(self, X):
        h = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_pred = np.where(h > 0.5, 1, 0)
        return y_pred


**LOGISTIC REGRESSION**

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

model = Logisticregression()

model.weights = [0.3, -0.3, -2.9]

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#Importing logistic regression function from scikit learn directly

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
# Evaluate the model
accuracy_LR = accuracy_score(y_test, y_pred)
print("Accuracy is:",accuracy_LR)
print(classification_report(y_test, y_pred))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#SVM INBUILT FUNCTION
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create an SVM instance and fit the training data
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svm.predict(X_test)
print(y_pred)
# Calculate the accuracy of the model
accuracy_SVM = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_SVM)
print(classification_report(y_test, y_pred))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

**PCA**

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
pca = PCA(n_components=6)
print(scaled_features)

reduced_features = pca.fit_transform(scaled_features)
print(reduced_features)

**KNN**

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)
accuracy_KNN = accuracy_score(y_test, y_pred)
print("Accuracy is :",accuracy)
print(classification_report(y_test, y_pred))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

**BOOTSTRAP**

from sklearn.utils import resample
# Define the machine learning model
model = LogisticRegression()
# Define the number of bootstrap iterations
n_iterations = 100
# Perform the bootstrap
scores_lgr = list()
for i in range(n_iterations):
    # Resample the training dataset
    X_sample, y_sample = resample(X_train, y_train)
    # Fit the model on the resampled dataset
    model.fit(X_sample, y_sample)
    # Evaluate the model on the testing dataset
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores_lgr.append(score)

iterations = [i for i in range(1, 101)]
plt.plot(iterations, scores_lgr)
# Add a title to the graph
plt.title("Accuracy Scores for 100 Iterations in logistic regression")
# Label the x-axis
plt.xlabel("Iterations")
# Label the y-axis
plt.ylabel("Accuracy Scores")
# Display the graph
plt.show()

from sklearn.utils import resample
# Define the machine learning model
model = Perceptron(random_state=42, max_iter=1000, tol=1e-3)
# Define the number of bootstrap iterations
n_iterations = 100
# Perform the bootstrap
scores_per = list()
for i in range(n_iterations):
    # Resample the training dataset
    X_sample, y_sample = resample(X_train, y_train)
    # Fit the model on the resampled dataset
    model.fit(X_sample, y_sample)
    # Evaluate the model on the testing dataset
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores_per.append(score)

iterations = [i for i in range(1, 101)]
plt.plot(iterations, scores_per)
# Add a title to the graph
plt.title("Accuracy Scores for 100 Iterations in Perceptron")
# Label the x-axis
plt.xlabel("Iterations")
# Label the y-axis
plt.ylabel("Accuracy Scores")
# Display the graph
plt.show()

from sklearn.utils import resample
# Define the machine learning model
model = SVC(kernel='linear', C=1, random_state=42)
# Define the number of bootstrap iterations
n_iterations = 100
# Perform the bootstrap
scores_svc= list()
for i in range(n_iterations):
    # Resample the training dataset
    X_sample, y_sample = resample(X_train, y_train)
    # Fit the model on the resampled dataset
    model.fit(X_sample, y_sample)
    # Evaluate the model on the testing dataset
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores_svc.append(score)

iterations = [i for i in range(1, 101)]
plt.plot(iterations, scores_svc)
# Add a title to the graph
plt.title("Accuracy Scores for 100 Iterations in SVM")
# Label the x-axis
plt.xlabel("Iterations")
# Label the y-axis
plt.ylabel("Accuracy Scores")
# Display the graph
plt.show()

from sklearn.utils import resample
# Define the machine learning model
model = KNeighborsClassifier(n_neighbors=3)
# Define the number of bootstrap iterations
n_iterations = 100
# Perform the bootstrap
scores_knn = list()
for i in range(n_iterations):
    # Resample the training dataset
    X_sample, y_sample = resample(X_train, y_train)
    # Fit the model on the resampled dataset
    model.fit(X_sample, y_sample)
    # Evaluate the model on the testing dataset
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores_knn.append(score)

iterations = [i for i in range(1, 101)]
plt.plot(iterations, scores_knn)
# Add a title to the graph
plt.title("Accuracy Scores for 100 Iterations in KNN ")
# Label the x-axis
plt.xlabel("Iterations")
# Label the y-axis
plt.ylabel("Accuracy Scores")
# Display the graph
plt.show()

algorithm_names = ['Perceptron network','Logistic Regression','Support Vector Machine','KNN Classifier']
accuracy_scores = [ accuracy_PM , accuracy_LR , accuracy_SVM,accuracy_KNN ]
# Plotting the scores
plt.figure( figsize = ( 10 , 5 ) )
# Accuracy scores plot
plt.plot(algorithm_names, accuracy_scores, label = 'Accuracy', marker = 'o', color = '#42253B')
plt.xlabel( 'Classification Algorithms' )
plt.ylabel( 'Accuracy Scores' )
plt.title( 'Performance Comparison' )
# Adding a legend
plt.legend()

# Rotating the x-axis labels for better visibility
plt.xticks( rotation = 45 )

# Displaying the plot

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Create a list of model names
model_names = ['Logistic regression', 'Perceptron model', 'SVM Classifier','KNN classifier']

# Calculate the mean and standard deviation for each model's accuracy scores
mean_scores = [np.mean(scores) for scores in [scores_lgr, scores_per,scores_svc,scores_knn]]
std_scores = [np.std(scores) for scores in [scores_lgr,scores_per,scores_svc,scores_knn]]

# Plot the mean accuracy scores over models
plt.figure(figsize=(10, 6))
plt.bar(model_names, mean_scores, yerr=std_scores, capsize=5, color=['blue', 'green', 'red','purple'])
plt.ylabel('Mean Accuracy')
plt.title('Mean Accuracy and Standard Deviation Over Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Calculate the confidence interval in logistic regression
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(scores_lgr, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(scores_lgr, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

mean_accuracy = np.mean(scores_lgr)
std_accuracy = np.std(scores_lgr)

print(f"Mean Accuracy(logistic regression): {mean_accuracy:.2f}")
print(f"Standard Deviation(logistic regression): {std_accuracy:.2f}")          #mean standard deviation code

# Calculate the confidence interval in perceptron learning
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(scores_per, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(scores_per, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

mean_accuracy = np.mean(scores_per)
std_accuracy = np.std(scores_per)

print(f"Mean Accuracy(perceptron learning): {mean_accuracy:.2f}")
print(f"Standard Deviation(perceptron learning): {std_accuracy:.2f}")

# Calculate the confidence interval in SVM
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(scores_svc, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(scores_svc, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

mean_accuracy = np.mean(scores_svc)
std_accuracy = np.std(scores_svc)

print(f"Mean Accuracy(SVM): {mean_accuracy:.2f}")
print(f"Standard Deviation(SVM): {std_accuracy:.2f}")

# Calculate the confidence interval in KNN
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(scores_knn, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(scores_knn, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

mean_accuracy = np.mean(scores_knn)
std_accuracy = np.std(scores_knn)

print(f"Mean Accuracy(KNN): {mean_accuracy:.2f}")
print(f"Standard Deviation(KNN): {std_accuracy:.2f}")
