import sys
# Trying to avoid an import error that may arise
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
except Exception as e:
    print(f"Shiddd dog what wrong {e}")
    sys.exit()
pd.options.display.width = 0
# Loading file
dataset_filename = "Churn_Modelling.csv"
dataset = pd.read_csv(dataset_filename)
#print(len(dataset.columns.values))
#Changing Categorical Data into numerical data. Doing it now because next step changes values into numpy array
data_dummies = pd.get_dummies(dataset.values[:,4])
#print(data_dummies)
x_merge = pd.concat([dataset,data_dummies], axis='columns')
#print(x_merge)
 #Has to be array type!!!
# Get x-values and y-values to look at. Y-values are answers while x is  input
#for i, n  in enumerate(x_merge.columns.values):
#    print(i,n)

drop_list = [x_merge.columns.values[i] for i in range(5)]
x_merge.drop(columns=drop_list, inplace = True)
#print(x_merge.columns.values[-4:])
#print(x_merge.columns.values[:8])
cols = list(x_merge.columns.values[-4:]) + list(x_merge.columns.values[:8])
print(cols)
print(x_merge.dtypes)
x_merge = x_merge[cols]
#print(x_merge)
x = x_merge.iloc[:, 1:].values
y = x_merge.iloc[:,0].values
#print(x[:,3])
changer = preprocessing.LabelEncoder()
x[:,3] = changer.fit_transform(x[:,3])
#print(y)
#print(x[:,3])
#Get out of the dummy variable trap
x = x[:,1:]

'''
The dummy variable trap:
This is when create dummy variables and there are multiple rows of 1s and 0s created.
It is suggested to remove one of the columns to remove redundancy and stop the variables from becoming highly correlated (not sure)
For example:
Male Female
1      0
0      1
0      1

If someone has a 1 for male, we know that female will be 0 so only one of the columns needs to be there.
In this code example, we can remove one country because if the other 2 are 0 we know that the other one is the country.
'''

#Scale all data
#print(x)
scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)
#print(len(x[0]))
#print(x)
#Split Data in to Training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.30, random_state= 42)

try:
    import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
except:
    print(f'Importing for keras went wrong')
    sys.exit()

classifier = Sequential()
#output_dim --> units, init --> kernel_initializer (changed from tutorial)
#Units is number of outputs from this layer, kernel_initializer is setting the kernels init weights
#activation function defines the output from each layer
#input_dim is the amount of inputs the function will get, only done for first dense
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=10))

#adding second hidden layer
classifier.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))
#Output layer
#For cases with more then soft-max?
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fit the classifier
classifier.fit(x_train, y_train, batch_size=15, nb_epoch=100)
classifier.save('my_model.h5')
'''
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
customer = np.array([[0,0,1,40,3,60000.00,2,1, 1,50000]])
customer = scaler.transform(customer)
customer.shape
'''
'''
Exited               int64
France               uint8
Germany              uint8
Spain                uint8
Gender              object
Age                  int64
Tenure               int64
Balance            float64
NumOfProducts        int64
HasCrCard            int64
IsActiveMember       int64
EstimatedSalary    float64
dtype: object
'''
'''
q = classifier.predict(customer)
print(q>0.5)
'''
