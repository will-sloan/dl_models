'''
Getting that gold metal for the challenge --> Get 86% accuracy on the model
'''
#Pre-processing!!!
import sys
# Trying to avoid an import error that may arise
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import GridSearchCV
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
except Exception as e:
    print(f"Shiddd dog what wrong {e}")
    sys.exit()
pd.options.display.width = 0
# Loading file
dataset_filename = "Churn_Modelling.csv"
dataset = pd.read_csv(dataset_filename)
#Changing Categorical Data into numerical data. Doing it now because next step changes values into numpy array
data_dummies = pd.get_dummies(dataset.values[:,4])
x_merge = pd.concat([dataset,data_dummies], axis='columns')
drop_list = ['RowNumber', 'CustomerId', 'Surname', 'Geography']
x_merge.drop(columns=drop_list, inplace = True)
cols = list(x_merge.columns.values[-4:]) + list(x_merge.columns.values[:9])
x_merge = x_merge[cols]

x = x_merge.iloc[:, 1:].values
y = x_merge.iloc[:,0].values
changer = preprocessing.LabelEncoder()
x[:,4] = changer.fit_transform(x[:,4])
#Get out of the dummy variable trap
x = x[:,1:]

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)

#Split Data in to Training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.30, random_state= 42)
#print(len(x_train[0,:]))


#The Model!!!
def build_classifier(optimizer):
    classifier = Sequential()
    #First Hidden
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    #classifier.add(Dropout(p=0.1))
    #Second Hidden
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    #classifier.add(Dropout(p=0.1))
    #Output line
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier_new = KerasClassifier(build_fn= build_classifier)
parameters = {'batch_size': [25, 32, 48],
              'epochs': [100, 300, 500],
              'optimizer': ['adam', 'sgd','rmsprop', 'adagrad']}
grid_search = GridSearchCV(estimator=classifier_new, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(x_train, y_train)
print('\n\n\n')
print('Best parameters')
print(grid_search.best_params_)
print('\n\n\nBest Score')
print(grid_search.best_score_)
classifier_new.save('ann_model.h5')
#UPDATE DROPOUT
