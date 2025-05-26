import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras import layers, models
from sklearn.linear_model import LogisticRegression


np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


data = pd.read_csv('MultipleFiles/data.csv')


data = data.drop(columns=['Unnamed: 32'], errors='ignore')


imputer = SimpleImputer(strategy='mean')
data.iloc[:, 2:] = imputer.fit_transform(data.iloc[:, 2:])


data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})


X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_predictions))

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

cnn_model = models.Sequential()
cnn_model.add(layers.Input(shape=(X_train.shape[1], 1)))  
cnn_model.add(layers.Conv1D(32, 3, activation='relu'))
cnn_model.add(layers.MaxPooling1D(2))
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(64, activation='relu'))
cnn_model.add(layers.Dense(1, activation='sigmoid'))

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.2)
cnn_predictions = (cnn_model.predict(X_test_cnn) > 0.5).astype("int32")
cnn_accuracy = accuracy_score(y_test, cnn_predictions)
print("CNN Accuracy:", cnn_accuracy)
print(classification_report(y_test, cnn_predictions))


ir_model = LogisticRegression(max_iter=1000)
ir_model.fit(X_train, y_train)
ir_predictions = ir_model.predict(X_test)
ir_accuracy = accuracy_score(y_test, ir_predictions)
print("IR Model (Logistic Regression) Accuracy:", ir_accuracy)
print(classification_report(y_test, ir_predictions))


print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"CNN Accuracy: {cnn_accuracy}")
print(f"IR Model Accuracy: {ir_accuracy}")


best_model = max(rf_accuracy, cnn_accuracy, ir_accuracy)
if best_model == rf_accuracy:
    print("Best Model: Random Forest")
elif best_model == cnn_accuracy:
    print("Best Model: CNN")
else:
    print("Best Model: IR Model (Logistic Regression)")