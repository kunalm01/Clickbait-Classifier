#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('clickbait_data.csv')


# In[3]:


df.head()


# In[4]:


X = df['headline'].values


# In[5]:


y = df['clickbait'].values


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=36)


# In[8]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[9]:


from tensorflow.keras.layers import TextVectorization


# In[10]:


vocab_size = 6000
maxlen = 600

vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=maxlen,output_mode="int")

vectorizer.adapt(X_train)

X_train = vectorizer(X_train)
X_test = vectorizer(X_test)


# In[11]:


X_train


# In[12]:


vectorizer.get_vocabulary()


# In[13]:


vectorizer('Can you believe that')[:10]


# In[14]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional,Flatten


# In[15]:


model = Sequential()
model.add(Embedding(vocab_size, output_dim = 32, input_length=maxlen))
model.add(Bidirectional(LSTM(32, return_sequences=True, activation = 'tanh')))
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation = 'relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()


# In[16]:


from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[17]:


callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='weights.h5',
        monitor='val_accuracy', 
        mode='max', 
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
]


# In[18]:


y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))


# In[19]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, validation_data=(X_test, y_test), epochs=10, callbacks=callbacks)


# In[20]:


model.load_weights('weights.h5')
model.save('model')


# In[21]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
x = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, acc, 'g', label='Training acc')
plt.plot(x, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, loss, 'g', label='Training loss')
plt.plot(x, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[22]:


preds = [round(i[0]) for i in model.predict(X_test)]


# In[24]:


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[25]:


cm = confusion_matrix(y_test, preds)


# In[26]:


plt.figure()
plot_confusion_matrix(cm, figsize=(8,6),  cmap=plt.cm.Greens)
plt.xticks(range(2), ['Not clickbait', 'Clickbait'], fontsize=16)
plt.yticks(range(2), ['Not clickbait', 'Clickbait'], fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()


# In[27]:


tn, fp, fn, tp = cm.ravel()


# In[28]:


accuracy = (tp+tn)/(tp+tn+fp+fn)
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f1_score = 2*precision*recall/(precision+recall)

print("Accuracy of the model is {:.4f}".format(accuracy))
print("Precision of the model is {:.4f}".format(precision))
print("Recall of the model is {:.4f}".format(recall))
print("F1 score of the model is {:.4f}".format(f1_score))


# In[29]:


test = ['My biggest laugh reveal ever!',
        'Learning game development with Unity',
        'A tour of Japan\'s Kansai region',
        '12 things NOT to do in Europe',
        'How to train your dog in 5 easy steps',
        'The ultimate guide to weight loss',
        'Study finds no link between cell phone use and cancer',
        'New treatment shows promise in fighting off antibiotic-resistant infections'
       ]
vec_text = vectorizer(test)
ans = [round(i[0]) for i in model.predict(vec_text)]
for (text,res) in zip(test,ans):
  label = 'Clickbait' if res == 1.0 else 'Not Clickbait'
  print("{} - \033[1m{}\033[0m".format(text,label))

