import numpy as np
import pandas as pd
import loadData, labelEncoder, testTrainSplit, inputTokenizer, model, plot

training_sentences = []
training_labels=[]
labels=[]
responses=[]

# Load the data from intents.json
training_sentences, training_labels, responses, labels = loadData.load_data()

# Encode the labels into integers for classification
training_labels = labelEncoder.label_encoder(training_labels)

# Create test and train data
training_sentences, training_labels, testing_sentences, testing_labels = testTrainSplit.createTestTrainSets(training_sentences, training_labels)

#define model parameters and layer dimensions
num_classes = len(labels)
vocab_size = 5000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"
epochs = 300

# tokenize and convert the sentences to vector representation. Also, add padding to make all sentences equal length
train_padded_sequences, test_padded_sequences = inputTokenizer.tokenizeInput(vocab_size, oov_token, training_sentences, testing_sentences, max_len)

# Define the model to be used for training
model = model.defineModel(vocab_size, embedding_dim, max_len, num_classes)
model.summary()

# Train the model
history = model.fit(train_padded_sequences, np.array(training_labels), epochs=epochs, validation_data=(test_padded_sequences,testing_labels))

# plot the training and validation accuracy to determine model performance
plot.plotMetrics(history)

# Save the model for future use
model.save("chat_bot")
