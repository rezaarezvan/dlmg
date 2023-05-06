from glob import glob
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import tensorflow as tf

SEQUENCE_LENGTH = 1024  # You can adjust this value based on your preference

# Load the training and evaluation data
train_data = [np.load(file, allow_pickle=True)
              for file in glob('../data/train/*.npy')]
eval_data = [np.load(file, allow_pickle=True)
             for file in glob('../data/eval/*.npy')]

# Pad/truncate train_data and eval_data
train_data = pad_sequences(train_data, maxlen=SEQUENCE_LENGTH, dtype='float32')
eval_data = pad_sequences(eval_data, maxlen=SEQUENCE_LENGTH, dtype='float32')

# Define the model architecture
input_layer = Input(shape=(SEQUENCE_LENGTH, 128))
x = LSTM(512, return_sequences=True)(input_layer)
x = LSTM(512, return_sequences=True)(x)
output_layer = Dense(128, activation='sigmoid')(x)

# Create the model
model = Model(input_layer, output_layer)
model.summary()

# Compile the model


@tf.function
def train_step(train_data):
    with tf.GradientTape() as tape:
        predictions = model(train_data)
        loss = model.compiled_loss(train_data, predictions)
    grads = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    model.compiled_metrics.update_state(train_data, predictions)


model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)

# Train the model
for epoch in range(50):
    print(f"Epoch {epoch+1}/50")
    for i, seq in enumerate(train_data):
        train_step(tf.expand_dims(seq, 0))
    model.reset_metrics()

# Save the model
model.save('../model/model.h5')

print("Training complete!")
