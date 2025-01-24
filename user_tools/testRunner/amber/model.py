import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import save_model as tf_save_model, load_model as tf_load_model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import (
    Layer, Add, Input, Conv1D, BatchNormalization, Activation,
    MaxPooling1D, Bidirectional, LSTM, Dense, Dropout,
    Reshape, Permute, Attention, GlobalMaxPooling1D, Concatenate, MultiHeadAttention
)
from amber.enhanced_fusion_layer import EnhancedFusionLayer
from amber.config import Config
import os

class Amber:
    def __init__(self, row_hidden, col_hidden, num_classes):
        self.row_hidden = row_hidden
        self.col_hidden = col_hidden
        self.num_classes = num_classes
        self.model = None

    def conv_block(self, in_layer, filters, kernel_size):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(in_layer)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        return conv

    def lstm_pipe(self, in_layer):
        b1 = self.conv_block(in_layer, filters=64, kernel_size=3)
        b1 = MaxPooling1D(pool_size=2)(b1)
        b2 = self.conv_block(b1, filters=128, kernel_size=3)
        b2 = MaxPooling1D(pool_size=2)(b2)
        b3 = self.conv_block(b2, filters=256, kernel_size=3)
        b3 = MaxPooling1D(pool_size=2)(b3)
        encoded_rows = Bidirectional(LSTM(self.row_hidden, return_sequences=True))(b3)
        return LSTM(self.col_hidden)(encoded_rows)

    def build_model(self, num_features, input_shape, num_heads=4, key_dim=64):
        input_layers = []
        lstm_outputs = []

        for i in range(num_features):
            input_layer = Input(shape=input_shape, name=f'input_feature_{i+1}')
            input_layers.append(input_layer)
            lstm_output = self.lstm_pipe(Permute(dims=(1, 2))(input_layer))
            lstm_output_reshaped = Reshape((-1,))(lstm_output)
            lstm_outputs.append(lstm_output_reshaped)

        attention_outputs = []
        for i, lstm_output in enumerate(lstm_outputs):
            lstm_output_reshaped = Reshape((-1, lstm_output.shape[-1]))(lstm_output)
            attention_output = Attention()([lstm_output_reshaped, lstm_output_reshaped])
            attention_outputs.append(attention_output)

        fused_features = EnhancedFusionLayer(num_heads=num_heads, key_dim=key_dim)(attention_outputs)

        dense_output = Dense(128, activation='relu', kernel_regularizer=l2(0.0001))(fused_features)
        dense_output = BatchNormalization()(dense_output)
        dense_output = Dropout(0.1)(dense_output)
        prediction = Dense(self.num_classes, activation='softmax')(GlobalMaxPooling1D()(dense_output))

        self.model = Model(inputs=input_layers, outputs=prediction)

    def compile_model(self):
        optimizer = RMSprop(learning_rate=Config.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def train_model(self, X_train_list, y_train, X_val_list, y_val, epochs=Config.epochs, batch_size=Config.batch_size):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

        history = self.model.fit(
            X_train_list, y_train,
            epochs=Config.epochs,
            batch_size=Config.batch_size,
            validation_data=(X_val_list, y_val),
            verbose=1,
            callbacks=[reduce_lr, early_stopping]
        )
        return history

    def save_model(self, path="Models/AMBER_Model.keras"):
        """Save the model, including custom layers, to a .keras file."""
        # Save the model architecture and weights
        self.model.save(path)
        
        # Extract custom layer metadata from the existing layers in the model
        custom_objects_metadata = {}
        for layer in self.model.layers:
            if isinstance(layer, EnhancedFusionLayer):
                custom_objects_metadata[layer.name] = layer.get_config()

        # Save custom layers metadata (if any)
        with open(f"{path}_custom_objects.json", "w") as file:
            json.dump(custom_objects_metadata, file)
                


    @staticmethod
    def load_model(path="Models/AMBER_Model.keras"):
        """Load a saved model, including custom layers."""
        try:
            # Load custom layer metadata
            metadata_path = f"{path}_custom_objects.json"
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Custom objects metadata not found at {metadata_path}.")

            with open(metadata_path, "r") as file:
                custom_objects_metadata = json.load(file)

            # Load the model architecture and weights, specifying the custom layers
            model = tf_load_model(path, custom_objects={**custom_objects_metadata})

            # Initialize and return an Amber model object
            amber_model = Amber(row_hidden=Config.row_hidden, col_hidden=Config.col_hidden, num_classes=Config.N_CLASSES)
            amber_model.model = model  # Assign the loaded model to amber_model
            
            return amber_model
        except Exception as e:
            raise Exception(f"Error loading model from {path}: {e}")
        
    def evaluate_model(self, X_test, y_test, batch_size=Config.batch_size):
        return self.model.evaluate(X_test, y_test, batch_size= Config.batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def architecture(self):
        return self.model.summary()