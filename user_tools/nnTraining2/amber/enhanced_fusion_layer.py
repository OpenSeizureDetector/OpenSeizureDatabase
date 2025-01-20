import json
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import (Layer, Add, Concatenate, MultiHeadAttention)
from amber.config import Config

# Register the custom layer to make it serializable
@register_keras_serializable(package='custom', name='EnhancedFusionLayer')
class EnhancedFusionLayer(Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(EnhancedFusionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads  # Store num_heads as an attribute
        self.key_dim = key_dim      # Store key_dim as an attribute
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        
    def call(self, inputs):
        # Concatenate inputs along the last axis
        concatenated_inputs = Concatenate()(inputs)
        # Apply multi-head attention to concatenated inputs
        attention_output = self.attention(concatenated_inputs, concatenated_inputs)
        # Add the original concatenated inputs to the attention output
        return Add()([concatenated_inputs, attention_output])
        
    def get_config(self):
        # Retrieve base Config and update with num_heads and key_dim
        Config = super(EnhancedFusionLayer, self).get_config()
        Config.update({
            "num_heads": self.num_heads,  # Use stored attribute
            "key_dim": self.key_dim       # Use stored attribute
         })
        return Config
