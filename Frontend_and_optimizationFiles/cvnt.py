import tensorflow as tf

# Step 1: Load the Keras model from the .h5 file
model = tf.keras.models.load_model('C:/Users/hp/Desktop/SIH/model2.h5')

# Step 2: Convert the model to TensorFlow Lite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable post-training quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: If you want to perform full integer quantization
# converter.target_spec.supported_types = [tf.float16]

# Convert the model to a TensorFlow Lite format
tflite_quantized_model = converter.convert()

# Step 3: Save the quantized model to a .tflite file
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized_model)

print("Quantized model saved as 'model_quantized.tflite'")
