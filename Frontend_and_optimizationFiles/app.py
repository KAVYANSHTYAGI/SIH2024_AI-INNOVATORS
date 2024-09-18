import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import custom_object_scope
from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import heapq
from collections import Counter

class EnsembleAdaptersLayer(Layer):
    def __init__(self, num_adapters, d_model, bottleneck_dim, **kwargs):
        super(EnsembleAdaptersLayer, self).__init__(**kwargs)
        self.num_adapters = num_adapters
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.adapters = []

    def build(self, input_shape):
        for _ in range(self.num_adapters):
            self.adapters.append(
                tf.keras.Sequential([
                    tf.keras.layers.Dense(self.bottleneck_dim, activation='relu'),
                    tf.keras.layers.Dense(self.d_model)
                ])
            )

    def call(self, inputs):
        adapter_outputs = [adapter(inputs) for adapter in self.adapters]
        outputs = tf.reduce_mean(tf.stack(adapter_outputs, axis=-1), axis=-1)
        return outputs

    def get_config(self):
        config = super(EnsembleAdaptersLayer, self).get_config()
        config.update({
            'num_adapters': self.num_adapters,
            'd_model': self.d_model,
            'bottleneck_dim': self.bottleneck_dim
        })
        return config

# Initialize Flask app
app = Flask(__name__)

model = None  # Initialize model variable

# Huffman decompression functions
def build_huffman_tree(freq):
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def decode_huffman(encoded_data, huffman_tree):
    reverse_mapping = {code: symbol for symbol, code in huffman_tree}
    current_code = ""
    decoded_bytes = bytearray()

    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_mapping:
            decoded_bytes.append(reverse_mapping[current_code])
            current_code = ""

    return bytes(decoded_bytes)

def huffman_decompress(compressed_file_path):
    with open(compressed_file_path, 'rb') as file:
        encoded_data = file.read()

    # Assuming huffman_tree is saved or can be reconstructed; 
    # for simplicity, use a predefined Huffman tree
    huffman_tree = [('A', '0'), ('B', '10'), ('C', '11')]  # Example tree
    huffman_tree = build_huffman_tree(Counter(dict(huffman_tree)))

    decoded_data = decode_huffman(encoded_data, huffman_tree)
    return decoded_data

# Load the model with custom object scope
try:
    # Decompress the Huffman-encoded model file
    decompressed_model_data = huffman_decompress('C:/Users/hp/Desktop/SIH/model_compressed.huffman')
    
    # Load the decompressed model
    with custom_object_scope({'EnsembleAdaptersLayer': EnsembleAdaptersLayer}):
        model = load_model(BytesIO(decompressed_model_data))  # Load model from BytesIO
        print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", e)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/detect', methods=['POST'])
def detect():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded."})

    try:
        data = request.json['image']
        image = process_image(data)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image)
        if prediction[0][0] > 0.5:
            return jsonify({"status": "success", "message": "Liveness Verified!"})
        else:
            return jsonify({"status": "failure", "message": "Liveness Failed!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def process_image(data):
    img_data = base64.b64decode(data)
    img = Image.open(BytesIO(img_data)).resize((224, 224))
    img_array = np.array(img) / 255.0
    # Ensure the image array has the correct number of channels
    if img_array.ndim == 2:  # If grayscale
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.repeat(img_array, 3, axis=-1)  # Convert to RGB
    elif img_array.shape[-1] != 3:  # If not RGB
        img_array = img_array[:, :, :3]  # Convert to RGB by slicing
    return img_array

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
