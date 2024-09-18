const express = require('express');
const bodyParser = require('body-parser');
const ort = require('onnxruntime-node');

const app = express();
app.use(bodyParser.json());

let session;

async function loadModel() {
    session = await ort.InferenceSession.create('modell.onnx');
}

app.post('/predict', async (req, res) => {
    if (!session) {
        return res.status(500).send('Model is not loaded');
    }

    try {
        const inputData = req.body.data;

        // Ensure the input data is in the format required by your model
        const inputs = { [session.inputNames[0]]: inputData };
        const results = await session.run(inputs);

        res.json(results);
    } catch (error) {
        res.status(500).send(error.message);
    }
});

app.listen(3000, () => {
    console.log('Server running on port 3001');
    loadModel().catch(err => console.error('Error loading model:', err));
});
