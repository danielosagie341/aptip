const express = require('express');
const bodyParser = require('body-parser');
const { loadModel, predictThreat } = require('./model_utils'); // Utility functions to handle model loading and prediction

const app = express();
const port = 5000;

app.use(bodyParser.json());

// Load the model and tokenizer once at server startup
let model, tokenizer;

loadModel().then(loadedModel => {
    model = loadedModel.model;
    tokenizer = loadedModel.tokenizer;
    console.log('Model and tokenizer loaded successfully');
}).catch(err => {
    console.error('Failed to load model and tokenizer:', err);
});

app.post('/predict', async (req, res) => {
    try {
        const { text } = req.body;
        if (!model || !tokenizer) {
            return res.status(500).json({ error: 'Model not loaded' });
        }
        const prediction = await predictThreat(model, tokenizer, text);
        res.json({ prediction });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
