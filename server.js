const express = require('express');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

const API_KEY = process.env.ABUSE_IP_API_KEY;
const BASE_URL = 'https://api.abuseipdb.com/api/v2';

app.post('/check-ip', async (req, res) => {
  const { ip } = req.body;

  if (!ip) {
    return res.status(400).json({ error: 'IP address is required' });
  }

  try {
    const response = await axios.get(`${BASE_URL}/check`, {
      params: { ipAddress: ip },
      headers: {
        'Key': API_KEY,
        'Accept': 'application/json',
      },
    });
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching data from AbuseIPDB:', error);
    res.status(500).json({ error: 'An error occurred while fetching data' });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});