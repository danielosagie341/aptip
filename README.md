# Advanced Persistent Threat Intelligence Platform (APTip)

APTip is a comprehensive, full-stack web application designed to provide users with a suite of advanced cybersecurity tools. It integrates artificial intelligence and various external APIs to offer real-time threat intelligence, making it a powerful tool for security analysts, researchers, and enthusiasts.

## Features

*   **AI-Powered Threat Analysis**: Paste any text (like emails, articles, or logs) to get an AI-driven analysis of potential cyber threats, complete with a confidence score.
*   **Live Threat Globe**: An interactive 3D globe visualizing mock real-time cyber threats originating from and targeting different countries.
*   **IP Reputation Checker**: Check the risk profile of any public IP address. The tool provides detailed information, including the abuse confidence score, country, and ISP, by integrating with the AbuseIPDB API.
*   **CVE Search**: Search the National Vulnerability Database (NVD) for specific Common Vulnerabilities and Exposures (CVEs) by keyword and date range.

## Tech Stack

*   **Frontend**: React, React Router, Recharts (for charts), React-Globe.gl (for 3D globe).
*   **Node.js Backend**: Express.js server to handle IP reputation checks.
*   **Python Backend**: Flask server to:
    *   Serve a fine-tuned Hugging Face `DistilBert` model for threat prediction.
    *   Proxy requests to the NVD CVE API.
    *   Provide data for the threat globe.
*   **Styling**: Custom CSS with a modern, glassmorphism-inspired design.

## Getting Started

To get the application running locally, you need to start its three main components in separate terminals: the Python backend, the Node.js backend, and the React frontend.

### 1. Start the Python Backend (Threat Analysis & CVEs)

This server runs on port `5001` and handles machine learning predictions, CVE searches, and threat globe data.

```bash
# Navigate to the project directory

# Run the Python server using its virtual environment
python model_server.py
```

### 2. Start the Node.js Backend (IP Check)

This server runs on port `5000` and is responsible for the IP reputation checks.

```bash
# In a new terminal, navigate to the project directory

# Start the Node.js server
node server.js
```

### 3. Start the React Frontend

This will start the user interface on port `3000`.

```bash
# In a third terminal, navigate to the project directory

# Install dependencies (if you haven't already)
npm install

# Start the React development server
npm start
```

Once all three services are running, you can access the application by opening [http://localhost:3000](http://localhost:3000) in your browser.
