import React, { useState } from 'react';
import './IPCheck.css';

function IPCheck() {
  const [ipAddress, setIpAddress] = useState('');
  const [ipDetails, setIpDetails] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setIpDetails(null);

    // LOGIC: If we are running on localhost, look for Node on 5000. 
    // If we are online (Render), look for the Node service at your NEW Render URL.
    const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    
    // REPLACE 'https://aptip-node.onrender.com' with the actual URL from your new Render service
    const API_URL = isLocal 
      ? 'http://localhost:5000' 
      : 'https://aptip-node.onrender.com'; 

    try {
      const response = await fetch(`${API_URL}/check-ip`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ip: ipAddress }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Network response was not ok');
      }

      const data = await response.json();
      setIpDetails(data.data); // The data is nested under a 'data' key
    } catch (error) {
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const getAbuseStatus = (abuseConfidence) => {
    if (abuseConfidence === 0) return { status: 'safe', label: 'Safe' };
    if (abuseConfidence < 50) return { status: 'suspicious', label: 'Suspicious' };
    return { status: 'malicious', label: 'Malicious' };
  };

  console.log(ipDetails)

  return (
    <div className="container">
      <div className="ip-check-container fade-in">
        <div className="card">
          <div className="page-header">
            <h1 className="page-title">🌐 IP Address Check</h1>
            <p className="page-subtitle">
              Check IP addresses for malicious activity and abuse reports
            </p>
          </div>

          <form onSubmit={handleSubmit} className="ip-form">
            <div className="form-section">
              <label htmlFor="ip-input" className="form-label">
                IP Address to Check
              </label>
              <div className="ip-input-group">
                <input
                  id="ip-input"
                  type="text"
                  value={ipAddress}
                  onChange={(e) => setIpAddress(e.target.value)}
                  className="ip-input"
                  placeholder="Enter IP address (e.g., 8.8.8.8)"
                  pattern="^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
                  required
                />
                <button type="submit" disabled={isLoading} className="check-btn">
                  {isLoading && <div className="loading-spinner"></div>}
                  {isLoading ? 'Checking...' : '🔍 Check IP'}
                </button>
              </div>
            </div>
          </form>

          {error && (
            <div className="error-message fade-in">
              <strong>⚠️ Check Failed:</strong> {error}
            </div>
          )}
          

          {ipDetails && (
            <div className="ip-results fade-in">
              <div className="result-card">
                <div className="result-header">
                  <div className="result-ip">
                    📍 {ipDetails.ipAddress}
                  </div>
                  <div className={`abuse-badge ${getAbuseStatus(ipDetails.abuseConfidenceScore || 0).status}`}>
                    {getAbuseStatus(ipDetails.abuseConfidenceScore || 0).label}
                  </div>
                </div>

                <div className="result-grid">
                  <div className="result-item">
                    <div className="result-label">Abuse Confidence</div>
                    <div className="result-value confidence-score">
                      {ipDetails.abuseConfidenceScore || 0}%
                    </div>
                  </div>

                  <div className="result-item">
                    <div className="result-label">Country Code</div>
                    <div className="result-value">
                      {ipDetails.countryCode || 'Unknown'}
                    </div>
                  </div>

                  <div className="result-item">
                    <div className="result-label">ISP</div>
                    <div className="result-value">
                      {ipDetails.isp || 'Unknown'}
                    </div>
                  </div>

                  <div className="result-item">
                    <div className="result-label">Usage Type</div>
                    <div className="result-value">
                      {ipDetails.usageType || 'Unknown'}
                    </div>
                  </div>

                  <div className="result-item">
                    <div className="result-label">Total Reports</div>
                    <div className="result-value">
                      {ipDetails.totalReports || 0}
                    </div>
                  </div>

                  <div className="result-item">
                    <div className="result-label">Is Whitelisted</div>
                    <div className="result-value">
                      {ipDetails.isWhitelisted ? '✅ Yes' : '❌ No'}
                    </div>
                  </div>
                </div>

                {ipDetails.domain && (
                  <div className="result-item" style={{ marginTop: '20px', gridColumn: '1 / -1' }}>
                    <div className="result-label">Associated Domain</div>
                    <div className="result-value">
                      {ipDetails.domain}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default IPCheck;

