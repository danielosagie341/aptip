import React, { useState } from 'react';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './ThreatAnalysis.css';

const COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b'];

function ThreatAnalysis() {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:5001/predict_detailed', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const chartData = prediction ? Object.entries(prediction.scores).map(([name, value]) => ({ 
    name: name.charAt(0).toUpperCase() + name.slice(1), 
    value: parseFloat((value * 100).toFixed(2)),
    fullValue: value
  })) : [];

  return (
    <div className="container">
      <div className="threat-analysis-container fade-in">
        <div className="card">
          <div className="page-header">
            <h1 className="page-title">🔍 Threat Analysis</h1>
            <p className="page-subtitle">
              Enter text to analyze for potential cybersecurity threats using AI-powered detection
            </p>
          </div>

          <form onSubmit={handleSubmit} className="analysis-form">
            <div className="form-section">
              <label htmlFor="threat-text" className="form-label">
                Text to Analyze
              </label>
              <textarea
                id="threat-text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="text-input"
                placeholder="Enter suspicious text, email content, or any text you want to analyze for cyber threats..."
                required
              />
            </div>
            
            <button type="submit" disabled={isLoading} className="analyze-btn">
              {isLoading && <div className="loading-spinner"></div>}
              {isLoading ? (
                <span className="loading-text">
                  <span className="pulse">Analyzing Threat</span>
                </span>
              ) : (
                '🛡️ Analyze Threat'
              )}
            </button>
          </form>

          {error && (
            <div className="error-message fade-in">
              <strong>⚠️ Analysis Failed:</strong> {error}
            </div>
          )}

          {prediction && (
            <div className="results-section fade-in">
              <div className="results-header">
                <h2>📊 Analysis Results</h2>
                <p>AI-powered threat detection results</p>
              </div>

              <div className="prediction-result">
                <div className="prediction-label">Detected Threat Type:</div>
                <div className="prediction-value">{prediction.prediction}</div>
              </div>

              <div className="charts-container">
                <div className="chart-card">
                  <h3 className="chart-title">📊 Confidence Distribution</h3>
                  <div className="chart-wrapper">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                        <XAxis 
                          dataKey="name" 
                          tick={{ fontSize: 12 }}
                          angle={-45}
                          textAnchor="end"
                          height={70}
                        />
                        <YAxis 
                          tick={{ fontSize: 12 }}
                          label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip 
                          formatter={(value, name) => [`${value}%`, 'Confidence']}
                          labelFormatter={(label) => `Threat Type: ${label}`}
                          contentStyle={{
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            border: 'none',
                            borderRadius: '8px',
                            boxShadow: '0 10px 20px rgba(0, 0, 0, 0.1)'
                          }}
                        />
                        <Bar 
                          dataKey="value" 
                          fill="url(#barGradient)"
                          radius={[4, 4, 0, 0]}
                        />
                        <defs>
                          <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#667eea" />
                            <stop offset="100%" stopColor="#764ba2" />
                          </linearGradient>
                        </defs>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="chart-card">
                  <h3 className="chart-title">🥧 Threat Probability</h3>
                  <div className="chart-wrapper">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={chartData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          outerRadius={90}
                          fill="#8884d8"
                          dataKey="value"
                          nameKey="name"
                          label={({ name, value }) => `${name}: ${value}%`}
                        >
                          {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip 
                          formatter={(value, name) => [`${value}%`, name]}
                          contentStyle={{
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            border: 'none',
                            borderRadius: '8px',
                            boxShadow: '0 10px 20px rgba(0, 0, 0, 0.1)'
                          }}
                        />
                        <Legend 
                          verticalAlign="bottom" 
                          height={36}
                          formatter={(value) => <span style={{color: '#333', fontWeight: 500}}>{value}</span>}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ThreatAnalysis;
