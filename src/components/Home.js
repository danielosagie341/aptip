import React, { useState, useEffect, useRef } from 'react';
import Globe from 'react-globe.gl';
import './Home.css';

function Home() {
  const [threats, setThreats] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const globeEl = useRef();

  useEffect(() => {
    // Generate sample threat data
    const generateThreats = () => {
      const threatTypes = ['Malware', 'Phishing', 'DDoS', 'Ransomware', 'Social Engineering'];
      const severities = ['low', 'medium', 'high', 'critical'];
      const countries = [
        { name: 'USA', lat: 39.8283, lng: -98.5795 },
        { name: 'Russia', lat: 61.5240, lng: 105.3188 },
        { name: 'China', lat: 35.8617, lng: 104.1954 },
        { name: 'Germany', lat: 51.1657, lng: 10.4515 },
        { name: 'UK', lat: 55.3781, lng: -3.4360 },
        { name: 'Brazil', lat: -14.2350, lng: -51.9253 },
        { name: 'India', lat: 20.5937, lng: 78.9629 },
        { name: 'Japan', lat: 36.2048, lng: 138.2529 }
      ];

      const threats = [];
      for (let i = 0; i < 20; i++) {
        const source = countries[Math.floor(Math.random() * countries.length)];
        const target = countries[Math.floor(Math.random() * countries.length)];
        
        if (source !== target) {
          threats.push({
            source,
            target,
            type: threatTypes[Math.floor(Math.random() * threatTypes.length)],
            severity: severities[Math.floor(Math.random() * severities.length)],
            timestamp: new Date(Date.now() - Math.random() * 86400000).toISOString()
          });
        }
      }
      return threats;
    };

    setTimeout(() => {
      setThreats(generateThreats());
      setIsLoading(false);
    }, 1000);
  }, []);

  useEffect(() => {
    if (globeEl.current) {
      globeEl.current.controls().autoRotate = true;
      globeEl.current.controls().autoRotateSpeed = 0.5;
    }
  }, [threats]);

  if (error) {
    return (
      <div className="container">
        <div className="card error-message fade-in">
          <h2>⚠️ Error Loading Threat Data</h2>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="hero-section fade-in">
        <div className="card">
          <div className="hero-content">
            <h1 className="hero-title">
              🛡️ Advanced Persistent Threat Intelligence Platform
            </h1>
            <p className="hero-subtitle">
              Real-time cybersecurity threat monitoring and analysis powered by AI
            </p>
            <div className="hero-stats">
              <div className="stat-item">
                <div className="stat-number">{threats.length}</div>
                <div className="stat-label">Active Threats</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">
                  {threats.filter(t => t.severity === 'critical').length}
                </div>
                <div className="stat-label">Critical Alerts</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">24/7</div>
                <div className="stat-label">Monitoring</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="globe-section fade-in">
        <div className="card">
          <div className="globe-header">
            <h2>🌍 Live Threat Globe</h2>
            <p>Real-time visualization of global cybersecurity threats</p>
            {isLoading && (
              <div className="loading-container">
                <div className="loading-spinner"></div>
                <span>Loading threat data...</span>
              </div>
            )}
          </div>
          
          <div className="globe-container">
            {!isLoading && (
              <Globe
                ref={globeEl}
                globeImageUrl="//unpkg.com/three-globe/example/img/earth-night.jpg"
                arcsData={threats}
                arcStartLat={d => d.source.lat}
                arcStartLng={d => d.source.lng}
                arcEndLat={d => d.target.lat}
                arcEndLng={d => d.target.lng}
                arcDashLength={() => Math.random()}
                arcDashGap={() => Math.random()}
                arcDashAnimateTime={() => Math.random() * 4000 + 500}
                arcColor={d => {
                  const severities = {
                    'low': '#4CAF50',
                    'medium': '#FF9800',
                    'high': '#F44336',
                    'critical': '#9C27B0'
                  };
                  return severities[d.severity] || '#ffffff';
                }}
                arcStroke={() => Math.random() * 2 + 1}
                arcLabel={d => `${d.type} threat from ${d.source.name} to ${d.target.name}`}
                onArcHover={arc => {
                  if (arc && arc.source && arc.target) {
                    // Handle hover
                  }
                }}
                width={800}
                height={500}
                backgroundColor="rgba(0,0,0,0)"
              />
            )}
          </div>

          <div className="threat-legend">
            <h3>Threat Severity Levels</h3>
            <div className="legend-items">
              <div className="legend-item">
                <div className="legend-color" style={{background: '#4CAF50'}}></div>
                <span>Low</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{background: '#FF9800'}}></div>
                <span>Medium</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{background: '#F44336'}}></div>
                <span>High</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{background: '#9C27B0'}}></div>
                <span>Critical</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="recent-threats fade-in">
        <div className="card">
          <h2>📊 Recent Threat Activity</h2>
          <div className="threats-grid">
            {threats.slice(0, 6).map((threat, index) => (
              <div key={index} className="threat-card">
                <div className={`threat-severity severity-${threat.severity}`}>
                  {threat.severity.toUpperCase()}
                </div>
                <div className="threat-info">
                  <h4>{threat.type}</h4>
                  <p>
                    <strong>From:</strong> {threat.source.name}<br/>
                    <strong>To:</strong> {threat.target.name}
                  </p>
                  <small>{new Date(threat.timestamp).toLocaleString()}</small>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;
