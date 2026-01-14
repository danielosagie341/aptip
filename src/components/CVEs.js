import React, { useState } from 'react';
import './CVEs.css';

function CVEs() {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [cves, setCves] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    setCves([]);

    const start = new Date(startDate);
    const end = new Date(endDate);
    const diffTime = Math.abs(end - start);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays > 120) {
      setError('The date range cannot be more than 120 days.');
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch(`http://localhost:5001/api/cves?startDate=${startDate}&endDate=${endDate}`);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Network response was not ok');
      }

      const data = await response.json();
      setCves(data);
    } catch (error) {
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const getSeverityClass = (severity) => {
    if (!severity) return 'severity-low';
    const sev = severity.toLowerCase();
    if (sev.includes('critical')) return 'severity-critical';
    if (sev.includes('high')) return 'severity-high';
    if (sev.includes('medium')) return 'severity-medium';
    return 'severity-low';
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <div className="container">
      <div className="cves-container fade-in">
        <div className="card">
          <div className="page-header">
            <h1 className="page-title">📋 Common Vulnerabilities and Exposures</h1>
            <p className="page-subtitle">
              Search for recent CVEs and security vulnerabilities by date range
            </p>
          </div>

          <form onSubmit={handleSubmit} className="date-form">
            <div className="date-inputs">
              <div className="date-group">
                <label htmlFor="start-date" className="form-label">
                  Start Date
                </label>
                <input
                  id="start-date"
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="date-input"
                  required
                />
              </div>

              <div className="date-group">
                <label htmlFor="end-date" className="form-label">
                  End Date
                </label>
                <input
                  id="end-date"
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="date-input"
                  required
                />
              </div>

              <button type="submit" disabled={isLoading} className="fetch-btn">
                {isLoading && <div className="loading-spinner"></div>}
                {isLoading ? 'Fetching CVEs...' : '📊 Fetch CVEs'}
              </button>
            </div>
          </form>

          {error && (
            <div className="error-message fade-in">
              <strong>⚠️ Fetch Failed:</strong> {error}
            </div>
          )}

          {cves.length > 0 && (
            <div className="cves-results fade-in">
              <div className="results-summary">
                <div className="summary-count">{cves.length}</div>
                <div className="summary-text">
                  CVEs found between {formatDate(startDate)} and {formatDate(endDate)}
                </div>
              </div>

              <div className="cves-table-container">
                <h3 style={{ textAlign: 'center', marginBottom: '20px', color: '#333' }}>
                  🔍 Vulnerability Details
                </h3>
                <div style={{ overflowX: 'auto' }}>
                  <table className="cves-table">
                    <thead>
                      <tr>
                        <th>CVE ID</th>
                        <th>Description</th>
                        <th>Published Date</th>
                        <th>Severity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cves.map((cve) => (
                        <tr key={cve.id}>
                          <td>
                            <a 
                              href={`https://cve.mitre.org/cgi-bin/cvename.cgi?name=${cve.id}`}
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="cve-id"
                            >
                              {cve.id}
                            </a>
                          </td>
                          <td>
                            <div className="description-cell" title={cve.description}>
                              {cve.description}
                            </div>
                          </td>
                          <td>{formatDate(cve.publishedDate)}</td>
                          <td>
                            <span className={`severity-badge ${getSeverityClass(cve.severity)}`}>
                              {cve.severity || 'Unknown'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {!isLoading && cves.length === 0 && startDate && endDate && !error && (
            <div className="card" style={{ textAlign: 'center', marginTop: '40px' }}>
              <h3>📝 No CVEs Found</h3>
              <p>No vulnerabilities were found for the selected date range.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default CVEs;

