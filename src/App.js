import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import Home from './components/Home';
import ThreatAnalysis from './components/ThreatAnalysis';
import IPCheck from './components/IPCheck';
import CVEs from './components/CVEs';
import './App.css';

function NavLink({ to, children }) {
  const location = useLocation();
  const isActive = location.pathname === to;
  
  return (
    <Link to={to} className={`nav-link ${isActive ? 'active' : ''}`}>
      {children}
    </Link>
  );
}

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="navbar-container">
            <Link to="/" className="navbar-brand">
              
            </Link>
            <ul className="navbar-nav">
              <li className="nav-item">
                <NavLink to="/">Home</NavLink>
              </li>
              <li className="nav-item">
                <NavLink to="/threat-analysis">Threat Analysis</NavLink>
              </li>
              <li className="nav-item">
                <NavLink to="/ip-check">IP Check</NavLink>
              </li>
              <li className="nav-item">
                <NavLink to="/cves">CVEs</NavLink>
              </li>
            </ul>
          </div>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/threat-analysis" element={<ThreatAnalysis />} />
            <Route path="/ip-check" element={<IPCheck />} />
            <Route path="/cves" element={<CVEs />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;

