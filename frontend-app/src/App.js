import React, { useState } from "react";
import axios from "axios";

const API_URL = "https://filingapp.onrender.com";

const LANGUAGES = [
  { code: "en-US", label: "English" },
  { code: "ko-KR", label: "Korean" },
  { code: "ja-JP", label: "Japanese" },
  { code: "es-ES", label: "Spanish" },
  { code: "zh-CN", label: "Chinese" },
  { code: "fr-FR", label: "French" },
  { code: "de-DE", label: "German" },
];

function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [filings, setFilings] = useState([]);
  const [loadingFilings, setLoadingFilings] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState({}); // key: filing.url, value: {audio_url, transcript, summary}
  const [loadingPodcast, setLoadingPodcast] = useState({}); // key: filing.url, value: boolean
  const [selectedLanguage, setSelectedLanguage] = useState({}); // key: filing.url, value: lang

  const handleSearch = async () => {
    setLoadingFilings(true);
    setError("");
    setFilings([]);
    try {
      const res = await axios.get(`${API_URL}/api/filings`, {
        params: { ticker: ticker.trim().toUpperCase() },
      });
      setFilings(res.data);
    } catch (err) {
      setError(err.response?.data || err.message);
    } finally {
      setLoadingFilings(false);
    }
  };

  const handleGeneratePodcast = async (filing) => {
    setLoadingPodcast((prev) => ({ ...prev, [filing.url]: true }));
    setError("");
    try {
      const lang = selectedLanguage[filing.url] || "en-US";
      const res = await axios.post(`${API_URL}/api/summarize`, {
        documentUrl: filing.url,
        language: lang,
      });
      setResults((prev) => ({ ...prev, [filing.url]: res.data }));
    } catch (err) {
      setError(err.response?.data || err.message);
    } finally {
      setLoadingPodcast((prev) => ({ ...prev, [filing.url]: false }));
    }
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: "#f4f6fc" }}>
      {/* Sidebar */}
      <div style={{ width: 300, background: "#232946", color: "#fff", padding: 24 }}>
        <h2>Search Filings</h2>
        <input
          style={{ width: "100%", padding: 8, borderRadius: 4, border: "1px solid #b8c1ec", background: "#232946", color: "#fff" }}
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          placeholder="Enter Stock Ticker"
        />
        <button
          style={{ marginTop: 16, width: "100%", background: "#6246ea", color: "#fff", border: "none", borderRadius: 8, padding: 10, fontWeight: "bold" }}
          onClick={handleSearch}
          disabled={loadingFilings}
        >
          {loadingFilings ? "Loading..." : "Search Filings"}
        </button>
        {error && <div style={{ color: "#ff6f61", marginTop: 16 }}>{error.toString()}</div>}
      </div>
      {/* Main Content */}
      <div style={{ flex: 1, padding: 32 }}>
        <h1>🎙️ Financial Filing Podcast Generator</h1>
        {filings.length === 0 && <div style={{ marginTop: 32, color: "#888" }}>Enter a ticker symbol and click 'Search Filings' to get started.</div>}
        {filings.length > 0 && (
          <div>
            <h2>Latest Filings for {ticker.toUpperCase()}</h2>
            {filings.map((filing, idx) => (
              <div key={filing.url} style={{ background: "#fff", border: "1px solid #b8c1ec", borderRadius: 12, margin: "24px 0", padding: 24, boxShadow: "0 2px 8px rgba(98,70,234,0.05)" }}>
                <div style={{ fontWeight: "bold", color: "#232946" }}>{filing.form} - {filing.date}</div>
                <div style={{ margin: "8px 0" }}><a href={filing.url} target="_blank" rel="noopener noreferrer">Document URL</a></div>
                <div style={{ margin: "8px 0" }}>
                  <label style={{ marginRight: 8 }}>Language:</label>
                  <select
                    value={selectedLanguage[filing.url] || "en-US"}
                    onChange={e => setSelectedLanguage(l => ({ ...l, [filing.url]: e.target.value }))}
                  >
                    {LANGUAGES.map(lang => (
                      <option key={lang.code} value={lang.code}>{lang.label}</option>
                    ))}
                  </select>
                </div>
                <button
                  style={{ marginTop: 8, background: "#3e54a3", color: "#fff", border: "none", borderRadius: 8, padding: 10, fontWeight: "bold" }}
                  onClick={() => handleGeneratePodcast(filing)}
                  disabled={loadingPodcast[filing.url]}
                >
                  {loadingPodcast[filing.url] ? "Generating..." : "Generate Podcast"}
                </button>
                {results[filing.url] && (
                  <div style={{ marginTop: 24 }}>
                    <audio controls src={`${API_URL}${results[filing.url].audio_url}`} style={{ width: "100%" }} />
                    <div style={{ marginTop: 16 }}>
                      <strong>Transcript:</strong>
                      <textarea style={{ width: "100%", height: 120, marginTop: 8 }} value={results[filing.url].transcript} readOnly />
                    </div>
                    <div style={{ marginTop: 16 }}>
                      <strong>Summary:</strong>
                      <div style={{ background: "#f4f6fc", borderRadius: 8, padding: 12, marginTop: 8 }}>{results[filing.url].summary}</div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App; 