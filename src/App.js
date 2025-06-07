import React, { useState } from "react";
import axios from "axios";

const API_URL = "https://filingapp.onrender.com";

const LANGUAGES = [
  { code: "en-US", label: "English (US)" },
  { code: "ko-KR", label: "Korean" },
  { code: "ja-JP", label: "Japanese" },
  { code: "es-ES", label: "Spanish" },
  { code: "zh-CN", label: "Chinese" },
  { code: "fr-FR", label: "French" },
  { code: "de-DE", label: "German" },
];

const mainBg = "#f5f7fb";
const headerBg = "#232946";
const accent = "#4f5dff";
const cardBg = "#fff";
const border = "#e0e6f7";
const text = "#232946";
const muted = "#6b7280";
const errorColor = "#ff6f61";
const fontFamily = "'Inter', 'Segoe UI', Arial, sans-serif";

function App() {
  const [ticker, setTicker] = useState("");
  const [filings, setFilings] = useState([]);
  const [loadingFilings, setLoadingFilings] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState({});
  const [loadingPodcast, setLoadingPodcast] = useState({});
  const [selectedLanguage, setSelectedLanguage] = useState({});
  const [visibleFilings, setVisibleFilings] = useState(5);

  const handleSearch = async () => {
    setLoadingFilings(true);
    setError("");
    setFilings([]);
    setResults({});
    setVisibleFilings(5);
    try {
      const res = await axios.get(`${API_URL}/api/filings`, {
        params: { ticker: ticker.trim().toUpperCase() },
      });
      setFilings(res.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message);
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
      setError(
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message);
    } finally {
      setLoadingPodcast((prev) => ({ ...prev, [filing.url]: false }));
    }
  };

  // Responsive styles
  const responsiveCard = {
    background: cardBg,
    border: `1.5px solid ${border}`,
    borderRadius: 18,
    margin: "32px 0",
    padding: 32,
    boxShadow: "0 2px 16px rgba(79,93,255,0.08)",
    color: text,
    width: "100%",
    maxWidth: 600,
    boxSizing: "border-box",
  };

  const responsiveContainer = {
    maxWidth: 700,
    margin: "0 auto",
    padding: "32px 8px 0 8px",
    width: "100%",
    boxSizing: "border-box",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  };

  return (
    <div style={{ minHeight: "100vh", background: mainBg, fontFamily, color: text }}>
      {/* Header/Search Area */}
      <div style={{ background: headerBg, padding: "36px 0 32px 0", width: "100%", boxSizing: "border-box" }}>
        <div style={{ maxWidth: 700, margin: "0 auto", padding: "0 8px" }}>
          <h1 style={{ color: "#fff", fontWeight: 700, fontSize: 40, marginBottom: 18, letterSpacing: 1.2 }}>FilingTalk</h1>
          <h1 style={{ color: "#fff", fontWeight: 700, fontSize: 22, marginBottom: 18, letterSpacing: 1.2 }}>Turn 10-Q & 10-K filings into engaging Podcasts</h1>
          <div style={{ background: headerBg, borderRadius: 16, boxShadow: "0 2px 8px rgba(79,93,255,0.08)", padding: 32, border: `1.5px solid ${border}` }}>
            <label htmlFor="ticker" style={{ color: "#fff", fontWeight: 500, fontSize: 16 }}>Stock Ticker or Company Name</label>
            <input
              id="ticker"
              style={{
                width: "100%",
                padding: "14px 18px",
                borderRadius: 10,
                border: `1.5px solid ${border}`,
                background: "#1a1c2c",
                color: "#fff",
                fontSize: 18,
                marginTop: 8,
                marginBottom: 18,
                fontFamily,
                boxSizing: "border-box",
                height: 52,
                lineHeight: 1.2,
                fontWeight: 500,
              }}
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="e.g. AAPL, GOOG, MSFT"
            />
            <button
              style={{ width: "100%", background: accent, color: "#fff", border: "none", borderRadius: 10, padding: 16, fontWeight: 700, fontSize: 18, marginTop: 4, letterSpacing: 0.5, boxShadow: "0 2px 8px rgba(79,93,255,0.10)" }}
              onClick={handleSearch}
              disabled={loadingFilings}
            >
              {loadingFilings ? <span>Loading...</span> : <span style={{ display: "flex", alignItems: "center", justifyContent: "center" }}><span style={{ fontSize: 22, marginRight: 10 }}>🔍</span>Search Filings</span>}
            </button>
            <div style={{ color: "#b8c1ec", marginTop: 18, textAlign: "center", fontSize: 15 }}>
              Each summary is freshly generated each time, so there may be slight variations.<br />
              FilingTalk is for informational use only and not intended for investment decisions. 
            </div>
            {error && <div style={{ color: errorColor, marginTop: 16, textAlign: "center", fontWeight: 500 }}>{error.toString()}</div>}
          </div>
        </div>
      </div>
      {/* Main Content */}
      <div style={responsiveContainer}>
        {filings.length > 0 && (
          <>
            <h2 style={{ color: text, textAlign: "center", fontWeight: 800, fontSize: 32, marginBottom: 28, letterSpacing: 1, width: "100%" }}>
              Recent Filings for {ticker.toUpperCase()}
            </h2>
            {filings.slice(0, visibleFilings).map((filing, idx) => (
              <div key={filing.url} style={responsiveCard}>
                <div style={{ fontWeight: 700, color: accent, fontSize: 20, marginBottom: 4 }}>{filing.form} - {filing.date}</div>
                <div style={{ color: muted, fontSize: 15, marginBottom: 8 }}>{filing.description || (filing.form === "10-K" ? "Annual Report pursuant to Section 13 or 15(d)" : "Quarterly report pursuant to Section 13 or 15(d)")}</div>
                <div style={{ margin: "10px 0", color: muted, fontSize: 15 }}>
                  <span style={{ fontWeight: 500 }}>Document URL:</span>
                  <span style={{ display: 'block', wordBreak: 'break-all', overflowWrap: 'anywhere' }}>
                    <a href={filing.url} target="_blank" rel="noopener noreferrer" style={{ color: accent, textDecoration: "underline" }}>{filing.url}</a>
                  </span>
                </div>
                <div style={{ margin: "18px 0 10px 0" }}>
                  <label style={{ marginRight: 8, color: text, fontWeight: 500, fontSize: 16 }}>Select Language</label>
                  <select
                    value={selectedLanguage[filing.url] || "en-US"}
                    onChange={e => setSelectedLanguage(l => ({ ...l, [filing.url]: e.target.value }))}
                    style={{ padding: 10, borderRadius: 7, border: `1.5px solid ${border}`, background: mainBg, color: text, fontWeight: 500, fontSize: 16 }}
                  >
                    {LANGUAGES.map(lang => (
                      <option key={lang.code} value={lang.code}>{lang.label}</option>
                    ))}
                  </select>
                </div>
                <button
                  style={{ marginTop: 10, background: "#3e54a3", color: "#fff", border: "none", borderRadius: 8, padding: 14, fontWeight: 700, width: "100%", fontSize: 17, letterSpacing: 0.5, boxShadow: "0 2px 8px rgba(79,93,255,0.10)" }}
                  onClick={() => handleGeneratePodcast(filing)}
                  disabled={loadingPodcast[filing.url]}
                >
                  {loadingPodcast[filing.url] ? "Generating..." : "Generate Podcast"}
                </button>
                {results[filing.url] && (
                  <div style={{ marginTop: 28 }}>
                    <audio controls src={`${API_URL}${results[filing.url].audio_url}`} style={{ width: "100%", borderRadius: 8, background: mainBg }} />
                    <div style={{ marginTop: 18 }}>
                      <strong style={{ color: accent, fontSize: 17 }}>Transcript:</strong>
                      <textarea
                        style={{
                          width: '100%',
                          height: 120,
                          marginTop: 8,
                          borderRadius: 8,
                          border: `1.5px solid ${border}`,
                          background: mainBg,
                          color: text,
                          fontSize: 15,
                          padding: 10,
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          overflowX: 'auto',
                          resize: 'vertical',
                        }}
                        value={results[filing.url].transcript}
                        readOnly
                      />
                    </div>
                    <div style={{ marginTop: 18 }}>
                      <strong style={{ color: accent, fontSize: 17 }}>Summary:</strong>
                      <div style={{ background: mainBg, borderRadius: 8, padding: 14, marginTop: 8, color: text, border: `1.5px solid ${border}` }}>
                        {results[filing.url].summary
                          .split(/\n\n|(?=ALEX:|JAMIE:)/)
                          .filter(p => p.trim())
                          .map((p, i) => (
                            <div key={i} style={{ marginBottom: 12, lineHeight: 1.7 }}>{p.trim()}</div>
                          ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
            {visibleFilings < filings.length && (
              <button
                style={{ margin: "16px auto 32px auto", display: "block", background: accent, color: "#fff", border: "none", borderRadius: 8, padding: "12px 32px", fontWeight: 700, fontSize: 16, letterSpacing: 0.5, boxShadow: "0 2px 8px rgba(79,93,255,0.10)" }}
                onClick={() => setVisibleFilings(v => v + 5)}
              >
                Load More
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App; 
