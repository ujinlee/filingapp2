import React, { useState } from 'react';

import {
  Container,
  Box,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Paper,
  CircularProgress,
} from '@mui/material';

function App() {
  const [formData, setFormData] = useState({
    companyTicker: '',
    filingType: '10-K',
    year: new Date().getFullYear(),
    quarter: 1,
    language: 'en-US',
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const API_URL = "http://localhost:8000";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      // First, get the list of filings
      const filingsResponse = await fetch(`${API_URL}/api/filings?ticker=${formData.companyTicker}`);
      if (!filingsResponse.ok) {
        throw new Error(`Failed to fetch filings: ${filingsResponse.statusText}`);
      }
      const filings = await filingsResponse.json();
      
      // Find the matching filing
      const matchingFiling = filings.find(f => 
        f.form === formData.filingType && 
        new Date(f.date).getFullYear() === parseInt(formData.year) &&
        (formData.filingType === '10-K' || 
         Math.ceil((new Date(f.date).getMonth() + 1) / 3) === parseInt(formData.quarter))
      );

      if (!matchingFiling) {
        throw new Error('No matching filing found for the selected criteria');
      }

      // Then, get the summary
      const summaryResponse = await fetch(`${API_URL}/api/summarize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          documentUrl: matchingFiling.url,
          language: formData.language,
        }),
      });

      if (!summaryResponse.ok) {
        throw new Error(`Failed to generate summary: ${summaryResponse.statusText}`);
      }

      const data = await summaryResponse.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Financial Filing Podcast Summarizer
        </Typography>
        
        <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Company Name or Ticker"
              value={formData.companyTicker}
              onChange={(e) => setFormData({ ...formData, companyTicker: e.target.value })}
              margin="normal"
              required
              helperText="Enter company name (e.g. NVIDIA) or ticker symbol (e.g. NVDA)"
            />
            
            <FormControl fullWidth margin="normal">
              <InputLabel>Filing Type</InputLabel>
              <Select
                value={formData.filingType}
                onChange={(e) => setFormData({ ...formData, filingType: e.target.value })}
                required
              >
                <MenuItem value="10-K">10-K (Annual Report)</MenuItem>
                <MenuItem value="10-Q">10-Q (Quarterly Report)</MenuItem>
              </Select>
            </FormControl>
            
            <TextField
              fullWidth
              label="Year"
              type="number"
              value={formData.year}
              onChange={(e) => setFormData({ ...formData, year: e.target.value })}
              margin="normal"
              required
            />
            
            {formData.filingType === '10-Q' && (
              <TextField
                fullWidth
                label="Quarter"
                type="number"
                value={formData.quarter}
                onChange={(e) => setFormData({ ...formData, quarter: e.target.value })}
                margin="normal"
                required
                inputProps={{ min: 1, max: 4 }}
              />
            )}
            
            <FormControl fullWidth margin="normal">
              <InputLabel>Language</InputLabel>
              <Select
                value={formData.language}
                onChange={(e) => setFormData({ ...formData, language: e.target.value })}
                required
              >
                <MenuItem value="en-US">English</MenuItem>
                <MenuItem value="es-ES">Spanish</MenuItem>
                <MenuItem value="ko-KR">Korean</MenuItem>
                <MenuItem value="ja-JP">Japanese</MenuItem>
                <MenuItem value="zh-CN">Chinese</MenuItem>
              </Select>
            </FormControl>
            
            <Button
              type="submit"
              variant="contained"
              color="primary"
              fullWidth
              size="large"
              sx={{ mt: 3 }}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Generate Podcast'}
            </Button>
          </form>
        </Paper>

        {result && (
          <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
            <Typography variant="h6" gutterBottom>
              Summary
            </Typography>
            <Typography paragraph>{result.summary}</Typography>
            
            <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
              Transcript
            </Typography>
            <Typography paragraph>{result.transcript}</Typography>
            
            <Box sx={{ mt: 3 }}>
              <audio controls src={`${API_URL}${result.audio_url}`} style={{ width: '100%' }}>
                Your browser does not support the audio element.
              </audio>
            </Box>
          </Paper>
        )}
      </Box>
    </Container>
  );
}

export default App; 