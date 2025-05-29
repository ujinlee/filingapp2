# Financial Filing Podcast Summarizer

This application converts SEC filings (10-K and 10-Q) into concise, engaging podcast-style summaries in multiple languages.

## Features

- Convert SEC filings into 10-minute podcast summaries
- Support for multiple languages (English, Spanish, Korean, Japanese, Chinese)
- Modern web interface
- Powered by Google's Gemini AI for intelligent summarization
- High-quality text-to-speech conversion

## Prerequisites

- Python 3.8+
- Node.js 14+
- Google Cloud account with the following APIs enabled:
  - Gemini API
  - Cloud Text-to-Speech API
  - Cloud Translation API

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd financial-filing-podcast-summarizer
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

4. Create a `.env` file in the backend directory with your Google Cloud credentials:
```
GOOGLE_API_KEY=your_api_key_here
```

## Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Usage

1. Enter a company ticker symbol (e.g., AAPL, GOOG)
2. Select the filing type (10-K or 10-Q)
3. Choose the year and quarter (if applicable)
4. Select your preferred language
5. Click "Generate Podcast" to create your summary

## Architecture

The application consists of three main components:

1. **Frontend**: React-based web interface with Material-UI components
2. **Backend**: FastAPI server handling API requests and processing
3. **AI Services**: Integration with Google's Gemini AI for summarization and translation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 