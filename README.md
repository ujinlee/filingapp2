# Financial Filing Podcast Summarizer

This application converts SEC filings (10-K and 10-Q) into concise, engaging podcast-style summaries in multiple languages.

## Core Features

- Convert SEC filings into podcast summaries
- Support for multiple languages (English, Spanish, Korean, Japanese, Chinese)
- Modern web interface
- Powered by OpenAI's GPT-3.5 for intelligent summarization
- High-quality text-to-speech conversion
- Hybrid architecture using Google ADK and custom Python code

## Hybrid Architecture: Google ADK + Custom Python

This app is part of the Google ADK project and is built using a hybrid approach:

### Features implemented with Google ADK
- User authentication and authorization (if enabled)
- Secure API gateway and request routing
- Scalable deployment and monitoring (if deployed on Google Cloud)
- Integration with Google Cloud services (e.g., Cloud Storage, Cloud Translation, Cloud Text-to-Speech) via ADK connectors
- (Optional) Frontend hosting and static asset delivery

### Features implemented with custom Python code
- Extraction of the Management's Discussion and Analysis (MDA) section from SEC filings
- Extraction of up to 5 full sentences after each table in the MDA section that contain BOTH a financial keyword and a number (for financial performance summary)
- Construction of a strict LLM prompt and direct interaction with OpenAI's GPT-3.5 for podcast script generation
- Extraction and formatting of current and prior period numbers for key financial metrics (Revenue, Net Income, Earnings per Share, etc.)
- All summarization, script generation, and post-processing logic
- Backend API endpoints for summarization and audio generation

## Agents and Their Roles

The backend uses several specialized agents to modularize and manage different parts of the processing pipeline:

- **SECAgent**: Handles all interactions with SEC data. Responsible for looking up company tickers, fetching filing documents, and extracting raw filing content for further processing.

- **SummarizationAgent**: Extracts the Management's Discussion and Analysis (MDA) section from filings. Provides robust logic to locate and parse the MDA section, which is then used for further sentence extraction and summarization. (Note: The actual LLM prompt construction and revenue sentence extraction is now handled in `main.py`.)

- **TranslationAgent**: Translates the generated podcast script into the target language using AI translation services. Ensures that speaker tags and important phrases are preserved and that the translation is suitable for a business podcast.

- **TTSAgent**: Converts the final podcast script (in the selected language) into high-quality audio using text-to-speech services. Handles language-specific formatting and ensures natural-sounding output.

These agents work together to fetch, process, summarize, translate, and synthesize SEC filings into engaging, multilingual podcast episodes.

## How it works

- The backend extracts the Management's Discussion and Analysis (MDA) section from the SEC filing.
- For the financial performance section, the backend extracts up to 5 full sentences immediately following each table in the MDA section that contain BOTH a financial keyword (increase, decrease, revenue, etc.) and a number.
- Only these sentences are provided to the LLM for the financial performance summary, and the LLM is strictly instructed to quote or restate them as-is (no paraphrasing, mixing, or inferring).
- The backend also extracts and provides both current and prior period numbers for key financial metrics (Revenue, Net Income, EPS, etc.) for comparison in the script.
- All summarization and script generation logic is handled directly in `main.py`. The old SummarizationAgent logic in `agent.py` is not used for podcast script generation.

## Prerequisites

- Python 3.8+
- Node.js 14+
- Google Cloud account with the following APIs enabled:
  - OpenAI API
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
3. **AI Services**: Integration with OpenAI's GPT-3.5 for summarization and translation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 