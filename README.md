# Financial Filing Podcast Summarizer

This application converts SEC filings (10-K and 10-Q) into concise, engaging podcast-style summaries in multiple languages.

## Core Features

- Convert SEC filings into podcast summaries
- Support for multiple languages (English, Spanish, Korean, Japanese, Chinese, French, German)
- Modern web interface (React)
- Powered by OpenAI's GPT-4o for intelligent summarization and translation
- High-quality text-to-speech conversion (Google Cloud TTS)
- Modular Python backend (FastAPI)

## Architecture

The application consists of three main components:

1. **Frontend**: React-based web interface (in `frontend-app/src`) with Material-UI components
2. **Backend**: FastAPI server (in `multi_tool_agent/`) handling API requests and processing
3. **AI Services**: Integration with OpenAI's GPT-4o for summarization and translation, and Google Cloud Text-to-Speech for audio

## Agents and Their Roles

- **SECAgent**: Handles all interactions with SEC data. Responsible for looking up company tickers, fetching filing documents, and extracting raw filing content for further processing.
- **SummarizationAgent**: Extracts the Management's Discussion and Analysis (MDA) section from filings. Provides robust logic to locate and parse the MDA section, which is then used for further sentence extraction and summarization. (Note: The actual LLM prompt construction and revenue sentence extraction is now handled in `main.py`.)
- **TranslationAgent**: Translates the generated podcast script into the target language using OpenAI's GPT-4o. Ensures that speaker tags and important phrases are preserved and that the translation is suitable for a business podcast.
- **TTSAgent**: Converts the final podcast script (in the selected language) into high-quality audio using Google Cloud Text-to-Speech. Handles language-specific formatting and ensures natural-sounding output.

These agents work together to fetch, process, summarize, translate, and synthesize SEC filings into engaging, multilingual podcast episodes.

## How it works

- The backend extracts the Management's Discussion and Analysis (MDA) section from the SEC filing.
- For the financial performance section, the backend extracts up to 5 full sentences immediately following each table in the MDA section that contain BOTH a financial keyword (increase, decrease, revenue, etc.) and a number.
- Only these sentences are provided to the LLM for the financial performance summary, and the LLM is strictly instructed to quote or restate them as-is (no paraphrasing, mixing, or inferring).
- The backend also extracts and provides both current and prior period numbers for key financial metrics (Revenue, Net Income, EPS, etc.) for comparison in the script.
- All summarization and script generation logic is handled directly in `main.py`. The old SummarizationAgent logic in `agent.py` is not used for podcast script generation.

## Number and Currency Normalization

- All currency values are read as digits, with "dollars" at the end (e.g., `19.4 billion dollars`, `0.13 dollars`).
- There is **no conversion to "cents"** for small values, and numbers are **not spelled out in words** for currency.
- This ensures natural and consistent TTS output for all financial values.

## Prerequisites

- Python 3.8+
- Node.js 14+
- OpenAI API key (for GPT-4o)
- Google Cloud account with the following API enabled:
  - Cloud Text-to-Speech API

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd filingapp2-main
```

2. Set up the backend:
```bash
cd multi_tool_agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r ../requirements.txt
```

3. Set up the frontend:
```bash
cd ../frontend-app
npm install
```

4. Create a `.env` file in the `multi_tool_agent` directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

## Running the Application

1. Start the backend server:
```bash
cd multi_tool_agent
uvicorn main:app --reload
```

2. Start the frontend development server:
```bash
cd ../frontend-app
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Usage

1. Enter a company ticker symbol (e.g., AAPL, GOOG) in the search box and click "Search Filings".
2. Browse the list of recent SEC filings (10-K, 10-Q) for the selected company.
3. For any filing, select your preferred language from the dropdown menu.
4. Click "Generate Podcast" to create a podcast-style summary for that filing.
5. Listen to the generated audio, read the transcript, and review the summary directly in the web interface.
6. Repeat for other filings or companies as needed. Previous results are cleared when you search for a new company.

## Google ADK and Agent Model

This application is architected using a modular, agent-based backend inspired by the Google ADK (Agent Development Kit) pattern. Each agent is responsible for a specific part of the processing pipeline:

- **SECAgent**: Handles all SEC data retrieval, including ticker lookup and document fetching.
- **SummarizationAgent**: Extracts and processes the Management's Discussion and Analysis (MDA) section from filings.
- **TranslationAgent**: Uses OpenAI GPT-4o to translate the podcast script into the target language, preserving speaker tags and business tone.
- **TTSAgent**: Uses Google Cloud Text-to-Speech to synthesize high-quality audio from the final script, with language-specific formatting.

**Note:**
- The "Google ADK" here refers to the architectural pattern of modular, specialized agents, not a runtime dependency or library. Only Google Cloud Text-to-Speech is used as an external Google service.
- The agent model makes the backend robust, maintainable, and easy to extend for new features or data sources.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 