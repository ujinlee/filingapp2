import os
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from multi_tool_agent.agent import SECAgent, SummarizationAgent, TranslationAgent, TTSAgent, extract_xbrl_facts_with_arelle
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import requests
import re
import logging
import uvicorn
from typing import Optional, List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)
print(f"FastAPI serving audio files from: {AUDIO_DIR}")

# Mount the audio directory using the absolute path
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Initialize agents
sec_agent = SECAgent()
summarization_agent = SummarizationAgent()
translation_agent = TranslationAgent()
tts_agent = TTSAgent()

class SummarizeRequest(BaseModel):
    ticker: str
    form_type: str
    filing_date: str
    language: Optional[str] = "en-US"

class SummarizeResponse(BaseModel):
    audio_url: str
    script: str
    error: Optional[str] = None

def extract_revenue_statements(mda_text: str) -> str:
    """
    Extract up to 5 complete sentences per table/section that contain both numbers and keywords.
    Maintains original order and meaning of sentences.
    """
    import re
    from collections import defaultdict
    
    # Split content into complete sentences
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', mda_text)
    
    # Define keywords for financial performance and drivers
    performance_keywords = [
        'increase', 'increased', 'decrease', 'decreased',
        'growth', 'decline', 'up', 'down',
        'revenue', 'revenues', 'sales', 'business', 'sector', 'segment'
    ]
    
    driver_keywords = [
        'driven by', 'due to', 'because of', 'as a result of',
        'primarily', 'mainly', 'primarily due to', 'mainly due to',
        'resulted from', 'attributed to', 'impacted by',
        'contributed to', 'led to', 'caused by'
    ]
    
    # Pattern to match financial numbers
    number_pattern = r'\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|%|percent|percentage))?'
    
    # Group sentences by their context (table/section)
    table_sentences = defaultdict(list)
    current_table = "general"
    sentence_count = defaultdict(int)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if this is a new table/section
        if re.search(r'table|exhibit|note|item', sentence.lower()):
            current_table = sentence[:50]
            sentence_count[current_table] = 0
            continue
            
        # Only process if we haven't reached 5 sentences for this table
        if sentence_count[current_table] < 5:
            # Check if sentence contains both numbers and keywords
            has_numbers = bool(re.search(number_pattern, sentence, re.IGNORECASE))
            has_performance = any(kw in sentence.lower() for kw in performance_keywords)
            has_driver = any(kw in sentence.lower() for kw in driver_keywords)
            
            # Only include complete sentences that have both numbers and either performance or driver keywords
            if has_numbers and (has_performance or has_driver):
                # Ensure the sentence is complete (starts with capital letter and ends with punctuation)
                if re.match(r'^[A-Z].*[.!?]$', sentence):
                    table_sentences[current_table].append(sentence)
                    sentence_count[current_table] += 1
    
    # Combine all sentences while maintaining original order
    all_sentences = []
    for table, sentences in table_sentences.items():
        all_sentences.extend(sentences)
    
    # Join the sentences with spaces
    return ' '.join(all_sentences)

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize_filing(request: SummarizeRequest):
    try:
        # Get filings for the company
        filings, error = sec_agent.list_filings(request.ticker)
        if error:
            raise HTTPException(status_code=404, detail=error)
        
        # Find the specific filing
        target_filing = None
        for filing in filings:
            if filing["form"] == request.form_type and filing["date"] == request.filing_date:
                target_filing = filing
                break
        
        if not target_filing:
            raise HTTPException(
                status_code=404,
                detail=f"No {request.form_type} filing found for {request.ticker} on {request.filing_date}"
            )
        
        # Fetch the document
        content, error = sec_agent.fetch_document(target_filing["url"])
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        # Extract revenue statements (up to 5 sentences per table)
        revenue_statements = extract_revenue_statements(content)
        
        # Extract XBRL facts
        xbrl_facts = summarization_agent.extract_xbrl_facts(content)
        
        # Generate summary using the extracted revenue statements
        summary = summarization_agent.summarize(content, revenue_statements)
        
        # Create podcast script with the following prompt
        prompt = (
            "Welcome to Filing Talk, the podcast where we break down the latest SEC filings. "
            "(IMPORTANT: Always say 'Filing Talk' in English, do not translate it, even in other languages.)\n\n"
            "For this script, use the following:\n"
            "1. For the Financial performance section, use ONLY the exact statements below that contain numbers and their drivers:\n"
            f"{revenue_statements}\n\n"
            "2. For the Details and strategic drivers and Risks, opportunities, and outlook sections, use the full MDA section below:\n"
            f"{content}\n\n"
            "Please create a podcast-style script (with Alex and Jamie) that is 2:30 to 3:30 minutes long, structured in three parts:\n"
            "1. Financial performance: Use ONLY the exact numbers and statements provided above. Do not create new sentences or modify the numbers.\n"
            "- Quote the exact sentences that contain both numbers and their drivers.\n"
            "- Use the precise numbers as stated in the original sentences.\n"
            "- Do not paraphrase or create new sentences.\n"
            "- Do not combine or modify the original statements.\n"
            "- If a statement includes a main driver (e.g., 'primarily driven by', 'mainly due to'), quote that part exactly.\n"
            "- Do not infer or add information not present in the original statements.\n"
            "- Do not mention 'MD&A', 'MDA', 'Management's Discussion and Analysis', or similar terms in the script.\n"
            "2. Details and strategic drivers: Summarize from the full MDA section above.\n"
            "3. Risks, opportunities, and outlook: Summarize from the full MDA section above.\n"
            "Each line of dialogue must start with either 'ALEX:' or 'JAMIE:' (all caps, followed by a colon, no extra spaces). Alternate lines between ALEX and JAMIE, always starting with ALEX.\n"
            "Do not use any other speaker names or formats.\n"
            "Make the discussion engaging, thorough, and human-like, focusing on what drove the numbers, company strategy, risks, and any forward-looking statements.\n\n"
            f"Official numbers for the current and previous period:\n"
            f"{xbrl_facts}\n"
            "Begin the podcast script now.\n\n"
        )
        
        script = summarization_agent.create_podcast_script(prompt)
        
        # Translate if needed
        if request.language != "en-US":
            script = translation_agent.translate(script, request.language)
        
        # Generate audio
        audio_url = tts_agent.synthesize(script, request.language)
        
        return SummarizeResponse(
            audio_url=audio_url,
            script=script
        )
        
    except Exception as e:
        logger.error("Error in summarize_filing: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

