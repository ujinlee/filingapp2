from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from multi_tool_agent.agent import SECAgent, SummarizationAgent, TranslationAgent, TTSAgent, AUDIO_DIR, extract_xbrl_facts_with_arelle
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import requests
import re

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # local React dev
        "https://front2.vercel.app",  # old Vercel frontend
        "https://front2-zeta.vercel.app",  # new Vercel frontend
        "https://filingapp.onrender.com"  # render.com backend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)
print(f"FastAPI serving audio files from: {AUDIO_DIR}")

# Mount the audio directory using the absolute path
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

class SummarizeRequest(BaseModel):
    documentUrl: str
    language: str = "en-US"

class SummarizeResponse(BaseModel):
    audio_url: str
    transcript: str
    summary: str

@app.get("/api/filings")
async def list_filings(ticker: str):
    filings, error = SECAgent.list_filings(ticker)
    if error:
        raise HTTPException(status_code=404, detail=error)
    return filings

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize_filing(request: SummarizeRequest):
    import traceback
    try:
        # 1. Fetch the raw HTML index page for XBRL extraction
        try:
            resp = requests.get(request.documentUrl)
            raw_html = resp.text
        except Exception as e:
            print("[ERROR] Exception fetching filing index page:", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to fetch filing index page: {str(e)}")

        # 2. Extract official numbers from Arelle/XBRL using the raw HTML index page
        try:
            xbrl_facts = extract_xbrl_facts_with_arelle(request.documentUrl)
            def get_latest_value(tag):
                if tag in xbrl_facts and xbrl_facts[tag]:
                    sorted_facts = sorted(xbrl_facts[tag], key=lambda x: x['period'] or '', reverse=True)
                    return sorted_facts[0]['value']
                return None
            revenue = get_latest_value('Revenues')
            net_income = get_latest_value('NetIncomeLoss')
            eps = get_latest_value('EarningsPerShareBasic')
            print(f"[DEBUG] Extracted values: Revenue={revenue}, Net Income={net_income}, EPS={eps}")
            print(f"[DEBUG] All XBRL facts: {xbrl_facts}")
        except Exception as e:
            print("[ERROR] Exception in XBRL extraction:", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"XBRL extraction failed: {str(e)}")

        # 3. Separately fetch the cleaned text for MDA extraction and summarization
        try:
            content, error = SECAgent.fetch_document(request.documentUrl)
            if error:
                raise HTTPException(status_code=404, detail=error)
            if not content or len(content) < 100:
                raise HTTPException(status_code=400, detail="Filing content is empty or too short to summarize.")
        except Exception as e:
            print("[ERROR] Exception fetching/cleaning filing text:", traceback.format_exc())
            raise

        # 4. Detect FilingSummary.xml and base URL
        filing_summary_url = None
        base_url = None
        if request.documentUrl.endswith('.htm') or request.documentUrl.endswith('.html'):
            base_url = request.documentUrl.rsplit('/', 1)[0] + '/'
            filing_summary_url = base_url + 'FilingSummary.xml'
        elif request.documentUrl.endswith('.xml'):
            # If the documentUrl is already FilingSummary.xml
            filing_summary_url = request.documentUrl
            base_url = request.documentUrl.rsplit('/', 1)[0] + '/'

        # 5. Extract MDA section using FilingSummary.xml if available
        mda_section = SummarizationAgent.extract_mda_section(content, filing_summary_url, base_url)
        if mda_section is None:
            mda_section = "[MDA section not found in filing.]"
        print(f"[DEBUG] First 500 chars of extracted MDA section: {mda_section[:500]}")

        # 6. Build the LLM prompt with both numbers and MDA
        prompt = (
            f"Here is the MDA section from the filing:\n\n{mda_section}\n\n"
            "Please create a podcast-style script (with Alex and Jamie) that is 2:30 to 3:30 minutes long, structured in three parts: "
            "1. Financial performance (summarize the key numbers and results using only the official numbers provided below from Arelle/XBRL). "
            "2. Details and strategic drivers (discuss what drove the numbers, management commentary, business segments, etc. from the MDA). "
            "3. Risks, opportunities, and outlook (cover forward-looking statements, risk factors, and opportunities from the MDA). "
            "The script must be engaging and insightful, weaving together numbers and narrative. Do not invent or guess any details not present in the text. If you are unsure, omit the detail. "
            "Each line of dialogue must start with either 'ALEX:' or 'JAMIE:' (all caps, followed by a colon, no extra spaces). Do not use any other speaker names or formats. "
            "Alternate lines between ALEX and JAMIE for a natural conversation, always starting with ALEX. "
            "Do NOT mention or refer to the MDA section, Management's Discussion and Analysis, or management commentary by name or description. Just incorporate the insights naturally, as if you are discussing the company's performance and outlook. "
            "Make the discussion engaging, thorough, and human-like, focusing on what drove the numbers, company strategy, risks, and any forward-looking statements.\n\n"
            f"Official numbers for the period:\n"
            f"Revenue: {revenue}\nNet Income: {net_income}\nEPS: {eps}\n\n"
            "Begin the podcast script now."
        )

        # 7. Summarize
        summary = SummarizationAgent.summarize(prompt)

        # After LLM output, post-process to remove 'Customer A', 'Customer B', etc.
        summary = re.sub(r'Customer [A-Z](,| and)?', 'a major customer', summary)

        # 8. Translate
        try:
            transcript = TranslationAgent.translate(summary, request.language)
            # After translation, enforce strict alternation of speaker tags, always starting with ALEX, by stripping all tags and reassigning
            lines = [line for line in transcript.split('\n') if line.strip()]
            normalized_lines = []
            speakers = ['ALEX', 'JAMIE']
            for i, line in enumerate(lines):
                # Remove any existing speaker tag
                content = re.sub(r'^(ALEX:|JAMIE:)', '', line, flags=re.IGNORECASE).strip()
                normalized_lines.append(f"{speakers[i % 2]}: {content}")
            transcript = '\n'.join(normalized_lines)
            tts_language = request.language
            if transcript == summary and request.language != 'en-US':
                print("[main] Translation failed or fell back to English, using English TTS.")
                tts_language = 'en-US'
        except Exception as e:
            print("[ERROR] Exception in translation:", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

        # 9. Generate audio
        try:
            audio_filename = TTSAgent.synthesize(transcript, tts_language)
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            if not os.path.exists(audio_path):
                raise HTTPException(status_code=500, detail=f"Audio file not found at {audio_path}")
            audio_url = f"/audio/{audio_filename}"
            print(f"Audio URL: {audio_url}")
        except Exception as e:
            print("[ERROR] Exception in TTS synthesis:", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Text-to-Speech failed: {str(e)}")

        return SummarizeResponse(
            audio_url=audio_url,
            transcript=transcript,
            summary=summary
        )
    except HTTPException as e:
        print("[ERROR] HTTPException in summarize_filing:", traceback.format_exc())
        raise e
    except Exception as e:
        import traceback
        print("[ERROR] Unhandled exception in /api/summarize:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}") 