from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from multi_tool_agent.agent import SECAgent, SummarizationAgent, TranslationAgent, TTSAgent, AUDIO_DIR, extract_xbrl_facts_with_arelle
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import requests

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

        # 4. Extract MDA section
        mda_section = SummarizationAgent.extract_mda_section(content)

        # 5. Build the LLM prompt with both numbers and MDA
        prompt = (
            f"Here are the official numbers for the period:\n"
            f"Revenue: {revenue}\nNet Income: {net_income}\nEPS: {eps}\n\n"
            f"Here is the Management's Discussion and Analysis (MDA) section from the filing:\n"
            f"{mda_section}\n\n"
            f"Please write a podcast script that uses both the numbers and the narrative from the MDA."
        )

        # 6. Summarize
        summary = SummarizationAgent.summarize(prompt)

        # 7. Translate
        try:
            transcript = TranslationAgent.translate(summary, request.language)
            tts_language = request.language
            if transcript == summary and request.language != 'en-US':
                print("[main] Translation failed or fell back to English, using English TTS.")
                tts_language = 'en-US'
        except Exception as e:
            print("[ERROR] Exception in translation:", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

        # 8. Generate audio
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