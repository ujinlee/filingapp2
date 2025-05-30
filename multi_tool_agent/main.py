from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from multi_tool_agent.agent import SECAgent, SummarizationAgent, TranslationAgent, TTSAgent, AUDIO_DIR
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # local React dev
        "https://your-vercel-domain.vercel.app"  # replace with your actual Vercel domain
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
    try:
        # 1. Fetch the document
        content, error = SECAgent.fetch_document(request.documentUrl)
        if error:
            raise HTTPException(status_code=404, detail=error)
        if not content or len(content) < 100:
            raise HTTPException(status_code=400, detail="Filing content is empty or too short to summarize.")

        # 2. Summarize with Gemini
        try:
            summary = SummarizationAgent.summarize(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini summarization failed: {str(e)}")

        # 3. Translate
        try:
            transcript = TranslationAgent.translate(summary, request.language)
            # If translation failed and fallback to English, force TTS to use en-US
            tts_language = request.language
            if transcript == summary and request.language != 'en-US':
                print("[main] Translation failed or fell back to English, using English TTS.")
                tts_language = 'en-US'
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

        # 4. Generate audio
        try:
            audio_filename = TTSAgent.synthesize(transcript, tts_language)
            # Verify the audio file exists
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            if not os.path.exists(audio_path):
                raise HTTPException(status_code=500, detail=f"Audio file not found at {audio_path}")
            audio_url = f"/audio/{audio_filename}"
            print(f"Audio URL: {audio_url}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text-to-Speech failed: {str(e)}")

        return SummarizeResponse(
            audio_url=audio_url,
            transcript=transcript,
            summary=summary
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        print("Error in /api/summarize:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}") 