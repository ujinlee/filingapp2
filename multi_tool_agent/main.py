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
        "https://filingapp.onrender.com",
        "https://filingtalk.com",
        "https://www.filingtalk.com"
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
            # Remove or comment out verbose debug prints
            # print(f"[DEBUG] All available XBRL tags: {list(xbrl_facts.keys())}")
            def get_latest_value(possible_tags, pick_largest=False):
                for tag in possible_tags:
                    if tag in xbrl_facts and xbrl_facts[tag]:
                        value = xbrl_facts[tag]
                        # If it's a list of dicts with 'value', get the latest by 'period'
                        if isinstance(value, list):
                            if all(isinstance(item, dict) and 'value' in item for item in value):
                                sorted_facts = sorted(value, key=lambda x: x.get('period') or '', reverse=True)
                                latest_period = sorted_facts[0]['period']
                                period_values = [float(x['value']) for x in sorted_facts if x['period'] == latest_period]
                                if pick_largest and len(period_values) > 1:
                                    return str(max(period_values))
                                else:
                                    return str(period_values[0])
                            # If it's a list of values, just return the first
                            elif all(not isinstance(item, dict) for item in value):
                                return value[0]
                        # If it's a dict with 'value'
                        elif isinstance(value, dict) and 'value' in value:
                            return value['value']
                        # If it's a single value
                        else:
                            return value
                return None
            base_revenue_tags = [
                'TotalRevenue',
                'TotalRevenues',
                'Revenues',
                'Revenue',
                'TotalSales',
                'Sales',
                'NetSales',
                'NetRevenue',
                'NetRevenues',
                'SalesRevenueNet',
                'SalesRevenueNetMember',
                'SalesRevenueServicesNet',
                'SalesRevenueGoodsNet',
                'RevenueFromContractWithCustomerExcludingAssessedTax',
                'RevenueFromContractWithCustomerMember',
                'RevenuesNetOfInterestExpense',
                'TotalRevenuesAndOtherIncome',
                'TopLineRevenue'
            ]
            revenue_tags = base_revenue_tags + [f'us-gaap:{tag}' for tag in base_revenue_tags]
            revenue = get_latest_value(revenue_tags, pick_largest=True)
            net_income = get_latest_value(['NetIncomeLoss'])
            eps = get_latest_value(['EarningsPerShareBasic'])
            # Print extracted XBRL numbers for debugging
            print(f"[XBRL] Extracted values: Revenue={revenue}, Net Income={net_income}, EPS={eps}")
            # Debug: Print all values for each revenue-related tag
            for tag in revenue_tags:
                if tag in xbrl_facts and xbrl_facts[tag]:
                    print(f"[XBRL] Tag: {tag} | Values: {xbrl_facts[tag]}")
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
        # Remove the verbose extract_mda_section debug print
        # print(f"[extract_mda_section] Filing content (first 1000 chars): {content[:1000]}")

        def humanize_large_number(n):
            try:
                n = float(n)
                if n >= 1_000_000_000:
                    return f"{n/1_000_000_000:.2f} billion"
                elif n >= 1_000_000:
                    return f"{n/1_000_000:.2f} million"
                else:
                    return str(int(n))
            except Exception:
                return str(n)

        # 6. Build the LLM prompt with both numbers and MDA
        numbers_section = ""
        if revenue:
            numbers_section += f"Revenue: {humanize_large_number(revenue)}\n"
        if net_income:
            numbers_section += f"Net Income: {humanize_large_number(net_income)}\n"
        if eps:
            numbers_section += f"EPS: {eps}\n"
        if not numbers_section:
            numbers_section = "(No official numbers were found for this period.)\n"

        prompt = (
            "Welcome to Filing Talk, the podcast where we break down the latest SEC filings. "
            "(IMPORTANT: Always say 'Filing Talk' in English, do not translate it, even in other languages.)\n\n"
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
            f"{numbers_section}\n"
            "Begin the podcast script now."
        )

        # 7. Summarize
        request_options = {
            'method': 'POST',
            'url': 'https://api.openai.com/v1/chat/completions',
            'headers': {
                'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                'Content-Type': 'application/json'
            },
            'json': {
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {'role': 'user', 'content': prompt}
                ]
            }
        }
        print(f"[DEBUG] LLM request: method={request_options['method']}, url={request_options['url']}")

        # Remove or comment out Arelle lock and HTTPS connection debug prints
        # (These are likely in the agent or XBRL extraction code, not main.py, but if present, comment out lines like:)
        # print(f"[DEBUG] Lock ...")
        # print(f"[DEBUG] Attempting to acquire lock ...")
        # print(f"[DEBUG] Attempting to release lock ...")
        # print(f"[DEBUG] Starting new HTTPS connection ...")

        response = requests.post(request_options['url'], headers=request_options['headers'], json=request_options['json'])
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to get a response from the LLM: {response.status_code}")
        data = response.json()
        summary = data['choices'][0]['message']['content']

        # After LLM output, post-process to remove 'Customer A', 'Customer B', etc.
        summary = re.sub(r'Customer [A-Z](,| and)?', 'a major customer', summary)

        # 8. Translate
        try:
            transcript = TranslationAgent.translate(summary, request.language)
            # After translation, preserve speaker tags if present, only alternate if missing
            lines = [line for line in transcript.split('\n') if line.strip()]
            normalized_lines = []
            for line in lines:
                if line.strip().startswith('ALEX:') or line.strip().startswith('JAMIE:'):
                    normalized_lines.append(line.strip())
                else:
                    tag = 'ALEX:' if len(normalized_lines) % 2 == 0 else 'JAMIE:'
                    normalized_lines.append(f"{tag} {line.strip()}")
            transcript = '\n'.join(normalized_lines)
            # Ensure 'Filing Talk' is always pronounced as '파일링 토크' in Korean transcript
            if request.language.startswith('ko'):
                transcript = re.sub(r'(Filing Talk|파일링 ?토크|필링 ?토크)', '파일링 토크', transcript, flags=re.IGNORECASE)
            # Remove or comment out verbose debug prints
            # print(f"[DEBUG] Final transcript before TTS:\n{transcript}")
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
            # Remove or comment out verbose debug prints
            # print(f"Audio URL: {audio_url}")
        except Exception as e:
            print("[ERROR] Exception in TTS synthesis:", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Text-to-Speech failed: {str(e)}")

        # Add a concise debug print for XBRL extraction results
        print(f"[DEBUG] XBRL extracted: Revenue={revenue}, Net Income={net_income}, EPS={eps}")

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