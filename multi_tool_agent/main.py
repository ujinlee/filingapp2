from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from multi_tool_agent.agent import SECAgent, SummarizationAgent, TranslationAgent, TTSAgent, AUDIO_DIR, extract_xbrl_facts_with_arelle
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import requests
import re
from bs4 import BeautifulSoup

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # local React dev
        "https://front2.vercel.app",  # old Vercel frontend
        "https://front2-zeta.vercel.app",  # new Vercel frontend
        "https://filingapp.onrender.com"
        # "https://filingtalk.com",
        # "https://www.filingtalk.com"
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
            def get_latest_and_previous_value(possible_tags, pick_largest=False, debug_label=None):
                all_facts = []
                for tag in possible_tags:
                    if tag in xbrl_facts and xbrl_facts[tag]:
                        value = xbrl_facts[tag]
                        if isinstance(value, list):
                            if all(isinstance(item, dict) and 'value' in item for item in value):
                                for item in value:
                                    period = item.get('period')
                                    val = item.get('value')
                                    if val is not None:
                                        all_facts.append({'period': period, 'value': val})
                            elif all(not isinstance(item, dict) for item in value):
                                for idx, val in enumerate(value):
                                    all_facts.append({'period': None, 'value': val})
                        elif isinstance(value, dict) and 'value' in value:
                            all_facts.append({'period': value.get('period'), 'value': value['value']})
                        else:
                            all_facts.append({'period': None, 'value': value})
                # Remove None values
                all_facts = [f for f in all_facts if f['value'] is not None]
                # Debug output for all periods/values
                if debug_label:
                    print(f"[XBRL DEBUG] {debug_label} - All periods/values: {all_facts}")
                # Sort by period (descending), fallback to order if period missing
                def sort_key(f):
                    return f['period'] or ''
                all_facts = sorted(all_facts, key=sort_key, reverse=True)
                if not all_facts:
                    return (None, None)
                latest = all_facts[0]
                previous = None
                for fact in all_facts[1:]:
                    if (fact['period'] != latest['period'] or not latest['period']) and str(fact['value']) != str(latest['value']):
                        previous = fact
                        break
                # For revenue, pick largest for each period if needed
                if pick_largest:
                    # Group by period
                    from collections import defaultdict
                    period_map = defaultdict(list)
                    for f in all_facts:
                        period_map[f['period']].append(float(f['value']))
                    periods_sorted = sorted(period_map.keys(), reverse=True)
                    latest_period = periods_sorted[0]
                    latest_value = str(max(period_map[latest_period]))
                    previous_value = None
                    if len(periods_sorted) > 1:
                        prev_period = periods_sorted[1]
                        previous_value = str(max(period_map[prev_period]))
                    return (latest_value, previous_value)
                else:
                    latest_value = str(latest['value'])
                    previous_value = str(previous['value']) if previous else None
                    return (latest_value, previous_value)
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
            
            # Add segment revenue tags
            segment_revenue_tags = [
                'SegmentRevenue',
                'SegmentRevenues',
                'SegmentSales',
                'SegmentNetSales',
                'SegmentNetRevenue',
                'SegmentNetRevenues',
                'SegmentSalesRevenueNet',
                'SegmentSalesRevenueNetMember',
                'SegmentSalesRevenueServicesNet',
                'SegmentSalesRevenueGoodsNet',
                'SegmentRevenueFromContractWithCustomerExcludingAssessedTax',
                'SegmentRevenueFromContractWithCustomerMember',
                'SegmentRevenuesNetOfInterestExpense',
                'SegmentTotalRevenuesAndOtherIncome',
                'SegmentTopLineRevenue',
                # Common segment names
                'EnergySegmentRevenue',
                'EnergySegmentRevenues',
                'EnergySegmentSales',
                'TechnologySegmentRevenue',
                'TechnologySegmentRevenues',
                'TechnologySegmentSales',
                'FinancialSegmentRevenue',
                'FinancialSegmentRevenues',
                'FinancialSegmentSales',
                'HealthcareSegmentRevenue',
                'HealthcareSegmentRevenues',
                'HealthcareSegmentSales',
                'ConsumerSegmentRevenue',
                'ConsumerSegmentRevenues',
                'ConsumerSegmentSales',
                'IndustrialSegmentRevenue',
                'IndustrialSegmentRevenues',
                'IndustrialSegmentSales'
            ]
            
            revenue_tags = base_revenue_tags + [f'us-gaap:{tag}' for tag in base_revenue_tags]
            segment_revenue_tags = segment_revenue_tags + [f'us-gaap:{tag}' for tag in segment_revenue_tags]
            
            # Get total revenue
            revenue, revenue_prev = get_latest_and_previous_value(revenue_tags, pick_largest=True, debug_label='Revenue')
            
            # Get segment revenues
            segment_revenues = {}
            for tag in segment_revenue_tags:
                if tag in xbrl_facts and xbrl_facts[tag]:
                    current, previous = get_latest_and_previous_value([tag], debug_label=f'Segment Revenue - {tag}')
                    if current is not None:
                        segment_name = tag.replace('SegmentRevenue', '').replace('SegmentRevenues', '').replace('SegmentSales', '')
                        if not segment_name:
                            segment_name = 'Unnamed Segment'
                        segment_revenues[segment_name] = {
                            'current': current,
                            'previous': previous
                        }
            
            net_income, net_income_prev = get_latest_and_previous_value(['NetIncomeLoss'], debug_label='Net Income')
            eps, eps_prev = get_latest_and_previous_value(['EarningsPerShareBasic'], debug_label='EPS')
            
            print(f"[XBRL] Extracted values: Revenue={revenue}, Net Income={net_income}, EPS={eps}")
            print(f"[XBRL] Previous values: Revenue={revenue_prev}, Net Income={net_income_prev}, EPS={eps_prev}")
            print(f"[XBRL] Segment revenues: {segment_revenues}")
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

        def format_number_pair(label, current, previous, always_float=False):
            if current is None and previous is None:
                return f"{label}: (not available)\n"
            def humanize(n):
                try:
                    n = float(n)
                    if always_float:
                        return f"{n:.2f}"
                    if n >= 1_000_000_000:
                        return f"{n/1_000_000_000:.2f} billion"
                    elif n >= 1_000_000:
                        return f"{n/1_000_000:.2f} million"
                    else:
                        return str(int(n))
                except Exception:
                    return str(n)
            c = humanize(current) if current is not None else "(not available)"
            p = humanize(previous) if previous is not None else "(not available)"
            return f"{label}: {c} (previous period: {p})\n"
        def is_valid_number(val):
            return val not in (None, [], "", "None")

        numbers_section = ""
        if is_valid_number(revenue):
            numbers_section += format_number_pair("Revenue", revenue, revenue_prev)
            
        # Add segment revenues to numbers section
        if segment_revenues:
            numbers_section += "\nSegment Revenues:\n"
            for segment, values in segment_revenues.items():
                if is_valid_number(values['current']):
                    numbers_section += format_number_pair(f"{segment} Revenue", values['current'], values['previous'])
            
        if is_valid_number(net_income):
            numbers_section += format_number_pair("Net Income", net_income, net_income_prev)
        if is_valid_number(eps):
            numbers_section += format_number_pair("Earnings per Share", eps, eps_prev, always_float=True)
        if not numbers_section:
            numbers_section = "(No official numbers were found for this period.)\n"

        # Extract up to 5 full sentences after each table in the MDA section that contain BOTH a keyword and a number
        def extract_post_table_sentences(mda_html, num_sentences=5):
            keywords = [
                'increase', 'increased', 'decrease', 'decreased',
                'driven by', 'due to',
                'revenue', 'revenues', 'sales', 'business', 'sector', 'segment'
            ]
            soup = BeautifulSoup(mda_html, 'html.parser')
            all_text = soup.get_text(separator=' ', strip=True)
            # Always extract from the entire MDA section
            sentences = re.split(r'(?<=[.!?])\s+|\n+', all_text)
            relevant = []
            for s in sentences:
                s_clean = s.strip()
                has_keyword = any(kw in s_clean.lower() for kw in keywords)
                has_number = re.search(r'\d', s_clean)
                print(f"[DEBUG][All MDA] Checking: '{s_clean}' | Keyword: {has_keyword} | Number: {has_number}")
                if has_keyword and has_number:
                    relevant.append(s_clean)
                if len(relevant) >= num_sentences:
                    break
            # If not enough, fallback to regex and sliding window as before
            results = relevant
            if not results:
                print("[DEBUG][Regex Fallback] No results from sentence split, trying regex chunk extraction.")
                pattern = r'([^.!?\n]*?(?:increase|decrease|revenue|revenues|sales|segment|driven by|due to)[^.!?\n]*?\d+[^.!?\n]*[.!?]|[^.!?\n]*?\d+[^.!?\n]*?(?:increase|decrease|revenue|revenues|sales|segment|driven by|due to)[^.!?\n]*[.!?])'
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                regex_chunks = []
                for m in matches:
                    chunk = m[0].strip() if isinstance(m, tuple) else m.strip()
                    print(f"[DEBUG][Regex Fallback] Checking chunk: '{chunk}'")
                    if chunk:
                        regex_chunks.append(chunk)
                print(f"[DEBUG][Regex Fallback] Regex-matched chunks: {regex_chunks}")
                results = regex_chunks[:num_sentences]
            if not results:
                print("[DEBUG][Sliding Window Fallback] No results from regex, trying sliding window.")
                words = all_text.split()
                window_size = 20
                for i in range(0, len(words) - window_size + 1):
                    chunk = ' '.join(words[i:i+window_size])
                    has_keyword = any(kw in chunk.lower() for kw in keywords)
                    has_number = re.search(r'\d', chunk)
                    if has_keyword and has_number:
                        results.append(chunk)
                    if len(results) >= num_sentences:
                        break
                print(f"[DEBUG][Sliding Window Fallback] Chunks: {results}")
            else:
                print(f"[DEBUG][All MDA] Sentences returned: {results}")
            return results

        def extract_revenue_statements(mda_html):
            post_table_sentences = extract_post_table_sentences(mda_html, num_sentences=5)
            print(f"[DEBUG] Extracted revenue sentences for LLM: {post_table_sentences}")
            return ' '.join(post_table_sentences)

        revenue_statements = extract_revenue_statements(mda_section)

        # Remove any mention of MDA, MD&A, or Management's Discussion and Analysis from the script
        def remove_mda_mentions(text):
            patterns = [
                r'MD&A', r'MDA', r'Management\'?s? Discussion and Analysis',
                r'md&a', r'mda', r'management\'?s? discussion and analysis'
            ]
            for pat in patterns:
                text = re.sub(pat, '', text, flags=re.IGNORECASE)
            return text

        prompt = (
            "Welcome to Filing Talk, the podcast where we break down the latest SEC filings. "
            "(IMPORTANT: Always say 'Filing Talk' in English, do not translate it, even in other languages.)\n\n"
            "For this script, use the following instructions:\n"
            "1. For the Financial performance section, ONLY use the statements below that contain BOTH a financial keyword (increase, increased, decrease, decreased, driven by, due to, revenue, revenues, sales, business, sector, or segment) AND a number.\n"
            "- Quote or restate each sentence exactly as written below.\n"
            "- Do NOT paraphrase, mix, combine, or create new sentences.\n"
            "- Do NOT use partial sentences.\n"
            "- Do NOT infer or add information.\n"
            f"{revenue_statements}\n\n"
            "2. For the Details and strategic drivers and Risks, opportunities, and outlook sections, use the full section below.\n"
            f"{mda_section}\n\n"
            "Create a podcast-style script (with Alex and Jamie) that is 2:30 to 3:30 minutes long, structured in three parts:\n"
            "1. Financial performance: Summarize revenue changes and their explicit explanations using ONLY the extracted statements above.\n"
            "2. Details and strategic drivers: Summarize from the full section above.\n"
            "3. Risks, opportunities, and outlook: Summarize from the full section above.\n"
        )
        prompt = remove_mda_mentions(prompt)

        # 7. Summarize
        request_options = {
            'method': 'POST',
            'url': 'https://api.openai.com/v1/chat/completions',
            'headers': {
                'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                'Content-Type': 'application/json'
            },
            'json': {
                'model': 'gpt-4o',
                'messages': [
                    {'role': 'user', 'content': prompt}
                ]
            }
        }
        # Remove or comment out the debug print for request_options
        # print(f"[DEBUG] LLM request: method={request_options['method']}, url={request_options['url']}")

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

