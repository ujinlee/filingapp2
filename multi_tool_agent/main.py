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

        # 2. Extract official Ts from Arelle/XBRL using the raw HTML index page
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

        # Improved MDA extraction logic
        mda_section = None
        # 1. Try regex extraction between known headers
        mda_match = re.search(r'(item\s*2[\s\S]+?)(item\s*3|item\s*4|item\s*7a|item\s*8|quantitative and qualitative disclosures|controls and procedures)', content, re.IGNORECASE)
        if not mda_match:
            mda_match = re.search(r'(item\s*7[\s\S]+?)(item\s*7a|item\s*8|quantitative and qualitative disclosures|controls and procedures)', content, re.IGNORECASE)
        if mda_match:
            mda_section = mda_match.group(1).strip()
            print(f"[DEBUG] Regex section fallback - MDA section (first 500 chars): {mda_section[:500]}")
        # 2. If regex result is too short or looks like a header, use alpha-ratio fallback
        if not mda_section or len(mda_section) < 500 or re.match(r'^item \d+management', mda_section.strip().lower()):
            print("[DEBUG] Regex result too short or only header, using alpha-ratio fallback.")
            candidates = re.findall(r'([\s\S]{0,10000})', content)
            best = ''
            best_alpha = 0
            for c in candidates:
                if 'management' in c.lower() and 'discussion' in c.lower():
                    alpha_ratio = sum(ch.isalpha() for ch in c) / max(1, len(c))
                    if alpha_ratio > 0.6 and len(c) > len(best):
                        best = c
                        best_alpha = alpha_ratio
            if best and len(best) > 500:
                mda_section = best.strip()
                print(f"[DEBUG] Alpha-ratio fallback - MDA section (first 500 chars): {mda_section[:500]}")
                print(f"[DEBUG] Alpha-ratio fallback - MDA section length: {len(mda_section)} chars, alpha ratio: {best_alpha:.2f}")
        # 3. If still not found, use large window after header
        if not mda_section or len(mda_section) < 500:
            print("[DEBUG] Alpha-ratio fallback failed, using large window after header.")
            header_match = re.search(r'(item\s*2[^\n\r]*)', content, re.IGNORECASE)
            if header_match:
                start = header_match.end()
                mda_section = content[start:start+10000].strip()
                print(f"[DEBUG] Large window fallback - MDA section (first 500 chars): {mda_section[:500]}")
        # 4. If all else fails, use entire filing text (up to 10,000 chars)
        if not mda_section or len(mda_section) < 500:
            print("[DEBUG] All MDA extraction failed, using entire filing text.")
            mda_section = content[:10000].strip()
            print(f"[DEBUG] Entire filing fallback - MDA section (first 500 chars): {mda_section[:500]}")

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

        # Improved extraction of driver sentences from MDA
        def extract_driver_sentences(mda_html, num_sentences=5):
            keywords = [
                'increase', 'increased', 'decrease', 'decreased',
                'driven by', 'due to',
                'revenue', 'revenues', 'sales', 'business', 'sector', 'segment'
            ]
            soup = BeautifulSoup(mda_html, 'html.parser')
            all_text = soup.get_text(separator=' ', strip=True)
            sentences = re.split(r'(?<=[.!?])\s+|\n+', all_text)
            relevant = []
            for i, s in enumerate(sentences):
                s_clean = s.strip()
                has_keyword = any(kw in s_clean.lower() for kw in keywords)
                has_number = re.search(r'\d', s_clean)
                if has_keyword or has_number:
                    # If sentence has a number, include previous and next for context
                    if has_number:
                        if i > 0:
                            relevant.append(sentences[i-1].strip())
                        relevant.append(s_clean)
                        if i < len(sentences)-1:
                            relevant.append(sentences[i+1].strip())
                    else:
                        relevant.append(s_clean)
                if len(relevant) >= num_sentences:
                    break
            # Remove duplicates and empty strings
            results = [s for i, s in enumerate(relevant) if s and s not in relevant[:i]]
            print(f"[DEBUG][Driver Extraction] Sentences returned: {results}")
            return results[:num_sentences]

        def extract_revenue_statements(mda_html):
            driver_sentences = extract_driver_sentences(mda_html, num_sentences=5)
            print(f"[DEBUG] Extracted driver sentences for LLM: {driver_sentences}")
            return ' '.join(driver_sentences)

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

        # After extracting entity/company name (assume variable company_name is available or set to None if not)
        company_name = None
        try:
            cik = SECAgent.get_cik_from_ticker(request.documentUrl.split('/')[-1].split('.')[0])
            if cik:
                url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    company_name = data.get('name', None)
        except Exception as e:
            print(f"[DEBUG] Could not extract company name for prompt: {e}")

        # Add debug print for extracted MDA section
        print(f"[DEBUG] Extracted MDA section (first 500 chars): {mda_section[:500]}")

        # Update the LLM prompt for a more conversational podcast script
        company_intro = f"The company discussed in this filing is {company_name}.\n" if company_name else ""
        prompt = (
            company_intro +
            "Welcome to Filing Talk, the podcast where we break down the latest SEC filings. "
            "(IMPORTANT: Always say 'Filing Talk' in English, do not translate it, even in other languages.)\n\n"
            "Write a podcast script as a natural conversation between Alex and Jamie, discussing the latest SEC filing for this company.\n"
            "- For the financial performance section, use the official numbers and the following sentences from the MDA that mention key drivers (e.g., 'driven by', 'due to', 'increase', 'decrease', etc.).\n"
            "- Make the conversation engaging and natural. Do not use block quotes or headings like PART 1/PART 2.\n"
            "- Paraphrase the sentences as needed to fit the flow of the conversation, but keep the facts accurate.\n"
            "- Do not invent numbers or drivers not present in the extracted sentences.\n"
            "- Here are the official numbers:\n"
            f"{numbers_section}\n"
            "- Here are the extracted driver sentences:\n"
            f"{revenue_statements}\n\n"
            "For details, strategic drivers, and risks/opportunities, use the following full section as context:\n"
            f"{mda_section}\n\n"
            "The script should be 2:30 to 3:30 minutes long, with Alex and Jamie alternating as speakers."
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

        # After LLM output, clean up excessive asterisks
        summary = re.sub(r'\*{2,}', '', summary)

        # After LLM output is generated (assume variable 'transcript' holds the script)
        def clean_transcript(transcript):
            # Remove redundant 'Alex:' or 'Jamie:' after speaker tag
            transcript = re.sub(r'^(ALEX:)\s*Alex:\s*', r'\1 ', transcript, flags=re.MULTILINE)
            transcript = re.sub(r'^(JAMIE:)\s*Jamie:\s*', r'\1 ', transcript, flags=re.MULTILINE)
            # Remove self-introductions at the start of the line after the speaker tag
            transcript = re.sub(r'^(ALEX:)\s*(Hey, everyone, )?(this is|I am|I'm|I\'m)?\s*Alex[,.!\s-]*', r'\1 ', transcript, flags=re.MULTILINE | re.IGNORECASE)
            transcript = re.sub(r'^(JAMIE:)\s*(Hey, everyone, )?(this is|I am|I'm|I\'m)?\s*Jamie[,.!\s-]*', r'\1 ', transcript, flags=re.MULTILINE | re.IGNORECASE)
            # Remove 'joined today by Jamie' or 'joined by Alex' at the start
            transcript = re.sub(r'^(ALEX:).*joined (today )?by Jamie[,.!\s-]*', r'\1 ', transcript, flags=re.MULTILINE | re.IGNORECASE)
            transcript = re.sub(r'^(JAMIE:).*joined (today )?by Alex[,.!\s-]*', r'\1 ', transcript, flags=re.MULTILINE | re.IGNORECASE)
            return transcript
        # ...
        # After you get the LLM output (e.g., 'transcript = ...')
        transcript = ""
        try:
            # ... LLM call that sets transcript ...
            transcript = summary  # or whatever variable holds the LLM output
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            transcript = ""

        if transcript:
            transcript = clean_transcript(transcript)
        else:
            print("[WARN] Transcript is empty after LLM call.")

        # Remove stage directions like [Intro Music] before translation and TTS
        def remove_stage_directions(transcript):
            lines = transcript.split('\n')
            filtered = [line for line in lines if not re.match(r'^\s*[A-Z]+:\s*\[.*\]\s*$', line)]
            return '\n'.join(filtered)
        transcript = remove_stage_directions(transcript)

        # 8. Translate
        try:
            # For Korean, ensure currency is spoken as '1.91달러' (number first)
            def fix_korean_currency(text):
                # Replace '달러 1.91' or '달러 1,000' with '1.91달러' or '1,000달러'
                return re.sub(r'달러\s*([\d,.]+)', r'\1달러', text)
            if request.language.startswith('ko'):
                transcript = fix_korean_currency(transcript)
            transcript = TranslationAgent.translate(transcript, request.language)
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
            # After translation, clean up excessive asterisks in transcript
            transcript = re.sub(r'\*{2,}', '', transcript)
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

