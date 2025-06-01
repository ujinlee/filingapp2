import logging
import pandas as pd
from bs4 import BeautifulSoup
from arelle import Cntlr

# Set up logging at the top of the file
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

from dotenv import load_dotenv
load_dotenv()
import os
print("GOOGLE_APPLICATION_CREDENTIALS:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
import requests
import re
from bs4 import BeautifulSoup
import google.generativeai as genai
from google.cloud import texttospeech
import time
import glob
from datetime import datetime, timedelta
import io
from concurrent.futures import ThreadPoolExecutor
from google.cloud import translate_v2 as translate
import html
import inflect
import openai
print("[DEBUG] openai module version:", getattr(openai, '__version__', 'unknown'))
print("[DEBUG] OPENAI_API_KEY is set:", bool(os.getenv("OPENAI_API_KEY")))
import tiktoken
import numpy as np

SEC_USER_AGENT_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL", "your-email@domain.com")
if not SEC_USER_AGENT_EMAIL or SEC_USER_AGENT_EMAIL == "your-email@domain.com":
    raise RuntimeError("SEC_USER_AGENT_EMAIL environment variable must be set to your real email address for SEC access.")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Use absolute path for audio directory
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
print(f"Audio directory path: {AUDIO_DIR}")

# Maximum number of audio files to keep
MAX_AUDIO_FILES = 500
# Maximum age of audio files in hours
MAX_FILE_AGE_HOURS = 24

def cleanup_old_files():
    """Remove old audio files to prevent the directory from growing too large."""
    try:
        # Get all MP3 files
        files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
        
        # Sort files by modification time (oldest first)
        files.sort(key=os.path.getmtime)
        
        # Remove files older than MAX_FILE_AGE_HOURS
        current_time = datetime.now()
        for file in files:
            file_time = datetime.fromtimestamp(os.path.getmtime(file))
            if current_time - file_time > timedelta(hours=MAX_FILE_AGE_HOURS):
                try:
                    os.remove(file)
                    print(f"Removed old file: {file}")
                except Exception as e:
                    print(f"Error removing file {file}: {e}")
        
        # If still too many files, remove oldest ones
        files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
        if len(files) > MAX_AUDIO_FILES:
            files_to_remove = files[:len(files) - MAX_AUDIO_FILES]
            for file in files_to_remove:
                try:
                    os.remove(file)
                    print(f"Removed excess file: {file}")
                except Exception as e:
                    print(f"Error removing file {file}: {e}")
                    
        # Print current file count
        remaining_files = len(glob.glob(os.path.join(AUDIO_DIR, "*.mp3")))
        print(f"Current number of audio files: {remaining_files}")
    except Exception as e:
        print(f"Error in cleanup_old_files: {e}")

def sec_get(url, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["User-Agent"] = f"Financial Filing Podcast Summarizer ({SEC_USER_AGENT_EMAIL})"
    headers["Accept-Encoding"] = "gzip, deflate"
    response = requests.get(url, headers=headers, **kwargs)
    if response.status_code == 403:
        print(f"[SEC 403 ERROR] Forbidden when accessing {url}\nHeaders: {headers}")
        raise Exception(f"SEC 403 Forbidden: {url}")
    time.sleep(1)  # Avoid SEC rate limiting
    return response

class SECAgent:
    @staticmethod
    def get_cik_from_ticker(ticker_or_name: str):
        url = "https://www.sec.gov/files/company_tickers.json"
        try:
            response = sec_get(url, timeout=10)
            response.raise_for_status()
            companies = response.json()
            
            # First try exact ticker match
            for entry in companies.values():
                if entry['ticker'].upper() == ticker_or_name.upper():
                    return str(entry['cik_str']).zfill(10)
            
            # If no exact ticker match, try company name match
            for entry in companies.values():
                if entry['title'].upper() == ticker_or_name.upper():
                    return str(entry['cik_str']).zfill(10)
            
            # If still no match, try partial company name match
            for entry in companies.values():
                if ticker_or_name.upper() in entry['title'].upper():
                    return str(entry['cik_str']).zfill(10)
            
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching CIK for {ticker_or_name}: {str(e)}")
            return None

    @staticmethod
    def list_filings(ticker_or_name: str):
        cik = SECAgent.get_cik_from_ticker(ticker_or_name)
        if not cik:
            return None, f"CIK not found for {ticker_or_name}"
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            response = sec_get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            forms = data.get("filings", {}).get("recent", {})
            filings = []
            for form, date, accession, doc in zip(
                    forms.get("form", []),
                    forms.get("filingDate", []),
                    forms.get("accessionNumber", []),
                    forms.get("primaryDocument", [])
                ):
                if form in ["10-K", "10-Q"]:
                    accession_clean = accession.replace("-", "")
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{doc}"
                    filings.append({
                        "form": form,
                        "date": date,
                        "url": doc_url
                    })
            return filings, None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching filings for {ticker_or_name}: {str(e)}")
            return None, f"Could not fetch filings for {ticker_or_name}: {str(e)}"

    @staticmethod
    def fetch_document(document_url: str):
        headers = {
            "User-Agent": f"Financial Filing Podcast Summarizer ({SEC_USER_AGENT_EMAIL})",
            "Accept-Encoding": "gzip, deflate"
        }
        max_retries = 3
        retry_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                import socket
                domain = "www.sec.gov"
                try:
                    socket.gethostbyname(domain)
                except socket.gaierror as e:
                    print(f"DNS resolution failed for {domain}: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    return None, f"Could not resolve SEC domain: {str(e)}"
                response = sec_get(document_url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.get_text()
                content = re.sub(r'\s+', ' ', content)
                content = re.sub(r'[^\w\s.,;:!?()-]', '', content)
                content = content.strip()
                if not content or len(content) < 100:
                    return None, "Filing content is empty or too short"
                return content, None
            except requests.exceptions.RequestException as e:
                print(f"Error fetching document (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return None, f"Failed to fetch filing after {max_retries} attempts: {str(e)}"

class SummarizationAgent:
    @staticmethod
    def summarize(content: str, allowed_numbers=None) -> str:
        start_time = time.time()
        # Directly send the prompt (already constructed in main.py) to the LLM
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": content}
            ],
            max_tokens=2048,
            temperature=0.5,
        )
        summary = response.choices[0].message.content.strip()
        summary = html.unescape(summary)
        elapsed = time.time() - start_time
        print(f"[Timing] Summarization took {elapsed:.2f} seconds")
        return summary

    @staticmethod
    def extract_mda_section(content: str) -> str:
        """
        Extract the Management's Discussion and Analysis (MDA) section from the filing text.
        Looks for common MDA section headers and extracts until the next major section.
        """
        import re
        # Try to find the MDA section using common headers
        mda_patterns = [
            r"management[’'`]s discussion and analysis[\s\S]{0,100}?of financial condition and results of operations",
            r"management[’'`]s discussion and analysis",
            r"item\s+2[.:-]?\s*management[’'`]s discussion and analysis",
            r"item\s+7[.:-]?\s*management[’'`]s discussion and analysis",
        ]
        end_patterns = [
            r"item\s+3[.:-]?", r"item\s+4[.:-]?", r"quantitative and qualitative disclosures", r"controls and procedures"
        ]
        content_lower = content.lower()
        mda_start = None
        for pat in mda_patterns:
            match = re.search(pat, content_lower)
            if match:
                mda_start = match.start()
                break
        if mda_start is None:
            return "[MDA section not found in filing.]"
        mda_text = content[mda_start:]
        # Find the end of the MDA section
        mda_end = len(mda_text)
        for pat in end_patterns:
            match = re.search(pat, mda_text.lower())
            if match:
                mda_end = match.start()
                break
        return mda_text[:mda_end].strip()

class TranslationAgent:
    _translation_cache = {}
    @staticmethod
    def translate(english_script: str, target_language: str) -> str:
        start_time = time.time()
        if target_language == "en-US":
            return english_script
        cache_key = f"{hash(english_script)}_{target_language}"
        if cache_key in TranslationAgent._translation_cache:
            return TranslationAgent._translation_cache[cache_key]

        # Always send the entire script as a single block unless extremely long
        max_block_size = 8000
        script_lines = english_script.split('\n')
        script_length = sum(len(line) for line in script_lines)
        if script_length < max_block_size:
            blocks = [(None, english_script)]
        else:
            # Fallback to old splitting logic for very long scripts
            blocks = []
            current_block = []
            current_speaker = None
            block_size = 0
            for line in script_lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('ALEX:') or line.startswith('JAMIE:'):
                    if current_block and block_size >= max_block_size:
                        blocks.append((current_speaker, '\n'.join(current_block)))
                        current_block = []
                        block_size = 0
                    current_speaker = line.split(':')[0]
                    current_block = [line]
                    block_size = len(line)
                else:
                    current_block.append(line)
                    block_size += len(line)
            if current_block:
                blocks.append((current_speaker, '\n'.join(current_block)))

        def translate_block(args):
            speaker, block = args
            prompt = (
                f"Translate the entire following podcast script to {target_language}. "
                f"Make the conversation sound natural and idiomatic in {target_language}, as if two native speakers are discussing the topic. "
                f"Adapt expressions and flow for naturalness, not just literal translation. "
                f"IMPORTANT: Translate the entire script below, do not skip any part. Keep the exact same dialogue structure with 'ALEX:' and 'JAMIE:' tags at the start of each line. "
                f"Do not modify or remove the speaker tags. Return the complete translated script with all lines preserved.\n\n"
                f"{block}"
            )
            print(f"[TranslationAgent] Starting translation for {target_language}.")
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a professional translator to {target_language}. Always preserve the speaker tags (ALEX: and JAMIE:) exactly as they appear. Translate the entire script, do not skip any part."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.5,
            )
            translated_text = response.choices[0].message.content.strip()
            print(f"[TranslationAgent] Translation complete for {target_language}.")

            # Process the translated text to ensure proper formatting
            lines = translated_text.split('\n')
            formatted_lines = []
            current_speaker = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('ALEX:') or line.startswith('JAMIE:'):
                    current_speaker = line.split(':')[0]
                    formatted_lines.append(line)
                else:
                    if formatted_lines:
                        formatted_lines[-1] = formatted_lines[-1] + ' ' + line
                    else:
                        formatted_lines.append(f'{speaker or "ALEX"}: {line}')
            return '\n'.join(formatted_lines)

        with ThreadPoolExecutor(max_workers=min(len(blocks), 4)) as executor:
            translated_blocks = list(executor.map(translate_block, blocks))

        result = '\n\n'.join(translated_blocks)
        result = html.unescape(result)

        print(f"[TranslationAgent] FINAL TRANSLATION RESULT for {target_language}: [content omitted for brevity]")

        # Only fall back to English if the result is empty
        if not result.strip():
            print(f"[TranslationAgent] GPT translation failed (empty result), using English.")
            result = english_script

        TranslationAgent._translation_cache[cache_key] = result
        elapsed = time.time() - start_time
        print(f"[Timing] Translation to {target_language} took {elapsed:.2f} seconds")
        return result

class TTSAgent:
    # Cache for TTS audio segments
    _tts_cache = {}
    
    @staticmethod
    def _naturalize_text(text):
        import re
        # Convert years like 2025 to 'twenty twenty-five'
        def year_to_words(match):
            year = int(match.group())
            if 2000 <= year <= 2099:
                first = 'twenty'
                second = str(year % 100)
                if second == '0':
                    return first + ' hundred'
                elif len(second) == 1:
                    second = 'oh ' + second
                return f"{first} {second}"
            return str(year)
        text = re.sub(r'20[0-9]{2}', year_to_words, text)
        # Replace 10-Q and 10-K with 'ten Q' and 'ten K'
        text = re.sub(r'10-([QK])', r'ten \1', text, flags=re.IGNORECASE)
        # Convert currency and large numbers to words (e.g., $104.169 billion -> one hundred four billion dollars)
        p = inflect.engine()
        def currency_to_words(match):
            num_str = match.group(1).replace(',', '')
            # Remove trailing non-numeric characters (like a period)
            num_str = re.sub(r'[^\d.-]+$', '', num_str)
            try:
                num = float(num_str)
            except Exception:
                num = 0
            unit = match.group(2)
            if unit:
                unit = unit.lower()
                if unit.startswith('b'):
                    num = int(num)
                    return f"{p.number_to_words(num, andword='', zero='zero', group=1)} billion dollars"
                elif unit.startswith('m'):
                    num = int(num)
                    return f"{p.number_to_words(num, andword='', zero='zero', group=1)} million dollars"
            return f"{p.number_to_words(int(num), andword='', zero='zero', group=1)} dollars"
        # $104.169 billion, $104 billion, $104,169,000,000
        text = re.sub(r'\$([\d,.]+)\s*(billion|million)?', currency_to_words, text, flags=re.IGNORECASE)
        # Convert large numbers to words (e.g., 1000000 -> one million)
        def number_to_words(match):
            num = int(match.group())
            if num >= 1000:
                return p.number_to_words(num, andword='', zero='zero', group=1)
            return str(num)
        text = re.sub(r'\b\d{4,}\b', number_to_words, text)
        # Always pronounce SEC as S-E-C
        text = re.sub(r'\bSEC\b', 'S-E-C', text)
        return text
    
    @staticmethod
    def synthesize(text: str, language: str) -> str:
        start_time = time.time()
        print(f"[TTSAgent] Starting synthesis for language: {language}")
        client = texttospeech.TextToSpeechClient()
        # Preprocess text for natural speech
        text = TTSAgent._naturalize_text(text)
        
        # Split text into speaker segments
        parts = []
        current_speaker = None
        current_text = []
        
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('ALEX:'):
                if current_speaker and current_text:
                    parts.append((current_speaker, ' '.join(current_text)))
                current_speaker = 'ALEX'
                current_text = [line[5:].strip()]
            elif line.startswith('JAMIE:'):
                if current_speaker and current_text:
                    parts.append((current_speaker, ' '.join(current_text)))
                current_speaker = 'JAMIE'
                current_text = [line[6:].strip()]
            else:
                current_text.append(line)
                
        if current_speaker and current_text:
            parts.append((current_speaker, ' '.join(current_text)))
        
        def add_sentence_pauses(text):
            # Add a 400ms pause after each sentence-ending punctuation
            return re.sub(r'([.!?])', r'\1<break time="400ms"/>', text)
        
        audio_segments = []
        lang_key = language.split('-')[0]
        
        # Map language to (ALEX, JAMIE) WaveNet voices
        voice_map = {
            'en': (('en-US', 'en-US-Wavenet-D', texttospeech.SsmlVoiceGender.MALE),
                   ('en-US', 'en-US-Wavenet-F', texttospeech.SsmlVoiceGender.FEMALE)),
            'ko': (('ko-KR', 'ko-KR-Wavenet-B', texttospeech.SsmlVoiceGender.MALE),
                   ('ko-KR', 'ko-KR-Wavenet-C', texttospeech.SsmlVoiceGender.FEMALE)),
            'ja': (('ja-JP', 'ja-JP-Wavenet-B', texttospeech.SsmlVoiceGender.MALE),
                   ('ja-JP', 'ja-JP-Wavenet-A', texttospeech.SsmlVoiceGender.FEMALE)),
            'es': (('es-ES', 'es-ES-Wavenet-B', texttospeech.SsmlVoiceGender.MALE),
                   ('es-ES', 'es-ES-Wavenet-A', texttospeech.SsmlVoiceGender.FEMALE)),
            'zh': (('cmn-CN', 'cmn-CN-Wavenet-B', texttospeech.SsmlVoiceGender.MALE),
                   ('cmn-CN', 'cmn-CN-Wavenet-A', texttospeech.SsmlVoiceGender.FEMALE)),
            'fr': (('fr-FR', 'fr-FR-Wavenet-B', texttospeech.SsmlVoiceGender.MALE),
                   ('fr-FR', 'fr-FR-Wavenet-A', texttospeech.SsmlVoiceGender.FEMALE)),
            'de': (('de-DE', 'de-DE-Wavenet-B', texttospeech.SsmlVoiceGender.MALE),
                   ('de-DE', 'de-DE-Wavenet-A', texttospeech.SsmlVoiceGender.FEMALE)),
        }
        
        def synthesize_segment(args):
            speaker, segment = args
            if not segment.strip():
                print(f"[TTSAgent] Empty segment for speaker {speaker}, skipping")
                return None
            # Check cache first
            cache_key = f"{hash(segment)}_{language}_{speaker}"
            if cache_key in TTSAgent._tts_cache:
                print(f"[TTSAgent] Using cached audio for speaker {speaker}")
                return TTSAgent._tts_cache[cache_key]
            try:
                print(f"[TTSAgent] Synthesizing segment for speaker {speaker} in {language}")
                if lang_key in voice_map:
                    if speaker == 'ALEX':
                        lang_code, voice_name, gender = voice_map[lang_key][0]
                        pitch = "+2st"
                        break_time = "800ms"
                    else:
                        lang_code, voice_name, gender = voice_map[lang_key][1]
                        pitch = "-1st"
                        break_time = "1000ms"
                    voice = texttospeech.VoiceSelectionParams(
                        language_code=lang_code,
                        name=voice_name,
                        ssml_gender=gender
                    )
                else:
                    print(f"[TTSAgent] Warning: No voice mapping for language {lang_key}, using default")
                    voice = texttospeech.VoiceSelectionParams(
                        language_code=language,
                        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                    )
                    pitch = "0st"
                    break_time = "800ms"
                # Add sentence-level pauses for naturalness
                segment_with_pauses = add_sentence_pauses(segment)
                ssml = f'<speak><prosody rate="medium" pitch="{pitch}">{segment_with_pauses}<break time="{break_time}"/></prosody></speak>'
                synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
                audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
                print(f"[TTSAgent] Calling Google TTS API for speaker {speaker}")
                tts_response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                if not tts_response.audio_content:
                    print(f"[TTSAgent] Warning: No audio content returned for segment: {segment[:100]}")
                    return None
                print(f"[TTSAgent] Successfully synthesized segment for speaker {speaker}")
                # Cache the result
                TTSAgent._tts_cache[cache_key] = tts_response.audio_content
                return tts_response.audio_content
            except Exception as e:
                print(f"[TTSAgent] Error synthesizing segment for speaker {speaker}: {str(e)}")
                print(f"[TTSAgent] Segment content: {segment[:100]}")
                return None
        # Use ThreadPoolExecutor for parallel TTS synthesis
        num_workers = max(1, min(len(parts), 4))
        print(f"[TTSAgent] Using {num_workers} workers for parallel synthesis")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            audio_segments = list(filter(None, executor.map(synthesize_segment, parts)))
        if not audio_segments:
            print("[TTSAgent] No audio segments were generated successfully")
            print(f"[TTSAgent] Input text: {text[:200]}...")
            raise Exception("No audio segments were generated successfully")
        filename = f"{language}_{int(time.time())}_{os.getpid() % 10000}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        print(f"[TTSAgent] Writing {len(audio_segments)} segments to file: {filepath}")
        with open(filepath, "wb") as out:
            for segment in audio_segments:
                out.write(segment)
        cleanup_old_files()
        print(f"[TTSAgent] Audio file written: {filepath}, size: {os.path.getsize(filepath)} bytes")
        elapsed = time.time() - start_time
        print(f"[Timing] TTS synthesis for {language} took {elapsed:.2f} seconds")
        return filename 

def download_and_extract_xbrl_facts(document_url):
    """
    Robustly download and extract XBRL facts from a plain XML file (not in a zip).
    This function handles plain XML files and extracts key financial metrics.
    """
    import requests
    import xml.etree.ElementTree as ET
    from datetime import datetime
    import re
    import os

    SEC_USER_AGENT_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL", "your-email@domain.com")
    headers = {
        "User-Agent": f"Financial Filing Podcast Summarizer ({SEC_USER_AGENT_EMAIL})",
        "Accept-Encoding": "gzip, deflate"
    }

    # Define the XBRL namespace
    ns = {'xbrli': 'http://www.xbrl.org/2003/instance',
          'us-gaap': 'http://fasb.org/us-gaap/2021-01-31'}

    # Download the XML file
    response = sec_get(document_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download XML file: {response.status_code}")

    # Parse the XML content
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        raise Exception(f"Failed to parse XML: {e}")

    # Initialize a dictionary to store extracted facts
    facts = {
        'Revenues': [],
        'NetIncomeLoss': [],
        'EarningsPerShareBasic': []
    }

    # Helper function to extract period from contextRef
    def extract_period(context_ref):
        # Try to extract period from contextRef using regex
        match = re.search(r'(\d{4})(\d{2})(\d{2})', context_ref)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"
        return None

    # Iterate over all elements in the XML
    for elem in root.findall('.//us-gaap:*', ns):
        tag = elem.tag.split('}')[-1]  # Remove namespace
        if tag in facts:
            context_ref = elem.get('contextRef', '')
            period = extract_period(context_ref)
            value = elem.text
            if value is not None:
                facts[tag].append({'period': period, 'value': value})

    return facts 

def extract_xbrl_facts_with_arelle(xbrl_path_or_url):
    cntlr = Cntlr.Cntlr()
    model_xbrl = cntlr.modelManager.load(xbrl_path_or_url)
    facts = {
        'Revenues': [],
        'NetIncomeLoss': [],
        'EarningsPerShareBasic': []
    }
    for fact in model_xbrl.facts:
        if fact.concept is not None:
            name = fact.concept.qname.localName
            if name in facts:
                period = None
                if hasattr(fact.context, 'endDatetime') and fact.context.endDatetime:
                    period = fact.context.endDatetime.strftime('%Y-%m-%d')
                value = fact.value
                if value is not None:
                    facts[name].append({'period': period, 'value': value})
    return facts 