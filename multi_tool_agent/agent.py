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

SEC_USER_AGENT_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL", "your-email@domain.com")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

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

class SECAgent:
    @staticmethod
    def get_cik_from_ticker(ticker: str):
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {
            "User-Agent": f"Financial Filing Podcast Summarizer ({SEC_USER_AGENT_EMAIL})",
            "Accept-Encoding": "gzip, deflate"
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            companies = response.json()
            for entry in companies.values():
                if entry['ticker'].upper() == ticker.upper():
                    return str(entry['cik_str']).zfill(10)
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching CIK for {ticker}: {str(e)}")
            return None

    @staticmethod
    def list_filings(ticker: str):
        cik = SECAgent.get_cik_from_ticker(ticker)
        if not cik:
            return None, f"CIK not found for ticker {ticker}"
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = {
            "User-Agent": f"Financial Filing Podcast Summarizer ({SEC_USER_AGENT_EMAIL})",
            "Accept-Encoding": "gzip, deflate"
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
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
            print(f"Error fetching filings for {ticker}: {str(e)}")
            return None, f"Could not fetch filings for {ticker}: {str(e)}"

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
                response = requests.get(document_url, headers=headers, timeout=30)
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
    def summarize(content: str) -> str:
        start_time = time.time()
        # Split content into chunks for faster processing
        max_chunk_size = 5000
        chunks = [content[i:i + max_chunk_size] for i in range(0, len(content), max_chunk_size)]
        def summarize_chunk(chunk):
            prompt = (
                "Extract key points from this section of an SEC filing. Focus on:\n"
                "- Financial metrics and performance\n"
                "- Major business developments\n"
                "- Risks and challenges\n"
                "- Strategic initiatives\n"
                "- Market position\n"
                "- Revenue and growth\n"
                "- Operational highlights\n"
                "- Future outlook\n"
                "- Competitive landscape\n\n"
                f"Content:\n{chunk}\n\n"
                "Key Points:"
            )
            chunk_start = time.time()
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.5,
            )
            chunk_elapsed = time.time() - chunk_start
            print(f"[Timing] Chunk summarization took {chunk_elapsed:.2f} seconds")
            return response['choices'][0]['message']['content']
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            summaries = list(executor.map(summarize_chunk, chunks))
        # Combine summaries and create podcast script
        combined_summary = "\n".join(summaries)
        podcast_prompt = (
            "Create a detailed 3-4 minute podcast script from these key points. "
            "Format as a natural conversation between Alex and Jamie where:\n"
            "- Alex asks focused questions about the most important developments\n"
            "- Jamie provides detailed, expert analysis\n"
            "- Include specific numbers and metrics when available\n"
            "- Cover at least 5-6 major topics from the filing\n"
            "- Each topic should have 2-3 exchanges between Alex and Jamie\n"
            "- Keep responses informative but conversational\n"
            "- Use natural transitions between topics\n"
            "- Focus on the most impactful information\n"
            "- IMPORTANT: Each line must start with either 'ALEX:' or 'JAMIE:' followed by their dialogue\n"
            "- Do not include any other text or formatting\n"
            "- Ensure the total script is long enough for a 3-4 minute podcast\n\n"
            f"Key Points:\n{combined_summary}\n\n"
            "Podcast Script:"
        )
        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": podcast_prompt}
            ],
            max_tokens=2048,
            temperature=0.5,
        )
        script = gpt_response['choices'][0]['message']['content'].strip()
        # Validate and fix the script format
        lines = script.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not (line.startswith('ALEX:') or line.startswith('JAMIE:')):
                if formatted_lines:
                    formatted_lines[-1] = formatted_lines[-1] + ' ' + line
                else:
                    formatted_lines.append('ALEX: ' + line)
            else:
                formatted_lines.append(line)
        summary = '\n'.join(formatted_lines)
        summary = html.unescape(summary)
        elapsed = time.time() - start_time
        print(f"[Timing] Summarization (all chunks + final) took {elapsed:.2f} seconds")
        return summary

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
        # Split script into blocks for parallel processing
        blocks = []
        current_block = []
        current_speaker = None
        block_size = 0
        max_block_size = 8000
        for line in english_script.split('\n'):
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
                f"Translate the following podcast script to {target_language}. "
                f"Make the conversation sound natural and idiomatic in {target_language}, as if two native speakers are discussing the topic. "
                f"Adapt expressions and flow for naturalness, not just literal translation. "
                f"Preserve the dialogue structure and keep the speaker tags (ALEX: and JAMIE:) at the start of each line. "
                f"Return only the translated script, with each line starting with the correct speaker tag.\n\n"
                f"{block}"
            )
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a professional translator to {target_language}."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.5,
            )
            translated_text = response['choices'][0]['message']['content'].strip()
            lines = translated_text.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if not (line.startswith('ALEX:') or line.startswith('JAMIE:')):
                    if formatted_lines:
                        formatted_lines[-1] = formatted_lines[-1] + ' ' + line
                    else:
                        formatted_lines.append(f'{speaker}: ' + line)
                else:
                    formatted_lines.append(line)
            return '\n'.join(formatted_lines)
        with ThreadPoolExecutor(max_workers=min(len(blocks), 4)) as executor:
            translated_blocks = list(executor.map(translate_block, blocks))
        result = '\n\n'.join(translated_blocks)
        result = html.unescape(result)
        num_alex = result.count('ALEX:')
        num_jamie = result.count('JAMIE:')
        if num_alex < 2 or num_jamie < 2 or len(result) < 100:
            print(f"[TranslationAgent] GPT translation failed, using English.")
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
        # Convert large numbers to words (e.g., 1000000 -> one million)
        p = inflect.engine()
        def number_to_words(match):
            num = int(match.group())
            if num >= 1000:
                return p.number_to_words(num, andword='', zero='zero', group=1)
            return str(num)
        text = re.sub(r'\b\d{4,}\b', number_to_words, text)
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