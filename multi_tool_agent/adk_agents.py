from google.agents import ToolAgent
from .agent import SummarizationAgent, TranslationAgent, TTSAgent

class SummarizationToolAgent(ToolAgent):
    def run(self, content, allowed_numbers=None):
        return SummarizationAgent.summarize(content, allowed_numbers)

class TranslationToolAgent(ToolAgent):
    def run(self, english_script, target_language):
        return TranslationAgent.translate(english_script, target_language)

class TTSToolAgent(ToolAgent):
    def run(self, text, language):
        return TTSAgent.synthesize(text, language) 