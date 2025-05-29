import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure()

models = genai.list_models()
for model in models:
    print(model.name) 