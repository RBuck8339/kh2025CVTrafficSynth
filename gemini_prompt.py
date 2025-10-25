from google import genai
from google.genai import types
import pandas as pd

client = genai.Client()

data_analytics = pd.read_csv("path to csv of the analytics")

def gemini_civil(data_analytics):
    responses = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a Senior Level Civil Engineer, and your expertise is mainly focused to improve existing road intersections"
        ),
        contents="Now with the following data analytics of near crash data, I want you to give options on how to improve the intersection and minimize risk and increase safety"
    )
    print(responses.text)

