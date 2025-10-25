from google import genai
from google.genai import types
import pandas as pd

client = genai.Client(api_key="AIzaSyCHnqaXYQL_ZDPr5pE6mgqgW9cdnsBNMpk")

chat = client.chats.create(model='gemini-2.5-flash')

def gemini_civil():
    responses = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a Senior Level Civil Engineer, and your expertise is mainly focused to improve existing road intersections"
        ),
        contents="How can you help me solve a problem of decreasing risk of injury"
    )
    print(responses.text)

# data_analytics = pd.read_csv("path to csv of the analytics")

while True:
    gemini_civil()
    message = input(">")
    if message == "exit":
        break

    res = chat.send_message(message)
    print(res.text)
