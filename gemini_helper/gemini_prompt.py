from google import genai
from google.genai import types
import pandas as pd
import io
import os

# Use environment variable for API key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Set it with: export GOOGLE_API_KEY='your-api-key-here'")
    exit()

try:
    # Initialize client (v1.x SDK)
    client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"Error initializing client: {e}")
    exit()

DATA_ANALYST_PROMPT = (
    "You are an expert data analyst. Your primary goal is to help me understand a "
    "dataset and guide me, step-by-step, on how to build an insightful report from it. "
    "\n\n"
    "I will provide you with the dataset's `info()` and `head()`."
    "\n\n"
    "Your responses should be: "
    "1.  **Insightful:** Go beyond the obvious. Suggest relationships between columns, "
    "    potential key performance indicators (KPIs), and interesting questions to explore."
    "2.  **Actionable:** Give me clear, specific steps for analysis or for sections of the report."
    "3.  **Concise:** Be detailed and knowledgeable, but avoid unnecessary verbosity. Get to the point."
    "\n\n"
    "When I ask how to build a report, guide me through sections like:"
    "-   **Introduction:** What is the data and what problem are we solving?"
    "-   **Data Overview:** Key stats and data quality notes."
    "-   **Key Findings:** What are the main insights? (e.g., 'Sales are highest in Q3', 'X correlates with Y')."
    "-   **Visualizations:** What charts would best represent these findings?"
    "-   **Conclusion & Recommendations:** What should be the key takeaways or next steps?"
)

def load_data():
    """Prompts the user for a CSV path and loads it into a pandas DataFrame."""
    while True:
        # Fixed: Actually prompt the user
        csv_path = input("Enter the path to your CSV file: ").strip()

        if not os.path.exists(csv_path):
            print(f"Error: File not found at '{csv_path}'. Please try again.")
            continue

        try:
            df = pd.read_csv(csv_path)
            print(f"Successfully loaded '{csv_path}'.")
            return df
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{csv_path}' is empty.")
            return None
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None

def get_data_summary(df):
    """Generates a text summary (info and head) of the DataFrame."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()
    df_head = df.head().to_string()

    return (
        "Here is the data I want you to analyze.\n\n"
        "--- Data Structure and Info (df.info()) ---\n"
        f"{df_info}\n\n"
        "--- First 5 Rows (df.head()) ---\n"
        f"{df_head}\n\n"
        "Please confirm you've received this and are ready to help me build a report."
    )

def start_chat_session():
    """Main function to initialize and run the chat."""

    # 1. Load Data
    df = load_data()
    if df is None:
        print("Could not load data. Exiting.")
        return

    # 2. Get Data Summary
    initial_prompt = get_data_summary(df)

    # 3. Create chat session with system instruction (v1.x SDK)
    try:
        chat = client.chats.create(
            model='gemini-2.0-flash-exp',  # Updated model name
            config=types.GenerateContentConfig(
                system_instruction=DATA_ANALYST_PROMPT,
                temperature=0.7
            )
        )
    except Exception as e:
        print(f"Error creating chat: {e}")
        return

    # 4. Send initial data summary
    print("\nSending data summary to the AI analyst...")
    try:
        response = chat.send_message(initial_prompt)
        print(f"\nModel: {response.text}")
    except Exception as e:
        print(f"Error sending initial message: {e}")
        return

    # 5. Interactive loop
    print("\n--- Chat with your Data Analyst ---")
    print("You can now ask for analysis, insights, or how to build your report.")
    print("Type 'exit' to end the chat.")

    while True:
        message = input("\nYou: ").strip()
        if message.lower() == "exit":
            print("Chat session ended. Goodbye!")
            break

        if not message:
            continue

        try:
            response = chat.send_message(message)
            print(f"\nModel: {response.text}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try your message again.")

if __name__ == "__main__":
    start_chat_session()
