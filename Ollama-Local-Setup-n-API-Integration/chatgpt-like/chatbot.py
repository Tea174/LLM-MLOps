import requests
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

# API configurations
APIS = {
    "1": {
        "name": "Llama-2-7b (HuggingFace)",
        "url": "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
        "token": os.getenv("HF_TOKEN"),
        "type": "huggingface"
    },
    "2": {
        "name": "Mistral-7B (HuggingFace)",
        "url": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        "token": os.getenv("HF_TOKEN"),
        "type": "huggingface"
    },
    "3": {
        "name": "Flan-T5 (HuggingFace)",
        "url": "https://api-inference.huggingface.co/models/google/flan-t5-xxl",
        "token": os.getenv("HF_TOKEN"),
        "type": "huggingface"
    },
    "4": {
        "name": "Groq Llama",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "token": os.getenv("GROQ_API_KEY"),
        "type": "groq"
    }
}


def query_huggingface(url, token, question):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": question}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.text}"


def query_groq(url, token, question):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": question}]
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.text}"


def select_api():
    print("\nAvailable APIs:")
    for key, api in APIS.items():
        print(f"{key}. {api['name']}")

    choice = input("\nSelect API (1-4): ")
    return APIS.get(choice)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-API Chatbot')
    parser.add_argument('--api', type=str, default='1',
                        help='API to use (1-4)')

    args = parser.parse_args()
    current_api = APIS.get(args.api)

    if not current_api:
        current_api = select_api()

    print(f"\nUsing: {current_api['name']}\n")

    while True:
        question = input("You: ")

        if question.lower() == 'switch':
            current_api = select_api()
            print(f"\nUsing: {current_api['name']}\n")
            continue

        if question.lower() in ['quit', 'exit']:
            break

        # Use appropriate query function based on API type
        if current_api['type'] == 'groq':
            response = query_groq(current_api['url'], current_api['token'], question)
        else:
            response = query_huggingface(current_api['url'], current_api['token'], question)

        print(f"\nBot: {response}\n")