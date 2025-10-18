import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
API_TOKEN = os.getenv("HF_TOKEN")


def query_huggingface(url, token, question):
    """Query HuggingFace API"""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": question}

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', str(result))
            return str(result)
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


def query_ollama(prompt, model="llama2"):
    """Query local Ollama API"""
    try:
        import ollama
        response = ollama.chat(model=model, messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"


def test_model(prompt, model_name, query_function):
    """Test a model and measure performance"""
    print(f"  Testing {model_name}...")
    start_time = time.time()
    response = query_function(prompt)
    end_time = time.time()

    return {
        "model": model_name,
        "prompt": prompt,
        "response": response[:200] + "..." if len(response) > 200 else response,
        "time": round(end_time - start_time, 2)
    }


# Test prompts
test_prompts = [
    "Explain quantum computing in simple terms",
    "Write a Python function to calculate fibonacci numbers",
    "What are the main causes of climate change?",
    "Translate 'Hello, how are you?' to French and Spanish",
    "Summarize the key points of machine learning"
]

# Run tests
print("=== STARTING MODEL COMPARISON ===\n")
results = []

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[{i}/{len(test_prompts)}] Testing: {prompt}\n")

    # Test Online API
    online_result = test_model(
        prompt,
        "HuggingFace Llama",
        lambda p: query_huggingface(API_URL, API_TOKEN, p)
    )

    # Test Ollama
    ollama_result = test_model(
        prompt,
        "Local Ollama",
        lambda p: query_ollama(p, "llama2")
    )

    results.append({"online": online_result, "ollama": ollama_result})

    print(f"\n  ✓ Online response time: {online_result['time']}s")
    print(f"  ✓ Ollama response time: {ollama_result['time']}s")
    print(f"  Speed difference: {abs(online_result['time'] - ollama_result['time']):.2f}s")

# Final Analysis
print("\n" + "=" * 50)
print("=== COMPARISON SUMMARY ===")
print("=" * 50 + "\n")

total_online_time = sum(r['online']['time'] for r in results)
total_ollama_time = sum(r['ollama']['time'] for r in results)

for i, result in enumerate(results, 1):
    print(f"\nPrompt {i}: {result['online']['prompt'][:50]}...")
    print(f"  Online time: {result['online']['time']}s")
    print(f"  Ollama time: {result['ollama']['time']}s")

    speed_diff = result['online']['time'] - result['ollama']['time']
    if speed_diff > 0:
        print(f"  → Ollama was {speed_diff:.2f}s faster")
    else:
        print(f"  → Online was {abs(speed_diff):.2f}s faster")

print("\n" + "=" * 50)
print(f"Total Online Time: {total_online_time:.2f}s")
print(f"Total Ollama Time: {total_ollama_time:.2f}s")
print(f"Average Online Time: {total_online_time / len(results):.2f}s")
print(f"Average Ollama Time: {total_ollama_time / len(results):.2f}s")
print("=" * 50)