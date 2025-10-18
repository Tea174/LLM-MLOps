import ollama


def query_ollama(question, model="llama2"):
    """Query local Ollama model"""
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': question,
        },
    ])
    return response['message']['content']


# Main chatbot
print("Ollama Local Chatbot\n")

while True:
    question = input("You: ")

    if question.lower() in ['quit', 'exit']:
        break

    response = query_ollama(question)
    print(f"\nBot: {response}\n")