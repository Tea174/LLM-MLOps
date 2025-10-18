
### Set Up Ollama Locally
Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
![img_2.png](img_2.png)
(I have my other window laptop with nvidia card 
but Ollama is not allowed to be there during exam 
so I use this laptop huhu)

Download a Model: ollama pull llama2
To test: ollama run llama2 (super SLOW here in my HP probook without nvidia)
![img_1.png](img_1.png)

### Create Ollama Chatbot Script
Install Ollama Python Library: pip install ollama
![img_3.png](img_3.png)

### Compare Ollama Performance 
*Test 5 prompts with both online and Ollama models, compare results.*
![img_4.png](img_4.png)
![img_5.png](img_5.png)
![img_6.png](img_6.png)

### Implement RAG with Ollama 
Install lib: pip install chromadb sentence-transformers pypdf

##### prepare the documents
install pip install wikipedia-api -> create .txt docs in documents folder

- run the RAG-system.py file:
![img_7.png](img_7.png)
![img_8.png](img_8.png)
![img_9.png](img_9.png)

