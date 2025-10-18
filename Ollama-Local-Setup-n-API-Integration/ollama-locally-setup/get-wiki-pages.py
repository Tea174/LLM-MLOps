import wikipediaapi


def download_wikipedia_page(page_title, output_file):
    """Download Wikipedia page as text"""
    wiki = wikipediaapi.Wikipedia(
        user_agent='WikiDownloader/1.0 (huongtra.tran@student.kdg.be)',
        language='en'
    )
    page = wiki.page(page_title)

    if page.exists():
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Title: {page.title}\n\n")
            f.write(page.text)
        print(f"Downloaded: {page_title}")
    else:
        print(f"Page not found: {page_title}")


# Download multiple related pages
topics = [
    "Machine Learning",
    "Artificial Intelligence",
    "Neural Network",
    "Deep Learning",
    "Natural Language Processing"
]

for topic in topics:
    filename = f"documents/wiki_{topic.replace(' ', '_').lower()}.txt"
    download_wikipedia_page(topic, filename)
