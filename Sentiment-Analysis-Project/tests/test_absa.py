#
# # testing Implementation 1 - Lexicon-Based with spaCy
# from src.lexicon_absa import LexiconABSA
# def test_lexicon_absa():
#     analyzer = LexiconABSA()
#     text = "The pizza was delicious but the service was terrible."
#     results = analyzer.analyze(text)
#
#     print(f"\nAnalyzing: '{text}'")
#     for result in results:
#         print(result)
#
#     assert len(results) > 0
#
#
# if __name__ == "__main__":
#     test_lexicon_absa()

# # Test Implementation 2 - Pre-trained Model
# from src.transformer_absa import TransformerABSA
# def test_transformer_absa():
#     analyzer = TransformerABSA()
#     text = "The pizza was delicious but the service was terrible."
#     results = analyzer.analyze(text)
#
#     print(f"\nAnalyzing: '{text}'")
#     for result in results:
#         print(result)
#
#
# if __name__ == "__main__":
#     test_transformer_absa()


# Test Implementation 3 - LLM with Ollama
from src.llm_absa import LLMABSA
def test_llm_absa():
    analyzer = LLMABSA()
    text = "The pizza was delicious but the service was terrible."
    results = analyzer.analyze(text)

    print(f"\nAnalyzing: '{text}'")
    for result in results:
        print(result)


if __name__ == "__main__":
    test_llm_absa()