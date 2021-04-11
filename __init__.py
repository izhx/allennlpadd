# Make sure that allennlp is installed
try:
    import allennlp

except ModuleNotFoundError:
    print(
        "Using this library requires AllenNLP to be installed. Please see "
        "https://github.com/allenai/allennlp for installation instructions."
    )
    raise

from .modules import AdapterTransformerMismatchedEmbedder, MultiFeatrueSpanExtractor
