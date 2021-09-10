# Make sure that allennlp is installed
try:
    import allennlp  # noqa

except ModuleNotFoundError:
    print(
        "Using this library requires AllenNLP to be installed. Please see "
        "https://github.com/allenai/allennlp for installation instructions."
    )
    raise

from .commands import tune  # noqa
from .modules import TransformerMismatchedEmbedder, MultiFeatrueSpanExtractor  # noqa
