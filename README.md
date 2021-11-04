# allennlpadd
Some additional classes for allennlp.

## 1 Overview
The modules and functions I have implemented are listed below.

### 1.1 Modules
1. Span extractors registered in `SpanExtractor` :
   - `PoolingSpanExtractor` : Represent spans as the pooling (mean or max) of all tokens' embeddings , registered as `"pooling"`.
   - `MultiFeatureSpanExtractor` : A extractor to combine representations from any kinds of enabled `SpanExtractor` (i.e., `BidirectionalEndpointSpanExtractor`, `EndpointSpanExtractor`, `PoolingSpanExtractor`, `SelfAttentiveSpanExtractor`), registered as `"multi_feat"`.


2. Token embedders registered in `TokenEmbedder` : 
   - `AdapterTransformerEmbedder` : A parameter-effeicient way to use pre-trained BERT model on the down-stream tasks [1], which freeze BERT parameters and insert `Adapter`s into every layer, registered as `"adapter_transformer"`.
   - `PgnAdapterTransformerEmbedder` : A special `AdapterTransformerEmbedder` with `domain_embedding`, could dynamically generate parameters for adapters inside [4], registered as `"pgn_adapter_transformer"`.
   - `TransformerMismatchedEmbedder` : The general mismatched version of various `*TransformerEmbedder`, registered as `"transformer_mismatched"`.

3. Seq2seq encoders registered in `Seq2SeqEncoder` : 
   - `ParameterGenerationLstmSeq2SeqEncoder`: A LSTM that could dynamically generate parameters for different domian to encode domain-aware representations [2, 3]. Coming soon.


### 1.2 Subcommands
1. `tune`: sub-command for hyperparameter optimization using [Optuna](https://github.com/optuna/optuna), modified from [allennlp-optuna](https://github.com/himkt/allennlp-optuna).
   - It combines the `hparams.json` and `optuna.json` in a single `jsonnet` config file.
   - It will delete all trial archives except the best trial in the end of optimization.


### 1.3 Functions
1. `common`
   - `util`
      - `construct_from_params` : Construct a subclass of `FromParams` from the given `kwargs` argument dict.

2. `nn`
   - `util`
      - `batched_linear` : A linear forward with batched weights, i.e.,ever samples in the batch have the corresponding `weight` and `bias`.
      - `batched_prune` : Prune elements based on the given `scores`.


## 2 Usage

1. If you are developing your project in a git repo, I recommend using this repo as a `submodule` by `git submodule add https://github.com/izhx/allennlpadd.git`.
If not, clone to your project root by `git clone https://github.com/izhx/allennlpadd.git`.

To use functions or modules in the `import` style:

2. Directly import it from python scripts, such as `from allnlpadd.nn.util import batched_prune`.

To use registered modules in config files:

2. Add `allennlpadd` to your `.allennlp_plugins` file in your project root, or add an argument `--include-package=allennlpadd`, so that allennlp will scan and register classes in this repo when running.

3. Use the above registered classes (mainly are modules) as you need.


## 3 References
1. Houlsby, Neil, et al. "[Parameter-efficient transfer learning for NLP](http://proceedings.mlr.press/v97/houlsby19a.html)." International Conference on Machine Learning. PMLR, 2019.
2. Platanios, Emmanouil Antonios, et al. "[Contextual Parameter Generation for Universal Neural Machine Translation](https://www.aclweb.org/anthology/D18-1039)." Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018.
3. Jia, Chen, Xiaobo Liang, and Yue Zhang. "[Cross-domain NER using cross-domain language modeling](https://www.aclweb.org/anthology/P19-1236)." Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019.
4. Xin Zhang, Guangwei Xu, Yueheng Sun, Meishan Zhang, Pengjun Xie. "[Crowdsourcing Learning as Domain Adaptation: A Case Study on Named Entity Recognition](https://arxiv.org/abs/2105.14980)." Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP). 2021.

## 4 Citation
If you use AllenNLP in your research, please cite [AllenNLP: A Deep Semantic Natural Language Processing Platform](https://www.semanticscholar.org/paper/AllenNLP%3A-A-Deep-Semantic-Natural-Language-Platform-Gardner-Grus/a5502187140cdd98d76ae711973dbcdaf1fef46d).
```
@inproceedings{Gardner2017AllenNLP,
  title={AllenNLP: A Deep Semantic Natural Language Processing Platform},
  author={Matt Gardner and Joel Grus and Mark Neumann and Oyvind Tafjord
    and Pradeep Dasigi and Nelson F. Liu and Matthew Peters and
    Michael Schmitz and Luke S. Zettlemoyer},
  year={2017},
  Eprint = {arXiv:1803.07640},
}
```

If you use modules I developed (e.g., Adapter-BERT, PGN-LSTM), please cite the corresponding paper in the above "References" section.
```
@misc{allennlpadd,
  title={allennlpadd: additional classes for allennlp},
  url={https://github.com/izhx/allennlpadd},
  note={Software available from https://github.com/izhx/allennlpadd},
  author={Xin Zhang},
  year={2021},
}
```
