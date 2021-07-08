# Transformers

Transformers are revolutionizing the field of NLP, and recently starting to attract attention in computer vision applcations.

### Architecture
Primary components of Transformer architecture (simplified):
* **Encoder**: receives an input, then builds a representation of the feature matrix (understanding the input)
* **Decoder**: using the encoder's representation of the input, target sequences are generated (producing output)

**Attention Layers**: allows transformer models to carefully identify the most meaningful or influential words in a given sentence...



### Use Cases
These components may be used individually, or in unison to build 'state-of-the-art' models optimized for certain tasks...
* Encoder models: sentence classification (sentiment analysis) or named entity recognition (NER)
  * BERT
  * ALBERT
  * DistilBERT
  * ELECTRA
* Decoder models: text generation (writing poems, finishing sentences)
* Encoder-Decoder models (sequence-to-sequence): language translation, summarizing text



