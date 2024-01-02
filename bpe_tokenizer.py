# bpe_tokenizer_module.py

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm
from datasets import Dataset

class BPETokenizer:
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
        self.tokenizer.pretokenizer = pre_tokenizers.ByteLevel()
        self.tokenizer.post_processor = processors.TemplateProcessing(
                                                                      single="[CLS] $A",
                                                                      special_tokens=[("[CLS]", 1)])
    @classmethod
    def dataset_iterator(cls, dataset, chunk_size=1000):
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

    def train(self, df):
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        dataset = Dataset.from_pandas(df[['text']])
        self.tokenizer.train_from_iterator(self.dataset_iterator(dataset), trainer=trainer)
        return self

    def encode(self, df):
        tokenized_texts = []
        for text in tqdm(df['text'].tolist()):
            tokenized_texts.append(self.tokenizer.encode(text))
        return tokenized_texts

    def get_fast_tokenizer(self):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        return tokenizer
