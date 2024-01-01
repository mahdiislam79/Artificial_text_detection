# bpe_tokenizer_module.py

from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers, processors
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFC, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import PreTrainedTokenizerFast
from datasets import Dataset
from tqdm import tqdm

class BPETokenizer:
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
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
        
    def tokenize(self, df):
        tokenized_texts = []
        for text in tqdm(df['text'].tolist()):
            tokenized_texts.append(self.tokenizer.encode(text))
        return tokenized_texts
        
    def get_fast_tokenizer(self, max_length):
        return PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            model_max_length=max_length
        )
