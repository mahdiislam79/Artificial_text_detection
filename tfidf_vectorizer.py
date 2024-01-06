class CustomTfidfVectorizer:
    def __init__(self, ngram_range=(3, 5), lowercase=False, sublinear_tf=True):
        self.vectorizer = None
        self.vocab = None
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.sublinear_tf = sublinear_tf

    def dummy(self, text):
        return text

    def fit(self, texts):
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
            sublinear_tf=self.sublinear_tf,
            analyzer='word',
            tokenizer=self.dummy,
            preprocessor=self.dummy,
            token_pattern=None,
            strip_accents='unicode'
        )
        self.vectorizer.fit(texts)
        self.vocab = self.vectorizer.vocabulary_
        
    def fit_transform(self, texts):
        if self.vectorizer is None:
            raise ValueError("fit() must be called before fit_transform()")
        
        # Use the fitted vectorizer for fit_transform
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        if self.vectorizer is None:
            raise ValueError("fit() must be called before transform()")
        
        # Use the already fitted vectorizer
        return self.vectorizer.transform(texts)