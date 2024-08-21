from bpe import Encoder


def whitespace_tokenize(text: str) -> list:
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    return text.strip().split() if text.strip() else []


class BPETokenizer:
    def __init__(
        self,
        special_tokens=None,
        vocab_file=None,
        vocab_size=10000,
        pct_bpe=0.2,
        lowercase=False,
    ):
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab_file = vocab_file
        self.encoder = Encoder(
            vocab_size=vocab_size,
            pct_bpe=pct_bpe,
            required_tokens=self.special_tokens,
            lowercase=lowercase,
        )
        if vocab_file:
            self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file: str):
        """Load vocabulary from a file."""
        self.encoder = Encoder.load(vocab_file)

    def save_vocab(self, vocab_file: str):
        """Save vocabulary to a file."""
        self.encoder.save(vocab_file)

    def fit(self, texts: list):
        """Learn the BPE vocabulary from a list of texts."""
        self.encoder.fit(texts)

    def tokenize(self, text: str) -> list:
        """Tokenize a piece of text and return token IDs."""
        return list(self.encoder.transform([text]))[0]

    def inv_tokenize(self, token_ids: list) -> str:
        """Inverse tokenization."""
        return list(self.encoder.inverse_transform([token_ids]))[0]

    def token_to_id(self, token: str) -> int:
        """Convert a token to its corresponding ID."""
        return self.encoder.token_to_id(token)
