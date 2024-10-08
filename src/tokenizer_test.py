from tokenizer import BPETokenizer,  whitespace_tokenize

input_wiki = open("./src/bpe.py", "r", encoding="utf-8").read()
input_wiki = whitespace_tokenize(input_wiki)
tokenizer = BPETokenizer(vocab_size=200, pct_bpe=0.4, lowercase=False)
tokenizer.fit(input_wiki)
tokenizer.save_vocab("vocab.json")
tokenizer.load_vocab("vocab.json")

toks = tokenizer.tokenize("def main():\n    Print('Hello, world!')")
print(toks)
for tok in toks:
    print(tokenizer.inv_tokenize([tok]))