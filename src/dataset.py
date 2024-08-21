from torch.utils.data import Dataset
import multiprocessing as mp
import torch


def create_samples_for_range(text, context_size, start_idx, end_idx):
    return [
        (text[i : i + context_size], text[i + context_size])
        for i in range(start_idx, end_idx)
    ]


def create_samples(text, context_size, pad_token, num_workers=None):
    if num_workers is None:
        num_workers = mp.cpu_count()
    padded_text = text + [pad_token] * context_size
    chunk_size = len(text) // num_workers
    ranges = [
        (i * chunk_size, min((i + 1) * chunk_size, len(text)))
        for i in range(num_workers)
    ]

    with mp.Pool(num_workers) as pool:
        results = pool.starmap(
            create_samples_for_range,
            [(padded_text, context_size, start, end) for start, end in ranges],
        )

    return [sample for sublist in results for sample in sublist]


class TextDataset(Dataset):
    def __init__(self, text, context_size, pad_token):
        self.samples = create_samples(text, context_size, pad_token)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer):
    max_context_length = max(len(context) for context, _ in batch)
    pad_token_id = tokenizer.token_to_id(tokenizer.encoder.PAD)

    contexts, targets = zip(*batch)
    contexts = [
        list(context) + [pad_token_id] * (max_context_length - len(context))
        for context in contexts
    ]

    return torch.LongTensor(contexts), torch.LongTensor(targets)
