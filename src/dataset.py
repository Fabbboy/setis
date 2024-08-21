from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
import torch
from tokenizer import whitespace_tokenize, BPETokenizer


def create_samples_for_range(
    text: list, context_size: int, start_idx: int, end_idx: int
) -> list:
    samples = []
    for i in range(start_idx, end_idx):
        context = text[i : i + context_size]
        target = text[i + context_size]
        samples.append((context, target))
    return samples


def create_samples(
    text: list, context_size: int, pad_token: int, num_workers: int = None
) -> list:
    if num_workers is None:
        num_workers = mp.cpu_count()
    padded_text = text + [pad_token] * context_size
    chunk_size = len(text) // num_workers
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]
    ranges[-1] = (ranges[-1][0], len(text))
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(
            create_samples_for_range,
            [(padded_text, context_size, start, end) for start, end in ranges],
        )
    samples = [sample for sublist in results for sample in sublist]
    return samples


class TextDataset(Dataset):
    def __init__(self, text: list, context_size: int, pad_token: int):
        self.samples = create_samples(text, context_size, pad_token)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, tokenizer):
    max_context_length = max(len(context) for context, _ in batch)
    pad_token_id = tokenizer.token_to_id(tokenizer.encoder.PAD)
    contexts, targets = [], []
    for context, target in batch:
        context = context + [pad_token_id] * (max_context_length - len(context))
        contexts.append(context)
        targets.append(target)
    return torch.LongTensor(contexts), torch.LongTensor(targets)

