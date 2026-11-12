
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# Load TinyStories
dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:100000]"
)

# Create BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# Split text at the byte level
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)
tokenizer.decoder = decoders.ByteLevel()

# Train
trainer = trainers.BpeTrainer(
    vocab_size=4096,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

def text_iterator():
    for example in dataset:
        yield example["text"]

tokenizer.train_from_iterator(
    text_iterator(),
    trainer=trainer,
    length=len(dataset)
)

# Save
tokenizer.save("vocab.json")

print("Vocab size:", tokenizer.get_vocab_size())
