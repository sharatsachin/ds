# Hugging Face

## What is Hugging Face and how does it differ from other ML libraries?

Hugging Face provides:
- Pre-trained transformer models for NLP/CV/Audio
- Model Hub for sharing and discovering models
- Datasets library for ML datasets
- Tools for training and fine-tuning models
- Inference API and model deployment
- AutoML capabilities with AutoTrain
- Spaces for ML app deployment

Key differences from other frameworks:
- Focus on transformer architectures
- Largest collection of pre-trained models
- Stronger community and sharing features
- Better standardization across models
- Simpler fine-tuning workflows
- Integrated deployment solutions

## How do you load and use pre-trained models?

Basic model usage:
```python
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    pipeline
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Basic tokenization and inference
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)

# Using pipelines (high-level API)
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")

ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
entities = ner("My name is Sarah and I live in London")

qa = pipeline("question-answering")
result = qa(question="Who was Jim Henson?",
           context="Jim Henson was a puppeteer")

# Specific task models
classifier = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

ner_model = AutoModelForTokenClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=9
)

qa_model = AutoModelForQuestionAnswering.from_pretrained(
    'bert-base-uncased'
)
```

## How do you handle tokenization?

Tokenization operations:
```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Basic tokenization
tokens = tokenizer.tokenize("Hello, how are you?")
input_ids = tokenizer.encode("Hello, how are you?")

# Batch tokenization
inputs = tokenizer(
    ["Hello, how are you?", "I'm fine, thanks!"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"  # or 'tf' for TensorFlow
)

# Access different components
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
token_type_ids = inputs['token_type_ids']  # for some models

# Special tokens
cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id

# Decode tokens back to text
text = tokenizer.decode(input_ids[0])

# Handle long sequences
inputs = tokenizer(
    long_text,
    max_length=512,
    truncation=True,
    stride=128,
    return_overflowing_tokens=True
)
```

## How do you implement fine-tuning?

Fine-tuning transformers:
```python
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import datasets

# Load dataset
dataset = datasets.load_dataset('imdb')

# Prepare data
def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=512
    )

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# Train model
trainer.train()

# Save model
trainer.save_model("./fine_tuned_model")
```

## How do you work with the datasets library?

Working with datasets:
```python
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    Features,
    Value,
    ClassLabel
)

# Load existing dataset
dataset = load_dataset('imdb')
squad_dataset = load_dataset('squad')

# Create custom dataset
data = {
    'text': ["Hello", "World"],
    'label': [0, 1]
}

features = Features({
    'text': Value('string'),
    'label': ClassLabel(num_classes=2, names=['neg', 'pos'])
})

dataset = Dataset.from_dict(data, features=features)

# Dataset operations
# Filter
filtered = dataset.filter(lambda x: len(x['text']) > 100)

# Map
def uppercase(example):
    return {'text': example['text'].upper()}

dataset = dataset.map(uppercase, batched=True)

# Shuffle and select
shuffled = dataset.shuffle(seed=42)
subset = dataset.select(range(100))

# Split dataset
train_test = dataset.train_test_split(test_size=0.2)

# Save and load
dataset.save_to_disk('path/to/dataset')
loaded = Dataset.load_from_disk('path/to/dataset')

# Stream large datasets
streamed_dataset = load_dataset('large_dataset', streaming=True)
for example in streamed_dataset:
    process_example(example)
```

## How do you handle model evaluation?

Model evaluation:
```python
from transformers import TrainerCallback
from evaluate import load

# Load evaluation metric
metric = load("accuracy")
f1_metric = load("f1")

# Custom evaluation function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    return metric.compute(predictions=predictions, references=labels)

# Add to trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Custom evaluation loop
model.eval()
for batch in eval_dataloader:
    with torch.no_grad():
        outputs = model(**batch)
        predictions = outputs.logits.argmax(-1)
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"]
        )

final_score = metric.compute()

# Custom callback for logging
class EvaluationCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"Step {state.global_step}: {metrics}")

trainer.add_callback(EvaluationCallback())
```

## How do you implement custom training loops?

Custom training:
```python
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Learning rate scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        print(f"Loss: {loss.item()}")
```

## How do you use Accelerate for distributed training?

Distributed training with Accelerate:
```python
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# Initialize accelerator
accelerator = Accelerator()
logger = get_logger(__name__)

# Prepare model, dataloaders, optimizer
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        optimizer.zero_grad()
        
    # Evaluation
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

# Save model
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), "./model_state.pt")
```

## How do you deploy models?

Model deployment options:
```python
# Local deployment with pipeline
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="fine_tuned_model",
    tokenizer="fine_tuned_model"
)

# Export to ONNX
from transformers.onnx import export
from pathlib import Path

export(
    tokenizer=tokenizer,
    model=model,
    output=Path("model.onnx"),
    opset=12
)

# Optimize with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input_ids": input_ids})

# Export to TorchScript
traced_model = torch.jit.trace(
    model,
    (input_ids, attention_mask)
)
torch.jit.save(traced_model, "model.pt")

# Gradio interface
import gradio as gr

def predict(text):
    results = classifier(text)
    return results[0]['label'], results[0]['score']

iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs=["label", "number"]
)
iface.launch()
```

## How do you use the Hub?

Interacting with the Hub:
```python
from huggingface_hub import (
    HfApi,
    Repository,
    create_repo,
    upload_file
)

# Initialize API
api = HfApi()

# Create repository
create_repo("my-model")

# Clone repository
repo = Repository("path/to/local/repo", "username/my-model")
repo.git_pull()

# Push to hub
api.upload_file(
    path_or_fileobj="model.pt",
    path_in_repo="model.pt",
    repo_id="username/my-model"
)

# Upload model
model.push_to_hub("username/my-model")
tokenizer.push_to_hub("username/my-model")

# Download from hub
api.snapshot_download(
    repo_id="username/my-model",
    revision="main"
)

# Model cards
from huggingface_hub import ModelCard

card_content = """
---
language: en
tags:
- sentiment-analysis
- bert
---
# Model Card for my-model
"""

card = ModelCard(card_content)
card.push_to_hub("username/my-model")
```

## How do you use AutoTrain?

AutoTrain usage:
```python
from autotrain.cli import AutoTrain

# Initialize AutoTrain project
project = AutoTrain(
    project_name="my_project",
    task="text_classification",
    model_name="bert-base-uncased",
    training_data="path/to/train.csv",
    validation_data="path/to/valid.csv"
)

# Configure training
project.config(
    num_epochs=3,
    learning_rate=2e-5,
    batch_size=16,
    max_seq_length=128
)

# Start training
project.train()

# Get best model
best_model = project.get_best_model()

# Deploy model
project.deploy()
```

## How do you implement text generation?

Text generation:
```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Basic generation
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Advanced generation parameters
outputs = model.generate(
    **inputs,
    max_length=100,
    num_beams=5,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    do_sample=True,
    num_return_sequences=3
)

# Using pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
results = generator(
    "Once upon a time",
    max_length=100,
    num_return_sequences=3
)
```