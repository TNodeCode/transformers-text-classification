import click
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from transformers import TextClassificationPipeline
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
import onnxruntime as ort
import numpy as np
import os


@click.group()
def cli():
    pass


@cli.command()
@click.option('--epochs', default=3, help='Number of training epochs')
@click.option('--output_dir', default='work_dirs', help='Directory to save the trained model')
@click.option('--train_csv', default='./data/imdb/imdb_train.csv', help='Path to the training CSV file')
@click.option('--val_csv', default='./data/imdb/imdb_test.csv', help='Path to the validation CSV file')
@click.option('--fine_tune', is_flag=True, default=False, help='If not set, only the prediction head will be trained')
@click.option('--random_init', is_flag=True, default=False, help='If set, initializes model with random weights instead of pretrained')
def train(epochs, output_dir, train_csv, val_csv, fine_tune, random_init):
    """Train a DistilBERT model on a dataset loaded from CSV files with validation after each epoch."""
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    num_classes = 2
    cache_path_train = os.path.splitext(train_csv)[0] + '.parquet'
    cache_path_val = os.path.splitext(val_csv)[0] + '.parquet'
    
    if not os.path.exists(cache_path_train):
        print("Tokenizing training data ...")
        csv_dataset = load_dataset('csv', data_files={'train': train_csv, 'test': val_csv})
        # Tokenize the dataset
        tokenized_datasets = csv_dataset.map(tokenize_function, batched=True)
        tokenized_datasets['train'].to_parquet(cache_path_train)
        tokenized_datasets['test'].to_parquet(cache_path_val)
    else:
        print("Load cached training dataset")
        tokenized_datasets = load_dataset('parquet', data_files={'train': cache_path_train, 'test': cache_path_val})

    # Set format for PyTorch
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Initialize model with or without pretrained weights
    model_init = "distilbert-base-uncased" if not random_init else None
    model = DistilBertForSequenceClassification.from_pretrained(
        model_init or "distilbert-base-uncased",  # Use pretrained or a new random initialization
        num_labels=num_classes,
    )

    if random_init:
        # If random initialization is selected, reset weights
        model.init_weights()    # Set up training arguments and Trainer

    # Freeze all layers except the prediction head if training head only
    if not fine_tune:
        for param in model.distilbert.parameters():
            param.requires_grad = False
        print("Only the prediction head will be trained.")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",  # Evaluate after every epoch
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
    )

    # Step 8: Check for a previous checkpoint to resume from
    last_checkpoint = None
    checkpoint_dirs = list(filter(lambda x: "checkpoint" in x, os.listdir(training_args.output_dir)))
    if os.path.exists(training_args.output_dir) and checkpoint_dirs:
        last_checkpoint = training_args.output_dir + "/" + checkpoint_dirs[-1]
        print(f"Resume training from {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Start training")
        trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model trained and saved in directory {output_dir}")


@cli.command()
@click.option('--model_dir', default='work_dirs', help='Directory of the trained model')
def test(model_dir):
    """Evaluate the model on the test dataset."""
    # Load dataset and tokenizer
    dataset = load_dataset("imdb")
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

    # Tokenize test dataset
    test_dataset = dataset["test"].map(lambda x: tokenizer(x["text"], padding=True, truncation=True), batched=True)
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)

    # Set up Trainer for evaluation
    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_eval_batch_size=16,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
    )

    # Evaluate model
    results = trainer.evaluate()
    click.echo(f"Test evaluation results: {results}")


@cli.command()
@click.argument('input_csv', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path())
@click.option('--model_dir', default='work_dirs', help='Directory of the trained model')
def inference(input_csv, output_csv, model_dir):
    """Run inference on a CSV file of sentences and save results."""
    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=-1)

    # Load input data
    data = pd.read_csv(input_csv)    
    if 'text' in data.columns:
        data = data.rename(columns={"text": "sentence"})
        print(data.columns)
    if 'sentence' not in data.columns:
        raise ValueError("The input CSV must contain a 'sentence' column")

    # Run inference
    max_tokens = 512
    data['label'] = data['sentence'].apply(lambda x: pipeline(x[:max_tokens])[0]['label'])

    # Save results
    data.to_csv(output_csv, index=False)
    click.echo(f"Inference results saved in {output_csv}")


@cli.command()
@click.argument('output_dir', type=click.Path())
@click.option('--model_dir', default='work_dirs', help='Directory of the trained model')
def export_onnx(output_dir, model_dir):
    """Export the trained model to ONNX format."""
    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

    # Make output dir
    os.makedirs(output_dir, exist_ok=True)

    # Export to ONNX
    dummy_input = tokenizer("dummy input", return_tensors="pt")
    onnx_path = os.path.join(output_dir, "distilbert_model.onnx")
    torch.onnx.export(model, (dummy_input['input_ids'], dummy_input['attention_mask']), onnx_path,
                      opset_version=14,
                      input_names=['input_ids', 'attention_mask'],
                      output_names=['output'],
                      dynamic_axes={'input_ids': {0: 'batch_size', 1: 'seq_length'}, 'attention_mask': {0: 'batch_size', 1: 'seq_length'}})
    click.echo(f"Model exported to ONNX format at {onnx_path}")


@cli.command()
@click.argument('onnx_model_path', type=click.Path(exists=True))
def inspect_onnx_model(onnx_model_path):
    import onnx
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    
    # Get the model's input information
    for input in model.graph.input:
        input_name = input.name
        input_type = input.type.tensor_type
        input_shape = [dim.dim_value for dim in input_type.shape.dim]
        
        print(f"Input Name: {input_name}")
        print(f"Input Type: {onnx.TensorProto.DataType.Name(input_type.elem_type)}")
        print(f"Input Shape: {input_shape}")
    
    # Get the model's input information
    for output in model.graph.output:
        output_name = output.name
        output_type = output.type.tensor_type
        output_shape = [dim.dim_value for dim in output_type.shape.dim]
        
        print(f"Output Name: {output_name}")
        print(f"Output Type: {onnx.TensorProto.DataType.Name(output_type.elem_type)}")
        print(f"Output Shape: {output_shape}")


@cli.command()
@click.argument('input_csv', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path())
@click.argument('onnx_model_path', type=click.Path(exists=True))
def inference_onnx(input_csv, output_csv, onnx_model_path):
    """Run inference on a CSV file of sentences using the ONNX model and save results."""
    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Load input data
    data = pd.read_csv(input_csv)
    if 'text' in data.columns:
        data = data.rename(columns={"text": "sentence"})
    if 'sentence' not in data.columns:
        raise ValueError("The input CSV must contain a 'sentence' column")
    
    # Prepare ONNX runtime session
    ort_session = ort.InferenceSession(onnx_model_path)

    # Run inference
    labels = []
    batch_size = 16  # Set your desired batch size
    for i in range(0, len(data), batch_size):
        batch_sentences = data['sentence'][i:i + batch_size].tolist()

        # Tokenize input
        inputs = tokenizer(batch_sentences, return_tensors='np', padding='max_length', truncation=True, max_length=512)

        # Ensure input_ids and attention_mask are int64
        input_ids = np.transpose(inputs['input_ids'].astype(np.int64), [1,0])
        attention_mask = np.transpose(inputs['attention_mask'].astype(np.int64), [1,0])

        print(input_ids.shape, attention_mask.shape)

        # Run the model
        ort_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        logits = ort_session.run(None, ort_inputs)[0]

        # Get predicted labels for the batch
        print("logits", logits.shape)
        predicted_labels = np.argmax(logits, axis=1)
        labels.extend(predicted_labels)

    # Save results
    data['label'] = labels
    data.to_csv(output_csv, index=False)
    click.echo(f"Inference results saved in {output_csv}")

if __name__ == '__main__':
    cli()