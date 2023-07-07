#!/bin/python3
import torch
import json
from datasets import Dataset, load_dataset
import evaluate
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback, TrainerState, TrainerControl
from copy import deepcopy
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
import wandb
import sys

test_mode=True
bf16=True
resume_from_checkpoint=True
model_name = "Salesforce/codet5p-16b"

if test_mode:
    model_name = "Salesforce/codet5p-220m"
    bf16=False
    resume_from_checkpoint=False

metric = evaluate.load('accuracy')
class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.model_max_length > 1e29:
    tokenizer.model_max_length = int(input("Enter model input max length: "))
print("Model input max length:", tokenizer.model_max_length)
if bf16:
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, pad_token_id=tokenizer.eos_token_id)
else:
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", trust_remote_code=True, pad_token_id=tokenizer.eos_token_id)

#model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", load_in_8bit=True, pad_token_id=tokenizer.eos_token_id)
#model = prepare_model_for_int8_training(model)

# Remove both split token and end token for seq2seq
datasetSplitToken = "\n\n###\n\n"
datasetEndToken = ' END'

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def preprocess_function(example):
    if not example['prompt'].endswith(datasetSplitToken):
        raise Exception("prompt does not contain split token")
    prompt = remove_suffix(example['prompt'], datasetSplitToken)
    
    if not example['completion'].endswith(datasetEndToken):
        raise Exception("completion does not contain end token")
    completion = remove_suffix(example['completion'], datasetEndToken)


    tok_input = tokenizer(prompt, truncation=True, max_length=tokenizer.model_max_length)
    if len(tok_input['input_ids']) == tokenizer.model_max_length:
        print("Warning: input length >= model max length, prompt truncated")

    with tokenizer.as_target_tokenizer():
        tok_completion=tokenizer(completion)

    tok_input["labels"] = tok_completion["input_ids"] # Set completion as labels (prediction) for input

    return tok_input#.to(device)

dataset = load_dataset("elwint/go-fuzzing-inputs")

train_dataset = dataset["train"].map(preprocess_function, batched=False)
test_dataset = dataset["test"].map(preprocess_function, batched=False)

data_collator = DataCollatorForSeq2Seq(tokenizer)

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    labels_noshift = labels.reshape(-1)
    preds_noshift = preds.reshape(-1)

    mask = labels_noshift != -100
    labels_noshift = labels_noshift[mask]
    preds_noshift = preds_noshift[mask]
    print("--- LABELS ---")
    print(tokenizer.decode(labels_noshift))
    print("--- PREDS ---")
    print(tokenizer.decode(preds_noshift))

    acc_noshift = metric.compute(predictions=preds_noshift, references=labels_noshift)

    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)

    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    acc = metric.compute(predictions=preds, references=labels)

    return {
            "accuracy": acc["accuracy"],
            "accuracy_noshift": acc_noshift["accuracy"],
    }

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

save_strategy="epoch"
evaluation_strategy="epoch"
eval_steps=None
logging_steps=5
load_best_model_at_end=True

if test_mode:
    save_strategy="no"
    evaluation_strategy="steps"
    eval_steps=1
    logging_steps=1
    load_best_model_at_end=False

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    #overwrite_output_dir=True,
    report_to="wandb",
    bf16=bf16,
    evaluation_strategy=evaluation_strategy,
    eval_steps=eval_steps,
    logging_steps=logging_steps,
    logging_first_step=True,

    save_strategy=save_strategy,
    save_total_limit=3,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    num_train_epochs=18,
    learning_rate=5e-6, # TODO: Fine-tune parameters
    lr_scheduler_type="cosine",
    weight_decay=0.005,
    per_device_train_batch_size=1, # TODO: Test with batch size?
    per_device_eval_batch_size=1,
    # gradient_accumulation_steps=16,

    # Memory saving strats
    eval_accumulation_steps=1,
    gradient_checkpointing=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.add_callback(CustomCallback(trainer))

trainer.train(resume_from_checkpoint=resume_from_checkpoint)

wandb.finish()

#trainer.save_model()
