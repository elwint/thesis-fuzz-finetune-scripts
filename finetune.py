#!/bin/python3
import torch
import json
from datasets import Dataset, load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback, TrainerState, TrainerControl
from copy import deepcopy
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
import wandb

peft_required=False
model_name = "Salesforce/codegen-350M-multi"
#model_name = "bigcode/starcoder"

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


#class LoadBestPeftModelCallback(TrainerCallback):
#    def on_train_end(
#        self,
#        args: TrainingArguments,
#        state: TrainerState,
#        control: TrainerControl,
#        **kwargs,
#    ):
#        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
#        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
#        adapters_weights = torch.load(best_model_path)
#        model = kwargs["model"]
#        set_peft_model_state_dict(model, adapters_weights)
#        return control

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
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
print("Model max length:", tokenizer.model_max_length)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, pad_token_id=tokenizer.eos_token_id)
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, pad_token_id=tokenizer.eos_token_id)

#model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", load_in_8bit=True, pad_token_id=tokenizer.eos_token_id)
#model = prepare_model_for_int8_training(model)


if peft_required:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["c_proj", "c_attn", "q_attn"]
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

datasetSplitToken = "\n\n###\n\n"
tok_split_token = tokenizer(datasetSplitToken)
datasetEndToken = ' END' # No need to check end token, because it will be replaced
print(tokenizer.eos_token, tokenizer.eos_token_id)

# This function preprocesses an example input, a dictionary with 'prompt' and 'completion' keys. 
# It ensures 'prompt' ends with a specific token (datasetSplitToken) and 'completion' ends with another (datasetEndToken).
# Both 'prompt' and 'completion' are tokenized, with 'prompt' being truncated if it exceeds permissible length, 
# ensuring that the datasetSplitToken is retained. 
# The function returns a dictionary that includes concatenated 'input_ids' and 'attention_mask' from tokenized 'prompt' and 'completion'.
# Attention mask of completion part is set to 0
def preprocess_function(example):
    if not example['prompt'].endswith(datasetSplitToken):
        raise Exception("prompt does not contain split token")
    prompt = example['prompt'].removesuffix(datasetSplitToken) # split token is re-added later
    
    if not example['completion'].endswith(datasetEndToken):
        raise Exception("completion does not contain end token")
    completion = example['completion'].replace(datasetEndToken, tokenizer.eos_token)

    tok_completion=tokenizer(completion, return_attention_mask=True)
    max_length = tokenizer.model_max_length - len(tok_split_token['input_ids']) - len(tok_completion['input_ids'])
    
    tok_input = tokenizer(prompt, truncation=True, max_length=max_length, return_attention_mask=True)
    tok_input['input_ids'] += tok_split_token['input_ids'] + tok_completion['input_ids']

    completion_idx = len(tok_input['input_ids'])-len(tok_completion['input_ids'])
    tok_input['attention_mask'] += tok_split_token['attention_mask'] + tok_completion['attention_mask']
    for idx in range(completion_idx, len(tok_input['input_ids'])):
        tok_input['attention_mask'][idx] = 0 # We want to predict the completion part (attention_mask = 0)

    if len(tok_input['input_ids']) == tokenizer.model_max_length:
        print("Warning: input length >= model max length, prompt truncated")

    return tok_input#.to(device)

dataset = load_dataset("elwint/go-fuzzing-inputs")

train_dataset = dataset["train"].map(preprocess_function, batched=False)
test_dataset = dataset["test"].map(preprocess_function, batched=False)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Post-process labels of default collator to only calculate loss for the completion part
# https://discuss.huggingface.co/t/labels-in-language-modeling-which-tokens-to-set-to-100/2346/2
def qgen_data_collator(examples) -> dict:
    if len(examples) != 1:
        raise Exception("zero or multiple examples")

    batch = data_collator(examples)

    last_idx = 0
    for idx, mask in enumerate(batch['attention_mask'][0]):
        if mask == 1:
            batch['labels'][0][idx] = -100 # Only calculate loss for the completion part, ignore prompt + split tokens TODO: Test if accuracy is idd too high if this is not set
            last_idx = idx

    if batch['input_ids'][0][last_idx] != tok_split_token['input_ids'][-1]:
        raise Exception("input not masked correctly")

    return batch

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    report_to="wandb",
    num_train_epochs=4,
    #bf16=True,
    output_dir="./results",
    evaluation_strategy="epoch",
    #evaluation_strategy="steps",
    #eval_steps=1,
    logging_steps=5,
    logging_first_step=True,

    save_strategy="epoch",

    learning_rate=2e-5, # TODO: Fine-tune parameters
    weight_decay=0.01,
    per_device_train_batch_size=1, # TODO: Test with batch size?
    per_device_eval_batch_size=1,

    # Memory saving strats
    eval_accumulation_steps=1,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=qgen_data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.add_callback(CustomCallback(trainer))
if peft_required:
    trainer.add_callback(SavePeftModelCallback)
    #trainer.add_callback(LoadBestPeftModelCallback)

trainer.train()

wandb.finish()

trainer.save_model()
