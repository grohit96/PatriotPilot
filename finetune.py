import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq

from peft import get_peft_model, LoraConfig, TaskType

from datasets import load_dataset

from accelerate import Accelerator



accelerator = Accelerator()

model_dir = "./Qwen-2.5-14B-Instruct"

dataset_file = "./qwen_loratune_data.json"



tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token



model = AutoModelForCausalLM.from_pretrained(

    model_dir,

    torch_dtype=torch.float16,

    local_files_only=True,

    trust_remote_code=True,

)



model = accelerator.prepare(model)



lora_config = LoraConfig(

    task_type=TaskType.CAUSAL_LM,

    inference_mode=False,

    r=16,

    lora_alpha=32,

    lora_dropout=0.1,

)



model = get_peft_model(model, lora_config)



dataset = load_dataset('json', data_files=dataset_file)



def preprocess_function(examples):

    inputs = [

        f"Instruction: {instr}\nContext: {ctx}\nResponse:"

        for instr, ctx in zip(examples['instruction'], examples['context'])

    ]

    responses = examples['response']

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')

    labels = tokenizer(responses, max_length=512, truncation=True, padding='max_length')['input_ids']

    labels = [

        [(label if label != tokenizer.pad_token_id else -100) for label in example]

        for example in labels

    ]

    model_inputs['labels'] = labels

    return model_inputs



tokenized_dataset = dataset.map(preprocess_function, batched=True)



data_collator = DataCollatorForSeq2Seq(

    tokenizer=tokenizer,

    model=model,

    padding=True

)



training_args = TrainingArguments(

    output_dir='./lora_qwen_finetuned',

    per_device_train_batch_size=4,

    per_device_eval_batch_size=8,

    gradient_accumulation_steps=16,

    num_train_epochs=5,

    learning_rate=1e-5,

    warmup_steps=200,

    logging_steps=50,

    evaluation_strategy="steps",

    eval_steps=500,

    save_steps=500,

    save_total_limit=5,

    fp16=True,

    report_to='none',

    resume_from_checkpoint=True

)



class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()

        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))

        shift_labels = shift_labels.view(-1)

        loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(shift_logits, shift_labels)

        if return_outputs:

            return loss, outputs

        return loss



trainer = CustomTrainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset['train'],

    eval_dataset=tokenized_dataset['train'],

    data_collator=data_collator

)



trainer.train()



model.save_pretrained('./lora_qwen_finetuned')

tokenizer.save_pretrained('./lora_qwen_finetuned')


