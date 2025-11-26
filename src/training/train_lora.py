import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from src.configs import config
from src.utils.tokenizer import get_tokenizer
from ETL.process_data import prepare_dataset_for_training


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    return model


def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


def apply_lora(model):
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj",
                        "fc1", "fc2", "dense", "lm_head"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def train_model():
    # 1️⃣ Load dataset from disk
    dataset = load_from_disk(config.DATASET_OUTPUT_DIR)

    # 2️⃣ Load tokenizer and prepare dataset
    tokenizer = get_tokenizer()
    train_ds, test_ds = prepare_dataset_for_training(dataset, tokenizer) #TODO change to cloud

    # 3️⃣ Load model
    model = load_model()
    print_trainable_parameters(model)

    # 4️⃣ Apply LoRA
    model = apply_lora(model)
    print_trainable_parameters(model)

    # 5️⃣ Training arguments
    training_args = TrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        lr_scheduler_type='cosine',
        max_steps=config.MAX_STEPS,
        learning_rate=config.LEARNING_RATE,
        optim="paged_adamw_8bit",
        logging_steps=config.LOGGING_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        do_eval=True,
        report_to="wandb",
    )

    # 6️⃣ Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 7️⃣ Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_ds.select(range(5)),
        eval_dataset=test_ds.select(range(5)),
        args=training_args,
        data_collator=data_collator
    )

    # 8️⃣ Train
    trainer.train()

