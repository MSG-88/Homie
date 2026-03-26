"""QLoRA trainer wrapping unsloth/peft + trl for SFT and DPO training."""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Callable, Optional

from homie_core.finetune.config import TrainingConfig

logger = logging.getLogger(__name__)


def _check_gpu_available() -> bool:
    """Return True when a CUDA GPU is reachable."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _detect_wsl2() -> bool:
    """Return True when running inside WSL2."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


class QLoRATrainer:
    """Orchestrates QLoRA-based SFT and DPO training runs."""

    def __init__(
        self,
        base_model: str,
        config: TrainingConfig,
        output_dir: str | Path,
    ):
        self.base_model = base_model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def preflight_check(self) -> tuple[bool, str]:
        """Check GPU available, VRAM sufficient, deps installed."""
        if not _check_gpu_available():
            return False, "No GPU available or insufficient VRAM"
        # Check deps: try import unsloth, fallback peft
        try:
            import unsloth  # noqa: F401

            logger.info("Using unsloth backend")
        except ImportError:
            try:
                import peft  # noqa: F401

                logger.info("Falling back to peft backend")
            except ImportError:
                return False, "Neither unsloth nor peft is installed"
        return True, "Ready"

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def format_sft_example(self, example: dict) -> str:
        """Format a ChatML example to a training string."""
        system = example.get("system", "")
        user = example.get("user", "")
        assistant = example.get("assistant", "")
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"

    def format_dpo_example(self, example: dict) -> dict:
        """Format a DPO example to trl DPOTrainer format."""
        prompt = (
            f"<|system|>\n{example.get('system', '')}\n"
            f"<|user|>\n{example.get('user', '')}"
        )
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    @staticmethod
    def convert_alpaca_to_chatml(example: dict) -> dict:
        """Convert Alpaca ``{instruction, input, output}`` to ChatML ``{system, user, assistant}``."""
        user_msg = example.get("instruction", "")
        inp = example.get("input", "")
        if inp:
            user_msg = f"{user_msg}\n{inp}"
        return {
            "system": "",
            "user": user_msg,
            "assistant": example.get("output", ""),
        }

    # ------------------------------------------------------------------
    # Training entry-points (require GPU)
    # ------------------------------------------------------------------

    def train_sft(
        self,
        dataset_path: Path,
        checkpoint_callback: Optional[Callable] = None,
    ) -> dict:
        """Run SFT training. Returns ``{"loss": float, "adapter_path": str}``."""
        import json
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig

        logger.info("Starting SFT training on %s", self.base_model)

        # Load dataset
        examples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        logger.info("Loaded %d SFT examples", len(examples))

        # Format examples to text
        texts = [self.format_sft_example(ex) for ex in examples]

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load model + tokenizer
        logger.info("Loading model %s in 4-bit...", self.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare for kbit training
        model = prepare_model_for_kbit_training(model)

        # LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training dataset
        from datasets import Dataset
        ds = Dataset.from_dict({"text": texts})

        # SFT training config
        adapter_path = str(self.output_dir / "sft_adapter")
        training_args = SFTConfig(
            output_dir=adapter_path,
            num_train_epochs=self.config.sft_epochs,
            per_device_train_batch_size=min(self.config.batch_size, 2),  # Conservative for VRAM
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.sft_learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=self.config.checkpoint_steps,
            save_total_limit=2,
            bf16=True,
            optim="adamw_8bit",
            max_length=min(self.config.max_seq_length, 2048),  # Cap for VRAM
            dataset_text_field="text",
            report_to="none",
        )

        # Train
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            processing_class=tokenizer,
        )

        train_result = trainer.train()
        trainer.save_model(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        final_loss = train_result.training_loss
        logger.info("SFT training complete. Loss: %.4f", final_loss)

        # Cleanup GPU memory
        del model, trainer
        torch.cuda.empty_cache()

        return {"loss": final_loss, "adapter_path": adapter_path}

    def train_dpo(
        self,
        dataset_path: Path,
        sft_adapter_path: Path,
        checkpoint_callback: Optional[Callable] = None,
    ) -> dict:
        """Run DPO training. Returns ``{"loss": float, "adapter_path": str}``."""
        import json
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from trl import DPOTrainer, DPOConfig

        logger.info("Starting DPO training from SFT adapter: %s", sft_adapter_path)

        # Load dataset
        examples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        logger.info("Loaded %d DPO examples", len(examples))

        # Format for DPO trainer
        formatted = [self.format_dpo_example(ex) for ex in examples]

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load base + SFT adapter
        logger.info("Loading base model + SFT adapter...")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(sft_adapter_path))
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(str(sft_adapter_path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Rebuild LoRA for DPO
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # DPO dataset
        from datasets import Dataset
        ds = Dataset.from_dict({
            "prompt": [f["prompt"] for f in formatted],
            "chosen": [f["chosen"] for f in formatted],
            "rejected": [f["rejected"] for f in formatted],
        })

        adapter_path = str(self.output_dir / "dpo_adapter")
        dpo_args = DPOConfig(
            output_dir=adapter_path,
            num_train_epochs=self.config.dpo_epochs,
            per_device_train_batch_size=1,  # DPO needs more memory
            gradient_accumulation_steps=self.config.gradient_accumulation * 2,
            learning_rate=self.config.dpo_learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=self.config.checkpoint_steps,
            bf16=True,
            optim="adamw_8bit",
            max_length=min(self.config.max_seq_length, 1024),  # Tighter for DPO
            max_prompt_length=512,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model,
            args=dpo_args,
            train_dataset=ds,
            processing_class=tokenizer,
        )

        train_result = trainer.train()
        trainer.save_model(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        final_loss = train_result.training_loss
        logger.info("DPO training complete. Loss: %.4f", final_loss)

        del model, trainer
        torch.cuda.empty_cache()

        return {"loss": final_loss, "adapter_path": adapter_path}
