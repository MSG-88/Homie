"""Bake Homie's identity into model weights via QLoRA fine-tuning.

This creates a TRUE Homie base model — no system prompt needed.
The model inherently knows it's Homie, its capabilities, personality,
and security boundaries from the weights themselves.

Pipeline:
1. Load base model in 4-bit (QLoRA)
2. Train on Homie identity data (SFT + DPO)
3. Merge LoRA adapters into base weights
4. Export merged model
5. Quantize to GGUF (Q4_K_M)
6. Create Ollama model with NO system prompt
7. Push to PyMasters/Homie:native
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def bake_homie_model(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str | Path = Path.home() / ".homie" / "baked_model",
    registry_name: str = "PyMasters/Homie",
    version: str = "native",
    lora_rank: int = 32,
    epochs: int = 5,
    learning_rate: float = 2e-4,
    push: bool = False,
) -> dict:
    """Full pipeline: fine-tune → merge → quantize → create Ollama model.

    Args:
        base_model: HuggingFace model ID
        output_dir: Where to save all artifacts
        registry_name: Ollama registry name
        version: Version tag for the model
        lora_rank: LoRA rank (higher = more capacity, more VRAM)
        epochs: Training epochs over the identity dataset
        learning_rate: Learning rate for SFT
        push: Whether to push to Ollama registry

    Returns dict with all paths and results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"base_model": base_model, "version": version}

    # Step 1: Generate training data
    print("\n[1/6] Generating Homie identity training data...")
    from homie_core.finetune.homie_identity import generate_all_identity_data
    identity = generate_all_identity_data(output_dir / "identity_data")

    # Repeat identity data for stronger baking (5x repetition)
    sft_path = output_dir / "sft_train.jsonl"
    with open(sft_path, "w", encoding="utf-8") as out:
        source = Path(identity["sft_path"]).read_text(encoding="utf-8").strip().split("\n")
        for _ in range(5):  # 5x repetition = 145 examples
            for line in source:
                out.write(line + "\n")

    sft_count = len(source) * 5
    print(f"  {identity['sft_count']} base examples × 5 = {sft_count} training examples")
    results["sft_count"] = sft_count

    # Step 2: QLoRA fine-tune
    print(f"\n[2/6] Fine-tuning {base_model} with QLoRA (rank={lora_rank})...")
    adapter_dir = output_dir / "adapter"

    try:
        _train_qlora(
            base_model=base_model,
            train_file=str(sft_path),
            output_dir=str(adapter_dir),
            lora_rank=lora_rank,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        results["adapter_path"] = str(adapter_dir)
        print(f"  Adapter saved to: {adapter_dir}")
    except Exception as e:
        results["error"] = f"Training failed: {e}"
        print(f"  ERROR: {e}")
        return results

    # Step 3: Merge LoRA into base
    print("\n[3/6] Merging LoRA adapter into base model weights...")
    merged_dir = output_dir / "merged"
    try:
        _merge_adapter(base_model, str(adapter_dir), str(merged_dir))
        results["merged_path"] = str(merged_dir)
        print(f"  Merged model: {merged_dir}")
    except Exception as e:
        results["error"] = f"Merge failed: {e}"
        print(f"  ERROR: {e}")
        return results

    # Free GPU memory
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Step 4: Quantize to GGUF
    print("\n[4/6] Quantizing to GGUF (Q4_K_M)...")
    gguf_path = output_dir / f"homie-{version}.gguf"
    try:
        _quantize_gguf(str(merged_dir), str(gguf_path))
        results["gguf_path"] = str(gguf_path)
        size_gb = gguf_path.stat().st_size / (1024**3)
        print(f"  GGUF: {gguf_path} ({size_gb:.1f} GB)")
    except Exception as e:
        results["error"] = f"Quantization failed: {e}"
        print(f"  ERROR: {e}")
        return results

    # Step 5: Create Ollama model with NO system prompt
    print("\n[5/6] Creating Ollama model (NO system prompt — identity is in weights)...")
    modelfile_path = output_dir / "Modelfile.native"
    modelfile_content = f"""FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER repeat_penalty 1.2
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER stop <|endoftext|>
PARAMETER stop <|im_end|>
PARAMETER stop <|im_start|>
"""
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    model_name = f"{registry_name}:{version}"
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        results["model_name"] = model_name
        print(f"  Created: {model_name}")
    else:
        results["error"] = f"Ollama create failed: {result.stderr}"
        print(f"  ERROR: {result.stderr}")
        return results

    # Step 6: Push (optional)
    if push:
        print(f"\n[6/6] Pushing {model_name} to registry...")
        r = subprocess.run(["ollama", "push", model_name],
                          capture_output=True, text=True, timeout=1800)
        results["pushed"] = r.returncode == 0
        print(f"  {'Pushed!' if r.returncode == 0 else 'Push failed'}")
    else:
        print("\n[6/6] Skipping push (use --push to upload)")

    results["success"] = True
    # Save results
    (output_dir / "bake_result.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def _train_qlora(
    base_model: str,
    train_file: str,
    output_dir: str,
    lora_rank: int = 32,
    epochs: int = 5,
    learning_rate: float = 2e-4,
) -> None:
    """QLoRA SFT training using peft + trl."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Load dataset
    examples = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Convert to chat format strings for training
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(examples).map(format_chat)

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    # LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=2048,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Training complete. Adapter saved to {output_dir}")

    # Free memory
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


def _merge_adapter(base_model: str, adapter_path: str, output_path: str) -> None:
    """Merge LoRA adapter back into base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load base model in float16 for merging
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",  # Merge on CPU to save VRAM
        trust_remote_code=True,
    )

    # Load and merge adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    # Save merged model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"  Merged model saved to {output_path}")

    del model
    gc.collect()


def _quantize_gguf(model_path: str, gguf_output: str, quant_type: str = "q4_k_m") -> None:
    """Convert merged safetensors model to GGUF using llama-cpp-python."""
    # Try llama.cpp convert script
    try:
        # Method 1: Use llama-cpp-python's convert
        from llama_cpp import llama_cpp
        convert_script = Path(llama_cpp.__file__).parent / "scripts" / "convert_hf_to_gguf.py"
        if convert_script.exists():
            subprocess.run(
                ["python", str(convert_script), model_path,
                 "--outfile", gguf_output, "--outtype", quant_type],
                check=True, timeout=1800,
            )
            return
    except Exception:
        pass

    # Method 2: Use transformers convert
    try:
        result = subprocess.run(
            ["python", "-m", "llama_cpp.convert",
             "--model", model_path, "--output", gguf_output,
             "--quantize", quant_type],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode == 0:
            return
    except Exception:
        pass

    # Method 3: Use huggingface_hub GGUF conversion
    try:
        from huggingface_hub import snapshot_download
        # Export as GGUF via the HF API
        subprocess.run(
            ["python", "-c", f"""
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('{model_path}', torch_dtype='float16')
model.save_pretrained('{model_path}', safe_serialization=True)
"""],
            check=True, timeout=600,
        )
        # Then use llama.cpp CLI if available
        for cmd in ["llama-quantize", "llama.cpp/quantize"]:
            if shutil.which(cmd):
                subprocess.run([cmd, model_path, gguf_output, quant_type],
                             check=True, timeout=1800)
                return
    except Exception:
        pass

    raise RuntimeError(
        "Could not quantize to GGUF. Install llama.cpp or use: "
        "pip install llama-cpp-python[convert]"
    )
