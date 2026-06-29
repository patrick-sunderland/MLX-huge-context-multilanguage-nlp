import os
import math
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from pathlib import Path

def get_model_limit_or_fail(model_name: str) -> int:
    """
    Attempts to discover the model's native context limit via its online config.
    Handles nested sub-configs (like text_config in Qwen3.5 MoE/Vision models) gracefully.
    """
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # 1. Check if properties are nested within a sub-config object
        target_config = config
        if hasattr(config, "text_config") and getattr(config, "text_config") is not None:
            target_config = getattr(config, "text_config")
        
        # 2. Comprehensive check list for context parameters
        context_attributes = [
            "max_position_embeddings",  # Llama, Mistral, Qwen, DeepSeek
            "max_kv_size",              # MLX specific allocations
            "n_positions",              # Older GPT variants
            "max_seq_len",              # MosaicML / MPT models
            "model_max_length",         # Tokenizer constraint layout
            "seq_length"                # Custom architectural backends
        ]
        
        for attr in context_attributes:
            if hasattr(target_config, attr) and getattr(target_config, attr) is not None:
                limit = getattr(target_config, attr)
                if isinstance(limit, int) and limit > 0:
                    return limit
                    
        # 3. Fallback: If properties are missing, catch explicit architecture flags
        model_type = getattr(config, "model_type", "").lower()
        architectures = [str(arch).lower() for arch in getattr(config, "architectures", [])]
        
        if "qwen3_5_moe" in model_type or any("qwen3_5moe" in a for a in architectures):
            return 131072  # Known baseline architectural ceiling for Qwen 3.5 MoE family
            
    except Exception as e:
        raise ValueError(
            f"Failed to fetch configuration for model '{model_name}'. "
            f"Ensure the repo name/path is valid. Error: {e}"
        )
    
    raise ValueError(
        f"Critical Failure: Context limit property is missing or unparseable from '{model_name}' config "
        f"(even after checking nested sub-configurations)."
    )

def process_and_save_tokens_numpy(model_name: str, input_filepath: str, output_filepath: str):
    """
    Reads text, tokenizes it using Hugging Face transformers, saves the token IDs 
    as a serialized 1D NumPy array (.npy), and evaluates strict exact RoPE requirements.
    """
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"The input file '{input_filepath}' does not exist.")

    # Map raw OpenAI names to Hugging Face configurations
    hf_model_mapping = {
        "gpt-4o": "Xenova/gpt-4o",
        "gpt-4": "openai-community/gpt4",
        "gpt-3.5-turbo": "openai-community/gpt3.5-turbo"
    }
    resolved_model_name = hf_model_mapping.get(model_name.lower(), model_name)

    # 1. Strict limit verification
    native_limit = get_model_limit_or_fail(resolved_model_name)

    # 2. Read source text file
    with open(input_filepath, "r", encoding="utf-8") as f:
        text = f.read()
    character_count = len(text)

    # 3. Load tokenizer and encode text without warning triggers
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"Could not load Hugging Face tokenizer for '{resolved_model_name}'. Error: {e}")

    # Pass verbose=False to silence length warnings during the initial analysis pass
    token_ids = tokenizer.encode(text, verbose=False)
    token_count = len(token_ids)

    # Dynamically update the tokenizer max length to match the exact size of your document
    if token_count > native_limit:
        tokenizer.model_max_length = token_count

    # 4. Convert token IDs to a 1D NumPy array and save as a binary file
    try:
        token_array = np.array(token_ids, dtype=np.int32)
        
        # Enforce extension naming convention if not provided by user
        if not output_filepath.endswith(".npy"):
            output_filepath += ".npy"
            
        np.save(output_filepath, token_array)
        print(f"\n[+] Successfully saved 1D NumPy array {token_array.shape} to: {output_filepath}")
    except Exception as e:
        raise IOError(f"Failed to write NumPy binary data to '{output_filepath}'. Error: {e}")
    
    # 5. Determine dynamic extension configurations using STRICT EXACT scaling
    requires_extension = token_count > native_limit
    
    if requires_extension:
        # Compute the exact float ratio required, with no minimum floor or rounding steps
        scaling_factor = float(token_count / native_limit)

        rope_config = {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": scaling_factor,
                "original_max_position_embeddings": native_limit
            },
            # Use math.ceil to ensure it rounds up to the nearest whole token integer
            "max_model_len": int(math.ceil(native_limit * scaling_factor))
        }
    else:
        rope_config = {
            "rope_scaling": {"rope_type": "default"},
            "max_model_len": native_limit
        }

    return {
        "character_count": character_count,
        "model_requested": model_name,
        "token_count": token_count,
        "native_limit": native_limit,
        "context_overflow": requires_extension,
        "rope_config": rope_config
    }

def main():
    parser = argparse.ArgumentParser(
        description="Strict Multi-LLM Tokenizer with Dynamic RoPE/YaRN Analyzer."
    )
    
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        required=True, 
        help="The Hugging Face repo ID or OpenAI shorthand ('gpt-4o')."
    )
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, 
        help="Path to the source plain-text input file (.txt) to process."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default=None, 
        help="Path to save the 1D NumPy output array binary. Defaults to './full-draft/pre_tokenized_[input_name].npy'."
    )

    args = parser.parse_args()

    try:
        print(f"Initializing pipeline...")
        print(f" -> Target Model: {args.model}")
        print(f" -> Input File:   {args.input}")
        
        # Resolve output path dynamically if the user didn't provide an explicit override
        if args.output is None:
            output_dir = Path("./full-draft")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            input_filename = os.path.basename(args.input)
            output_path = output_dir / f"pre_tokenized_{input_filename}"
            output_path = output_path.with_suffix(".npy")
        else:
            output_path = Path(args.output)
            if output_path.parent:
                output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f" -> Output File:  {output_path}")
        
        metrics = process_and_save_tokens_numpy(
            model_name=args.model,
            input_filepath=args.input,
            output_filepath=str(output_path)
        )
        
        print("\n--- Pipeline Run Metrics ---")
        print(f"Total Characters Tokenized:   {metrics['character_count']}")
        print(f"Total Tokens Saved:   {metrics['token_count']}")
        print(f"Native Model Capacity: {metrics['native_limit']}")
        print(f"Context Window Breach: {metrics['context_overflow']}")
        print(f"Target Scaling Config: {metrics['rope_config']}")

    except Exception as error:
        print(f"\n[!] Pipeline Error: {error}")

if __name__ == "__main__":
    main()
