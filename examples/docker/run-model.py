#!/usr/bin/env python3
"""
Simple inference runner for TensorRT-LLM engines
Usage:
    python run-model.py --engine /workspace/engines/qwen3 --prompt "Hello world"
    python run-model.py --engine /workspace/engines/qwen3 --interactive
"""

import argparse
import sys
from pathlib import Path

try:
    import tensorrt_llm
    import tensorrt_llm.runtime as trt_runtime
    from tensorrt_llm.runtime import ModelRunner
except ImportError:
    print("❌ TensorRT-LLM not installed. Make sure you're running inside the container.")
    sys.exit(1)


def run_inference(engine_dir: str, prompt: str, max_new_tokens: int = 512):
    """Run inference on a TensorRT engine."""
    engine_path = Path(engine_dir)
    if not engine_path.exists():
        print(f"❌ Engine directory not found: {engine_dir}")
        print("   Build an engine first with trtllm-build")
        return
    
    print(f"🔥 Loading TensorRT engine from {engine_dir}...")
    
    try:
        runner = ModelRunner.from_dir(
            engine_dir=str(engine_path),
            rank=0,
        )
        
        print(f"📝 Prompt: {prompt}")
        print("🤖 Generating...\n")
        
        outputs = runner.generate(
            batch_input_ids=[prompt],
            max_new_tokens=max_new_tokens,
            end_id=None,
            pad_id=None,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            streaming=True,
        )
        
        for output in outputs:
            print(output, end="", flush=True)
        
        print("\n")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        print("Make sure the engine is compatible with your model and GPU")


def interactive_mode(engine_dir: str):
    """Interactive chat mode."""
    print(f"🤖 TensorRT-LLM Interactive Mode")
    print(f"Engine: {engine_dir}")
    print("Type 'exit' or 'quit' to end session")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\n📝 You: ")
            
            if prompt.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            if not prompt.strip():
                continue
            
            run_inference(engine_dir, prompt)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM Inference Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run-model.py --engine /workspace/engines/qwen3 --prompt "What is AI?"
  python run-model.py --engine /workspace/engines/qwen3 --interactive
  python run-model.py --engine /workspace/engines/qwen3 --prompt "Code" --max-tokens 1024
        """
    )
    
    parser.add_argument(
        "--engine", "-e",
        required=True,
        help="Path to TensorRT engine directory"
    )
    parser.add_argument(
        "--prompt", "-p",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.engine)
    elif args.prompt:
        run_inference(args.engine, args.prompt, args.max_tokens)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
