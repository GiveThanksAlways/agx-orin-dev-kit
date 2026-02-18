#!/usr/bin/env python3
"""
Simple Flask API server for TensorRT-LLM
OpenAI-compatible API endpoint

Usage:
    python api-server.py --engine /workspace/engines/qwen3 --port 8000
"""

import argparse
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ENGINE_DIR = None


@app.route("/v1/completions", methods=["POST"])
def completions():
    """OpenAI-compatible completions endpoint."""
    try:
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 512)
        
        # Import here to avoid loading at startup
        from tensorrt_llm.runtime import ModelRunner
        
        runner = ModelRunner.from_dir(engine_dir=ENGINE_DIR, rank=0)
        
        outputs = runner.generate(
            batch_input_ids=[prompt],
            max_new_tokens=max_tokens,
            end_id=None,
            pad_id=None,
            temperature=data.get("temperature", 0.7),
            top_k=data.get("top_k", 50),
            top_p=data.get("top_p", 0.9),
        )
        
        output_text = "".join(outputs)
        
        return jsonify({
            "id": "completion-1",
            "object": "text_completion",
            "created": 0,
            "model": ENGINE_DIR,
            "choices": [{
                "text": output_text,
                "index": 0,
                "finish_reason": "stop"
            }]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    try:
        data = request.json
        messages = data.get("messages", [])
        
        # Simple prompt construction from messages
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt += "\nassistant:"
        
        # Use completions endpoint
        completion_data = {
            "prompt": prompt,
            "max_tokens": data.get("max_tokens", 512),
            "temperature": data.get("temperature", 0.7),
        }
        
        # Call internal completions
        request.json = completion_data
        response = completions()
        
        # Convert to chat format
        completion = response.get_json()
        if "error" in completion:
            return response
        
        chat_response = {
            "id": "chat-completion-1",
            "object": "chat.completion",
            "created": 0,
            "model": ENGINE_DIR,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion["choices"][0]["text"]
                },
                "finish_reason": "stop"
            }]
        }
        
        return jsonify(chat_response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "engine": ENGINE_DIR})


def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM API Server")
    parser.add_argument("--engine", "-e", required=True, help="TensorRT engine directory")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    
    args = parser.parse_args()
    
    global ENGINE_DIR
    ENGINE_DIR = args.engine
    
    engine_path = Path(ENGINE_DIR)
    if not engine_path.exists():
        print(f"❌ Engine directory not found: {ENGINE_DIR}")
        return
    
    print(f"🚀 TensorRT-LLM API Server")
    print(f"Engine: {ENGINE_DIR}")
    print(f"Listening on http://{args.host}:{args.port}")
    print()
    print("OpenAI-compatible endpoints:")
    print(f"  POST /v1/completions")
    print(f"  POST /v1/chat/completions")
    print()
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
