import os
import argparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.routes import router
from inference.generate import load_generator

def create_app(checkpoint_path: str, tokenizer_path: str, device: str = None) -> FastAPI:
    """Creates and configures the FastAPI application state."""
    
    app = FastAPI(
        title="BUVN-1.1 Inference API",
        description="API for interacting with the BUVN-1.1 Decoder-Language Model.",
        version="1.1.0"
    )

    # Allow all CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach router
    app.include_router(router)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load generator models into app state memory
    try:
        model, tokenizer = load_generator(checkpoint_path, tokenizer_path, device)
        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.device = device
        print(f"Successfully loaded model on {device}")
    except Exception as e:
        print(f"WARNING: Could not load model/tokenizer. Ensure checkpoint exists. Details: {e}")
        app.state.model = None
        app.state.tokenizer = None
        app.state.device = device

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start BUVN-1.1 Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=str, default="BUVN-1.1/checkpoints/ckpt.pt")
    parser.add_argument("--tokenizer", type=str, default="BUVN-1.1/tokenizer/tokenizer.model")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda. Auto-detects if None")
    
    args = parser.parse_args()

    app = create_app(args.checkpoint, args.tokenizer, args.device)
    
    uvicorn.run(app, host=args.host, port=args.port)
