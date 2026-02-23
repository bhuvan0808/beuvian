import traceback
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
import time

# Create a router
router = APIRouter()

# Input schema
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input text to complete", example="The future of artificial intelligence is")
    max_tokens: int = Field(default=100, ge=1, le=512, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling parameter")

# Output schema
class GenerateResponse(BaseModel):
    generated_text: str
    usage: dict
    latency_ms: float

# The model will be injected via FastAPI's request.app.state
def get_model(request: Request):
    return request.app.state.model

def get_tokenizer(request: Request):
    return request.app.state.tokenizer

def get_device(request: Request):
    return request.app.state.device

@router.post("/generate", response_model=GenerateResponse, summary="Generate text from BUVN-1.1 model")
async def generate_text(
    req: GenerateRequest,
    model=Depends(get_model),
    tokenizer=Depends(get_tokenizer),
    device=Depends(get_device)
):
    """
    Endpoint to generate text unconditionally or conditionally from a prompt using the BUVN-1.1 Decoder model.
    """
    start_time = time.time()
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model currently unavailable. Server is starting up or has no checkpoint.")
        
    try:
        # Import the generate function here to avoid circular imports during startup
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from inference.sample import generate
        
        text, usage = generate(
            model=model,
            tokenizer=tokenizer,
            prompt_str=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            device=device
        )
        
        latency = (time.time() - start_time) * 1000
        
        return GenerateResponse(
            generated_text=text,
            usage=usage,
            latency_ms=latency
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
