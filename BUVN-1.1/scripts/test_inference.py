"""Quick inference test on 283MB WikiText model."""
import warnings; warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference.generate import load_generator
from inference.sample import generate

model, tokenizer = load_generator('BUVN-1.1/checkpoints/ckpt.pt', 'BUVN-1.1/tokenizer/tokenizer.json')

results = []
prompts = ['The history of', 'Science is important because', 'In the year 1990', 'Artificial intelligence']
for p in prompts:
    text, usage = generate(model, tokenizer, p, 80, 0.8, 40, 'cpu')
    line = f'Prompt: "{p}"\nOutput: "{text}"\nUsage: {usage}\n---'
    results.append(line)
    print(line)

with open('BUVN-1.1/inference_output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))
print('\nSaved to BUVN-1.1/inference_output.txt')
