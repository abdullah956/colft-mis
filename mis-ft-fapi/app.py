from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model_path = "../mistral-finetuned"  # adjust if needed

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": "", "user_input": ""})

@app.post("/", response_class=HTMLResponse)
async def generate_response(request: Request, user_input: str = Form(...)):
    input_text = f"### Prompt:\n{user_input}\n\n### Response:\n"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return templates.TemplateResponse("index.html", {"request": request, "response": response, "user_input": user_input})
