
from transformers import AutoTokenizer, T5ForConditionalGeneration

model_name = "t5-base" #  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def translate(text, model, tokenizer):
    inputs = tokenizer.encode("translate English to Urdu: " + text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0])
    return translated_text

text = "The future is generative ai"
print(translate(text, model, tokenizer))

