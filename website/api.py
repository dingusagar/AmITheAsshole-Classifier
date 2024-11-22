from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import (
    DistilBertForSequenceClassification, 
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

app = Flask(__name__)
CORS(app)

# Load AITA classifier model
def load_classifier_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("model/fine_tuned_aita_classifier")
    model.eval()
    return model, tokenizer

# Load Llama model (small variant for CPU)
def load_llama_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with optimized settings for CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map='cpu',
        low_cpu_mem_usage=True
    )
    return model, tokenizer

print("Loading classifier model...")
classifier_model, classifier_tokenizer = load_classifier_model()
print("Loading LLM model...")
llama_model, llama_tokenizer = load_llama_model()
print("Models loaded successfully!")

def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = ' '.join(text.split())
    text = ''.join([c if c.isalnum() or c.isspace() or c in '.,!?' else ' ' for c in text])
    return text

def predict_verdict(text):
    """Predict verdict with preprocessing"""
    clean_text = preprocess_text(text)
    
    inputs = classifier_tokenizer(
        clean_text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = classifier_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item()

    return prediction, confidence

def generate_explanation(text, verdict, confidence):
    """Generate explanation using Llama"""
    prompt = f"""Below is an AITA (Am I The A-hole) submission. The AI verdict is {verdict} with {confidence:.2%} confidence. 
    Please provide a brief, 2-3 sentence explanation for this verdict based on the story:

    Story: {text}

    Explanation:"""

    # Generate response with more conservative parameters for CPU
    generator = pipeline(
        "text-generation",
        model=llama_model,
        tokenizer=llama_tokenizer,
        max_length=1000,
        temperature=0.95,
        top_p=0.95,
        repetition_penalty=1.15,
        device_map='cpu',
        batch_size=4
    )

    response = generator(
        prompt, 
        max_new_tokens=150, 
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=llama_tokenizer.pad_token_id
    )[0]['generated_text']
    
    # Extract just the explanation part
    explanation = response.split("Explanation:")[-1].strip()
    
    return explanation

@app.route('/message', methods=['POST'])
def process_message():
    try:
        data = request.json
        message = data.get('message', '')
        
        print("Getting verdict...")
        prediction, confidence = predict_verdict(message)
        verdict = "NTA" if prediction == 1 else "YTA"
        
        print("Generating explanation...")
        explanation = generate_explanation(message, verdict, confidence)
        print("Explanation generated!")
        
        response = {
            'verdict': verdict,
            'confidence': f"{confidence:.2%}",
            'explanation': explanation,
            'message': f"Verdict: {verdict} (Confidence: {confidence:.2%})\n\nExplanation: {explanation}"
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'An error occurred while processing your request'
        }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)