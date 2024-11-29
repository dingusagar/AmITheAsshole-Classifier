import gradio as gr
import subprocess
import time
from ollama import chat
from ollama import ChatResponse

# Default model
OLLAMA_MODEL = "llama3.2:3b"
# OLLAMA_MODEL = "llama3.2:1b"

# Load BERT MODEL
from transformers import pipeline, DistilBertTokenizerFast

# Path to your locally saved model
# bert_model_path = "fine_tuned_aita_classifier"
bert_model_path = "dingusagar/distillbert-aita-classifier"

tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_path)
classifier = pipeline(
    "text-classification",
    model=bert_model_path,  # Path to your locally saved model
    tokenizer=tokenizer,  # Use the tokenizer saved with the model
    truncation=True
)

bert_label_map = {
    'LABEL_0': 'YTA',
    'LABEL_1': 'NTA',
}

def ask_bert(prompt):
    print(f"Getting response from Fine-tuned BERT")
    result = classifier([prompt])[0]
    label = bert_label_map.get(result['label'])
    confidence = f"{result['score']*100:.2f}"
    return label, confidence

def start_ollama_server():
    # Start Ollama server in the background
    print("Starting Ollama server...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)  # Give some time for the server to start

    # Pull the required model
    print(f"Pulling the model: {OLLAMA_MODEL}")
    subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)

    print("Starting the required model...")
    subprocess.Popen(["ollama", "run", OLLAMA_MODEL], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Ollama started model.")

def ask_ollama(question, expected_class=""):
    print(f"Getting response from Ollama")
    classify_and_explain_prompt = f"""
### You are an unbiased expert from subreddit community r/AmItheAsshole. In this community people post their life situations and ask if they are the asshole or not. 
The community uses the following acronyms. 
AITA : Am I the asshole? Usually posted in the question. 
YTA : You are the asshole in this situation.  
NTA : You are not the asshole in this situation.

### The task for you label YTA or NTA for the given text. Give a short explanation for the label. Be brutally honest and unbiased. Base your explanation entirely on the given text only. 

If the label is YTA, also explain what could the user have done better.  
### The output format is as follows:
"YTA" or "NTA", a short explanation. 

### Situation :  {question}
### Response :"""

    explain_only_prompt =  f"""
### You know about the subreddit community r/AmItheAsshole. In this community people post their life situations and ask if they are the asshole or not. 
The community uses the following acronyms. 
AITA : Am I the asshole? Usually posted in the question. 
YTA : You are the asshole in this situation.  
NTA : You are not the asshole in this situation.

### The task for you explain why a particular situation was tagged as NTA or YTA by most users. I will give the situation as well as the NTA or YTA tag. just give your explanation for the label. Be nice but give a brutally honest and unbiased view. Base your explanation entirely on the given text and the label tag only. Do not assume anything extra.  
Use second person terms like you in the explanation.

### Situation :  {question}
### Label Tag : {expected_class}
### Explanation for {expected_class} :"""

    if expected_class == "":
        prompt = classify_and_explain_prompt
    else:
        prompt = explain_only_prompt

    print(f"Prompt to llama : {prompt}")
    response: ChatResponse = chat(model=OLLAMA_MODEL, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    print(response['message']['content'])
    return response['message']['content']

def gradio_interface(prompt, selected_model):
    if selected_model == MODEL_CHOICE_LLAMA:
        response = ask_ollama(prompt)
    elif selected_model == MODEL_CHOICE_BERT:
        response, confidence = ask_bert(prompt)
        response = f"{response} with confidence {confidence}"
    elif selected_model == MODEL_CHOICE_BERT_LLAMA:
        bert_response, confidence = ask_bert(prompt)
        ollama_response = ask_ollama(prompt, expected_class=bert_response)
        response = f"{bert_response} with {confidence}% confidence. \n {ollama_response}"
    else:
        response = "Something went wrong. Select the correct model configuration from settings. "
    return response

MODEL_CHOICE_BERT_LLAMA = "Fine-tuned BERT (classification) + Llama 3.2 3B (explanation)"
MODEL_CHOICE_BERT = "Fine-tuned BERT (classification only)"
MODEL_CHOICE_LLAMA = "Llama 3.2 3B (classification + explanation)"

MODEL_OPTIONS = [MODEL_CHOICE_BERT_LLAMA, MODEL_CHOICE_LLAMA, MODEL_CHOICE_BERT]

# Example texts
EXAMPLES = [
    "I refused to invite my coworker to my birthday party even though we’re part of the same friend group. AITA?",
    "I didn't attend my best friend's wedding because I couldn't afford the trip. Now they are mad at me. AITA?",
    "I told my coworker they were being unprofessional during a meeting in front of everyone. AITA?",
    "I told my kid that she should become an engineer like me, she is into painting and wants to pursue arts. AITA? "
]

# Build the Gradio app
# with gr.Blocks(theme="JohnSmith9982/small_and_pretty")  as demo:
with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.purple)) as demo:
    gr.Markdown("# AITA Classifier")
    gr.Markdown(
        """### Ask this AI app if you are wrong in a situation. Describe the conflict you experienced, give both sides of the story and find out if you are right (NTA) or, you are the a**shole (YTA). Inspired by the subreddit [r/AmItheAsshole](https://www.reddit.com/r/AmItheAsshole/), this app tries to provide honest and unbiased assessments of user's life situations.
        <sub>**Disclaimer:** The responses generated by this AI model are based on the training data derived from the subreddit posts and do not represent the views or opinions of the creators or authors. This was our fun little project, please don't take the generated responses too seriously :) </sub>
        """)

    # Add Accordion for settings
    # with gr.Accordion("Settings", open=True):
    #     model_selector = gr.Dropdown(
    #         label="Select Models",
    #         choices=MODEL_OPTIONS,
    #         value=MODEL_CHOICE_BERT_LLAMA
    #     )

    with gr.Row():
        model_selector = gr.Dropdown(
                label="Selected Model",
                choices=MODEL_OPTIONS,
                value=MODEL_CHOICE_BERT_LLAMA
            )

    with gr.Row():
        input_prompt = gr.Textbox(label="Enter your situation here", placeholder="Am I the a**hole for...", lines=5)

    with gr.Row():
        # Add example texts
        example = gr.Examples(
            examples=EXAMPLES,
            inputs=input_prompt,
            label="Want to quickly try some example situations ?",
        )

    with gr.Row():
        submit_button = gr.Button("Check A**hole or not!", variant="primary")

    with gr.Row():
        output_response = gr.Textbox(label="Response", lines=10, placeholder="""Result will be YTA (you are the A**hole) or NTA(Not the A**shole)""")

    # Link the button click to the interface function
    submit_button.click(gradio_interface, inputs=[input_prompt, model_selector], outputs=output_response)

# Launch the app
if __name__ == "__main__":
    start_ollama_server()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
