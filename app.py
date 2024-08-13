import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer

# Orion-14B 모델을 로드
model_name = "/home/hansol/anaconda3/envs/mistral/custom_LLM_mistral-7B-it_final"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    #load_in_4bit=True,
    #load_in_8bit=True,
    #device_map="auto",
    device_map={"": 0},
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_name)

text_streamer = TextStreamer(tokenizer)

def answer(state, state_chatbot, text):
    input_text = f"### 질문: {text}\n\n### 답변:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=512,
            temperature=0.1,
            top_p=0.9,
            top_k=100,
            repetition_penalty=1.2,
            streamer=text_streamer,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            do_sample=True
        )
    msg = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    new_state = [{"role": "이전 질문", "content": text}, {"role": "이전 답변", "content": msg}]
    state = state + new_state
    state_chatbot = state_chatbot + [(text, msg)]

    return state, state_chatbot, state_chatbot

with gr.Blocks(css="#chatbot .overflow-y-auto{height:750px}") as demo:
    state = gr.State(
        [
            {"role": "맥락", "content": "모델 설명..."},
            {"role": "명령어", "content": "당신은 실내건축 및 마감하자 전문 챗봇 AI 입니다."}
        ]
    )
    state_chatbot = gr.State([])

    with gr.Row():
        gr.HTML("<h1>Hansoldeco llama3-70B ChatBot </h1>")
        
    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Send a message...").style(container=False)

    txt.submit(answer, [state, state_chatbot, txt], [state, state_chatbot, chatbot])
    txt.submit(lambda: "", None, txt)

demo.launch(debug=True, server_name="0.0.0.0", share=True)
