
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
#from .embed_index import build_prompt
import openai

def build_prompt(context_chunks, user_question):
    """
    Builds a prompt from retrieved context and user question.
    """
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a financial analyst assistant for CrediTrust. "
        "Your task is to answer questions about customer complaints.\n\n"
        "Use the following retrieved complaint excerpts to formulate your answer. "
        "Focus only on the information provided. If the context does not contain enough information, "
        "state that you don't have enough information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{user_question}\n\n"
        f"Answer:"
    )
    return prompt


def load_local_llm1(model_name="facebook/opt-1.3b", device_map="auto", offload_folder="./offload"):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        offload_folder=offload_folder,
        trust_remote_code=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_answer(context_chunks, user_question, llm_pipeline, max_tokens=300):
    """
    Generate an answer using LLM and prompt.
    """
    prompt = build_prompt(context_chunks, user_question)
    response = llm_pipeline(prompt, max_new_tokens=max_tokens, do_sample=False)[0]["generated_text"]

    # Extract only the answer portion after the prompt
    answer = response.split("Answer:")[-1].strip()
    return answer

def generate_answer_openai(context_chunks, user_question, model="gpt-4"):
    prompt = build_prompt(context_chunks, user_question)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful financial analyst assistant for CrediTrust."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response['choices'][0]['message']['content'].strip()

