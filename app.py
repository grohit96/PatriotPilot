from flask import Flask, request, jsonify, send_from_directory

import json

import numpy as np

import faiss

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sentence_transformers import SentenceTransformer



app = Flask(__name__, static_folder=".", static_url_path="")



model_dir = "./Qwen-2.5-14B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



faiss_index = faiss.read_index('faiss_index.index')



with open('metadata.json', 'r') as f:

    metadata = json.load(f)



embedding_model = SentenceTransformer('intfloat/e5-large-v2')



qwen_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

qwen_model = AutoModelForCausalLM.from_pretrained(

    model_dir,

    torch_dtype=torch.float16,

    local_files_only=True,

    trust_remote_code=True

).to(device)



def search_faiss(query, k=5):


    query=query.lower()
    query_embedding = embedding_model.encode([query])

    D, I = faiss_index.search(np.array(query_embedding).astype("float32"), k=k)

    retrieved_texts = [metadata[i] for i in I[0]]

    filtered_texts = [text for text in retrieved_texts if any(keyword.lower() in text.lower() for keyword in query.split())]

    return filtered_texts[:3]



def generate_response(query, retrieved_texts, max_context_length=2048):

    context = " ".join(retrieved_texts[:3])[:max_context_length]

    prompt = (

        f"Context extracted from relevant sources:\n{context}\n\n"

        "Answer the following question concisely, using only the facts provided in the context. Avoid assumptions or inferred information. "

        "Keep the response professional, direct, and no longer than necessary to answer the query. Do not repeat the query in your answer.\n"

        "Do not frame your own questions and answers.\n"

        "Do not provide additional notes or disclaimers.\n"

        f"Question: {query}\nAnswer:"

    )

    inputs = qwen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    outputs = qwen_model.generate(

        **inputs,

        max_new_tokens=150,

        num_return_sequences=1,

        temperature=0.5,

        top_k=50,

        pad_token_id=qwen_tokenizer.eos_token_id

    )

    response = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = response.split("Answer:")[-1].strip()

    response = response.split("Based on the provided context")[0].strip()

    if not response.endswith("?"):

        response += "\n\nDo you need help with anything else?"

    return response



@app.route("/")

def html():

    return send_from_directory(".", "CS_Department.html")



@app.route("/api/chat", methods=["POST"])

def patriotpilot():

    data = request.json

    query = data.get("query", "")

    if not query:

        return jsonify({"response": "Please provide a query"}), 400

    try:

        retrieved_texts = search_faiss(query)

        response = generate_response(query, retrieved_texts)

        return jsonify({"response": response})

    except Exception as e:

        return jsonify({"response": f"An error occurred: {str(e)}"}), 500



if __name__ == "__main__":

    app.run(host="0.0.0.0", port=39507)


