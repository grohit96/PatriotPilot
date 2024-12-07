PatriotPilot: Your Virtual GMU Guide
PatriotPilot is a comprehensive virtual assistant designed to provide seamless access to information about George Mason University (GMU). This guide explains the steps required to reproduce and deploy the project in your environment.

Prerequisites
Python 3.9.9 is recommended.
Install all dependencies using:
pip install -r requirements.txt

Data Preparation:
Place the structured JSON data files in the datasets folder. Run preprocessing.py to process the data and generate preprocessed_data.json.

Embedding Generation:
Run embedding.py to create embeddings from the preprocessed data. The embeddings are stored in FAISS for efficient similarity search.

Model Setup:
Download the Qwen-2.5 14B Instruct model from Hugging Face and place it in the appropriate directory.

Run CLI Version:
Use llm.py to interact with the virtual assistant via the command-line interface (CLI). You will be prompted to enter questions, and the model will generate answers.

Run GUI Version:
For a graphical interface, use app.py. This Flask-based web application provides a user-friendly front end:

On GPU clusters, the app starts a Python Flask server.
Note the exposed port, open the homepage in your browser, and start querying.
The GUI is pre-configured for GMU’s Hopper system.
Optional Fine-Tuning:
If desired, fine-tune the Qwen model using finetune.py. This uses LoRA for efficient adaptation. However, due to limited dataset size, significant performance improvement is not guaranteed.

Evaluation:
The project includes evaluation for both retrieval and text generation:

Retrieval Evaluation: A separate dataset is available to assess retrieval performance. Run retrieval_eval.py to calculate Precision, Recall, and F1-Score.
Text Generation Evaluation: Evaluate the model's text generation quality using rouge_bleu_eval.py. This script computes ROUGE and BLEU scores.

Notes
The CLI mode is quick and easy for testing purposes.
The GUI mode provides an enhanced user experience with backend support on Hopper.
Fine-tuning may require additional resources and configuration adjustments.
