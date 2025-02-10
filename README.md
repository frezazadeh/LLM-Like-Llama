# LLM-Like-Llama

📂 Project Structure

llm_project/
├── README.md
├── config.py
├── data_utils.py
├── tokenizer.py
├── model.py
├── train.py
├── inference.py
└── main.py

1️⃣ README.md

This README.md provides setup instructions and commands to run the project.

# 🧠 LLM Training and Inference

This project contains a small LLM (Large Language Model) that can be trained and used for text generation.

---

## 🚀 **Setup**
Before running anything, **install dependencies**:

```bash
pip install torch sentencepiece tqdm

🔗 Step 1: Tokenizer Preparation

Before training the model, you must prepare the tokenizer:

python tokenizer.py

This step ensures that the tokenizer is trained and ready to be used.
📈 Step 2: Training the Model

To train the model, run:

python main.py --DEVICE cuda --mode train

If you want to train on CPU, run:

python main.py --DEVICE cpu --mode train

🤖 Step 3: Running Inference

Once trained, you can generate text:

python main.py --DEVICE cuda --mode inference

If you trained the model on CPU, you can also infer on CPU:

python main.py --DEVICE cpu --mode inference

