# 🧠 LLM Training and Inference

This project contains a small LLM (Large Language Model) that can be trained and used for text generation.

---

## 🚀 **Setup**
Before running anything, **install dependencies**:

```bash
pip install torch sentencepiece tqdm
```

---

## 🔗 Step 1: Tokenizer Preparation

Create a file named wiki.txt and add the your content (text) to it. Before training the model, you must prepare the tokenizer:

```bash
python tokenizer.py
```

This step ensures that the tokenizer is trained and ready to be used.

---

## 📈 Step 2: Training the Model

To train the model, run:

```bash
python main.py --DEVICE cuda --mode train
```

If you want to train on CPU, run:

```bash
python main.py --DEVICE cpu --mode train
```

---

## 🤖 Step 3: Running Inference

Once trained, you can generate text:

```bash
python main.py --DEVICE cuda --mode inference
```

If you trained the model on CPU, you can also infer on CPU:

```bash
python main.py --DEVICE cpu --mode inference
```

---

## 📚 Reference

For more information, visit: [Ideami](https://ideami.com/)

