# Dr Data V3 - Bulletproof Setup

## Step 1: Install Ollama (ONE TIME)
1. Go to https://ollama.com
2. Download for your OS
3. Install it

## Step 2: Create Project
```bash
mkdir dr-data-v3
cd dr-data-v3
# Copy all files from this prompt into here
```

## Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Step 4: Run Setup Wizard
```bash
streamlit run app.py
```

## Step 5: Follow On-Screen Instructions
The app will NOT let you proceed until:
- ✅ Ollama server is running
- ✅ nomic-embed-text is installed
- ✅ llama3.1:8b is installed
- ✅ (Optional) phi4 and qwen2.5 installed

Click the **Install** buttons in the UI or run the commands shown.

## Step 6: Use the App
Once you see 🎉 Setup Complete, click "Enter Application" and start using RAG.

## Troubleshooting

**"Ollama not running"**
- Open terminal, run: `ollama serve`
- Keep that terminal open
- Refresh the app

**GPU memory / VRAM errors (laptop)**
- Run Ollama in CPU-only mode (slower but stable):
  ```powershell
  $env:OLLAMA_NO_GPU=1
  ollama serve
  ```
- Use smaller models: `phi4:latest` (fastest) or `qwen2.5:7b`
- Avoid `qwen2.5:14b` on laptops (needs 10GB+ VRAM)

**Model downloads fail**
- They are large files (GBs)
- Use the terminal commands shown in the app instead

**PDF not working**
- Install: `pip install pdfplumber`
- Restart app
