# ⚙️ **MechBro**

## 📌 **Overview**
**MechBro** is an AI-powered assistant developed to streamline **appliance troubleshooting** using technical manuals 🛠️. Using a Retrieval-Augmented Generation (RAG) architecture with Google's Gemini models, the application provides high-fidelity, accurate answers based **exclusively on the content of uploaded documents**.

This project helps support technicians by providing instant, precise guidance, leading to faster and more reliable repairs.

## 📚 **Source Documents & Processing**
- **Source**:  
  - 📄 User-uploaded PDF technical and service manuals for any appliance.
- **Content**:  
  - Contains technical specifications, troubleshooting flowcharts, error codes, and repair procedures.
  - The model's knowledge is limited *only* to the documents provided by the user.
- **Processing**:
  - **Text Extraction**: Content is parsed directly from PDF files.
  - **Chunking & Embedding**: Text is split into manageable chunks and converted into vector embeddings for semantic search.
  - **Vector Storage**: Embeddings are stored in a FAISS vector store for fast and efficient retrieval.

## 🛠️ **Installation**
```bash
git clone https://github.com/adityaverulkar/MechBro.git
cd MechBro
pip install -r requirements.txt
```
*Note: You will also need to get a [Google AI API Key](https://aistudio.google.com/app/apikey) and enter it in the app's sidebar to use the service.*

## ▶️ Running the Application
Once the installation is complete, run the application from your terminal with the following command:
```bash
streamlit run app.py
```
This will launch the application and open it in a new tab in your default web browser.

## 🚀 **Benefits & Usage**
MechBro delivers significant value for field service technicians, DIY enthusiasts, and repair shops:

1.  **Rapid Diagnostics:** Instantly get solutions for complex error codes or symptoms without manually flipping through dense manuals.
2.  **Increased First-Time Fix Rate:** Reduce callbacks by ensuring the correct troubleshooting steps are followed from the start.
3.  **Reduced Human Error:** Provides answers directly from the manufacturer's text, minimizing guesswork and procedural mistakes.
4.  **On-Demand Expertise:** Puts the entire knowledge base of a technical manual at your fingertips, accessible via a simple chat interface.
5.  **Standardized Training:** Helps new technicians learn manufacturer-approved procedures quickly and effectively.


## **🔬 Technology & Architecture**

 - **Architecture:** Retrieval-Augmented Generation (RAG)
 - **Frameworks:** LangChain & Streamlit 🔧
 - **LLM:** Google Gemini (`gemini-1.5-flash-latest`)
 - **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)

## **📈 Results**

 - ✅ Delivers high-fidelity, context-aware answers grounded exclusively in the source manuals.
 - ⚙️ Generates structured outputs with a "Problem Analysis" and clear, "Actionable Steps" for technicians.
 - 🗣️ Handles conversational follow-ups and recognizes user intent (e.g., technical question vs. feedback).

## **🤝 Contributing**
Pull requests and forks are welcome! If you have ideas to improve the prompt, add features, or enhance the processing pipeline, feel free to contribute.
## **📜 License**
This project is licensed under the MIT License.
