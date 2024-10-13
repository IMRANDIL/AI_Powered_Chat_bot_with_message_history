

# Conversational RAG with PDF Uploads and Chat History

This project is a **Conversational Retrieval-Augmented Generation (RAG) system** that allows users to upload PDF files, extract the content, and ask questions about the uploaded content. The system keeps track of chat history and contextualizes user questions based on the past conversation to provide relevant answers.

The application is built with **Streamlit** for the user interface, integrates **Groq** for the LLM (language model), and **Chroma** as the vector store to handle document embeddings and retrieval. The conversational flow is maintained with chat history to enable stateful question-answering.

## Features

- **PDF Upload**: Users can upload multiple PDF files, and the content is processed and indexed.
- **Question Answering**: Ask questions about the uploaded PDF documents, and the assistant will answer based on the content.
- **Chat History**: Keeps track of past questions and answers to contextualize future queries.
- **Integration with Groq**: Utilizes the Groq API for language modeling and response generation.
- **Embeddings**: Uses `HuggingFaceEmbeddings` to create document embeddings and retrieve relevant content.

## Requirements

To run this project, you need the following:

- Python 3.7+
- A Groq API key
- A HuggingFace API token

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/conversational-rag-with-pdf.git
   cd conversational-rag-with-pdf
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   Create a `.env` file in the root directory and add your API keys:

   ```bash
   touch .env
   ```

   Inside the `.env` file, add the following:

   ```bash
   HF_TOKEN=your_huggingface_token
   GROQ_API_KEY=your_groq_api_key
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

## Usage

### Step 1: Enter the Groq API Key

When you launch the app, you'll first need to enter your Groq API key. This is required to perform the language modeling and question answering.

### Step 2: Upload PDFs

Click on the **Choose A PDF file** button to upload one or more PDF files. The content from the PDFs will be processed and split into chunks for efficient retrieval.

### Step 3: Ask Questions

After uploading PDFs, you can ask questions related to the content of the uploaded files. The assistant will answer based on the context provided by the PDFs and previous chat history.

### Step 4: Manage Chat History

The system tracks the conversation history. You can view the session history, and the assistant uses this history to answer follow-up questions in context.

### Example Walkthrough:

1. **Start the app** and input your **Groq API key**.
2. **Upload PDFs** that contain the content you'd like to ask about.
3. Type a question in the input field like, "What is the summary of chapter 3?"
4. The assistant will respond based on the uploaded PDF content and provide a concise answer.
5. Ask follow-up questions, and the system will use the past conversation to provide context-aware responses.

## Key Components

- **Streamlit**: Provides the user interface and form interactions.
- **Chroma**: Used as the vector store to store embeddings of the split PDF documents for retrieval.
- **Groq API**: Handles the question-answering task using the `Gemma2-9b-It` model.
- **Langchain**: Manages the retrieval and document combination process, along with chat history handling.



## Future Improvements

- Add support for other document types (e.g., Word, text files).
- Improve the UI for better user interaction.
- Add multi-language support for non-English documents.
- Implement a system for saving and loading chat history across sessions.

## Troubleshooting

If you run into issues such as API key errors or file upload issues, ensure that:

- You have a valid **Groq API key** and **Hugging Face token**.
- The uploaded PDFs are valid and can be read by the PyPDFLoader.
- All dependencies are installed correctly via `pip install -r requirements.txt`.



## Acknowledgements

- [Langchain](https://github.com/hwchase17/langchain) for the amazing retrieval and LLM integration framework.
- [Streamlit](https://streamlit.io) for providing the easy-to-use UI framework.
