

# Chat with Multiple PDFs

## Overview

This application allows users to chat with multiple uploaded PDF documents using a conversational AI. The app processes the PDF documents and enables the user to ask questions about their content. The answers are provided by a chatbot powered by LangChain and OpenAI's language model, and the chat history is displayed in a user-friendly interface.

## Features

- Upload and process multiple PDF documents.
- Automatically extract text from the PDFs.
- Split the extracted text into manageable chunks.
- Create a vector store for efficient retrieval.
- Chat with a bot about the content of the uploaded PDFs.
- User-friendly interface with avatars for both user and bot.

## Technologies Used

- **Streamlit**: For creating the web interface.
- **LangChain**: For handling conversational AI and document retrieval.
- **OpenAI**: For natural language processing (ChatOpenAI).
- **PyPDF2**: For extracting text from PDFs.
- **FAISS**: For storing and querying the document embeddings.

## Installation

### Prerequisites

1. Python 3.7+
2. Install the required Python libraries using `pip` or `conda`.

### Clone the Repository

```bash
git clone https://github.com/username/your-repository.git
cd your-repository
```

### Install Dependencies

Use `pip` to install the required libraries:

```bash
pip install -r requirements.txt
```

Or if you're using `conda`:

```bash
conda install --file requirements.txt
```

### Set up the Environment

Make sure you have your OpenAI API key in a `.env` file. The `.env` file should contain:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the App

Once the dependencies are installed, run the app using:

```bash
streamlit run app.py
```

This will start the app and open it in your default browser.

## Usage

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files.
2. **Process PDFs**: Click on "Process" to extract and prepare the content from the uploaded PDFs.
3. **Ask Questions**: In the main section of the app, enter your questions related to the content of the PDFs. The bot will answer based on the document text.

## File Structure

```
project/
│
├── app.py              # The main Streamlit app
├── htmlTemplates.py    # HTML templates for chat interface
├── pics/               # Directory containing bot and user images
│   ├── bot.jpeg
│   └── human.jpg
├── requirements.txt    # List of required Python libraries
└── .env                # Environment variables for API keys
```

## Troubleshooting

- **Images not displaying**: Make sure the images are located in the `pics/` folder and are named `bot.jpeg` and `human.jpg`. Also, check that your file paths are correct.
- **PDFs not processed correctly**: If no text is extracted, make sure the PDFs are not scanned images but rather contain machine-readable text.

