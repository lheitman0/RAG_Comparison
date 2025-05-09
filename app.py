import streamlit as st
import os
import re
import json
from pathlib import Path
import markdown
import base64
from bs4 import BeautifulSoup
from typing import Dict, Any  # ADD IMPORT

# Retrieval framework (new)
from src.retrieval.base import RetrievalRecipe, retrieve
from src.utils.language_id import detect_language
from openai import OpenAI

# Set page config
st.set_page_config(
    page_title="RAG Comparison",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set light theme
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #262730;
    }
    .stSidebar {
        background-color: #F8F9FA;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1E3A8A;
    }
    /* Ensure all text is dark, not white */
    p, div, span, li, td, th {
        color: #262730 !important;
    }
    .metric-card {
        background-color: #F1F5F9;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #3B82F6;
    }
    .metric-value {
        font-size: 16px;
        font-weight: bold;
        color: #262730 !important;
    }
    .stButton button {
        background-color: #F1F5F9;
        color: #262730;
        border: none;
        text-align: left;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #E5E7EB;
    }
    .nav-button-selected {
        background-color: #E5E7EB;
        font-weight: bold;
    }
    /* Chat container styling */
    .chat-area {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    /* Message styling */
    .stChatMessage {
        background-color: #F8F9FA !important;
        border-radius: 10px !important;
        padding: 10px !important;
        margin-bottom: 10px !important;
    }
    /* JSON Code block styling */
    .document-container pre {
        background-color: #F1F5F9;
        border-radius: 5px;
        padding: 15px;
        overflow-x: auto;
        font-family: 'Courier New', monospace;
        line-height: 1.4;
        font-size: 14px;
        color: #374151 !important;
    }
    .document-container code {
        background-color: #F1F5F9;
        border-radius: 3px;
        padding: 2px 5px;
        font-family: 'Courier New', monospace;
        color: #374151 !important;
        font-size: 14px;
    }
    /* Specific JSON styling */
    .json-content {
        color: #374151 !important;
        background-color: #F8F9FA;
        border-radius: 5px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        overflow-x: auto;
    }
    .json-key {
        color: #1E3A8A !important;
        font-weight: bold;
    }
    .json-string {
        color: #047857 !important;
    }
    .json-number {
        color: #C026D3 !important;
    }
    .json-boolean {
        color: #B91C1C !important;
    }
    /* Ensure document container has dark text */
    .document-container {
        padding: 20px;
        background-color: white;
        border-radius: 5px;
        color: #262730 !important;
    }
    .document-container * {
        color: #262730 !important;
    }
    .document-container img {
        max-width: 100%;
        height: auto;
        margin: 20px 0;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .document-container table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
    }
    .document-container th, .document-container td {
        border: 1px solid #E5E7EB;
        padding: 8px;
        text-align: left;
    }
    .document-container th {
        background-color: #F1F5F9;
    }
    .document-container tr:nth-child(even) {
        background-color: #F8F9FA;
    }
    .document-container h1, .document-container h2 {
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 8px;
    }
    /* Override black backgrounds in code blocks */
    div.stCodeBlock {
        background-color: #F1F5F9 !important;
    }
    div.stCodeBlock pre {
        background-color: #F1F5F9 !important;
    }
    /* Style the selectbox to match light theme */
    div[data-baseweb="select"] {
        background-color: #F8F9FA !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #F8F9FA !important;
        border-color: #E5E7EB !important;
    }
    /* Override default dark sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 1px solid #E5E7EB;
        z-index: 150;
        height: calc(100vh - 50px);
    }
    /* Make the dropdown match light theme */
    div[role="listbox"] {
        background-color: #FFFFFF !important;
    }
    div[role="listbox"] div[role="option"] {
        background-color: #FFFFFF !important;
        color: #262730 !important;
    }
    div[role="listbox"] div[role="option"]:hover {
        background-color: #F1F5F9 !important;
    }
    /* Style the chat input */
    textarea[aria-label="Ask a question about the technical documentation..."] {
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
    }
    /* Adjust footer display */
    footer {
        display: none !important;
    }
    /* Fix tab style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F8F9FA !important;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        border: 1px solid #E5E7EB;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        border-bottom: 2px solid #3B82F6 !important;
    }
    /* Wider chat container */
    div[data-testid="stChatMessageContent"] {
        max-width: 100% !important;
    }
    /* Center main content */
    .main-content {
        max-width: 800px;
        margin: 0 auto;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F1F5F9 !important;
        color: #1E3A8A !important;
        border-radius: 5px;
        font-weight: 500;
    }
    .streamlit-expanderContent {
        background-color: white !important;
        color: #262730 !important;
        border: 1px solid #E5E7EB;
        border-top: none;
        border-radius: 0 0 5px 5px;
        padding: 10px;
    }
    /* Image display in the chat */
    .retrieved-image {
        margin: 10px 0;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        max-width: 100%;
    }
    .retrieved-image-container {
        padding: 10px;
        background-color: #F8F9FA;
        border-radius: 5px;
        margin: 10px 0;
    }
    .image-caption {
        font-size: 14px;
        color: #4B5563;
        margin-top: 5px;
        text-align: center;
    }
    .source-info {
        background-color: #F1F5F9;
        padding: 8px 12px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 3px solid #3B82F6;
        font-size: 14px;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    .source-chunk {
        font-family: monospace;
        background-color: #F8F9FA;
        padding: 12px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #E5E7EB;
    }
    .source-chunk pre.code-content {
        white-space: pre-wrap;
        overflow-x: auto;
        padding: 10px;
        border-radius: 4px;
        background-color: #F1F5F9;
    }
    /* JSON-specific formatting */
    .source-chunk pre.code-content span {
        font-family: 'Courier New', monospace;
    }
    /* Figure display in chunks */
    .chunk-figures {
        margin-top: 15px;
        border-top: 1px solid #E5E7EB;
        padding-top: 10px;
    }
    .chunk-figures h4 {
        margin-bottom: 10px;
        color: #1E3A8A;
        font-size: 15px;
    }
    .chunk-figure-container {
        display: inline-block;
        margin: 10px;
        vertical-align: top;
        text-align: center;
        max-width: 300px;
    }
    .chunk-figure {
        max-width: 100%;
        max-height: 200px;
        border-radius: 5px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .chunk-figure-caption {
        margin-top: 5px;
        font-size: 12px;
        color: #4B5563;
    }
    .chunk-figure-error {
        color: #DC2626;
        font-size: 14px;
        padding: 8px;
        background: #FEF2F2;
        border-radius: 4px;
        margin: 5px 0;
    }
    .chunk-title {
        font-weight: bold;
        margin-bottom: 8px;
        color: #1E3A8A;
    }
    .source-details {
        padding: 10px;
        background-color: #F8F9FA;
        border-radius: 5px;
        margin: 10px 0;
    }
    .source-details summary {
        cursor: pointer;
        padding: 8px 12px;
        background-color: #E5E7EB;
        border-radius: 5px;
        display: inline-block;
        font-weight: 500;
        color: #1E3A8A;
        margin-bottom: 10px;
    }
    .source-details summary:hover {
        background-color: #D1D5DB;
    }
    /* Sticky chat input at the bottom */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 22rem;
        right: 0;
        padding: 20px;
        background-color: white;
        border-top: 1px solid #E5E7EB;
        z-index: 100;
    }
    /* Add padding at the bottom of the chat to prevent content from being hidden behind the input box */
    .chat-content-area {
        padding-bottom: 100px;
    }
    /* Adjust for collapsed sidebar state */
    [data-testid="collapsedControl"] ~ .input-container {
        left: 4rem;
    }
    /* Media query for small screens */
    @media (max-width: 992px) {
        .input-container {
            left: 0;
        }
        [data-testid="collapsedControl"] ~ .input-container {
            left: 0;
        }
    }
    /* Give the chat input a slight shadow for depth */
    .input-container {
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
    /* Hide the sidebar collapse button to make the sidebar always visible */
    section[data-testid="stSidebar"] [data-testid="collapsedControl"] {
        display: none !important;
    }
    /* Always show the sidebar in expanded state */
    section[data-testid="stSidebar"] {
        width: 22rem !important;
        min-width: 22rem !important;
        margin-left: 0px !important;
    }
    /* Make the main content area respect the fixed sidebar */
    .main .block-container {
        padding-left: 22rem;
        max-width: 100%;
    }
    /* Fix the input container position with always-expanded sidebar */
    .input-container {
        left: 22rem !important;
    }
    /* Remove need for media query handling sidebar state since it's now fixed */
    @media (max-width: 992px) {
        section[data-testid="stSidebar"] {
            width: 18rem !important;
            min-width: 18rem !important;
        }
        .main .block-container {
            padding-left: 18rem;
        }
        .input-container {
            left: 18rem !important;
        }
    }
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 15rem !important;
            min-width: 15rem !important;
        }
        .main .block-container {
            padding-left: 15rem;
        }
        .input-container {
            left: 15rem !important;
        }
    }
    /* Figures display in response */
    .response-figures {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin: 15px 0;
        justify-content: flex-start;
    }
    .response-figure-container {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 10px;
        background-color: #F9FAFB;
        width: calc(33.333% - 10px);
        min-width: 250px;
        max-width: 350px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .response-figure {
        max-width: 100%;
        max-height: 200px;
        border-radius: 5px;
        margin-bottom: 8px;
    }
    .response-figure-caption {
        margin-top: 5px;
        font-size: 14px;
        color: #4B5563;
        text-align: center;
    }
    /* Media query for mobile */
    @media (max-width: 768px) {
        .response-figure-container {
            width: 100%;
            max-width: 100%;
        }
    }
    .chunk-figures summary {
        cursor: pointer;
        color: #1E3A8A;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .chunk-figure-reference {
        padding: 5px 10px;
        margin: 5px 0;
        background-color: #F1F5F9;
        border-radius: 4px;
        font-size: 13px;
    }
    /* Better nested list styling in tradeoffs doc */
    .document-container ol {
        margin-left: 1.5em;
        padding-left: 0.5em;
    }
    .document-container ol ol {
        margin-left: 1.5em;
        list-style-type: lower-alpha;
    }
    .document-container ol ol ol {
        margin-left: 1.5em;
        list-style-type: lower-roman;
    }
    .document-container ol ul {
        list-style-type: disc;
        margin-left: 1.2em;
    }
    .document-container pre code.json-content {
        white-space: pre-wrap;
        line-height: 1.4;
        word-break: break-word;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_approach" not in st.session_state:
    st.session_state.selected_approach = "OpenAI CLIP RAG"  # Default approach

if "current_page" not in st.session_state:
    st.session_state.current_page = "Chat Interface"

# Initialize OpenAI client once
openai_client = OpenAI()

# ----------------------------------------------------------------------
# Helper: generate answer using GPT-4o (vision-capable if figures present)
# ----------------------------------------------------------------------
def generate_answer(query: str, context_hits, figure_paths=None, language: str = "english"):
    """Generate an answer conditioned on retrieved context and optional images."""
    if figure_paths is None:
        figure_paths = []

    # Assemble textual context from hits
    context_text = ""
    for i, hit in enumerate(context_hits):
        content = hit.content if hasattr(hit, "content") else hit.get("content", "")
        meta = hit.metadata if hasattr(hit, "metadata") else hit.get("metadata", {})
        source = meta.get("document", "unknown")
        section = meta.get("section_id", "")
        context_text += f"\n--- Document {i+1} [Source: {source}, Section: {section}] ---\n{content}\n"

    language_instruction = f"Please respond in {language}." if language and language.lower() != "english" else ""

    has_figures = bool(figure_paths)

    if has_figures:
        # Build mixed content (text + images)
        user_content = [{"type": "text", "text": f"""Based on the following technical documentation and images, please answer the question:\n\nQuestion: {query}\n\nRelevant Documentation:\n{context_text}"""}]

        import base64, mimetypes, os as _os
        for path in figure_paths[:4]:  # clip to first few images for token economy
            if _os.path.exists(path):
                with open(path, "rb") as img_f:
                    mime = mimetypes.guess_type(path)[0] or "image/png"
                    b64 = base64.b64encode(img_f.read()).decode()
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        messages = [
            {"role": "system", "content": f"You are a helpful technical documentation assistant. Use the provided context and images to answer questions accurately and comprehensively. {language_instruction}"},
            {"role": "user", "content": user_content}
        ]
    else:
        messages = [
            {"role": "system", "content": f"You are a helpful technical documentation assistant. Use the provided context to answer questions accurately and comprehensively. {language_instruction}"},
            {"role": "user", "content": f"""Based on the following technical documentation, please answer the question:\n\nQuestion: {query}\n\nRelevant Documentation:\n{context_text}"""}
        ]

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=1000 if has_figures else 800,
    )

    answer_text = response.choices[0].message.content
    return answer_text, {
        "generation_tokens": getattr(response.usage, "total_tokens", 0),
        "model": "gpt-4o",
        "has_figures": has_figures,
        "num_figures": len(figure_paths)
    }

def load_markdown_file(file_path):
    """Load and parse a markdown file, returning both text and section headers."""
    with open(file_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    # Extract headers for navigation
    headers = []
    for line in md_text.split('\n'):
        if line.startswith('#'):
            level = len(re.match(r'^#+', line).group())
            title = line.strip('#').strip()
            headers.append((level, title))
    
    return md_text, headers


def convert_md_to_html(md_text):
    """Convert markdown to HTML with better code formatting."""
    # Apply standard markdown conversion
    html = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code', 'codehilite']
    )
    
    # Process with BeautifulSoup for better control
    soup = BeautifulSoup(html, 'html.parser')
    
    # Enhance code blocks, especially JSON
    for pre in soup.find_all('pre'):
        code_block = pre.find('code')
        if code_block:
            # Save original classes
            original_classes = code_block.get('class', [])
            
            # Check if this is a JSON code block by class or content patterns
            is_json = False
            if original_classes:
                is_json = 'language-json' in original_classes or 'json' in ' '.join(original_classes).lower()
            elif '{' in code_block.text and '}' in code_block.text and '"' in code_block.text and ':' in code_block.text:
                is_json = True
            
            if is_json:
                # Instead of injecting inner spans (which Streamlit may strip),
                # embed the pretty JSON as plain text and rely on CSS monospace.
                import html as _html
                text = code_block.text
                pretty = text
                new_pre = soup.new_tag('pre')
                new_code = soup.new_tag('code')
                new_code['class'] = 'json-content'
                new_code.string = pretty
                new_pre.append(new_code)
                code_block.parent.replace_with(new_pre)
    
    # Process images - base64 encode them
    for img in soup.find_all('img'):
        src = img.get('src')
        if os.path.exists(src):
            with open(src, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
                img['src'] = f"data:image/png;base64,{encoded_string}"
    
    return str(soup)


def get_rag_approach():
    """Return a RetrievalRecipe instance for the selected approach."""
    approach_name = st.session_state.selected_approach
    
    # If OpenSource RAG is somehow still selected (from previous sessions), 
    # reset to default OpenAI CLIP RAG
    if approach_name == "OpenSource RAG":
        approach_name = "OpenAI CLIP RAG"
        st.session_state.selected_approach = approach_name
        st.warning("OpenSource RAG is not available in the cloud deployment. Using OpenAI CLIP RAG instead.")

    if approach_name == "OpenAI CLIP RAG":
        return RetrievalRecipe.openai_clip()
    elif approach_name == "OpenAI Vision RAG":
        return RetrievalRecipe.openai_vision()
    elif approach_name == "Hybrid RAG":
        return RetrievalRecipe.hybrid()
    else:
        return RetrievalRecipe.openai_clip()  # Default fallback


def main():
    # Remove large RAG Comparison title from sidebar
    # st.sidebar.title("RAG Comparison")
    
    # App navigation - simpler tabs
    st.sidebar.markdown("### Navigation")
    tabs = ["Chat", "Approaches + Tradeoffs"]
    
    col1, col2 = st.sidebar.columns(2)
    
    # Chat tab
    if col1.button(
        "Chat", 
        key="nav_chat",
        type="primary" if st.session_state.current_page == "Chat Interface" else "secondary",
        use_container_width=True
    ):
        st.session_state.current_page = "Chat Interface"
        st.rerun()
    
    # Documentation tab
    if col2.button(
        "Approaches + Tradeoffs", 
        key="nav_docs",
        type="primary" if st.session_state.current_page == "Approach Tradeoffs" else "secondary",
        use_container_width=True
    ):
        st.session_state.current_page = "Approach Tradeoffs"
        st.rerun()
    
    st.sidebar.markdown("---")
    
    if st.session_state.current_page == "Chat Interface":
        display_chat_interface()
    else:
        display_tradeoffs_document()


def display_chat_interface():
    # Centered header with more space
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 32px;'>RAG Technical Documentation Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px;'>Ask questions about the VM and WiFi technical manuals</p>", unsafe_allow_html=True)
    
    # Model selection - simplified UI
    st.sidebar.markdown("### RAG Approach")
    approach_options = [
        "OpenAI CLIP RAG",
        "OpenAI Vision RAG",
        "Hybrid RAG"
        # "OpenSource RAG" - Removed for cloud deployment
    ]
    
    st.session_state.selected_approach = st.sidebar.selectbox(
        "Select RAG approach",
        approach_options,
        index=approach_options.index(st.session_state.selected_approach)
    )
    
    # Display performance metrics for selected approach
    st.sidebar.markdown("### Average Metrics (dev-set)")
    
    # Look up the latest evaluation metrics from disk
    latest_metrics = load_latest_evaluation_metrics()
    selected_metrics = latest_metrics.get(st.session_state.selected_approach, {})
    for metric, value in selected_metrics.items():
        st.sidebar.markdown(
            f"""<div class="metric-card">
                {metric}<br>
                <span class="metric-value">{value}</span>
            </div>""", 
            unsafe_allow_html=True
        )
    
    # Chat container with proper styling and centering
    chat_container = st.container()
    
    with st.container():
        st.markdown("<div class='main-content chat-content-area'>", unsafe_allow_html=True)
        
        # Display chat messages
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            # Empty space for better layout when no messages
            st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Sticky chat input at the bottom
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    
    # Chat input with proper width
    if prompt := st.chat_input("Ask a question about the technical documentation..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing query..."):
                # Select retrieval recipe
                recipe = get_rag_approach()

                try:
                    # -------------------- Retrieval --------------------
                    context_hits, retrieval_metrics = retrieve(
                        query=prompt,
                        recipe=recipe,
                        k=8
                    )

                    # ------------------ Extract slices ------------------
                    retrieved_sections = []
                    retrieved_sections_metadata = []
                    retrieved_figures = []
                    retrieved_figures_metadata = []

                    for hit in context_hits:
                        content = hit.content if hasattr(hit, "content") else hit.get("content", "")
                        metadata = hit.metadata if hasattr(hit, "metadata") else hit.get("metadata", {})
                        retrieved_sections.append(content)
                        retrieved_sections_metadata.append(metadata)
                        if "figure_path" in metadata:
                            fp = metadata["figure_path"]
                            # The merge step may leave a single string or a list of paths
                            if isinstance(fp, list):
                                for p in fp:
                                    if os.path.exists(p):
                                        retrieved_figures.append(p)
                                        retrieved_figures_metadata.append(metadata)
                            else:
                                if os.path.exists(fp):
                                    retrieved_figures.append(fp)
                                    retrieved_figures_metadata.append(metadata)

                    # Detect language for generation
                    lang = detect_language(prompt)

                    # -------------------- Generation --------------------
                    answer, generation_metrics = generate_answer(
                        query=prompt,
                        context_hits=context_hits,
                        figure_paths=retrieved_figures,
                        language=lang
                    )

                    # Assemble performance dict compatible with legacy UI
                    performance = {
                        "retrieved_sections": retrieved_sections,
                        "retrieved_sections_metadata": retrieved_sections_metadata,
                        "retrieved_figures": retrieved_figures,
                        "retrieved_figures_metadata": retrieved_figures_metadata,
                        "retrieval_tokens": retrieval_metrics.get("retrieval_tokens", 0),
                        "generation_tokens": generation_metrics.get("generation_tokens", 0)
                    }
                    
                    # Get relevant sections and figures for display
                    retrieved_sections = performance.get("retrieved_sections", [])
                    retrieved_figures = performance.get("retrieved_figures", [])
                    
                    # Determine the primary document from the sections
                    primary_document = None
                    try:
                        if "retrieved_sections_metadata" in performance and performance["retrieved_sections_metadata"]:
                            primary_document = performance["retrieved_sections_metadata"][0].get("document")
                    except (IndexError, KeyError, TypeError):
                        pass
                    
                    # Build a response with the answer and retrieved data
                    response_content = []
                    
                    # Add the answer
                    response_content.append(answer)
                    
                    # Collect all figures referenced in the retrieved sections
                    all_referenced_figures = []
                    for section in retrieved_sections:
                        figure_refs = extract_figure_references(section, performance.get("retrieved_figures_metadata", []))
                        figure_paths = find_image_paths(figure_refs, retrieved_figures)
                        all_referenced_figures.extend(figure_paths)
                    
                    # Also try to use the retrieved figures directly if available
                    if retrieved_figures and not all_referenced_figures:
                        print("Using retrieved figures directly")
                        for fig_path in retrieved_figures:
                            if os.path.exists(fig_path):
                                # Extract caption and ID from filename
                                filename = os.path.basename(fig_path)
                                caption = filename.replace('_', ' ').replace('.png', '')
                                
                                # Look for matching metadata
                                matching_metadata = None
                                if "retrieved_figures_metadata" in performance:
                                    for metadata in performance["retrieved_figures_metadata"]:
                                        if isinstance(metadata, dict) and metadata.get("filename", "") in filename:
                                            matching_metadata = metadata
                                            break
                                
                                # Create figure info
                                figure_info = {
                                    "path": fig_path,
                                    "caption": matching_metadata.get("caption", caption) if matching_metadata else caption,
                                    "figure_id": matching_metadata.get("figure_id", "") if matching_metadata else ""
                                }
                                all_referenced_figures.append(figure_info)
                    
                    # Remove duplicates from all_referenced_figures based on path
                    unique_figures = []
                    unique_paths = set()
                    for fig in all_referenced_figures:
                        if fig["path"] not in unique_paths:
                            unique_paths.add(fig["path"])
                            unique_figures.append(fig)
                    
                    print(f"Found {len(unique_figures)} unique figures to display")
                    
                    # Display all referenced figures from source chunks directly under the answer
                    if unique_figures:
                        response_content.append("\n\n**Figures:**")
                        figures_html = """<div class='response-figures'>"""
                        
                        for img in unique_figures:
                            try:
                                with open(img["path"], "rb") as img_file:
                                    encoded_img = base64.b64encode(img_file.read()).decode()
                                
                                caption = img.get("caption", "") or img.get("figure_id", "") or "Figure"
                                figures_html += (
                                    f"<div class='response-figure-container'>"
                                    f"<a href='data:image/png;base64,{encoded_img}' target='_blank'>"
                                    f"<img src='data:image/png;base64,{encoded_img}' class='response-figure' alt='{caption}'>"
                                    f"</a>"
                                    f"<div class='response-figure-caption'>{caption}</div>"
                                    f"</div>"
                                )
                            except Exception as e:
                                # If image loading fails, skip it
                                pass
                        
                        figures_html += "</div>"
                        response_content.append(figures_html)
                    
                    # --------------------------------------------------
                    # Show up to three *additional* figures that were in
                    # retrieved_figures but weren't explicitly referenced
                    # --------------------------------------------------
                    additional_paths = [p for p in retrieved_figures if p not in unique_paths][:6]
                    if additional_paths:
                        response_content.append("\n\n**Additional Figures in Context:**")
                        add_fig_html = "<div class='response-figures'>"
                        for p in additional_paths:
                            try:
                                with open(p, "rb") as fimg:
                                    enc = base64.b64encode(fimg.read()).decode()
                                # Attempt caption lookup from metadata list
                                caption = os.path.basename(p).replace('_',' ').replace('.png','')
                                for meta in retrieved_figures_metadata:
                                    if isinstance(meta, dict) and meta.get("figure_path", meta.get("image_path", "")) == p:
                                        caption = meta.get("caption", caption)
                                        break
                                add_fig_html += (
                                    f"<div class='response-figure-container'>"
                                    f"<a href='data:image/png;base64,{enc}' target='_blank'>"
                                    f"<img src='data:image/png;base64,{enc}' class='response-figure' alt='{caption}'>"
                                    f"</a>"
                                    f"<div class='response-figure-caption'>{caption}</div>"
                                    f"</div>"
                                )
                            except Exception:
                                pass
                        add_fig_html += "</div>"
                        response_content.append(add_fig_html)
                    
                    # Add source information
                    if retrieved_sections:
                        # Create source sections info
                        section_info = []
                        max_sources_to_show = 3  # show top-N chunks for quick reference
                        for i, section in enumerate(retrieved_sections[:max_sources_to_show]):
                            # Try to extract section title and page numbers from metadata
                            section_title = f"Source {i+1}"  # Ensure proper numbering starts at 1
                            page_info = ""
                            document_name = ""
                            
                            try:
                                # Check different paths for metadata in the performance dictionary
                                if "retrieved_sections_metadata" in performance:
                                    # Try to get metadata directly
                                    section_metadata = performance["retrieved_sections_metadata"][i]
                                    
                                    if "section_title" in section_metadata:
                                        section_title = section_metadata["section_title"]
                                    
                                    if "document" in section_metadata:
                                        document_name = f" - {section_metadata['document']}"
                                    
                                    if "original_page_numbers" in section_metadata:
                                        page_info = f" (Pages: {', '.join(map(str, section_metadata['original_page_numbers']))})"
                                    elif "page_range" in section_metadata:
                                        page_info = f" (Page range: {section_metadata['page_range']})"
                            except (IndexError, KeyError, TypeError):
                                # If metadata extraction fails, continue without page info
                                pass
                            
                            # Extract any figure references in this chunk
                            figure_refs = extract_figure_references(section, performance.get("retrieved_figures_metadata", []))
                            figure_paths = find_image_paths(figure_refs, retrieved_figures)
                            
                            # Create HTML for any figures found, but simpler since we show them in the main response
                            figures_html = ""
                            if figure_paths:
                                figures_html = "<div class='chunk-figures'><details><summary>Referenced Figures</summary>"
                                for img in figure_paths:
                                    cap = img.get("caption", "") or img.get("figure_id", "") or "Figure"
                                    figures_html += f"<div class='chunk-figure-reference'>{cap}</div>"
                                figures_html += "</details></div>"
                            
                            # Create a div with the chunk title, content, and figures; embed figures_html inside
                            chunk_html = f"""<div class=\"source-chunk\">\n<div class=\"chunk-title\">{section_title}{document_name}{page_info}</div>\n<pre class=\"code-content\">{format_code_content(section)}</pre>\n{figures_html}\n</div>"""
                            section_info.append(chunk_html)
                        
                        # Add source info to response with unsafe_allow_html
                        response_content.append("\n\n---\n\n**Sources Used:**")
                        response_content.append("<details class='source-details'><summary>Click to view source chunks</summary><div class='source-info'>" + "".join(section_info) + "</div></details>")
                    
                    # Filter figures to only show those from the primary document
                    relevant_figures = []
                    relevant_metadata = []
                    
                    if primary_document and retrieved_figures:
                        for i, fig_path in enumerate(retrieved_figures):
                            # Check if figure is from primary document
                            is_relevant = False
                            try:
                                if "retrieved_figures_metadata" in performance:
                                    figure_metadata = performance["retrieved_figures_metadata"][i]
                                    if figure_metadata.get("document") == primary_document:
                                        is_relevant = True
                                # If metadata doesn't have document info, check filepath
                                elif primary_document.lower() in fig_path.lower():
                                    is_relevant = True
                            except (IndexError, KeyError, TypeError):
                                # If can't determine relevance, include by default
                                pass
                            
                            if is_relevant and os.path.exists(fig_path):
                                relevant_figures.append(fig_path)
                                if "retrieved_figures_metadata" in performance:
                                    try:
                                        relevant_metadata.append(performance["retrieved_figures_metadata"][i])
                                    except (IndexError, KeyError):
                                        relevant_metadata.append({})
                    
                    # Display only relevant figures if available
                    if relevant_figures:
                        response_content.append("\n\n**Relevant Figures:**")
                        
                        # If more than one figure, create a grid layout
                        if len(relevant_figures) > 1:
                            figure_html = ["<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;'>"]
                        else:
                            figure_html = ["<div class='retrieved-image-container'>"]
                        
                        for i, fig_path in enumerate(relevant_figures):
                            # Extract figure caption if available
                            caption = f"Figure {i+1}"
                            try:
                                # Try to get metadata
                                if i < len(relevant_metadata) and relevant_metadata[i]:
                                    figure_metadata = relevant_metadata[i]
                                    if "caption" in figure_metadata:
                                        caption = figure_metadata["caption"]
                                    elif "figure_id" in figure_metadata:
                                        caption = figure_metadata["figure_id"]
                                # Also check filename for potential caption
                                elif "/" in fig_path:
                                    filename = fig_path.split("/")[-1].replace("_", " ").replace(".png", "")
                                    if filename:
                                        caption = f"Figure: {filename}"
                            except (IndexError, KeyError, TypeError):
                                # If extraction fails, keep default caption
                                pass
                            
                            # Convert image to base64
                            with open(fig_path, "rb") as img_file:
                                encoded_img = base64.b64encode(img_file.read()).decode()
                            
                            # Add image with caption
                            figure_html.append(
                                f"<div class='retrieved-image-container' style='margin-bottom: 15px;'>"
                                f"<img src='data:image/png;base64,{encoded_img}' class='retrieved-image' alt='{caption}'>"
                                f"<div class='image-caption'>{caption}</div>"
                                f"</div>"
                            )
                        
                        figure_html.append("</div>")
                        response_content.append("\n".join(figure_html))
                    
                    # Join all response components and display
                    full_response = "\n\n".join(response_content)
                    st.markdown(full_response, unsafe_allow_html=True)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def display_tradeoffs_document():
    # Simple header
    st.markdown("<h1 style='text-align: center;'>RAG Approaches Tradeoffs</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px;'>Technical analysis of different RAG approaches for documentation question-answering</p>", unsafe_allow_html=True)
    
    # Create columns for document layout
    col1, col2 = st.columns([1, 3])
    
    # Load the markdown file
    md_file_path = "RAG_Approaches_Tradeoffs.md"
    md_text, headers = load_markdown_file(md_file_path)
    
    # Filter for main sections and subsections
    main_sections = [h for h in headers if h[0] == 2]  # Level 2 headers (##)
    
    # Display sections in the sidebar
    with col1:
        st.markdown("### Document Sections")
        
        # Initialize current section if not in session state
        if "active_section" not in st.session_state:
            st.session_state.active_section = main_sections[0][1] if main_sections else ""
        
        # Display main sections
        for i, (level, title) in enumerate(main_sections):
            is_active = st.session_state.active_section == title
            button_style = "primary" if is_active else "secondary"
            
            if st.button(f"{title}", key=f"section_{i}", type=button_style):
                st.session_state.active_section = title
                st.rerun()
        
        # Get subsections for active section
        if st.session_state.active_section:
            # Find index of current section
            section_indices = [i for i, (_, title) in enumerate(headers) if title == st.session_state.active_section]
            if section_indices:
                current_idx = section_indices[0]
                # Find next main section index
                next_section_indices = [i for i, (level, _) in enumerate(headers) if level == 2 and i > current_idx]
                end_idx = next_section_indices[0] if next_section_indices else len(headers)
                
                # Get subsections
                subsections = headers[current_idx+1:end_idx]
                
                # Display subsections if any
                if subsections:
                    st.markdown(f"**{st.session_state.active_section} Subsections:**")
                    for j, (sublevel, subtitle) in enumerate(subsections):
                        if sublevel > 2:  # Only show level 3+ headers
                            indent = "  " * (sublevel - 3)
                            if st.button(f"{indent}• {subtitle}", key=f"sub_{j}"):
                                st.session_state.scroll_to = subtitle
                                st.rerun()
    
    # Convert markdown to HTML with better code block handling
    html_content = convert_md_to_html(md_text)
    
    # Display the document content
    with col2:
        st.markdown(f'<div class="document-container">{html_content}</div>', unsafe_allow_html=True)


def format_code_content(content):
    """Format code content, especially for JSON."""
    try:
        # Check if content is JSON by attempting to parse it
        if (content.strip().startswith('{') and content.strip().endswith('}')) or \
           (content.strip().startswith('[') and content.strip().endswith(']')):
            try:
                # Try to parse and pretty-print JSON
                parsed = json.loads(content)
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                # If it fails to parse as JSON, return the original content
                pass
        
        # Check for [object Object] patterns and try to fix them
        if '[object Object]' in content:
            # This is a JavaScript string representation issue
            # Try to replace with proper JSON notation
            import re
            # Try to identify patterns like: key: [object Object] or "key": [object Object]
            pattern = r'(["\']\w+["\']\s*:\s*|[\w\.]+\s*:\s*)\[object Object\]'
            content = re.sub(pattern, r'\1{ ... }', content)
        
        # Look for stringified JSON objects within the content
        if '"content":' in content or '"metadata":' in content or '"figures":' in content:
            # This might be a document chunk with JSON structure
            try:
                # Add syntax highlighting to make JSON more readable
                import re
                # Highlight keys in JSON
                content = re.sub(r'("[\w\s\-]+")(\s*:)', r'<span style="color:#0c5460">\1</span>\2', content)
                # Highlight string values in JSON
                content = re.sub(r':\s*("[\w\s\-\.,\?\/\\\(\)]+")([,}])', r': <span style="color:#155724">\1</span>\2', content)
                # Highlight numbers and booleans
                content = re.sub(r':\s*(\d+|true|false|null)([,}])', r': <span style="color:#721c24">\1</span>\2', content)
            except Exception:
                # If regex fails, return plain content
                pass
        
        return content
    except Exception:
        # If any error occurs, return the original content
        return content


def extract_figure_references(text, retrieved_figures_metadata=None):
    """Extract figure references from text and return their file paths."""
    figure_references = []
    
    # Debug info
    print("Extracting figure references from text chunk")
    
    # Try to extract figure references from JSON-formatted chunks
    try:
        # Check if this is a JSON chunk with a "figures" field
        if '"figures":' in text and (text.strip().startswith('{') or text.strip().startswith('[')):
            try:
                # Try to parse as JSON
                import json
                data = json.loads(text)
                
                # Look for figures field in the dictionary
                if isinstance(data, dict) and "figures" in data and isinstance(data["figures"], list):
                    print(f"Found {len(data['figures'])} figures in JSON")
                    for fig in data["figures"]:
                        if isinstance(fig, dict):
                            # Extract information
                            figure_info = {
                                "figure_id": fig.get("figure_id", ""),
                                "filename": fig.get("filename", ""),
                                "caption": fig.get("caption", ""),
                                "page": fig.get("page", "")
                            }
                            print(f"JSON Figure: {figure_info['filename']}")
                            figure_references.append(figure_info)
            except json.JSONDecodeError:
                # Not valid JSON, try regex
                print("Failed to parse as JSON, will try regex")
                pass
    except Exception as e:
        # If any error in JSON parsing, continue with regex approach
        print(f"Error in JSON parsing: {str(e)}")
        pass
    
    # If no figures found in JSON, try regex approach
    if not figure_references:
        print("Using regex to find figure references")
        # Use regex to find patterns like "Figure X: description" or "filename: something.png"
        import re
        
        # Look for filename patterns (typically in the format: manual_name_figure_X.png)
        filename_pattern = r'(\w+_manual_figure_[\w\-]+\.png)'
        filename_matches = re.findall(filename_pattern, text)
        
        # Capture figure id and optional caption (text after dash or colon)
        figure_pattern = re.compile(r'(?:Figura|Figure)\s+(\d+)\s*(?:[-–:]\s*([^\n]+))?', re.IGNORECASE)
        figure_matches = figure_pattern.findall(text)
        figure_id_matches = [fid for fid, _ in figure_matches]
        
        print(f"Found {len(filename_matches)} filename matches and {len(figure_id_matches)} id matches with regex")
        
        # Process filename matches
        for filename in filename_matches:
            # Check if this image exists in retrieved_figures_metadata
            matching_metadata = None
            if retrieved_figures_metadata:
                for metadata in retrieved_figures_metadata:
                    if isinstance(metadata, dict) and metadata.get("filename") == filename:
                        matching_metadata = metadata
                        break
            
            figure_info = {
                "figure_id": matching_metadata.get("figure_id", "") if matching_metadata else "",
                "filename": filename,
                "caption": matching_metadata.get("caption", "") if matching_metadata else "",
                "page": matching_metadata.get("page", "") if matching_metadata else ""
            }
            print(f"Regex Figure: {figure_info['filename']}")
            figure_references.append(figure_info)
        
        # Process figure-id matches (with optional caption)
        for (fid, cap) in figure_matches:
            if any(fr.get("figure_id") == fid and fr.get("caption") == (cap or "") for fr in figure_references):
                continue
            figure_references.append({
                "figure_id": fid,
                "filename": "",
                "caption": (cap or "").strip(),
                "page": ""
            })
        print(f"Total extracted figure references: {len(figure_references)}")
    
    return figure_references


def find_image_paths(figure_references, retrieved_figures=None):
    """Find actual image file paths for figure references."""
    image_paths = []
    
    # Print debug info
    print(f"Looking for {len(figure_references)} figure references")
    if retrieved_figures:
        print(f"We have {len(retrieved_figures)} retrieved figures")
    
    # First check if we already have the paths in retrieved_figures
    if retrieved_figures:
        for ref in figure_references:
            filename = ref.get("filename", "")
            fig_id = ref.get("figure_id", "")
            slug = "".join(re.findall(r'[a-z0-9]+', ref.get("caption", "").lower()))[:30]
            if filename:
                print(f"Looking for filename: {filename}")
            # Check direct filename match in retrieved_figures
            for fig_path in retrieved_figures:
                if (filename and filename in fig_path) or (fig_id and f"_figure_{fig_id}" in fig_path) or (slug and slug in fig_path.lower()):
                    if os.path.exists(fig_path):
                        print(f"Found match in retrieved_figures: {fig_path}")
                        image_paths.append({
                            "path": fig_path,
                            "caption": ref.get("caption", ""),
                            "figure_id": fig_id
                        })
                        break
    
    # If we didn't find all images, search for image files in common directories
    if len(image_paths) < len(figure_references):
        print(f"Still missing {len(figure_references) - len(image_paths)} images. Searching directories...")
        # Common directories to search for images
        image_dirs = [
            "data/VM_manual/figures", 
            "data/wifi_manual/figures",
            "data/figures", 
            "data/images",
            "vector_db",
            "src/data/figures"
        ]
        
        for ref in figure_references:
            # Skip if we already found this image
            if any(ref.get("filename", "") in img["path"] for img in image_paths):
                continue
                
            filename = ref.get("filename", "")
            fig_id = ref.get("figure_id", "")
            slug = "".join(re.findall(r'[a-z0-9]+', ref.get("caption", "").lower()))[:30]
            if not filename and not fig_id:
                continue
                
            print(f"Searching directories for: {filename}")
            # Search for the file
            for dir_path in image_dirs:
                if not os.path.exists(dir_path):
                    print(f"Directory {dir_path} does not exist")
                    continue
                    
                print(f"Checking directory: {dir_path}")
                # Look for the exact filename
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        match_file = False
                        if filename:
                            match_file = (filename == file or filename in file)
                        if not match_file and fig_id:
                            match_file = f"_figure_{fig_id}" in file
                        if not match_file and slug:
                            match_file = slug in file.lower()
                        if match_file:
                            img_path = os.path.join(root, file)
                            if os.path.exists(img_path):
                                print(f"Found image: {img_path}")
                                image_paths.append({
                                    "path": img_path,
                                    "caption": ref.get("caption", ""),
                                    "figure_id": fig_id
                                })
                                break
    
    print(f"Found {len(image_paths)} image paths out of {len(figure_references)} references")
    return image_paths


# ----------------------------------------------------------------------
# Helper: load latest evaluation metrics for sidebar
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_latest_evaluation_metrics(results_dir: str = "evaluation_results") -> Dict[str, Dict[str, str]]:
    """Parse the newest combined_results_*.json file and return sidebar-ready metrics.

    The returned dictionary is keyed by RAG approach and contains pretty-formatted
    strings for the small sidebar metric cards.
    """
    if not os.path.isdir(results_dir):
        return {}

    files = [f for f in os.listdir(results_dir)
             if f.startswith("combined_results_") and f.endswith(".json")]
    if not files:
        return {}

    # Newest file by modification time
    latest_path = max(
        (os.path.join(results_dir, f) for f in files),
        key=os.path.getmtime,
    )

    try:
        with open(latest_path, "r", encoding="utf-8") as fp:
            data: Dict[str, Any] = json.load(fp)
    except Exception:
        return {}

    approaches: Dict[str, Any] = data.get("approaches", {})
    if not approaches:
        return {}

    # --- Simple cost estimation ------------------------------------------------
    # We approximate cost using billable token counts and rough per-1K-token prices.
    # Prices are *approximate* and meant only for high-level comparison in the UI.
    PRICE_PER_1K = {
        "gpt-4o": 0.005,          # USD per 1K tokens (mixed in/out average)
        "openai_clip": 0.00013,   # text-embedding-3-small
        "openai_vision": 0.005,   # treat vision embedding similarly
        "hybrid": 0.00013,        # hybrid uses small text embedding
    }

    sidebar_metrics: Dict[str, Dict[str, str]] = {}
    for approach_name, details in approaches.items():
        overall = details.get("overall", {})
        token_usage = details.get("token_usage", {})
        model_token_usage = details.get("model_token_usage", {})

        total_questions = overall.get("total_questions", 1) or 1

        # Compute latency & overall score
        avg_latency = overall.get("avg_response_time", 0.0)
        avg_score = overall.get("avg_overall_score", 0.0)

        # Estimate cost ---------------------------------------------------------
        total_cost = 0.0
        for model_name, m_info in model_token_usage.items():
            if not m_info.get("billable", False):
                continue
            tokens = m_info.get("tokens", 0)
            price = PRICE_PER_1K.get(model_name, PRICE_PER_1K.get(model_name.split("_")[0], 0.0))
            total_cost += (tokens / 1000.0) * price

        cost_per_query = total_cost / total_questions if total_questions else 0.0

        sidebar_metrics[approach_name] = {
            "Overall (/10)": f"{avg_score:.2f}",
            "Latency (s)": f"{avg_latency:.1f}",
            "Cost / Query ($)": f"${cost_per_query:.3f}",
        }

    return sidebar_metrics


if __name__ == "__main__":
    main() 