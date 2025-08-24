import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db_books = Chroma.from_documents(
    documents,
    embedding=embeddings
)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    if not query.strip():
        return []
    
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

def get_random_suggestion():
    suggestions = [
        "A story about time travel and second chances",
        "An adventure through mystical lands with talking animals",
        "A mystery set in a Victorian mansion during a storm",
        "A romance between two people from different worlds",
        "A tale of friendship that overcomes all obstacles",
        "A journey of self-discovery in a post-apocalyptic world",
        "A magical school where students learn ancient arts",
        "A detective story with supernatural elements",
        "A coming-of-age story in a small coastal town",
        "An epic fantasy with dragons and lost kingdoms"
    ]
    return np.random.choice(suggestions)

def clear_all():
    """Clear all inputs and results"""
    return "", "All", "All", []

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Simple clean CSS
custom_css = """
.gradio-container {
    background: #1a1a2e;
    color: #ffffff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-header {
    text-align: center;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.input-section {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.search-button {
    background: #667eea !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.8rem 2rem !important;
    font-weight: bold !important;
    color: white !important;
}

.search-button:hover {
    background: #5a67d8 !important;
}

.gallery-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.gr-textbox, .gr-dropdown {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 5px !important;
    color: white !important;
}

.gr-textbox:focus, .gr-dropdown:focus {
    border-color: #667eea !important;
}

label {
    color: #e0e0e0 !important;
    font-weight: 600 !important;
}

h1 {
    color: #ffffff !important;
    font-size: 2.5rem !important;
    font-weight: bold !important;
    text-align: center !important;
    margin-bottom: 1rem !important;
}

h2 {
    color: #f0f0f0 !important;
    text-align: center !important;
    font-size: 1.8rem !important;
    margin-bottom: 1.5rem !important;
}

.suggestion-button {
    background: #48bb78 !important;
    border: none !important;
    border-radius: 5px !important;
    color: white !important;
    font-weight: bold !important;
    padding: 0.8rem 1.5rem !important;
}

.suggestion-button:hover {
    background: #38a169 !important;
}

.clear-button {
    background: #f56565 !important;
    border: none !important;
    border-radius: 5px !important;
    color: white !important;
    font-weight: bold !important;
    padding: 0.8rem 1.5rem !important;
}

.clear-button:hover {
    background: #e53e3e !important;
}

.gr-gallery img {
    border-radius: 5px !important;
}

.gr-textbox input::placeholder {
    color: rgba(255, 255, 255, 0.6) !important;
}
"""

# Create the interface
with gr.Blocks(css=custom_css, title="Book Recommender", theme=gr.themes.Base()) as dashboard:
    # Header section
    with gr.Column(elem_classes="main-header"):
        gr.Markdown("# Book Recommender")
        gr.Markdown("Discover your next great read using AI-powered recommendations")
    
    # Input section
    with gr.Column(elem_classes="input-section"):
        gr.Markdown("### Search for Books")
        
        with gr.Row():
            with gr.Column(scale=2):
                user_query = gr.Textbox(
                    label="Describe what you're looking for:",
                    placeholder="e.g., A story about forgiveness and redemption",
                    lines=2,
                    max_lines=3
                )
            with gr.Column(scale=1):
                suggestion_btn = gr.Button("Random Suggestion", elem_classes="suggestion-button", size="sm")
            with gr.Column(scale=1):
                clear_btn = gr.Button("Clear All", elem_classes="clear-button", size="sm")
        
        with gr.Row():
            with gr.Column():
                category_dropdown = gr.Dropdown(
                    choices=categories, 
                    label="Category:", 
                    value="All"
                )
            with gr.Column():
                tone_dropdown = gr.Dropdown(
                    choices=tones, 
                    label="Emotional tone:", 
                    value="All"
                )
        
        with gr.Row():
            submit_button = gr.Button(
                "Find Books", 
                elem_classes="search-button",
                size="lg"
            )
    
    # Results section
    with gr.Column(elem_classes="gallery-container"):
        gr.Markdown("## Recommended Books")
        
        output = gr.Gallery(
            label="Book Recommendations", 
            columns=4, 
            rows=4,
            height="auto",
            object_fit="cover",
            show_label=False,
            allow_preview=True
        )
    
    # Event handlers
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )
    
    suggestion_btn.click(
        fn=get_random_suggestion,
        outputs=user_query
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[user_query, category_dropdown, tone_dropdown, output]
    )
    
    # Auto-submit on Enter
    user_query.submit(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        favicon_path=None
    )