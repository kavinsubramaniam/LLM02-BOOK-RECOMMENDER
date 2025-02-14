import pandas as pd
import numpy as np
import gradio as gr
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from transformers import pipeline
category_prediction = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device="cuda")


# Path where your ChromaDB is stored
persist_directory = "./chroma_db"

# Load the existing ChromaDB
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model="deepseek-r1"), collection_name="book_recommendation")


books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


def retrieve_semantic_recommendations(
        query: str,
        tone: str = None,
        initial_top_k: int = 50,
        metadata_filter: str = None,
) -> pd.DataFrame:

    recs = vectorstore.similarity_search(query, k=initial_top_k, filter=metadata_filter)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

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


def get_prediction(sequence, categories):
    prediction = category_prediction(sequence, categories)
    max_index = np.argmax(prediction["scores"])
    max_label = prediction["labels"][max_index]
    return max_label


def recommend_books(
        query: str,
        tone: str
):

    predicted_cat = get_prediction(query, ["Fiction", "Nonfiction", "Children's Fiction", "Children's Nonfiction"])
    recommendations = retrieve_semantic_recommendations(query, tone, predicted_cat)
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

        caption = f"{row['title_and_subtitle']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()