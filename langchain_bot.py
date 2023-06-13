from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import requests
# using vector space search engine
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
# split sources to small chunks
from langchain.text_splitter import CharacterTextSplitter
# connect to github repo
import pathlib
import subprocess
import tempfile


def get_wiki_data(title, first_paragraph_only):
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
    if first_paragraph_only:
        url += "&exintro=1"
    data = requests.get(url).json()
    return Document(
        page_content=list(data["query"]["pages"].values())[0]["extract"],
        metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
    )

def get_github_docs(repo_owner, repo_name):
    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f"git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git .",
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)
        markdown_files = list(repo_path.glob("*/*.md")) + list(
            repo_path.glob("*/*.mdx")
        )
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url})

# Sources to bot answer question from Wiki
# sources = [
#     get_wiki_data("Unix", False),
#     get_wiki_data("Microsoft_Windows", False),
#     get_wiki_data("Linux", False),
#     get_wiki_data("Seinfeld", False),
#     get_wiki_data("Matchbox_Twenty", False),
#     get_wiki_data("Roman_Empire", False),
#     get_wiki_data("London", False),
#     get_wiki_data("Python_(programming_language)", False),
#     get_wiki_data("Monty_Python", False),
# ]

# Source to answer from github repo
sources = get_github_docs("dagster-io", "dagster")


# Split sources to small chunks
source_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
for source in sources:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

# Create a chain to work with GPT3 and map_reduce chain
# chain = load_qa_with_sources_chain(OpenAI(temperature=0),
#                                    chain_type="map_reduce")
# def print_answer(question):
#     print(
#         chain(
#             {
#                 "input_documents": sources,
#                 "question": question,
#             },
#             return_only_outputs=True,
#         )["output_text"]
#     )

# Use vector space search engine
# create a Faiss search index for all of our sources
# search_index = FAISS.from_documents(sources, OpenAIEmbeddings())

chain = load_qa_with_sources_chain(OpenAI(temperature=0))

def print_answer(question):
    print(
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )

