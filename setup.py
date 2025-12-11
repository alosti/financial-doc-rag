from setuptools import setup, find_packages

setup(
    name="financial-doc-rag",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pypdf>=3.17.0",
        "pdfplumber>=0.10.0",
        "langchain-text-splitters>=0.2.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.9",
    author="Alessandro Osti",
    description="RAG system for financial document Q&A",
    keywords="rag nlp financial-analysis",
)