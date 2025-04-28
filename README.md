# Semantic Spotter Project Submission

## 1. Background

This project demonstrate "Build a RAG System" in insurance domain
using  [LangChain](https://python.langchain.com/docs/introduction/).

## 2. Problem Statement

The goal of the project is to build a robust generative search system capable of effectively and accurately
answering questions from a bunch of policy documents.

## 3. Document

1. The policy documents can be found [here](https://github.com/DeepenKumarSahoo/Semantic-Spotter-Project/tree/main/Policy%2BDocuments)

## 4. Approach

LangChain is a framework that simplifies the development of LLM applications LangChain offers a suite of tools,
components, and interfaces that simplify the construction of LLM-centric applications. LangChain enables developers to
build applications that can generate creative and contextually relevant content LangChain provides an LLM class designed
for interfacing with various language model providers, such as OpenAI, Cohere, and Hugging Face.

LangChain's versatility and flexibility enable seamless integration with various data sources, making it a comprehensive
solution for creating advanced language model-powered applications.

LangChain's open-source framework is available to build applications in Python or JavaScript/TypeScript. Its core design
principle is composition and modularity. By combining modules and components, one can quickly build complex LLM-based
applications. LangChain is an open-source framework that makes it easier to build powerful and personalizeable
applications with LLMs relevant to user’s interests and needs. It connects to external systems to access information
required to solve complex problems. It provides abstractions for most of the functionalities needed for building an LLM
application and also has integrations that can readily read and write data, reducing the development speed of the
application. LangChains's framework allows for building applications that are agnostic to the underlying language model.
With its ever expanding support for various LLMs, LangChain offers a unique value proposition to build applications and
iterate continuously.

LangChain framework consists of the following:

- **Components**: LangChain provides modular abstractions for the components necessary to work with language models.
  LangChain also has collections of implementations for all these abstractions. The components are designed to be easy
  to use, regardless of whether you are using the rest of the LangChain framework or not.
- **Use-Case Specific Chains**: Chains can be thought of as assembling these components in particular ways in order to
  best accomplish a particular use case. These are intended to be a higher level interface through which people can
  easily get started with a specific use case. These chains are also designed to be customizable.

The LangChain framework revolves around the following building blocks:

* Model I/O: Interface with language models (LLMs & Chat Models, Prompts, Output Parsers)
* Retrieval: Interface with application-specific data (Document loaders, Document transformers, Text embedding models,
  Vector stores, Retrievers)
* Chains: Construct sequences/chains of LLM calls
* Memory: Persist application state between runs of a chain
* Agents: Let chains choose which tools to use given high-level directives
* Callbacks: Log and stream intermediate steps of any chain

## 5. System Layers

- **Reading & Processing PDF Files:** We will be
  using
  LangChain [PyPDFDirectoryLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFDirectoryLoader.html)
  to read and process the PDF files from specified directory.

- **Document Chunking:**  We will be
  using LangChain [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/).
  This text
  splitter is the recommended one for generic text. It is parameterized by a list of
  characters. It tries to split on them in order until the chunks are small enough. The default list
  is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words)
  together as long as possible, as those would generically seem to be the strongest semantically related pieces of
  text..

- **Generating Embeddings:**  We will be
  using [OpenAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/openai/) from LangChain
  package. The Embeddings class
  is a class designed for interfacing with text embedding models.
  LangChain provides support for most of the embedding model providers (OpenAI, Cohere) including sentence transformers
  library from Hugging Face. Embeddings create a vector representation of a piece of text and supports all the
  operations such as similarity search, text comparison, sentiment analysis etc. The base Embeddings class in LangChain
  provides two methods: one for embedding documents and one for embedding a query.

- **Store Embeddings In ChromaDB:** In this section we will store embedding in ChromaDB. This embedding is backed by
  LangChain [CacheBackedEmbeddings](https://python.langchain.com/api_reference/langchain/embeddings/langchain.embeddings.cache.CacheBackedEmbeddings.html)

- **Retrievers:** Retrievers provide Easy way to combine documents with language models.A retriever is an interface that
  returns documents given an unstructured query. It is more general than a vector store. A retriever does not need to be
  able to store documents, only to return (or retrieve) them. Retriever stores data for it to be queried by a language
  model. It provides an interface that will return documents based on an unstructured query. Vector stores can be used
  as the backbone of a retriever, but there are other types of retrievers as well. There are many different types of
  retrievers, the most widely supported is
  the [VectoreStoreRetriever](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html).

- **Re-Ranking with a Cross Encoder:** Re-ranking the results obtained from the semantic search will sometime
  significantly improve the relevance of the retrieved results. This is often done by passing the query paired with each
  of the retrieved responses into a cross-encoder to score the relevance of the response w.r.t. the query. The above
  retriever is associated
  with [HuggingFaceCrossEncoder](https://python.langchain.com/api_reference/community/cross_encoders/langchain_community.cross_encoders.huggingface.HuggingFaceCrossEncoder.html)
  with model BAAI/bge-reranker-base

- **Chains:** LangChain provides Chains that can be used to combine multiple components together to create a single,
  coherent application. For example, we can create a chain that takes user input, formats it with a PromptTemplate, and
  then passes the formatted response to an LLM. We can build more complex chains by combining multiple chains together,
  or by combining chains with other components. We are using pulling prompt <b>rlm/rag-promp</b> from langchain hub to
  use in RAG chain.

## 6. System Architecture

![](https://github.com/DeepenKumarSahoo/Semantic-Spotter-Project/blob/main/System%20Architecture%20Images/arch1.png)
![](https://github.com/DeepenKumarSahoo/Semantic-Spotter-Project/blob/main/System%20Architecture%20Images/arch2.png)

## 7. Prerequisites

- Python 3.7+
- langchain 0.3.13
- Please ensure that you add your OpenAI API key to the empty text file named "OpenAI_API_Key.txt" in order to access
  the
  OpenAI API.

## 8. Running

- Clone the github repository
  ```shell
  $ git clone https://github.com/DeepenKumarSahoo/Semantic-Spotter-Project.git
  ```
- Open
  the [notebook](https://github.com/DeepenKumarSahoo/Semantic-Spotter-Project/blob/main/Semantic_Spotter_DeepenKumarSahoo.ipynb)
  in jupyter and run all cells.