# cl_ml_tools: Command-Line Machine Learning Toolkit

`cl_ml_tools` is a Python package designed to provide a collection of robust, reusable, and command-line friendly tools for common machine learning tasks. The goal is to offer modular components that can be easily integrated into larger applications or used as standalone utilities. These tools are specifically developed and tested for deployment on edge devices and low-resource systems, ensuring efficient performance where computational power is limited.

## Features

*   **Modular Design:** Each tool is self-contained and designed for a specific purpose.
*   **Extensible:** Easily integrate your own components, such as custom inference engines or data stores.
*   **Command-Line Friendly:** Built with CLI operations in mind for easy scripting and automation.

---

## Available Tools

### 1. VectorCore

`VectorCore` is the first tool in this collection. It provides a generic and powerful framework for vector-based similarity search. It is designed to be backend-agnostic, allowing you to plug in different machine learning models for inference and various vector databases for storage.

#### Key Features of VectorCore:

*   **Pluggable Inference Engine:** Bring your own model to generate embeddings or feature vectors.
*   **Pluggable Vector Store:** Use different vector databases (like Qdrant, Milvus, etc.) for storing and searching vectors.
*   **Generic:** Not limited to images. It can handle any data type as long as you provide the appropriate preprocessing and inference logic.
*   **Batch Processing:** Efficiently process large amounts of data.

---

## Installation

To install the package and its dependencies, clone the repository and run the following command from the root directory:

```bash
pip install .
```

## Getting Started with VectorCore

Here is a high-level example of how to use `VectorCore`. For a detailed, runnable example, please see the `example/` directory.

```python
from cl_ml_tools import VectorCore
from your_project.inference import MyInferenceEngine
from your_project.store import MyVectorStore

# 1. Initialize your custom components
inference_engine = MyInferenceEngine(model_path="path/to/your/model.hef")
vector_store = MyVectorStore(collection_name="my_collection")

# 2. Initialize VectorCore with your components
vector_core = VectorCore(
    inference_engine=inference_engine,
    store_interface=vector_store,
)

# 3. Add items to the index
files_to_add = {
    1: "/path/to/image1.jpg",
    2: "/path/to/image2.png"
}
payloads = {
    1: {"filename": "image1.jpg"},
    2: {"filename": "image2.png"}
}
vector_core.add_all(files=files_to_add, payload=payloads)

# 4. Perform a similarity search
search_results = vector_core.search(data="/path/to/query_image.jpg", limit=5)

print(search_results)
```

## Future Development

We plan to expand `cl_ml_tools` with more utilities to address other common machine learning tasks. Stay tuned for updates!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
