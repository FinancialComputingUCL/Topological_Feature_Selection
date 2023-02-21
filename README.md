# Topological Feature Selection: A Graph-Based Filter Feature Selection Approach

This repository contains the code to reproduce experiments and results discussed in the paper *Topological Feature Selection: A Graph-Based Filter Feature Selection Approach*. 

In this work, we present a novel unsupervised, graph-based filter feature selection technique which exploits the power of topologically constrained network representations. We model dependency structures among features using a family of chordal graphs (the Triangulated Maximally Filtered Graph) and we maximise the likelihood of features' relevance by studying their relative position inside the network. Such an approach presents three aspects that are particularly satisfactory compared to its alternatives: (i) it is highly tunable and easily adaptable to the nature of input data; (ii) it is fully explainable, maintaining, at the same time, a remarkable level of simplicity; (iii) it is computationally cheaper compared to its alternatives. We test our algorithm on 16 benchmark datasets from different applicative domains showing that it outperforms or matches the current state-of-the-art under heterogeneous evaluation conditions.

# Getting Started

1. Unpack the content of the two folders ```./data/entire_datasets.zip``` and ```./data/splitted_datasets.zip```. **(MANDATORY STEP)**
2. Run ```python3 main.py --stage 'SM_COMPUTATION' --dataset '<dataset_name>' --cc_type '<similarity_measure>'``` for each dataset and similarity measure you are interested in. Make sure all the similarity matrices are in the main directory (not in sub-directories).
3. Run ```python3 main.py --stage 'TFS' --dataset '<dataset_name>' --classification_algo '<classification_algorithm>'``` to perform **[Topological Feature Selection](https://arxiv.org/abs/2302.09543) + Training Stage** for each specific dataset and classification algorithm you are interested in.
4. Run ```python3 main.py --stage 'IFS' --dataset '<dataset_name>' --classification_algo '<classification_algorithm>'``` to perform **[Infinite Feature Selection](https://ieeexplore.ieee.org/iel7/34/4359286/09119168.pdf?casa_token=-I8btXZw0_8AAAAA:VU5GJmZ2V1Zty08uMdx2vi8aixWudPenTdxBcHKEQK2pmHBUpgXS3HjR9wEQJr5ZegzMTVKd) + Training Stage** for each specific dataset and classification algorithm you are interested in.
5. Run ```python3 main.py --stage 'TFS_TEST' --dataset '<dataset_name>' --classification_algo '<classification_algorithm>'``` to perform **[Topological Feature Selection](https://arxiv.org/abs/2302.09543) + Test Stage** for each specific dataset and classification algorithm you are interested in.
6. Run ```python3 main.py --stage 'IFS_TEST' --dataset '<dataset_name>' --classification_algo '<classification_algorithm>'``` to perform **[Infinite Feature Selection](https://ieeexplore.ieee.org/iel7/34/4359286/09119168.pdf?casa_token=-I8btXZw0_8AAAAA:VU5GJmZ2V1Zty08uMdx2vi8aixWudPenTdxBcHKEQK2pmHBUpgXS3HjR9wEQJr5ZegzMTVKd) + Test Stage** for each specific dataset and classification algorithm you are interested in.
7. Run ```python3 main.py --stage 'STATISTICAL_TEST' --dataset '<dataset_name>'``` to perform **Statistical Test Stage** for each specific dataset you are interested in.

Once mandatory steps have been executed, the user can try the entire pipeline on the *lung_small* dataset, for which the similarity matrices are already provided.

# Our paper

Read the paper *[Topological Feature Selection: A Graph-Based Filter Feature Selection Approach](https://arxiv.org/abs/2302.09543)* for more information about the setup (or contact us). If you use our method, please cite us using:

```
@misc{https://doi.org/10.48550/arxiv.2302.09543,
  doi = {10.48550/ARXIV.2302.09543},
  url = {https://arxiv.org/abs/2302.09543},
  author = {Briola, Antonio and Aste, Tomaso},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Topological Feature Selection: A Graph-Based Filter Feature Selection Approach},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}

```

# License

Copyright 2023 Antonio Briola, Tomaso Aste.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

```http://www.apache.org/licenses/LICENSE-2.0```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.