# Softwareproject Bioinformatics -  Sequence Embedding for Shallow Learners
Researchgroup Bioinformatics and Computational Biology @ University of Vienna

[Softwareproject for Bioinformatics](https://ufind.univie.ac.at/de/course.html?lv=053531&semester=2023W 
) by Fabio Pfähler and Klaus Hartmann-Baruffi

Supervisor: [Prof.Dr.Thomas Rattei](mailto:thomas.rattei@univie.ac.at), [Dr.Alexander Pfundner](alexander.pfundner@univie.ac.at)

Ressources used:

- [BioEmbeddings](https://github.com/sacdallago/bio_embeddings) 
- [ProtTrans](https://github.com/agemagician/ProtTrans)
- [SeqVec](https://github.com/Rostlab/SeqVec)
- [VOGDB - Virus Orthologous Groups](https://vogdb.org/download) : Database of the University of Vienna, Dept. of Microbiology and Ecosystems,
  - vog.members.tsv.gz
  - vog.faa.tar.gz

- used a small subset (VOG00024) as a PoC (Proof of Concept) for applying Machine Learning techniques on the dataset for VOG-classification 

Project aims:
- Get a small subset of VOGs and protein sequences
- compute embedding for sequence
- train a scikit-learn classifier
- use a workflow management tool (NextFlow, snakemake) for creation of a ML-pipeline
- use of a version control system, e.g. GIT :-)

The project includes:
- General purpose python embedders based on open models trained on biological sequence representations (SeqVec, ProtTrans, BioEmbeddings,...)
- A pipeline which:
  - embeds sequences into vector-representations that can be used to train a ML-model
  - dimensionality-reduction for representation and visualisation using t-SNE (optional UMAP)
  - visualisation for 3D interactive plots
 
## Installation

- creation of a python environment using venv or Conda

## Dependencies

we have recognized a major dependency of the used frameworks (ProtTrans, SeqVec and bio_embeddings) on the python version used, and the required libraries like torch, allennlp, h5py and so on. 

also, it is strongly recommended to run the program on a  [CUDA](https://developer.nvidia.com/cuda-zone) capable environment.

