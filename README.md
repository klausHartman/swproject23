# Softwareproject Bioinformatics -  Sequence Embedding for Shallow Learners
Researchgroup Bioinformatics and Computational Biology @ University of Vienna

[Softwareproject for Bioinformatics](https://ufind.univie.ac.at/de/course.html?lv=053531&semester=2023W 
) by Fabio Pfaehler and Klaus Hartmann-Baruffi

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

- installation of a jre (java runtime environment), at least version 11 for nextflow:
-   `sudo apt install default-jre`

- creation of a python environment using venv or Conda:
-   `conda create --name myenv python=3.8`
-   
     `Conda activate myenv`

- installation of nextflow: (https://github.com/nextflow-io/nextflow)
-   `pip install nextflow` (installed nextflow-23.10.1)


### Installation notes

This program was developed for Linux machines with GPU capabilities and [CUDA](https://developer.nvidia.com/cuda-zone) installed. If your setup diverges from this, you may encounter some inconsistencies (e.g. speed is significantly affected by the absence of a GPU and CUDA). 

For Windows users, we strongly recommend the use of [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

## Dependencies

we have recognized a major dependency of the used frameworks (ProtTrans, SeqVec and bio_embeddings) on the python version used, and the required libraries like torch, allennlp, h5py and so on. 

It is strongly recommended to run the program on a  [CUDA](https://developer.nvidia.com/cuda-zone) capable environment.

The benchmarks between the 3 different frameworks for embedding are mentioned in the paper [Survey of Protein Sequence Embedding Models](mdpi.com/1422-0067/24/4/3775)

## What model is a good one?

 We were using the `bio_embeddings` library with following embedders:
 - ProtTransBertBFDEmbedder()
 - SeqVecEmbedder()
 
 as a 'PoC' (proof of concept) and for runtime reasons we used it only on the `VOG00024.faa` sequence of the VOGDB [FASTA-Files ](https://www.ncbi.nlm.nih.gov/genbank/fastaformat/)

 ## Dimensionality reduction

 We used two different algorithms for dimensionality reduction with the aim, to visualise the embeddings:
 - tSNE

For a detailled view on the steps of the process, you can take a look at our [`jupyter notebook file of the project`](https://github.com/klausHartman/swproject23/blob/main/SPNotebook.ipynb)

## Contributors

- Alexander Pfunder
- Fabio Pfaehler
- Klaus Hartmann-Baruffi
