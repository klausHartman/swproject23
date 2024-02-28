# %% [markdown]
# # Embedding via Bio-Embeddings Library
# *Implementation by Klaus Baruffi, Fabio Pfaehler* <br>
# *Library by Dallago et al.* ([Bio-Embeddings Repository](https://github.com/sacdallago/bio_embeddings/tree/develop))
# 
# Useful Resources:
# - [Comparison of different embedders](https://www.mdpi.com/1422-0067/24/4/3775)
# - [SeqVec Repository](https://github.com/Rostlab/SeqVec/blob/master/seqvec/seqvec.py)
# - [ProtTrans Repository](https://github.com/sacdallago/bio_embeddings/tree/develop)

# %% [markdown]
# ## 1) Setting up a conda environment and dependencies
# 1) install anaconda
# 2) deactivate the base env 
#     - $conda deactivate
# 3) create and activate python 3.8 conda env
#     - $conda create --name myenv python=3.8
#     - $conda activate myenv
# 5) $pip install bio-embeddings[all]
# 6) Install other dependencies (which for what: see 'Error encounter' notifications below the particular notebook cells)
#     - $pip install allennlp==0.9.0
#     - $conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge 
#     - $pip install --upgrade nbformat (eventually restart kernel afterwards)
#     - Eventually you need to restart the jupyter kernel

# %% [markdown]
# ## 2) Choose input fasta file

# %%
# Test file for download (tiny_sampled.fasta)
# !wget http://data.bioembeddings.com/public/embeddings/notebooks/custom_data/tiny_sampled.fasta --output-document fasta_files/tiny_sampled.fasta


# %% [markdown]
# ## 3) Generate Embeddings
# also see [embed_fasta_sequences.ipynb](https://github.com/sacdallago/bio_embeddings/tree/develop/notebooks)

# %%
def load_libraries():
    
    # Load libraries
    import numpy as np
    from Bio import SeqIO
    from bio_embeddings.embed.seqvec_embedder import SeqVecEmbedder
    from bio_embeddings.embed import ProtTransBertBFDEmbedder

    # %%
    # Extract sequences from fasta file and store them as a list
    sequences = []
    for record in SeqIO.parse(filepath, "fasta"):
        sequences.append(record)

    # Sanity-check (First 3 and last 3 sequences)
    print(f"Member-ID     Identifier\t\tLength\t    Sequence\n")
    for i,s in enumerate(sequences[:3]): # s:SeqIO-object
        print(f"Protein {i+1:<6}{(s.id):<28}{len(s.seq):<10}{s.seq}") # :<6 for proper output alignment
    print(". . .")
    for i,s in enumerate(sequences[-3:], start=len(sequences)-2):
        print(f"Protein {i+1:<6}{(s.id):<28}{len(s.seq):<10}{s.seq}")

# %%
# Choose Embedder
def choose_embedder():
    embedder = SeqVecEmbedder()
    # embedder = ProtTransBertBFDEmbedder()

    # %% [markdown]
    # Error encounters:
    # - NVIDIA GeForce RTX 3050 Laptop GPU with CUDA capability sm_86 is not compatible with the current PyTorch installation.
    #     1) $pip install allennlp==0.9.0
    #     2) $conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge 
    #     3) Eventually restart jupyter kernel
    #     
    # *pytorch version GPU specific, search for proper version ([https://pytorch.org/](https://pytorch.org/))*
    # 

    # %%
    # Compute Amino Acid Level Embedding
    aa_embd = embedder.embed_many([str(s.seq) for s in sequences])
    # `embed_many` returns a generator. We want to keep both RAW (amino acid) embeddings and reduced (protein) embeddings in memory.
    # To do so, we simply turn the generator into a list (this will start embedding the sequences!).
    # Needs certain amount of GPU RAM, if not sufficient CPU is used (slower).
    aa_embd = list(aa_embd)

# %%
# Helper function: returns the number of dimensions of an array
def dimension_number(array):
    dim_num = 0
    sublist = array
    while isinstance(sublist, (np.ndarray, list)):
        dim_num += 1
        sublist = sublist[0]
    return dim_num

def print_shape():
    # Print Shape of Amino Acid Level Embedding (shape/no. of dimension dependent on embedder)
    if dimension_number(aa_embd) == 4: # SeqVec
        print(f"Amino acid level embeddings shape:")
        print(f"( {len(aa_embd)} | {len(aa_embd[0])} | variable | {len(aa_embd[0][0][0])} )")
        print("( no. of sequences | NN layers | sequence length | embedding dimensions)")
    elif dimension_number(aa_embd) == 3: # ProtTransBERTEmbedder
        print(f"Amino acid level embeddings object shape:")
        print(f"( {len(aa_embd)} | variable | {len(aa_embd[0][0])} )")
        print("( no. of sequences | sequence length | embedding dimensions)")


def compute_protein_embedding():
    # %%
    # Compute Protein Level Embedding
    protein_embd = [embedder.reduce_per_protein(e) for e in aa_embd]
    # mean of amino acid level vectors

    # Print Shape of Protein Level Embedding
    print("Protein level embeddings shape:")
    print(np.shape(protein_embd))
    print("( no. of sequences | embedding dimensions )")


def print_embedding_shapes():
    # %%
    # Print Summary of Embedding Shapes:  Sequence | AA Level Embedding | Protein Level Embedding
    print("Member ID\tAA Level Embedding\tProtein Level Embedding")
    for i, (per_amino_acid, per_protein) in enumerate(zip(aa_embd[:3], protein_embd[:3])):
        print(f"Protein {i+1}\t{per_amino_acid.shape}\t\t{per_protein.shape}")
    print(". . .")
    for i, (per_amino_acid, per_protein) in enumerate(zip(aa_embd[-3:], protein_embd[-3:]), start=len(aa_embd)-2):
        print(f"Protein {i+1}\t{per_amino_acid.shape}\t\t{per_protein.shape}")

# %% [markdown]
# ## 4) Projection/Dimensionality Reduction
# also see [project_visualize_pipeline_embeddings.ipynb](https://github.com/sacdallago/bio_embeddings/tree/develop/notebooks)

def bio_embeddings():
    # %%
    import numpy as np
    from bio_embeddings.project import tsne_reduce

    # Configure tsne options
    options = {
        'perplexity': 3, # Low perplexity values (e.g., 3) cause t-SNE to focus more on preserving the local structure of the data (high, e.g. 30).
        'n_iter': 500 # number of iterations for the tsne algorithm
    }

    # Apply TSNE Projection 
    projected_p_embd = tsne_reduce(np.array(protein_embd), **options) # list

    # Display Projected Embedding (from 1024 dimensional (Protein Level) vectors to 3 dimensional coordinate vectors)
    print(f"\nShape of projected/dimensionality-reduced protein level embedding: {projected_p_embd.shape}\n")
    for i,embedding in enumerate(projected_p_embd[:3]): # first 3
        print(f"Protein {i+1}\t{embedding}")
    print(". . .")
    for i,embedding in enumerate(projected_p_embd[-3:]): # last 3
        print(f"Protein {i+len(projected_p_embd)-2}\t{embedding}")
    print() 

def embedding_visualisation():
    # %% [markdown]
    # ## 5) Visualization
    # 
    # ### 5.1) Via Plotly Express

    # %%
    import plotly.express as px

    fig = px.scatter_3d(
        projected_p_embd, x=0, y=1, z=2,
        labels={'0': 'dim 1', '1': 'dim 2', '2': 'dim 3'}
    )
    fig.show()

    # %% [markdown]
    # Error Encounter:
    # - ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed
    #     1) pip install --upgrade nbformat (eventually restart kernel afterwards)
    # 

    # %% [markdown]
    # ### 5.2) Via Bio-Embeddings library

    # %%
    import pandas as pd
    from bio_embeddings.visualize import render_3D_scatter_plotly

    column_names = ['component_0', 'component_1', 'component_2'] # this format is mandatory
    df = pd.DataFrame(projected_p_embd, columns=column_names)

    figure = render_3D_scatter_plotly(df)
    figure.show()

# %% [markdown]
# ### 5.3) Via Pyplot Scatterplot
def plotly_visualisation():
    # %%
    import matplotlib.pyplot as plt

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, and z coordinates from the data
    x = projected_p_embd[:, 0]
    y = projected_p_embd[:, 1]
    z = projected_p_embd[:, 2]

    # Plot the points
    ax.scatter(x, y, z)

    # Set labels for each axis
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.set_zlabel('dim 3')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    import argparse
    import sys
    
    filepath = "fasta_files/VOG00024.faa"
    parser = argparse.ArgumentParser(description="Bio-Embeddings Argument Parser")
    parser.add_argument("--i", dest="input_file", type=str, help="Input fasta file path")
    parser.add_argument("--o", dest="output_file", type=str, help="Output file path")
    args = parser.parse_args()

    input_file  = args.input_file
    output_file = args.output_file
    print("hello world!")
