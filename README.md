<h1>📝 Article GNN - Predicting the Number of Article Reads </h1>

<b>Article GNN</b> is a project implementing Graph Neural Networks, for the purpose of predicting the number of reads an article is going to get. 

<h2> ℹ️ General Information </h2>

This project aims to understand the effect of locality graphs could create.

Graph generation is very simple in itself - articles of the same domain (source) are connected to each other, forming the connected components. This creates an isolated environment for each domain, and no communication is happening amongst the articles.

Leveraging these connected components, GNNs learn that the environment also matters, aside from the content. This leads to the obvious point - even if two different sources published the article with exactly the same content, we don't expect the same number of reads, simply because of the popularity and timing of the article publishing.

The architecture is relatively simple: <b>GNN block</b>, followed by an <b>MLP block</b>. <br/>
GNN block consists of several <b>Graph Attention Network (GAT)</b> layers, whilst the MLP consists of several <b>Fully Connected Layers</b>.

<b>Testing</b> is performed by adding a single article into the graph, and then evaluating the precision of the inferred value, compared to the ground truth value present within the dataset.

<h2> 🚀 Quick Start </h2>
<pre>
<code>git clone https://github.com/LukaNedimovic/article_gnn.git
cd article_gnn
source ./setup.sh</code></pre>

<h2> 📁 Project Structure </h2>
<pre><code>article_gnn
├── data
│   ├── make_graph.py        # Graph generation script  
│   └── preprocess.py        # Preprocess the dataset
├── model
│   ├── gnn.py               # Core GNN block implementation
│   ├── mlp.py               # Core MLP block implementation
│   └── model.py             # Combining GNN -> MLP into general model
├── requirements.txt        
├── setup.sh                 # Use this to set up the project!      
├── train
│   ├── train.py             # Main code used for training the model
│   └── train.sh             # Run training, as described in `train.py`
└── utils
    ├── argparser.py         # Parse cmdline arguments
    ├── merge.py             # Utility file, used to merge CSVs together
    ├── merge.sh             # Run CSV merging
    └── path.py              # Utility file, expand environment variables within the path</code></pre>

<h2> 🔍 References </h2>
This model leverages the Graph Attention Networks, as described in the original paper <i>"Graph Attention Networks"</i> (Veličković et al.):
<pre><code>@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={accepted as poster},
}</code></pre>
