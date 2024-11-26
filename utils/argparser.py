# This file contains utility functions for creation of argparsers
# and parsing the cmdline arguments.

import argparse

def setup_parser_train():
    """ Set up the parser for training the model. """
    
    parser = argparse.ArgumentParser('Train Parser')
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        help='Directory of CSV files to merge.', 
        required=True,
    )
    parser.add_argument(
        '--domain_id', 
        type=str, 
        default='domain_id', 
        help='Path to save file.', 
        required=False,
    )
    parser.add_argument(
        '--content_id', 
        type=str, 
        default='content', 
        help='Path to save file.', 
        required=False,
    )
    parser.add_argument(
        '--label_id', 
        type=str, 
        default='article_reads', 
        help='Path to save file.', 
        required=False,
    )
    parser.add_argument(
        '--embedding_id', 
        type=str, 
        default='bert_embedding', 
        help='Name of BERT embedding column to be created.', 
        required=False,
    )
    parser.add_argument(
        '--base_model', 
        type=str, 
        default='xlm-roberta-base', 
        help='Base model to use for encoding text.', 
        required=False,
    )

    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.25, 
        help='Percentage of test samples.', 
        required=False,
    )
    parser.add_argument(
        '--samples_per_domain', 
        type=int, 
        help='Number of samples per domain (i.e. 25 each).',
        required=True,
    )
    
    # Model arguments
    # GNN
    parser.add_argument(
        "--gcn_embed_dims", 
        nargs="*",
        type=int, 
        default=[1, 1],
        help='Embedding dimensions of GNN block.',
        required=False,
    )
    parser.add_argument(
        '--gcn_layer',
        dest='gcn_layer',
        type=str,
        default='GCNConv', 
        help='Type of convolution operation to use.',
        required=False,
    )
    parser.add_argument(
        '--gcn_act',
        dest='gcn_act',
        type=str,
        default='ReLU',
        help='Activation function after each GNN convolution.',
        required=False,
    )
    # MLP
    parser.add_argument(
        "--mlp_embed_dims", 
        nargs="*",
        type=int, 
        default=[1, 1],
        help='Embedding dimensions of MLP block.'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs.',
        required=True,
    )
    parser.add_argument(
        '--learning_rate',
        type=float, 
        help='Training learning rate.',
        required=True,
    )
    parser.add_argument(
        '--optimizer',
        type=str, 
        choices={'Adam', 'AdamW'},
        help='Training optimizer.',
        required=True,
    )
    parser.add_argument(
        '--device',
        type=str, 
        default='cuda',
        choices={'cuda', 'cpu'},
        help='Device to train on.',
        required=False,
    )
    
    return parser


def setup_parser_merge():
    """ Set up the parser for the purpose of merging the directory of CSV files into a single one. """
    parser = argparse.ArgumentParser('Merge Parser')
    
    parser.add_argument('--data_dir', type=str, help='Directory of CSV files to merge.', required=True,)
    parser.add_argument('--save_path', type=str, help='Path to save file.', required=True,)

    return parser
    

# Cleaner access to parsing functions
NAME_TO_PARSER_FUNCTION = {
    'train': setup_parser_train,
    'merge': setup_parser_merge,
}

def parse_args(parser_name: str):
    assert parser_name in NAME_TO_PARSER_FUNCTION, 'Parser function not found.'
    
    # Get the parser setup function and create the adequate parser
    setup_parser = NAME_TO_PARSER_FUNCTION[parser_name]
    parser = setup_parser()
    
    # Parse the arguments
    args = parser.parse_args()
    
    return args