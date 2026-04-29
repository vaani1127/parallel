#!/usr/bin/env python3
"""
Dataset Generator for Parallel Document Similarity Search
Generates documents.txt and query.txt for testing
"""
import random
import argparse

def generate_dataset(num_docs, dict_size, max_val=100, seed=42):
    """Generate random document dataset"""
    random.seed(seed)
    
    with open("documents.txt", "w") as f:
        for i in range(1, num_docs + 1):
            features = [random.randint(1, max_val) for _ in range(dict_size)]
            f.write(f"{i}: {' '.join(map(str, features))}\n")
    
    with open("query.txt", "w") as f:
        query = [random.randint(1, max_val) for _ in range(dict_size)]
        f.write(f"{' '.join(map(str, query))}\n")
    
    print(f"Generated {num_docs} documents with {dict_size} features each")
    print(f"Files: documents.txt, query.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test dataset for docsim")
    parser.add_argument("--docs", type=int, default=1000, help="Number of documents (default: 1000)")
    parser.add_argument("--features", type=int, default=10, help="Number of features per document (default: 10)")
    parser.add_argument("--max-val", type=int, default=100, help="Maximum feature value (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    generate_dataset(args.docs, args.features, args.max_val, args.seed)
