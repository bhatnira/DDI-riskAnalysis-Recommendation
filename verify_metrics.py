#!/usr/bin/env python3
"""Verify DDI Knowledge Graph metrics using CSV export."""

import pandas as pd
import networkx as nx
import numpy as np

# Load from CSV files (more portable than pickle)
print("Loading data from CSV exports...")
drugs_df = pd.read_csv('knowledge_graph_enriched/neo4j_export/drugs.csv')
ddi_df = pd.read_csv('knowledge_graph_enriched/neo4j_export/ddi_edges.csv')

print(f"Drugs CSV: {len(drugs_df)} drugs")
print(f"DDI CSV: {len(ddi_df)} DDI edges")

# Create undirected simple graph
print("\n--- Creating DDI undirected simple graph ---")
G = nx.Graph()

# Add drug nodes (using drugbank_id)
drug_ids = drugs_df['drugbank_id'].tolist()
G.add_nodes_from(drug_ids)

# Add DDI edges (undirected) - using drug1_id and drug2_id
for _, row in ddi_df.iterrows():
    G.add_edge(row['drug1_id'], row['drug2_id'])

n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
density = nx.density(G)

degrees = [d for n, d in G.degree()]
avg_degree = np.mean(degrees)
median_degree = np.median(degrees)
std_degree = np.std(degrees)

clustering = nx.average_clustering(G)
connected = nx.is_connected(G)

print('='*50)
print('ACTUAL KG METRICS (DDI-only, undirected):')
print('='*50)
print(f'Total Drug Nodes:       {n_nodes:,}')
print(f'Total DDI Edges:        {n_edges:,}')
print(f'Network Density:        {density:.4f}')
print(f'Average Degree:         {avg_degree:.1f}')
print(f'Median Degree:          {median_degree:.0f}')
print(f'Standard Deviation:     {std_degree:.1f}')
print(f'Clustering Coefficient: {clustering:.4f}')
print(f'Is Connected:           {connected}')

if connected and n_nodes < 10000 and n_nodes > 0:
    print("\nComputing diameter and avg path length (this may take a while)...")
    diameter = nx.diameter(G)
    avg_path = nx.average_shortest_path_length(G)
    print(f'Network Diameter:       {diameter}')
    print(f'Avg Path Length:        {avg_path:.2f}')
