#!/usr/bin/env python3
"""
GPU-Accelerated Semantic Severity Recalibration

Optimized for maximum throughput using:
- GPU acceleration for embeddings (RTX PRO 5000, 48GB VRAM)
- Batched encoding (process all descriptions at once)
- Multiprocessing for CPU-bound drug class scoring
- Vectorized similarity computation

This version processes 760k interactions in minutes, not hours.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import time
import multiprocessing as mp
from functools import partial


@dataclass
class GPURecalibrationConfig:
    """Configuration for GPU-accelerated recalibration"""
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "cuda"  # Use GPU
    batch_size: int = 8192  # Large batches for GPU efficiency
    n_workers: int = 24  # All CPU cores
    
    # Similarity thresholds
    contraindicated_threshold: float = 0.65
    major_threshold: float = 0.55
    moderate_threshold: float = 0.45
    
    # Component weights
    semantic_weight: float = 0.45
    confidence_weight: float = 0.25
    drug_class_weight: float = 0.30
    
    contraindicated_min_confidence: float = 0.65
    major_min_confidence: float = 0.50


SEVERITY_PROTOTYPES = {
    'contraindicated': [
        "This combination causes fatal cardiac arrhythmias including torsades de pointes",
        "Co-administration leads to QT prolongation and sudden cardiac death",
        "The combination results in potentially lethal ventricular arrhythmias",
        "Combined use causes life-threatening serotonin syndrome",
        "This interaction produces potentially fatal serotonergic toxicity",
        "The drugs together cause dangerous hyperserotonergic state",
        "This combination is absolutely contraindicated due to fatality risk",
        "Never use together - deaths have been reported",
        "Co-administration causes potentially fatal reactions",
        "The combination leads to neuroleptic malignant syndrome",
        "This interaction causes severe malignant hyperthermia",
    ],
    'major': [
        "This combination significantly increases the risk of serious bleeding",
        "Co-administration causes major hemorrhagic complications",
        "The interaction substantially enhances anticoagulant effect leading to bleeding",
        "Combined use results in gastrointestinal bleeding requiring hospitalization",
        "This combination causes dangerous hyperkalemia",
        "The interaction leads to severe hypoglycemia",
        "Co-administration results in rhabdomyolysis and muscle breakdown",
        "The combination causes acute renal failure",
        "This interaction produces severe hepatotoxicity",
        "Co-administration leads to agranulocytosis and bone marrow suppression",
        "The interaction requires immediate medical attention",
        "Combined use may require hospitalization",
        "This combination causes serious adverse events requiring intervention",
    ],
    'moderate': [
        "This combination may increase serum concentration of the drug",
        "Co-administration can decrease the metabolism of the affected drug",
        "The interaction may enhance the therapeutic effect requiring monitoring",
        "Combined use may reduce drug clearance",
        "This combination may cause hypertension requiring monitoring",
        "Co-administration can lead to hypotension",
        "The interaction may affect blood pressure control",
        "Use together with caution and monitor for adverse effects",
        "Dose adjustment may be needed when using these drugs together",
        "Consider monitoring for increased or decreased drug effects",
        "The interaction has moderate clinical significance",
    ],
    'minor': [
        "This interaction is unlikely to be clinically significant",
        "The combination has minimal impact on drug effects",
        "Co-administration may cause minor changes in drug levels",
        "The interaction has theoretical significance but rarely causes problems",
        "This is a minor interaction with negligible clinical effect",
        "The combination may slightly alter therapeutic efficacy",
        "Drug interaction is of low clinical significance",
        "Slight changes in drug levels are not expected to be significant",
    ]
}


HIGH_RISK_DRUG_CLASSES = {
    'anticoagulants': [
        'warfarin', 'heparin', 'enoxaparin', 'rivaroxaban', 'apixaban',
        'dabigatran', 'edoxaban', 'fondaparinux', 'argatroban', 'bivalirudin'
    ],
    'antiplatelets': [
        'aspirin', 'clopidogrel', 'ticagrelor', 'prasugrel', 'dipyridamole',
        'ticlopidine', 'cangrelor', 'vorapaxar'
    ],
    'qt_prolonging': [
        'amiodarone', 'sotalol', 'dofetilide', 'dronedarone', 'quinidine',
        'procainamide', 'ibutilide', 'azithromycin', 'erythromycin',
        'haloperidol', 'methadone', 'ondansetron'
    ],
    'maois': [
        'phenelzine', 'tranylcypromine', 'isocarboxazid', 'selegiline',
        'rasagiline', 'safinamide', 'moclobemide'
    ],
    'strong_cyp3a4_inhibitors': [
        'ketoconazole', 'itraconazole', 'clarithromycin', 'ritonavir',
        'nelfinavir', 'indinavir', 'nefazodone', 'cobicistat'
    ],
    'serotonergics': [
        'fluoxetine', 'sertraline', 'paroxetine', 'citalopram', 'escitalopram',
        'venlafaxine', 'duloxetine', 'tramadol', 'fentanyl', 'triptans'
    ]
}


def compute_drug_class_score(drug_pair: tuple) -> float:
    """Compute drug class risk score for a single pair (for multiprocessing)"""
    drug1, drug2 = drug_pair
    d1_lower = str(drug1).lower() if pd.notna(drug1) else ''
    d2_lower = str(drug2).lower() if pd.notna(drug2) else ''
    
    d1_classes = set()
    d2_classes = set()
    
    for cls, drugs in HIGH_RISK_DRUG_CLASSES.items():
        if any(d in d1_lower for d in drugs):
            d1_classes.add(cls)
        if any(d in d2_lower for d in drugs):
            d2_classes.add(cls)
    
    overlap = d1_classes & d2_classes
    
    if 'maois' in d1_classes and 'serotonergics' in d2_classes:
        return 4.0
    if 'maois' in d2_classes and 'serotonergics' in d1_classes:
        return 4.0
    if 'maois' in overlap:
        return 4.0
    if 'anticoagulants' in overlap or 'qt_prolonging' in overlap:
        return 3.5
    if d1_classes and d2_classes:
        return 3.0
    if d1_classes or d2_classes:
        return 2.5
    return 2.0


class GPUSeverityRecalibrator:
    """
    GPU-accelerated semantic severity recalibration.
    
    Strategy:
    1. Batch-encode ALL 760k descriptions on GPU (fast!)
    2. Compute similarity matrix via vectorized operations
    3. Parallel drug class scoring on all 24 CPU cores
    4. Vectorized final score computation
    """
    
    SEVERITY_NUMERIC = {
        'Contraindicated interaction': 4,
        'Major interaction': 3,
        'Moderate interaction': 2,
        'Minor interaction': 1
    }
    
    def __init__(self, config: Optional[GPURecalibrationConfig] = None):
        self.config = config or GPURecalibrationConfig()
        self.model = None
        self.prototype_centroids = {}
        self.stats = {}
        
    def _load_model(self):
        """Load sentence transformer on GPU"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            
            device = self.config.device
            if device == "cuda" and not torch.cuda.is_available():
                print("WARNING: CUDA not available, falling back to CPU")
                device = "cpu"
            
            print(f"Loading model on {device.upper()}...")
            self.model = SentenceTransformer(self.config.embedding_model, device=device)
            
            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   GPU: {gpu_name} ({gpu_mem:.1f}GB)")
            
            # Pre-compute prototype centroids
            print("Computing prototype centroids...")
            for severity, prototypes in SEVERITY_PROTOTYPES.items():
                embeddings = self.model.encode(prototypes, convert_to_numpy=True,
                                               show_progress_bar=False)
                self.prototype_centroids[severity] = np.mean(embeddings, axis=0)
            
            # Stack centroids for vectorized computation
            self.centroid_matrix = np.stack([
                self.prototype_centroids['contraindicated'],
                self.prototype_centroids['major'],
                self.prototype_centroids['moderate'],
                self.prototype_centroids['minor']
            ])
            # Normalize for cosine similarity
            self.centroid_matrix = self.centroid_matrix / np.linalg.norm(
                self.centroid_matrix, axis=1, keepdims=True
            )
            print("   ✓ Prototype centroids ready")
    
    def _batch_encode_descriptions(self, descriptions: List[str]) -> np.ndarray:
        """Batch encode all descriptions on GPU"""
        print(f"\n📡 Encoding {len(descriptions):,} descriptions on GPU...")
        start = time.time()
        
        # Handle NaN and empty
        clean_descriptions = [
            str(d) if pd.notna(d) and str(d).strip() else "No interaction description"
            for d in descriptions
        ]
        
        embeddings = self.model.encode(
            clean_descriptions,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Pre-normalize for cosine sim
        )
        
        elapsed = time.time() - start
        rate = len(descriptions) / elapsed
        print(f"   ✓ Encoded in {elapsed:.1f}s ({rate:,.0f} descriptions/sec)")
        
        return embeddings
    
    def _compute_semantic_scores_vectorized(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute semantic similarity scores via matrix multiplication"""
        print("\n🧮 Computing semantic similarities (vectorized)...")
        start = time.time()
        
        # Cosine similarity via dot product (both normalized)
        # Shape: (n_samples, 4) for 4 severity classes
        similarities = embeddings @ self.centroid_matrix.T
        
        # Map to scores based on thresholds
        scores = np.full(len(embeddings), 1.5)  # Default minor
        
        # Apply thresholds (order matters - most severe first)
        scores = np.where(similarities[:, 0] >= self.config.contraindicated_threshold, 4.0, scores)
        scores = np.where(
            (similarities[:, 1] >= self.config.major_threshold) & (scores < 4.0),
            3.2, scores
        )
        scores = np.where(
            (similarities[:, 2] >= self.config.moderate_threshold) & (scores < 3.2),
            2.0, scores
        )
        
        elapsed = time.time() - start
        print(f"   ✓ Computed in {elapsed:.2f}s")
        
        return scores, similarities
    
    def _compute_drug_class_scores_parallel(self, df: pd.DataFrame) -> np.ndarray:
        """Compute drug class scores using all CPU cores"""
        print(f"\n🔧 Computing drug class scores ({self.config.n_workers} workers)...")
        start = time.time()
        
        drug_pairs = list(zip(df['drug_name_1'].values, df['drug_name_2'].values))
        
        with mp.Pool(self.config.n_workers) as pool:
            scores = pool.map(compute_drug_class_score, drug_pairs, chunksize=10000)
        
        elapsed = time.time() - start
        print(f"   ✓ Computed in {elapsed:.1f}s")
        
        return np.array(scores)
    
    def _compute_confidence_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized confidence adjustment scores"""
        print("\n📊 Computing confidence adjustments...")
        
        severity_map = df['severity_label'].map(self.SEVERITY_NUMERIC).fillna(2).values
        confidence = df['severity_confidence'].fillna(0.5).values
        
        scores = severity_map.astype(float)
        
        # Downgrade low-confidence contraindicated
        mask_contra = (severity_map == 4) & (confidence < self.config.contraindicated_min_confidence)
        scores[mask_contra] = 3.0
        
        # Partial downgrade low-confidence major
        mask_major = (severity_map >= 3) & (confidence < self.config.major_min_confidence) & ~mask_contra
        scores[mask_major] = 2.5
        
        print(f"   ✓ {mask_contra.sum():,} contraindicated downgraded")
        print(f"   ✓ {mask_major.sum():,} major/contra partially downgraded")
        
        return scores
    
    def recalibrate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recalibrate entire dataset with GPU acceleration and multiprocessing.
        """
        print("\n" + "="*70)
        print("GPU-ACCELERATED SEMANTIC SEVERITY RECALIBRATION")
        print("="*70)
        print(f"Dataset: {len(df):,} interactions")
        print(f"Device: {self.config.device.upper()}")
        print(f"Workers: {self.config.n_workers} CPU cores")
        print(f"Batch size: {self.config.batch_size:,}")
        
        total_start = time.time()
        
        # Load model on GPU
        self._load_model()
        
        # Step 1: Batch encode ALL descriptions on GPU
        descriptions = df['interaction_description'].tolist()
        desc_embeddings = self._batch_encode_descriptions(descriptions)
        
        # Step 2: Vectorized semantic similarity computation
        semantic_scores, similarities = self._compute_semantic_scores_vectorized(desc_embeddings)
        
        # Step 3: Parallel drug class scoring on all CPU cores
        drug_class_scores = self._compute_drug_class_scores_parallel(df)
        
        # Step 4: Vectorized confidence adjustment
        confidence_scores = self._compute_confidence_scores(df)
        
        # Step 5: Compute final weighted scores (vectorized)
        print("\n⚡ Computing final scores...")
        final_scores = (
            self.config.semantic_weight * semantic_scores +
            self.config.confidence_weight * confidence_scores +
            self.config.drug_class_weight * drug_class_scores
        )
        
        # Step 6: Map to severity categories (vectorized)
        severities = np.full(len(df), 'Minor interaction', dtype=object)
        severities = np.where(final_scores >= 2.0, 'Moderate interaction', severities)
        severities = np.where(final_scores >= 2.5, 'Major interaction', severities)
        severities = np.where(final_scores >= 3.2, 'Contraindicated interaction', severities)
        
        # Build result dataframe
        df_recal = df.copy()
        df_recal['severity_recalibrated'] = severities
        df_recal['recal_confidence'] = np.minimum(0.95, df['severity_confidence'].fillna(0.5) + 0.1)
        df_recal['recal_method'] = 'semantic_hybrid_gpu'
        df_recal['semantic_score'] = semantic_scores
        df_recal['drug_class_score'] = drug_class_scores
        df_recal['confidence_score'] = confidence_scores
        df_recal['final_score'] = final_scores
        
        total_elapsed = time.time() - total_start
        
        # Compute and display statistics
        self._compute_stats(df, df_recal, total_elapsed)
        
        return df_recal
    
    def _compute_stats(self, df_original: pd.DataFrame, df_recal: pd.DataFrame, elapsed: float):
        """Compute and display recalibration statistics"""
        print("\n" + "="*70)
        print("RECALIBRATION RESULTS")
        print("="*70)
        
        orig_dist = df_original['severity_label'].value_counts(normalize=True)
        recal_dist = df_recal['severity_recalibrated'].value_counts(normalize=True)
        
        target_dist = {
            'Contraindicated interaction': 0.05,
            'Major interaction': 0.25,
            'Moderate interaction': 0.60,
            'Minor interaction': 0.10
        }
        
        print(f"\n{'Severity':<30} {'Original':>10} {'Recalibrated':>12} {'Target':>10}")
        print("-"*65)
        
        for sev in ['Contraindicated interaction', 'Major interaction', 
                    'Moderate interaction', 'Minor interaction']:
            orig = orig_dist.get(sev, 0)
            recal = recal_dist.get(sev, 0)
            target = target_dist[sev]
            delta = recal - target
            print(f"{sev:<30} {orig:>9.1%} {recal:>11.1%} {target:>9.1%}  ({delta:+.1%})")
        
        changes = (df_original['severity_label'] != df_recal['severity_recalibrated']).sum()
        print(f"\n📊 Total changes: {changes:,} ({changes/len(df_original)*100:.1f}%)")
        print(f"⏱️  Total time: {elapsed:.1f}s ({len(df_original)/elapsed:,.0f} interactions/sec)")
        
        self.stats = {
            'original_distribution': orig_dist.to_dict(),
            'recalibrated_distribution': recal_dist.to_dict(),
            'target_distribution': target_dist,
            'total_changes': int(changes),
            'change_rate': float(changes / len(df_original)),
            'processing_time_seconds': elapsed,
            'throughput_per_second': len(df_original) / elapsed
        }


def main():
    """Run GPU-accelerated semantic recalibration"""
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description='GPU-accelerated severity recalibration')
    parser.add_argument('--data', type=str,
                       default='data/ddi_cardio_or_antithrombotic_labeled (1).csv')
    parser.add_argument('--output', type=str,
                       default='data/ddi_semantic_recalibrated_gpu.csv')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--workers', type=int, default=mp.cpu_count())
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DDI SEVERITY RECALIBRATION - GPU ACCELERATED")
    print("="*70)
    
    # Check resources
    print(f"\n🖥️  System Resources:")
    print(f"   CPU Cores: {mp.cpu_count()}")
    
    if torch.cuda.is_available() and not args.cpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU: {gpu_name} ({gpu_mem:.1f}GB VRAM)")
        device = "cuda"
    else:
        print(f"   GPU: Not available, using CPU")
        device = "cpu"
    
    print(f"\n📂 Loading data: {args.data}")
    df = pd.read_csv(args.data)
    print(f"   Loaded {len(df):,} interactions")
    
    config = GPURecalibrationConfig(
        embedding_model=args.model,
        device=device,
        batch_size=args.batch_size,
        n_workers=args.workers
    )
    
    recalibrator = GPUSeverityRecalibrator(config)
    df_recal = recalibrator.recalibrate_dataset(df)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_recal.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    # Save statistics
    stats_path = output_path.with_suffix('.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'config': {
                'model': config.embedding_model,
                'device': config.device,
                'batch_size': config.batch_size,
                'n_workers': config.n_workers,
                'semantic_weight': config.semantic_weight,
                'confidence_weight': config.confidence_weight,
                'drug_class_weight': config.drug_class_weight
            },
            'stats': recalibrator.stats
        }, f, indent=2)
    print(f"📊 Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
