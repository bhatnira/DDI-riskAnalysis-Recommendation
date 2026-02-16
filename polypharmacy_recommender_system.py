"""
================================================================================
AI-based Polypharmacy Risk-aware Drug Recommender System
================================================================================

System Architecture:
               ┌──────────────────┐
               │  Drug Input List │
               └─────────┬────────┘
                         ↓
              ┌────────────────────┐
              │ Interaction Engine │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Severity Predictor │ (ML / GNN)
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Alternative Finder │ (Embedding + Filtering)
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Explanation LLM    │
              └────────────────────┘

Author: AI-based Polypharmacy Research
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support,
                            roc_auc_score)
from sklearn.neighbors import NearestNeighbors

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


# ============================================================================
# MODULE 1: DRUG INPUT LIST
# ============================================================================

class DrugInputProcessor:
    """
    📦 Module 1: Drug Input List
    
    Processes and validates drug input lists.
    Maps drug names to DrugBank IDs and retrieves metadata.
    """
    
    def __init__(self, ddi_dataframe):
        self.df = ddi_dataframe
        self._build_drug_database()
    
    def _build_drug_database(self):
        """Build a lookup database of all drugs"""
        drugs_1 = self.df[['drugbank_id_1', 'drug_name_1', 'atc_1', 
                           'is_cardiovascular_1', 'is_antithrombotic_1']].copy()
        drugs_1.columns = ['drugbank_id', 'drug_name', 'atc', 'is_cardiovascular', 'is_antithrombotic']
        
        drugs_2 = self.df[['drugbank_id_2', 'drug_name_2', 'atc_2',
                           'is_cardiovascular_2', 'is_antithrombotic_2']].copy()
        drugs_2.columns = ['drugbank_id', 'drug_name', 'atc', 'is_cardiovascular', 'is_antithrombotic']
        
        self.drug_db = pd.concat([drugs_1, drugs_2]).drop_duplicates(subset=['drugbank_id'])
        self.drug_names = set(self.drug_db['drug_name'].str.lower())
        self.name_to_id = dict(zip(self.drug_db['drug_name'].str.lower(), self.drug_db['drugbank_id']))
        self.id_to_name = dict(zip(self.drug_db['drugbank_id'], self.drug_db['drug_name']))
        
        print(f"✅ Drug database built: {len(self.drug_db):,} unique drugs")
    
    def validate_drugs(self, drug_list):
        """Validate a list of drug names"""
        validated = []
        unrecognized = []
        
        for drug in drug_list:
            drug_lower = drug.lower().strip()
            if drug_lower in self.drug_names:
                validated.append({
                    'input_name': drug,
                    'drugbank_id': self.name_to_id.get(drug_lower),
                    'status': 'valid'
                })
            else:
                # Try fuzzy matching
                matches = [d for d in self.drug_names if drug_lower in d or d in drug_lower]
                if matches:
                    best_match = matches[0]
                    validated.append({
                        'input_name': drug,
                        'matched_name': best_match.title(),
                        'drugbank_id': self.name_to_id.get(best_match),
                        'status': 'fuzzy_match'
                    })
                else:
                    unrecognized.append(drug)
        
        return validated, unrecognized
    
    def get_drug_info(self, drug_name):
        """Get detailed info for a drug"""
        drug_lower = drug_name.lower()
        if drug_lower in self.name_to_id:
            drug_id = self.name_to_id[drug_lower]
            info = self.drug_db[self.drug_db['drugbank_id'] == drug_id].iloc[0]
            return info.to_dict()
        return None
    
    def process_input(self, drug_list):
        """Main entry point - process a drug list"""
        print("\n" + "="*60)
        print("📦 MODULE 1: DRUG INPUT PROCESSING")
        print("="*60)
        print(f"\nInput drugs: {drug_list}")
        
        validated, unrecognized = self.validate_drugs(drug_list)
        
        print(f"\n✅ Validated: {len(validated)} drugs")
        print(f"❌ Unrecognized: {len(unrecognized)} drugs")
        
        if unrecognized:
            print(f"   Unrecognized: {unrecognized}")
        
        return {
            'validated_drugs': validated,
            'unrecognized': unrecognized,
            'drug_count': len(validated)
        }


# ============================================================================
# MODULE 2: INTERACTION ENGINE
# ============================================================================

class InteractionEngine:
    """
    ⚙️ Module 2: Interaction Engine
    
    Detects drug-drug interactions from a list of drugs.
    Queries the DDI database for all pairwise interactions.
    """
    
    def __init__(self, ddi_dataframe):
        self.df = ddi_dataframe
        self._build_interaction_index()
    
    def _build_interaction_index(self):
        """Build a fast lookup index for interactions"""
        self.interactions = defaultdict(list)
        
        for _, row in self.df.iterrows():
            drug1 = row['drug_name_1'].lower()
            drug2 = row['drug_name_2'].lower()
            
            interaction_data = {
                'drug_1': row['drug_name_1'],
                'drug_2': row['drug_name_2'],
                'drugbank_id_1': row['drugbank_id_1'],
                'drugbank_id_2': row['drugbank_id_2'],
                'description': row['interaction_description'],
                'severity_label': row['severity_label'],
                'severity_confidence': row['severity_confidence'],
                'severity_numeric': row['severity_numeric'],
                'is_cardiovascular_1': row['is_cardiovascular_1'],
                'is_cardiovascular_2': row['is_cardiovascular_2'],
                'is_antithrombotic_1': row['is_antithrombotic_1'],
                'is_antithrombotic_2': row['is_antithrombotic_2']
            }
            
            self.interactions[(drug1, drug2)].append(interaction_data)
            self.interactions[(drug2, drug1)].append(interaction_data)
        
        print(f"✅ Interaction index built: {len(self.df):,} interactions indexed")
    
    def find_interaction(self, drug1, drug2):
        """Find interaction between two drugs"""
        return self.interactions.get((drug1.lower(), drug2.lower()), [])
    
    def detect_all_interactions(self, drug_list):
        """Detect all pairwise interactions in a drug list"""
        print("\n" + "="*60)
        print("⚙️ MODULE 2: INTERACTION ENGINE")
        print("="*60)
        
        interactions_found = []
        n_drugs = len(drug_list)
        pairs_checked = 0
        
        for i in range(n_drugs):
            for j in range(i + 1, n_drugs):
                drug1, drug2 = drug_list[i], drug_list[j]
                pairs_checked += 1
                
                interactions = self.find_interaction(drug1, drug2)
                for interaction in interactions:
                    interactions_found.append({
                        'pair': f"{drug1} ↔ {drug2}",
                        'drug_1': drug1,
                        'drug_2': drug2,
                        **interaction
                    })
        
        # Sort by severity
        severity_order = {'Contraindicated interaction': 0, 'Major interaction': 1, 
                         'Moderate interaction': 2, 'Minor interaction': 3}
        interactions_found.sort(key=lambda x: severity_order.get(x['severity_label'], 4))
        
        print(f"\n📊 Analysis Summary:")
        print(f"   Drugs analyzed: {n_drugs}")
        print(f"   Pairs checked: {pairs_checked}")
        print(f"   Interactions found: {len(interactions_found)}")
        
        if interactions_found:
            severity_counts = Counter(i['severity_label'] for i in interactions_found)
            print(f"\n⚠️ Severity Breakdown:")
            for sev, count in severity_counts.items():
                emoji = "🔴" if "Contraindicated" in sev else "🟠" if "Major" in sev else "🟡" if "Moderate" in sev else "🟢"
                print(f"   {emoji} {sev}: {count}")
        
        return {
            'interactions': interactions_found,
            'total_interactions': len(interactions_found),
            'drugs_analyzed': n_drugs,
            'pairs_checked': pairs_checked
        }


# ============================================================================
# MODULE 3: SEVERITY PREDICTOR (ML / GNN)
# ============================================================================

class SeverityPredictor:
    """
    🧠 Module 3: Severity Predictor (ML / GNN)
    
    Machine Learning based severity prediction for drug interactions.
    Uses ensemble methods and can be extended to GNN for graph-based prediction.
    """
    
    def __init__(self, ddi_dataframe):
        self.df = ddi_dataframe
        self.models = {}
        self.encoders = {}
        self.is_trained = False
        
    def prepare_features(self):
        """Prepare feature matrix for ML models"""
        print("\n" + "="*60)
        print("🧠 MODULE 3: SEVERITY PREDICTOR - Training")
        print("="*60)
        
        ml_df = self.df.copy()
        
        # Encode categorical variables
        self.encoders['drug1'] = LabelEncoder()
        self.encoders['drug2'] = LabelEncoder()
        self.encoders['severity'] = LabelEncoder()
        
        ml_df['drug1_encoded'] = self.encoders['drug1'].fit_transform(ml_df['drug_name_1'])
        ml_df['drug2_encoded'] = self.encoders['drug2'].fit_transform(ml_df['drug_name_2'])
        ml_df['severity_encoded'] = self.encoders['severity'].fit_transform(ml_df['severity_label'])
        
        # Create interaction type feature
        def categorize_interaction(desc):
            desc_lower = str(desc).lower()
            if 'anticoagulant' in desc_lower: return 0
            elif 'bleeding' in desc_lower or 'hemorrhage' in desc_lower: return 1
            elif 'therapeutic efficacy' in desc_lower: return 2
            elif 'metabolism' in desc_lower: return 3
            elif 'serum concentration' in desc_lower: return 4
            elif 'cardiotoxic' in desc_lower or 'arrhythmia' in desc_lower: return 5
            else: return 6
        
        ml_df['interaction_type'] = ml_df['interaction_description'].apply(categorize_interaction)
        
        # Features
        self.features = ['drug1_encoded', 'drug2_encoded', 'is_cardiovascular_1', 
                        'is_cardiovascular_2', 'is_antithrombotic_1', 'is_antithrombotic_2',
                        'interaction_type']
        
        self.X = ml_df[self.features].astype(float)
        self.y = ml_df['severity_encoded']
        
        print(f"✅ Features prepared: {len(self.features)} features, {len(self.X):,} samples")
        return self.X, self.y
    
    def train_models(self):
        """Train multiple ML models"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        self.X_test, self.y_test = X_test, y_test
        
        model_configs = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        }
        
        results = []
        print(f"\nTraining on {len(X_train):,} samples, testing on {len(X_test):,} samples\n")
        
        for name, model in model_configs.items():
            print(f"Training {name}...", end=" ")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            self.models[name] = model
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        self.results = pd.DataFrame(results)
        self.best_model = max(self.models.items(), key=lambda x: accuracy_score(y_test, x[1].predict(X_test)))
        self.is_trained = True
        
        print(f"\n🏆 Best Model: {self.best_model[0]}")
        return self.results
    
    def predict_severity(self, drug1_name, drug2_name, interaction_desc, is_cv1, is_cv2, is_at1, is_at2):
        """Predict severity for a new drug pair"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Encode inputs (use -1 for unknown drugs)
        try:
            drug1_enc = self.encoders['drug1'].transform([drug1_name])[0]
        except:
            drug1_enc = -1
        try:
            drug2_enc = self.encoders['drug2'].transform([drug2_name])[0]
        except:
            drug2_enc = -1
        
        # Categorize interaction
        def categorize(desc):
            desc_lower = str(desc).lower()
            if 'anticoagulant' in desc_lower: return 0
            elif 'bleeding' in desc_lower or 'hemorrhage' in desc_lower: return 1
            elif 'therapeutic efficacy' in desc_lower: return 2
            elif 'metabolism' in desc_lower: return 3
            elif 'serum concentration' in desc_lower: return 4
            elif 'cardiotoxic' in desc_lower or 'arrhythmia' in desc_lower: return 5
            else: return 6
        
        int_type = categorize(interaction_desc)
        
        features = np.array([[drug1_enc, drug2_enc, is_cv1, is_cv2, is_at1, is_at2, int_type]])
        
        # Predict
        model = self.best_model[1]
        pred_encoded = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0]
        
        severity_label = self.encoders['severity'].inverse_transform([pred_encoded])[0]
        confidence = max(pred_proba)
        
        return {
            'predicted_severity': severity_label,
            'confidence': confidence,
            'all_probabilities': dict(zip(self.encoders['severity'].classes_, pred_proba))
        }
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest"""
        if 'Random Forest' in self.models:
            rf = self.models['Random Forest']
            importance = pd.DataFrame({
                'Feature': self.features,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            return importance
        return None


# ============================================================================
# MODULE 4: ALTERNATIVE FINDER (Embedding + Filtering)
# ============================================================================

class AlternativeFinder:
    """
    🔄 Module 4: Alternative Finder (Embedding + Filtering)
    
    Finds safer alternative drugs using embeddings and interaction filtering.
    Uses drug similarity based on ATC codes and interaction profiles.
    """
    
    def __init__(self, ddi_dataframe, drug_processor):
        self.df = ddi_dataframe
        self.drug_processor = drug_processor
        self._build_drug_profiles()
        self._build_embeddings()
    
    def _build_drug_profiles(self):
        """Build drug interaction profiles"""
        self.drug_profiles = {}
        
        # Count interactions and severity for each drug
        for _, row in self.df.iterrows():
            for drug_col, sev_col in [('drug_name_1', 'severity_numeric'), ('drug_name_2', 'severity_numeric')]:
                drug = row[drug_col]
                if drug not in self.drug_profiles:
                    self.drug_profiles[drug] = {
                        'total_interactions': 0,
                        'contraindicated': 0,
                        'major': 0,
                        'moderate': 0,
                        'minor': 0,
                        'avg_severity': [],
                        'is_cardiovascular': row.get(f'is_cardiovascular_{drug_col[-1]}', False),
                        'is_antithrombotic': row.get(f'is_antithrombotic_{drug_col[-1]}', False)
                    }
                
                self.drug_profiles[drug]['total_interactions'] += 1
                self.drug_profiles[drug]['avg_severity'].append(row['severity_numeric'])
                
                if 'Contraindicated' in row['severity_label']:
                    self.drug_profiles[drug]['contraindicated'] += 1
                elif 'Major' in row['severity_label']:
                    self.drug_profiles[drug]['major'] += 1
                elif 'Moderate' in row['severity_label']:
                    self.drug_profiles[drug]['moderate'] += 1
                else:
                    self.drug_profiles[drug]['minor'] += 1
        
        # Calculate average severity
        for drug in self.drug_profiles:
            sevs = self.drug_profiles[drug]['avg_severity']
            self.drug_profiles[drug]['avg_severity'] = np.mean(sevs) if sevs else 0
            self.drug_profiles[drug]['risk_score'] = (
                self.drug_profiles[drug]['contraindicated'] * 4 +
                self.drug_profiles[drug]['major'] * 3 +
                self.drug_profiles[drug]['moderate'] * 2 +
                self.drug_profiles[drug]['minor'] * 1
            ) / max(self.drug_profiles[drug]['total_interactions'], 1)
        
        print(f"✅ Drug profiles built: {len(self.drug_profiles):,} drugs profiled")
    
    def _build_embeddings(self):
        """Build drug embeddings for similarity search"""
        # Create feature vectors for each drug
        drugs = list(self.drug_profiles.keys())
        features = []
        
        for drug in drugs:
            profile = self.drug_profiles[drug]
            features.append([
                profile['total_interactions'] / 1000,  # Normalized
                profile['contraindicated'] / 100,
                profile['major'] / 100,
                profile['moderate'] / 100,
                profile['minor'] / 100,
                profile['avg_severity'] / 4,
                int(profile['is_cardiovascular']),
                int(profile['is_antithrombotic']),
                profile['risk_score']
            ])
        
        self.drug_list = drugs
        self.embeddings = np.array(features)
        
        # Build KNN model for similarity search
        self.knn = NearestNeighbors(n_neighbors=min(20, len(drugs)), metric='cosine')
        self.knn.fit(self.embeddings)
        
        print(f"✅ Drug embeddings built: {self.embeddings.shape}")
    
    def find_similar_drugs(self, drug_name, n=10):
        """Find similar drugs based on embeddings"""
        if drug_name not in self.drug_profiles:
            return []
        
        idx = self.drug_list.index(drug_name)
        distances, indices = self.knn.kneighbors([self.embeddings[idx]], n_neighbors=n+1)
        
        similar = []
        for i, dist in zip(indices[0][1:], distances[0][1:]):  # Skip self
            similar.append({
                'drug': self.drug_list[i],
                'similarity': 1 - dist,
                'risk_score': self.drug_profiles[self.drug_list[i]]['risk_score']
            })
        
        return similar
    
    def find_alternatives(self, problematic_drug, other_drugs, n=5):
        """
        Find safer alternatives for a problematic drug.
        Filters by checking interactions with other drugs in the regimen.
        """
        print("\n" + "="*60)
        print("🔄 MODULE 4: ALTERNATIVE FINDER")
        print("="*60)
        print(f"\nFinding alternatives for: {problematic_drug}")
        print(f"Must be safe with: {other_drugs}")
        
        # Get similar drugs
        similar = self.find_similar_drugs(problematic_drug, n=20)
        
        # Filter by checking interactions with other drugs
        safe_alternatives = []
        
        for candidate in similar:
            candidate_drug = candidate['drug']
            is_safe = True
            interaction_count = 0
            worst_severity = 0
            
            for other_drug in other_drugs:
                # Check interaction
                interactions = self.df[
                    ((self.df['drug_name_1'].str.lower() == candidate_drug.lower()) & 
                     (self.df['drug_name_2'].str.lower() == other_drug.lower())) |
                    ((self.df['drug_name_1'].str.lower() == other_drug.lower()) & 
                     (self.df['drug_name_2'].str.lower() == candidate_drug.lower()))
                ]
                
                if len(interactions) > 0:
                    interaction_count += len(interactions)
                    max_sev = interactions['severity_numeric'].max()
                    worst_severity = max(worst_severity, max_sev)
                    
                    # Check if contraindicated
                    if (interactions['severity_label'] == 'Contraindicated interaction').any():
                        is_safe = False
                        break
            
            if is_safe:
                safe_alternatives.append({
                    'drug': candidate_drug,
                    'similarity': candidate['similarity'],
                    'risk_score': candidate['risk_score'],
                    'interactions_with_regimen': interaction_count,
                    'worst_severity': worst_severity
                })
        
        # Sort by risk score (lower is better) and similarity (higher is better)
        safe_alternatives.sort(key=lambda x: (x['risk_score'], -x['similarity']))
        
        print(f"\n✅ Found {len(safe_alternatives)} safe alternatives")
        
        return safe_alternatives[:n]


# ============================================================================
# MODULE 5: EXPLANATION LLM
# ============================================================================

class ExplanationGenerator:
    """
    📝 Module 5: Explanation LLM
    
    Generates human-readable explanations for drug interactions and recommendations.
    Uses template-based generation (can be extended to use actual LLM APIs).
    """
    
    def __init__(self):
        self.severity_explanations = {
            'Contraindicated interaction': "🔴 CRITICAL: These drugs should NOT be used together. This combination poses serious health risks.",
            'Major interaction': "🟠 WARNING: This is a significant interaction that may require medical intervention or alternative therapy.",
            'Moderate interaction': "🟡 CAUTION: This interaction may worsen the patient's condition. Monitor closely.",
            'Minor interaction': "🟢 NOTICE: Minor interaction with limited clinical significance. Monitor as needed."
        }
        
        self.interaction_templates = {
            'anticoagulant': "The combination may {effect} anticoagulant effects, {risk}.",
            'bleeding': "There is an increased risk of bleeding and hemorrhage when these drugs are combined.",
            'efficacy': "The therapeutic efficacy of {drug} may be {change} by {other_drug}.",
            'concentration': "The serum concentration of {drug} may be {change} by {other_drug}.",
            'cardiac': "This combination may increase the risk of cardiac adverse effects including arrhythmias."
        }
    
    def generate_interaction_explanation(self, interaction_data):
        """Generate explanation for a single interaction"""
        severity = interaction_data.get('severity_label', 'Unknown')
        desc = interaction_data.get('description', '')
        drug1 = interaction_data.get('drug_1', 'Drug 1')
        drug2 = interaction_data.get('drug_2', 'Drug 2')
        confidence = interaction_data.get('severity_confidence', 0)
        
        explanation = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DRUG INTERACTION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💊 Drug Pair: {drug1} ↔ {drug2}

⚠️ Severity: {severity}
📊 Confidence: {confidence:.1%}

{self.severity_explanations.get(severity, '')}

📋 Clinical Description:
{desc}

💡 Recommendation:
"""
        if 'Contraindicated' in severity:
            explanation += "Avoid this combination. Consider alternative medications."
        elif 'Major' in severity:
            explanation += "Use with extreme caution. Consult with a clinical pharmacist."
        elif 'Moderate' in severity:
            explanation += "Monitor patient closely. Adjust dosage if necessary."
        else:
            explanation += "Continue with standard monitoring protocols."
        
        return explanation
    
    def generate_polypharmacy_report(self, drug_list, interactions, risk_score, alternatives=None):
        """Generate comprehensive polypharmacy report"""
        print("\n" + "="*60)
        print("📝 MODULE 5: EXPLANATION GENERATOR")
        print("="*60)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    POLYPHARMACY RISK ASSESSMENT REPORT                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

📅 Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT MEDICATION LIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for i, drug in enumerate(drug_list, 1):
            report += f"  {i}. {drug}\n"
        
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL RISK ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Total Risk Score: {risk_score:.2f}
"""
        
        if risk_score > 10:
            report += "⚠️ Risk Level: HIGH - Immediate review recommended\n"
        elif risk_score > 5:
            report += "⚠️ Risk Level: MODERATE - Review within 24 hours\n"
        else:
            report += "✅ Risk Level: LOW - Standard monitoring\n"
        
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETECTED INTERACTIONS ({len(interactions)} total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Group by severity
        severity_groups = defaultdict(list)
        for inter in interactions:
            severity_groups[inter['severity_label']].append(inter)
        
        for severity in ['Contraindicated interaction', 'Major interaction', 'Moderate interaction', 'Minor interaction']:
            if severity in severity_groups:
                emoji = "🔴" if "Contraindicated" in severity else "🟠" if "Major" in severity else "🟡" if "Moderate" in severity else "🟢"
                report += f"\n{emoji} {severity}:\n"
                for inter in severity_groups[severity]:
                    report += f"   • {inter['pair']}: {inter['description'][:80]}...\n"
        
        if alternatives:
            report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED ALTERNATIVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            for drug, alts in alternatives.items():
                report += f"\n🔄 Alternatives for {drug}:\n"
                for alt in alts[:3]:
                    report += f"   • {alt['drug']} (Risk Score: {alt['risk_score']:.2f}, Similarity: {alt['similarity']:.2%})\n"
        
        report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLINICAL RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Review all contraindicated combinations immediately
2. Consider therapeutic alternatives for high-risk pairs
3. Implement appropriate monitoring for moderate interactions
4. Document all interactions in patient record
5. Schedule follow-up review within appropriate timeframe

╔══════════════════════════════════════════════════════════════════════════════╗
║                              END OF REPORT                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        return report


# ============================================================================
# INTEGRATED PIPELINE
# ============================================================================

class PolypharmacyRecommenderSystem:
    """
    🏗️ Integrated Polypharmacy Risk-aware Drug Recommender System
    
    Combines all modules into a single pipeline:
    Drug Input → Interaction Engine → Severity Predictor → Alternative Finder → Explanation
    """
    
    def __init__(self, data_path):
        print("="*70)
        print("🏗️ INITIALIZING POLYPHARMACY RECOMMENDER SYSTEM")
        print("="*70)
        
        # Load data
        print(f"\n📂 Loading data from: {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(self.df):,} drug-drug interactions")
        
        # Initialize modules
        print("\n📦 Initializing modules...")
        self.drug_processor = DrugInputProcessor(self.df)
        self.interaction_engine = InteractionEngine(self.df)
        self.severity_predictor = SeverityPredictor(self.df)
        self.alternative_finder = AlternativeFinder(self.df, self.drug_processor)
        self.explanation_generator = ExplanationGenerator()
        
        # Train ML models
        print("\n🧠 Training ML models...")
        self.severity_predictor.prepare_features()
        self.severity_predictor.train_models()
        
        print("\n✅ System initialized successfully!")
    
    def analyze_prescription(self, drug_list):
        """
        Main entry point: Analyze a prescription for polypharmacy risks.
        
        Returns comprehensive analysis with:
        - Validated drugs
        - Detected interactions
        - Risk assessment
        - Alternative recommendations
        - Human-readable report
        """
        print("\n" + "="*70)
        print("🔍 ANALYZING PRESCRIPTION")
        print("="*70)
        
        # Step 1: Process drug input
        input_result = self.drug_processor.process_input(drug_list)
        valid_drugs = [d['input_name'] for d in input_result['validated_drugs']]
        
        if len(valid_drugs) < 2:
            return {"error": "Need at least 2 valid drugs for interaction analysis"}
        
        # Step 2: Detect interactions
        interaction_result = self.interaction_engine.detect_all_interactions(valid_drugs)
        
        # Step 3: Calculate risk score
        risk_score = 0
        severity_weights = {
            'Contraindicated interaction': 4,
            'Major interaction': 3,
            'Moderate interaction': 2,
            'Minor interaction': 1
        }
        
        for inter in interaction_result['interactions']:
            weight = severity_weights.get(inter['severity_label'], 1)
            risk_score += weight * inter['severity_confidence']
        
        # Step 4: Find alternatives for high-risk drugs
        alternatives = {}
        problematic_drugs = set()
        
        for inter in interaction_result['interactions']:
            if inter['severity_label'] in ['Contraindicated interaction', 'Major interaction']:
                problematic_drugs.add(inter['drug_1'])
                problematic_drugs.add(inter['drug_2'])
        
        for drug in problematic_drugs:
            other_drugs = [d for d in valid_drugs if d != drug]
            alts = self.alternative_finder.find_alternatives(drug, other_drugs)
            if alts:
                alternatives[drug] = alts
        
        # Step 5: Generate report
        report = self.explanation_generator.generate_polypharmacy_report(
            valid_drugs, 
            interaction_result['interactions'],
            risk_score,
            alternatives
        )
        
        print(report)
        
        return {
            'validated_drugs': input_result,
            'interactions': interaction_result,
            'risk_score': risk_score,
            'alternatives': alternatives,
            'report': report
        }


# ============================================================================
# COMPREHENSIVE DATA ANALYSIS
# ============================================================================

def run_comprehensive_analysis(data_path):
    """Run comprehensive analysis on the DDI dataset"""
    
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE DDI DATASET ANALYSIS")
    print("="*70)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\n✅ Loaded {len(df):,} drug-drug interactions")
    
    # ========== DATASET OVERVIEW ==========
    print("\n" + "-"*50)
    print("DATASET OVERVIEW")
    print("-"*50)
    print(f"Total DDI Records: {len(df):,}")
    print(f"Number of Features: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    
    all_drugs = set(df['drug_name_1'].unique()) | set(df['drug_name_2'].unique())
    print(f"Unique Drugs: {len(all_drugs):,}")
    
    # ========== SEVERITY ANALYSIS ==========
    print("\n" + "-"*50)
    print("SEVERITY DISTRIBUTION")
    print("-"*50)
    severity_counts = df['severity_label'].value_counts()
    for label, count in severity_counts.items():
        pct = count / len(df) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    # ========== DRUG CATEGORY ANALYSIS ==========
    print("\n" + "-"*50)
    print("DRUG CATEGORY ANALYSIS")
    print("-"*50)
    cv_count = df[df['is_cardiovascular_1'] | df['is_cardiovascular_2']].shape[0]
    at_count = df[df['is_antithrombotic_1'] | df['is_antithrombotic_2']].shape[0]
    print(f"  Cardiovascular interactions: {cv_count:,} ({cv_count/len(df)*100:.1f}%)")
    print(f"  Antithrombotic interactions: {at_count:,} ({at_count/len(df)*100:.1f}%)")
    
    # ========== CONFIDENCE STATISTICS ==========
    print("\n" + "-"*50)
    print("CONFIDENCE SCORE STATISTICS")
    print("-"*50)
    print(f"  Mean: {df['severity_confidence'].mean():.4f}")
    print(f"  Median: {df['severity_confidence'].median():.4f}")
    print(f"  Std Dev: {df['severity_confidence'].std():.4f}")
    print(f"  Min: {df['severity_confidence'].min():.4f}")
    print(f"  Max: {df['severity_confidence'].max():.4f}")
    
    # ========== TOP DRUGS ==========
    print("\n" + "-"*50)
    print("TOP 10 DRUGS BY INTERACTION COUNT")
    print("-"*50)
    all_drug_counts = pd.concat([
        df['drug_name_1'].value_counts(),
        df['drug_name_2'].value_counts()
    ]).groupby(level=0).sum().sort_values(ascending=False)
    
    for i, (drug, count) in enumerate(all_drug_counts.head(10).items(), 1):
        print(f"  {i}. {drug}: {count:,}")
    
    # ========== HIGH RISK COMBINATIONS ==========
    print("\n" + "-"*50)
    print("TOP 10 HIGHEST-RISK DRUG COMBINATIONS")
    print("-"*50)
    contraindicated = df[df['severity_label'] == 'Contraindicated interaction']
    high_risk = contraindicated.nlargest(10, 'severity_confidence')
    
    for i, (_, row) in enumerate(high_risk.iterrows(), 1):
        print(f"  {i}. {row['drug_name_1']} + {row['drug_name_2']}")
        print(f"     Confidence: {row['severity_confidence']:.4f}")
        print(f"     {row['interaction_description'][:60]}...")
    
    # ========== GENERATE FIGURES ==========
    print("\n" + "-"*50)
    print("GENERATING FIGURES")
    print("-"*50)
    
    # Figure 1: Severity Distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
    
    axes[0].pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%',
                colors=colors[:len(severity_counts)], startangle=90)
    axes[0].set_title('Figure 1a: DDI Severity Distribution', fontsize=14, fontweight='bold')
    
    bars = axes[1].bar(severity_counts.index, severity_counts.values, color=colors[:len(severity_counts)], edgecolor='black')
    axes[1].set_xlabel('Severity Label')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Figure 1b: DDI Severity Counts', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    for bar, count in zip(bars, severity_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                     f'{count:,}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figure1_severity_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved figure1_severity_distribution.png")
    
    # Figure 2: Confidence Distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].hist(df['severity_confidence'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['severity_confidence'].mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {df["severity_confidence"].mean():.3f}')
    axes[0, 0].set_xlabel('Confidence Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Figure 2a: Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    
    df.boxplot(column='severity_confidence', by='severity_label', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Severity Label')
    axes[0, 1].set_ylabel('Confidence Score')
    axes[0, 1].set_title('Figure 2b: Confidence by Severity', fontsize=14, fontweight='bold')
    plt.suptitle('')
    
    sns.violinplot(data=df, x='severity_label', y='severity_confidence', ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_xlabel('Severity Label')
    axes[1, 0].set_ylabel('Confidence Score')
    axes[1, 0].set_title('Figure 2c: Confidence Violin Plot', fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=15)
    
    for label in df['severity_label'].unique():
        subset = df[df['severity_label'] == label]['severity_confidence']
        sns.kdeplot(subset, ax=axes[1, 1], label=label, linewidth=2)
    axes[1, 1].set_xlabel('Confidence Score')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Figure 2d: Confidence KDE by Severity', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('figure2_confidence_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved figure2_confidence_distribution.png")
    
    # Figure 3: Top Drugs
    fig, ax = plt.subplots(figsize=(12, 8))
    top20 = all_drug_counts.head(20)
    colors_drugs = plt.cm.viridis(np.linspace(0, 1, 20))
    ax.barh(top20.index[::-1], top20.values[::-1], color=colors_drugs, edgecolor='black')
    ax.set_xlabel('Number of Interactions')
    ax.set_ylabel('Drug Name')
    ax.set_title('Figure 3: Top 20 Drugs by Interaction Count', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure3_top_drugs.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved figure3_top_drugs.png")
    
    # Figure 4: Interaction Type Analysis
    def categorize_interaction(desc):
        desc_lower = str(desc).lower()
        if 'anticoagulant' in desc_lower: return 'Anticoagulant Effect'
        elif 'bleeding' in desc_lower or 'hemorrhage' in desc_lower: return 'Bleeding Risk'
        elif 'therapeutic efficacy' in desc_lower: return 'Efficacy Change'
        elif 'metabolism' in desc_lower: return 'Metabolism Effect'
        elif 'serum concentration' in desc_lower: return 'Concentration Change'
        elif 'cardiotoxic' in desc_lower or 'arrhythmia' in desc_lower: return 'Cardiac Effect'
        else: return 'Other'
    
    df['interaction_type'] = df['interaction_description'].apply(categorize_interaction)
    int_counts = df['interaction_type'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors_int = plt.cm.tab20(np.linspace(0, 1, len(int_counts)))
    
    axes[0].pie(int_counts.values, labels=int_counts.index, autopct='%1.1f%%', colors=colors_int, startangle=90)
    axes[0].set_title('Figure 4a: Interaction Types Distribution', fontsize=14, fontweight='bold')
    
    axes[1].bar(int_counts.index, int_counts.values, color=colors_int, edgecolor='black')
    axes[1].set_xlabel('Interaction Type')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Figure 4b: Interaction Type Counts', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('figure4_interaction_types.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved figure4_interaction_types.png")
    
    # Figure 5: Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap_data = pd.crosstab(df['interaction_type'], df['severity_label'], normalize='index') * 100
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='Blues', ax=ax, cbar_kws={'label': 'Percentage (%)'})
    ax.set_title('Figure 5: Interaction Type vs Severity Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Severity Label')
    ax.set_ylabel('Interaction Type')
    plt.tight_layout()
    plt.savefig('figure5_heatmap.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved figure5_heatmap.png")
    
    # Figure 6: Risk by Drug Category
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cv_severity = pd.crosstab(df['is_cardiovascular_1'] | df['is_cardiovascular_2'], 
                              df['severity_label'], normalize='index') * 100
    cv_severity.index = ['Non-Cardiovascular', 'Cardiovascular']
    cv_severity.plot(kind='bar', ax=axes[0], colormap='RdYlGn_r', edgecolor='black')
    axes[0].set_xlabel('Drug Category')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_title('Figure 6a: Severity by Cardiovascular Involvement', fontsize=14, fontweight='bold')
    axes[0].legend(title='Severity', bbox_to_anchor=(1.02, 1))
    axes[0].tick_params(axis='x', rotation=0)
    
    at_severity = pd.crosstab(df['is_antithrombotic_1'] | df['is_antithrombotic_2'], 
                              df['severity_label'], normalize='index') * 100
    at_severity.index = ['Non-Antithrombotic', 'Antithrombotic']
    at_severity.plot(kind='bar', ax=axes[1], colormap='RdYlGn_r', edgecolor='black')
    axes[1].set_xlabel('Drug Category')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('Figure 6b: Severity by Antithrombotic Involvement', fontsize=14, fontweight='bold')
    axes[1].legend(title='Severity', bbox_to_anchor=(1.02, 1))
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('figure6_risk_by_category.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved figure6_risk_by_category.png")
    
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*70)
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Data path
    DATA_PATH = "ddi_cardio_or_antithrombotic_labeled (1).csv"
    
    # Run comprehensive analysis
    print("\n" + "🔬"*35)
    print("  AI-BASED POLYPHARMACY RISK-AWARE DRUG RECOMMENDER SYSTEM")
    print("🔬"*35)
    
    # Step 1: Comprehensive data analysis
    df = run_comprehensive_analysis(DATA_PATH)
    
    # Step 2: Initialize the recommender system
    print("\n")
    system = PolypharmacyRecommenderSystem(DATA_PATH)
    
    # Step 3: Test with example prescription
    print("\n" + "="*70)
    print("🧪 TESTING WITH EXAMPLE PRESCRIPTION")
    print("="*70)
    
    test_drugs = ['Warfarin', 'Aspirin', 'Metoprolol', 'Atorvastatin', 'Lisinopril']
    result = system.analyze_prescription(test_drugs)
    
    # Print model performance summary
    print("\n" + "="*70)
    print("📊 ML MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(system.severity_predictor.results.to_string(index=False))
    
    # Print feature importance
    print("\n" + "-"*50)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("-"*50)
    importance = system.severity_predictor.get_feature_importance()
    if importance is not None:
        print(importance.to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ SYSTEM DEMONSTRATION COMPLETE")
    print("="*70)
