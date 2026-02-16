"""
Severity Agent - Predicts and classifies drug interaction severity using ML
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from collections import Counter
import re
import pickle
import os

from .base_agent import BaseAgent, AgentResult, AgentStatus


class SeverityAgent(BaseAgent):
    """
    ⚠️ Severity Agent
    
    Responsible for:
    - ML-based severity prediction from interaction descriptions
    - Risk scoring and classification
    - Confidence assessment
    
    Architecture:
    ┌─────────────────────────────────────────┐
    │            Severity Agent               │
    ├─────────────────────────────────────────┤
    │  Input: Interactions from InteractionAgent │
    │  Output: Severity predictions & scores  │
    ├─────────────────────────────────────────┤
    │  - TF-IDF text vectorization            │
    │  - Ensemble ML classification           │
    │  - Risk score aggregation               │
    └─────────────────────────────────────────┘
    """
    
    SEVERITY_LEVELS = {
        'Contraindicated interaction': {'numeric': 4, 'risk': 'CRITICAL', 'color': '🔴'},
        'Major interaction': {'numeric': 3, 'risk': 'HIGH', 'color': '🟠'},
        'Moderate interaction': {'numeric': 2, 'risk': 'MODERATE', 'color': '🟡'},
        'Minor interaction': {'numeric': 1, 'risk': 'LOW', 'color': '🟢'}
    }
    
    SEVERITY_KEYWORDS = {
        'contraindicated': ['contraindicated', 'never use', 'do not use', 'fatal', 'death'],
        'major': ['serious', 'severe', 'dangerous', 'significant', 'life-threatening', 'hospitaliz'],
        'moderate': ['caution', 'monitor', 'adjust', 'moderate', 'avoid'],
        'minor': ['mild', 'minor', 'minimal', 'unlikely']
    }
    
    def __init__(self):
        super().__init__(
            name="SeverityAgent",
            description="Predicts and classifies drug interaction severity"
        )
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.model_trained = False
        
    def initialize(self, ddi_dataframe: pd.DataFrame = None, 
                   model_path: str = None, 
                   train_model: bool = False) -> bool:
        """Initialize with optional pre-trained model or training data"""
        print(f"⚠️ [{self.name}] Initializing...")
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        elif train_model and ddi_dataframe is not None:
            self._train_model(ddi_dataframe)
        else:
            # Use rule-based system by default
            print(f"📋 [{self.name}] Using rule-based severity classification")
        
        self._initialized = True
        print(f"✅ [{self.name}] Ready")
        return True
    
    def _train_model(self, df: pd.DataFrame):
        """Train ML model for severity prediction"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            
            print(f"🎓 [{self.name}] Training ML model...")
            
            # Prepare data
            X = df['interaction_description'].fillna('')
            y = df['severity_label']
            
            # Vectorize text
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X_vec = self.vectorizer.fit_transform(X)
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_enc = self.label_encoder.fit_transform(y)
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                X_vec, y_enc, test_size=0.2, random_state=42, stratify=y_enc
            )
            
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = self.model.score(X_test, y_test)
            print(f"📊 [{self.name}] Model accuracy: {accuracy:.2%}")
            
            self.model_trained = True
            
        except ImportError:
            print(f"⚠️ [{self.name}] sklearn not available, using rule-based system")
    
    def _load_model(self, path: str):
        """Load pre-trained model"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.model_trained = True
            print(f"📦 [{self.name}] Loaded model from {path}")
        except Exception as e:
            print(f"⚠️ [{self.name}] Failed to load model: {e}")
    
    def save_model(self, path: str):
        """Save trained model"""
        if self.model_trained:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'label_encoder': self.label_encoder
                }, f)
            print(f"💾 [{self.name}] Model saved to {path}")
    
    def validate_input(self, input_data: Dict[str, Any]) -> tuple:
        """Validate input contains interactions"""
        if 'interactions' not in input_data:
            return False, "Missing 'interactions' key"
        return True, ""
    
    def _rule_based_severity(self, description: str) -> Dict[str, Any]:
        """Rule-based severity classification"""
        description_lower = description.lower() if description else ""
        
        # Check keywords for each severity level
        for severity, keywords in self.SEVERITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in description_lower:
                    if severity == 'contraindicated':
                        return {
                            'predicted_severity': 'Contraindicated interaction',
                            'confidence': 0.85,
                            'method': 'rule_based',
                            'matched_keyword': keyword
                        }
                    elif severity == 'major':
                        return {
                            'predicted_severity': 'Major interaction',
                            'confidence': 0.80,
                            'method': 'rule_based',
                            'matched_keyword': keyword
                        }
                    elif severity == 'moderate':
                        return {
                            'predicted_severity': 'Moderate interaction',
                            'confidence': 0.75,
                            'method': 'rule_based',
                            'matched_keyword': keyword
                        }
                    elif severity == 'minor':
                        return {
                            'predicted_severity': 'Minor interaction',
                            'confidence': 0.70,
                            'method': 'rule_based',
                            'matched_keyword': keyword
                        }
        
        # Default to moderate if no keywords matched
        return {
            'predicted_severity': 'Moderate interaction',
            'confidence': 0.50,
            'method': 'default',
            'matched_keyword': None
        }
    
    def _ml_severity(self, description: str) -> Dict[str, Any]:
        """ML-based severity prediction"""
        if not self.model_trained:
            return self._rule_based_severity(description)
        
        X = self.vectorizer.transform([description])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        predicted_label = self.label_encoder.inverse_transform([pred])[0]
        confidence = float(max(proba))
        
        return {
            'predicted_severity': predicted_label,
            'confidence': confidence,
            'method': 'ml_model',
            'probabilities': dict(zip(self.label_encoder.classes_, proba.tolist()))
        }
    
    def predict_severity(self, description: str) -> Dict[str, Any]:
        """Predict severity for a single interaction"""
        if self.model_trained:
            return self._ml_severity(description)
        return self._rule_based_severity(description)
    
    def calculate_risk_score(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Calculate overall polypharmacy risk score"""
        if not interactions:
            return {
                'overall_score': 0,
                'risk_level': 'NO_RISK',
                'max_severity': None
            }
        
        # Weight by severity
        weights = {
            'Contraindicated interaction': 10,
            'Major interaction': 7,
            'Moderate interaction': 4,
            'Minor interaction': 1
        }
        
        total_weight = 0
        max_severity = 'Minor interaction'
        severity_order = ['Minor interaction', 'Moderate interaction', 
                         'Major interaction', 'Contraindicated interaction']
        
        for inter in interactions:
            sev = inter.get('severity_label') or inter.get('predicted_severity', 'Moderate interaction')
            total_weight += weights.get(sev, 4)
            if severity_order.index(sev) > severity_order.index(max_severity):
                max_severity = sev
        
        # Normalize score (0-100)
        max_possible = len(interactions) * 10
        score = min(100, (total_weight / max_possible) * 100) if max_possible > 0 else 0
        
        # Risk level
        if score >= 70 or max_severity == 'Contraindicated interaction':
            risk_level = 'CRITICAL'
        elif score >= 50 or max_severity == 'Major interaction':
            risk_level = 'HIGH'
        elif score >= 25:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        return {
            'overall_score': round(score, 2),
            'risk_level': risk_level,
            'max_severity': max_severity,
            'total_weighted_severity': total_weight,
            'interaction_count': len(interactions)
        }
    
    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Main processing: Analyze severity of all interactions
        """
        interactions = input_data['interactions']
        
        if not interactions:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                data={
                    'analyzed_interactions': [],
                    'risk_assessment': {
                        'overall_score': 0,
                        'risk_level': 'NO_INTERACTIONS',
                        'max_severity': None
                    }
                }
            )
        
        # Analyze each interaction
        analyzed = []
        for inter in interactions:
            # Use existing severity if available, otherwise predict
            if 'severity_label' in inter and inter['severity_label']:
                severity_info = {
                    'severity_label': inter['severity_label'],
                    'confidence': inter.get('severity_confidence', 1.0),
                    'method': 'database'
                }
            else:
                severity_info = self.predict_severity(inter.get('description', ''))
            
            analyzed.append({
                **inter,
                'severity_analysis': severity_info,
                'severity_info': self.SEVERITY_LEVELS.get(
                    inter.get('severity_label') or severity_info.get('predicted_severity'),
                    {'numeric': 2, 'risk': 'UNKNOWN', 'color': '⚪'}
                )
            })
        
        # Calculate overall risk
        risk_assessment = self.calculate_risk_score(analyzed)
        
        # Group by severity
        severity_groups = {}
        for inter in analyzed:
            sev = inter.get('severity_label') or inter['severity_analysis'].get('predicted_severity')
            if sev not in severity_groups:
                severity_groups[sev] = []
            severity_groups[sev].append(inter)
        
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                'analyzed_interactions': analyzed,
                'risk_assessment': risk_assessment,
                'severity_groups': severity_groups
            },
            metadata={
                'model_used': 'ml_model' if self.model_trained else 'rule_based'
            }
        )
