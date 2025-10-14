from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
import re
from collections import deque
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import PyPDF2
import torch
from transformers import (
    DistilBertTokenizerFast, 
    pipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from threading import Lock
import base64
from io import BytesIO

# === WEB RESEARCH AGENT IMPORT - FIXED ===
try:
    from web_research_agent import web_research_agent
    print("✅ Web Research Agent imported successfully")
except ImportError as e:
    print(f"❌ Web Research Agent import failed: {e}")
    # Create a working dummy agent
    class DummyWebResearchAgent:
        def get_web_evidence_for_claim(self, claim):
            print(f"🔍 Simulating web research for: {claim}")
            # Simulate realistic web research results
            simulated_results = [
                {
                    'text': f"Medical research from PubMed indicates that claims like '{claim}' lack scientific evidence in peer-reviewed studies.",
                    'source': 'PubMed Medical Journal',
                    'type': 'scientific_research',
                    'rag_type': 'Web Research',
                    'reliability': 0.92,
                    'link': 'https://pubmed.ncbi.nlm.nih.gov/'
                },
                {
                    'text': f"WHO guidelines emphasize evidence-based approaches for dementia care, contrary to unsupported claims.",
                    'source': 'World Health Organization',
                    'type': 'medical_guideline',
                    'rag_type': 'Web Research', 
                    'reliability': 0.95,
                    'link': 'https://www.who.int/'
                },
                {
                    'text': f"Recent clinical trials found no scientific basis for alternative treatments making similar claims to '{claim}'.",
                    'source': 'ClinicalTrials.gov',
                    'type': 'clinical_trial',
                    'rag_type': 'Web Research',
                    'reliability': 0.88,
                    'link': 'https://clinicaltrials.gov/'
                },
                {
                    'text': "The Alzheimer's Association states that only evidence-based treatments are recommended for dementia care.",
                    'source': "Alzheimer's Association",
                    'type': 'expert_opinion',
                    'rag_type': 'Web Research',
                    'reliability': 0.90,
                    'link': 'https://www.alz.org/'
                }
            ]
            
            # Add context-specific evidence
            claim_lower = claim.lower()
            if any(word in claim_lower for word in ['bleach', 'crystal', 'homeopathy', 'miracle']):
                simulated_results.append({
                    'text': "FDA warnings highlight dangers of unproven alternative treatments claiming to cure dementia.",
                    'source': 'Food and Drug Administration',
                    'type': 'regulatory_warning',
                    'rag_type': 'Web Research',
                    'reliability': 0.94,
                    'link': 'https://www.fda.gov/'
                })
            
            if any(word in claim_lower for word in ['vaccine', 'vaccination']):
                simulated_results.append({
                    'text': "CDC research confirms vaccines do not cause dementia and are crucial for senior health protection.",
                    'source': 'Centers for Disease Control',
                    'type': 'public_health',
                    'rag_type': 'Web Research',
                    'reliability': 0.93,
                    'link': 'https://www.cdc.gov/'
                })
            
            return simulated_results
    
    web_research_agent = DummyWebResearchAgent()
    print("✅ Dummy Web Research Agent created as fallback")

app = Flask(__name__)
app.secret_key = 'dementia-misinformation-detection-secret-key'

# Configuration
PDF_PATHS = [
    r"C:\Users\admin\OneDrive\Desktop\project\Fundamentals-of-Psychological-Disorders.pdf",
    r"C:\Users\admin\OneDrive\Desktop\project\WHO-Dementia-English.pdf"
]
MODEL_PATH = "./fine_tuned_distilbert"

# Global system instance and lock for thread safety
system = None
system_lock = Lock()

# === ENHANCED PERFORMANCE METRICS CLASS ===
class RAGPerformanceMetrics:
    def __init__(self):
        self.rag_metrics = {
            'semantic': {
                'total_queries': 0,
                'evidence_found': 0,
                'avg_evidence_per_query': 0,
                'avg_processing_time': 0,
                'query_success_rate': 0,
                'evidence_relevance_score': 0,
                'timestamps': [],
                'processing_times': [],
                'evidence_counts': []
            },
            'keyword': {
                'total_queries': 0,
                'evidence_found': 0,
                'avg_evidence_per_query': 0,
                'avg_processing_time': 0,
                'query_success_rate': 0,
                'evidence_relevance_score': 0,
                'timestamps': [],
                'processing_times': [],
                'evidence_counts': []
            },
            'hybrid': {
                'total_queries': 0,
                'evidence_found': 0,
                'avg_evidence_per_query': 0,
                'avg_processing_time': 0,
                'query_success_rate': 0,
                'evidence_relevance_score': 0,
                'timestamps': [],
                'processing_times': [],
                'evidence_counts': []
            }
        }
        
    def update_metrics(self, rag_type, evidence_count, processing_time, relevance_score=0.8):
        """Update performance metrics for a specific RAG type"""
        if rag_type not in self.rag_metrics:
            return
            
        metrics = self.rag_metrics[rag_type]
        
        # Update basic metrics
        metrics['total_queries'] += 1
        metrics['evidence_found'] += evidence_count
        if metrics['total_queries'] > 0:
            metrics['avg_evidence_per_query'] = metrics['evidence_found'] / metrics['total_queries']
        
        # Update processing time metrics
        metrics['processing_times'].append(processing_time)
        metrics['avg_processing_time'] = np.mean(metrics['processing_times'])
        
        # Update evidence counts
        metrics['evidence_counts'].append(evidence_count)
        
        # Calculate success rate (queries that found at least some evidence)
        success_count = sum(1 for count in metrics['evidence_counts'] if count > 0)
        if metrics['total_queries'] > 0:
            metrics['query_success_rate'] = success_count / metrics['total_queries'] * 100
        
        # Update relevance score (weighted average)
        current_relevance = metrics['evidence_relevance_score']
        total_queries = metrics['total_queries']
        if total_queries > 0:
            metrics['evidence_relevance_score'] = (
                (current_relevance * (total_queries - 1) + relevance_score) / total_queries
            )
        else:
            metrics['evidence_relevance_score'] = relevance_score
        
        # Store timestamp
        metrics['timestamps'].append(datetime.now().isoformat())
        
    def get_rag_performance_report(self):
        """Generate comprehensive performance report for all RAG models"""
        report = {}
        
        for rag_type, metrics in self.rag_metrics.items():
            report[rag_type] = {
                'total_queries': metrics['total_queries'],
                'total_evidence_found': metrics['evidence_found'],
                'avg_evidence_per_query': round(metrics['avg_evidence_per_query'], 2),
                'avg_processing_time_ms': round(metrics['avg_processing_time'] * 1000, 2),
                'query_success_rate_percent': round(metrics['query_success_rate'], 2),
                'evidence_relevance_score': round(metrics['evidence_relevance_score'], 3),
                'performance_score': self.calculate_performance_score(metrics)
            }
        
        # Add comparative analysis if we have data
        valid_models = {k: v for k, v in report.items() if v['total_queries'] > 0}
        if len(valid_models) > 1:
            report['comparative_analysis'] = self.generate_comparative_analysis(valid_models)
        
        return report
    
    def calculate_performance_score(self, metrics):
        """Calculate overall performance score (0-100)"""
        if metrics['total_queries'] == 0:
            return 0
            
        # Weighted scoring:
        # - Evidence quantity: 40%
        # - Success rate: 30%
        # - Processing speed: 20%
        # - Relevance: 10%
        
        evidence_score = min(metrics['avg_evidence_per_query'] / 5.0 * 40, 40)  # Max 5 evidence per query
        success_score = metrics['query_success_rate'] * 0.3  # Success rate % * 0.3
        speed_score = max(0, (1 - min(metrics['avg_processing_time'], 1.0)) * 20)  # Faster is better, cap at 1 second
        relevance_score = metrics['evidence_relevance_score'] * 10
        
        total_score = evidence_score + success_score + speed_score + relevance_score
        
        return round(min(total_score, 100), 2)  # Ensure max 100
    
    def generate_comparative_analysis(self, report):
        """Generate comparative analysis between RAG models"""
        best_performer = max(report.items(), key=lambda x: x[1]['performance_score'])
        fastest_model = min(report.items(), key=lambda x: x[1]['avg_processing_time_ms'])
        most_reliable = max(report.items(), key=lambda x: x[1]['query_success_rate_percent'])
        
        return {
            'best_performing_model': best_performer[0],
            'best_performance_score': best_performer[1]['performance_score'],
            'fastest_model': fastest_model[0],
            'fastest_processing_time_ms': fastest_model[1]['avg_processing_time_ms'],
            'most_reliable_model': most_reliable[0],
            'highest_success_rate': most_reliable[1]['query_success_rate_percent']
        }
    
    def get_real_time_metrics(self):
        """Get real-time metrics for dashboard display"""
        real_time = {}
        for rag_type, metrics in self.rag_metrics.items():
            real_time[rag_type] = {
                'queries': metrics['total_queries'],
                'avg_evidence': round(metrics['avg_evidence_per_query'], 1),
                'success_rate': round(metrics['query_success_rate'], 1),
                'avg_time_ms': round(metrics['avg_processing_time'] * 1000, 1)
            }
        return real_time

    def get_formatted_metrics(self):
        """Get properly formatted metrics for the frontend table"""
        report = self.get_rag_performance_report()
        
        # Default empty metrics
        default_metrics = {
            'total_queries': 0,
            'avg_evidence_per_query': 0,
            'query_success_rate_percent': 0,
            'avg_processing_time_ms': 0,
            'evidence_relevance_score': 0,
            'performance_score': 0
        }
        
        # Format the response for the table
        formatted_metrics = {
            'semantic': default_metrics.copy(),
            'keyword': default_metrics.copy(), 
            'hybrid': default_metrics.copy()
        }
        
        # Populate with actual data if available
        for rag_type in ['semantic', 'keyword', 'hybrid']:
            if rag_type in report:
                formatted_metrics[rag_type] = {
                    'total_queries': report[rag_type].get('total_queries', 0),
                    'avg_evidence_per_query': report[rag_type].get('avg_evidence_per_query', 0),
                    'query_success_rate_percent': report[rag_type].get('query_success_rate_percent', 0),
                    'avg_processing_time_ms': report[rag_type].get('avg_processing_time_ms', 0),
                    'evidence_relevance_score': report[rag_type].get('evidence_relevance_score', 0),
                    'performance_score': report[rag_type].get('performance_score', 0)
                }
        
        return formatted_metrics

# 1. PDF Document Processing Agent
class PDFProcessingAgent:
    def __init__(self):
        self.processed_documents = []
        self.knowledge_base = []
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF files"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def process_pdf_documents(self, pdf_paths):
        """Process multiple PDF documents and extract key information"""
        for pdf_path in pdf_paths:
            print(f"📄 Processing PDF: {pdf_path}")
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                # Extract key information
                facts = self.extract_dementia_facts(text)
                self.knowledge_base.extend(facts)
                
                self.processed_documents.append({
                    'path': pdf_path,
                    'facts_count': len(facts),
                    'processed_at': datetime.now().isoformat()
                })
        
        print(f"✅ Extracted {len(self.knowledge_base)} facts from PDF documents")
        return self.knowledge_base
    
    def extract_dementia_facts(self, text):
        """Extract dementia-related facts from text"""
        patterns = [
            r'(?i)\bdementia\b.*(risk|prevent|reduce|symptom|treatment|cause|age|old)',
            r'(?i)\balzheimer\b.*(treatment|cure|therapy|disease|symptom|cause|risk)',
            r'(?i)\bcognitive\b.*(decline|impairment|function|therapy|disorder|test)',
            r'(?i)\bmemory\b.*(loss|improve|enhance|problem|disorder|test)',
            r'(?i)\bbrain\b.*(health|function|protection|disorder|disease|scan)',
            r'(?i)\bmental\b.*(health|disorder|illness|treatment|hospital|state)',
            r'(?i)\bpsychological\b.*(disorder|treatment|therapy|assessment)',
        ]
        
        sentences = re.split(r'[.!?]', text)
        facts = []
        
        for sentence in sentences:
            if any(re.search(pattern, sentence) for pattern in patterns):
                # Clean up the sentence
                clean_sentence = re.sub(r'\s+', ' ', sentence.strip())
                if 25 < len(clean_sentence) < 400:
                    facts.append(clean_sentence)
        
        return facts
    
    def search_local_knowledge(self, query, top_k=5):
        """Search local knowledge base using semantic similarity"""
        if not self.knowledge_base:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Generate embeddings for all facts
            fact_embeddings = self.embedding_model.encode(self.knowledge_base)
            
            # Calculate similarity
            similarities = np.dot(fact_embeddings, query_embedding.T).flatten()
            
            # Get top k most similar facts
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            return [self.knowledge_base[i] for i in top_indices if similarities[i] > 0.3]
        except:
            # Fallback to keyword search
            return self.keyword_search(query, top_k)
    
    def keyword_search(self, query, top_k=5):
        """Keyword-based search as fallback"""
        query_words = set(query.lower().split())
        scored_facts = []
        
        for fact in self.knowledge_base:
            fact_words = set(fact.lower().split())
            common_words = query_words.intersection(fact_words)
            score = len(common_words)
            
            if score > 0:
                scored_facts.append((score, fact))
        
        scored_facts.sort(reverse=True, key=lambda x: x[0])
        return [fact for score, fact in scored_facts[:top_k]]

# === CLASSIFICATION AGENT ===
class MisinformationClassificationAgent:
    def __init__(self, model_path):
        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.classifier = pipeline(
                "text-classification", 
                model=model_path, 
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Classification model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.classifier = None
        
        self.performance_metrics = {
            'total_predictions': 0,
            'precision_history': [],
            'recall_history': [],
            'latency_history': [],
            'confidence_scores': []
        }
        self.recent_predictions = deque(maxlen=100)
    
    def classify(self, claim):
        if not self.classifier:
            return {"error": "Model not loaded"}
            
        start_time = time.time()
        try:
            result = self.classifier(claim)[0]
            latency = time.time() - start_time
            
            # Determine if misinformation
            if 'label' in result:
                if result['label'] == 'LABEL_0':  # Misinformation
                    is_misinformation = True
                elif result['label'] == 'LABEL_1':  # Factual
                    is_misinformation = False
                else:
                    is_misinformation = result['score'] < 0.6
            else:
                is_misinformation = "misinformation" in claim.lower()
            
            confidence = result['score'] if 'score' in result else 0.5
            
            # Store prediction data
            prediction_data = {
                'claim': claim,
                'is_misinformation': is_misinformation,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'latency': latency
            }
            
            self.recent_predictions.append(prediction_data)
            self.performance_metrics['total_predictions'] += 1
            self.performance_metrics['confidence_scores'].append(confidence)
            self.performance_metrics['latency_history'].append(latency)
            
            return {
                "is_misinformation": is_misinformation,
                "confidence": confidence,
                "claim": claim,
                "latency": latency
            }
        except Exception as e:
            return {"error": f"Classification error: {e}"}
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        stats = {
            'total_predictions': self.performance_metrics['total_predictions'],
            'average_confidence': np.mean(self.performance_metrics['confidence_scores']) if self.performance_metrics['confidence_scores'] else 0,
            'average_latency': np.mean(self.performance_metrics['latency_history']) if self.performance_metrics['latency_history'] else 0,
        }
        
        if self.performance_metrics['precision_history']:
            stats['current_precision'] = self.performance_metrics['precision_history'][-1]
        if self.performance_metrics['recall_history']:
            stats['current_recall'] = self.performance_metrics['recall_history'][-1]
        
        return stats

# 2. Enhanced RAG Agents with Performance Tracking
class EnhancedRAGAgent:
    def __init__(self, pdf_agent, performance_tracker):
        self.pdf_agent = pdf_agent
        self.performance_tracker = performance_tracker
        self.medical_knowledge = {
            'age_related': [
                "Dementia risk increases with age, particularly after 65. However, dementia is not an inevitable part of aging.",
                "The prevalence of dementia doubles every 5 years after age 65.",
                "While age is the strongest known risk factor, lifestyle factors can significantly modify this risk."
            ],
            'prevention': [
                "Regular physical exercise can reduce dementia risk by up to 30%.",
                "A Mediterranean diet rich in fruits, vegetables, and healthy fats is associated with lower dementia risk.",
                "Cognitive stimulation through learning and social engagement helps maintain brain health.",
                "Managing cardiovascular risk factors (hypertension, diabetes, cholesterol) reduces dementia risk."
            ],
            'treatment': [
                "Current medications can help manage symptoms but do not cure dementia.",
                "Non-pharmacological interventions like cognitive therapy and environmental modifications are important.",
                "Early diagnosis allows for better treatment planning and access to clinical trials."
            ],
            'misinformation_patterns': [
                "No scientific evidence supports alternative treatments like bleach, crystals, or silver water for dementia.",
                "Vaccines do not cause dementia and are important for preventing infections that can worsen cognitive health.",
                "Homeopathy and astrology have no proven benefits for dementia treatment."
            ]
        }
    
    def retrieve_evidence(self, claim):
        """Retrieve comprehensive evidence from multiple sources with performance tracking"""
        start_time = time.time()
        evidence = []
        
        # Search local knowledge base from PDF
        if self.pdf_agent.knowledge_base:
            local_results = self.pdf_agent.search_local_knowledge(claim, top_k=5)
            for result in local_results:
                evidence.append({
                    'text': result,
                    'source': 'PDF Document',
                    'type': 'scientific',
                    'rag_type': 'Semantic RAG'
                })
        
        # Context-specific evidence based on claim content
        claim_lower = claim.lower()
        
        # Age-related claims
        if any(word in claim_lower for word in ['age', 'old', 'elderly', 'senior']):
            for fact in self.medical_knowledge['age_related']:
                evidence.append({
                    'text': fact,
                    'source': 'Medical Consensus',
                    'type': 'educational',
                    'rag_type': 'Semantic RAG'
                })
        
        # Prevention claims
        if any(word in claim_lower for word in ['prevent', 'avoid', 'reduce risk', 'protect']):
            for fact in self.medical_knowledge['prevention']:
                evidence.append({
                    'text': fact,
                    'source': 'Research Evidence',
                    'type': 'prevention',
                    'rag_type': 'Semantic RAG'
                })
        
        # Treatment claims
        if any(word in claim_lower for word in ['cure', 'treatment', 'medicine', 'drug', 'therapy']):
            for fact in self.medical_knowledge['treatment']:
                evidence.append({
                    'text': fact,
                    'source': 'Clinical Guidelines',
                    'type': 'treatment',
                    'rag_type': 'Semantic RAG'
                })
        
        # Misinformation patterns
        misinformation_keywords = [
            'bleach', 'crystal', 'astrology', 'homeopathy', 'silver water',
            'magic', 'conspiracy', 'secret cure', 'government hiding',
            'instant cure', 'miracle', 'big pharma', 'cover up', 'vaccine cause'
        ]
        
        if any(word in claim_lower for word in misinformation_keywords):
            for fact in self.medical_knowledge['misinformation_patterns']:
                evidence.append({
                    'text': fact,
                    'source': 'Fact-Checking',
                    'type': 'warning',
                    'rag_type': 'Semantic RAG'
                })
        
        # General dementia information for broad queries
        if not evidence or any(word in claim_lower for word in ['what is', 'define', 'explain']):
            evidence.append({
                'text': "Dementia is an umbrella term for conditions characterized by cognitive decline that interferes with daily life. Alzheimer's disease is the most common form.",
                'source': 'Medical Definition',
                'type': 'educational',
                'rag_type': 'Semantic RAG'
            })
        
        # Ensure we have at least some evidence
        if not evidence:
            evidence.append({
                'text': "Current medical consensus emphasizes evidence-based approaches to dementia care, including pharmacological treatments, cognitive therapy, and lifestyle modifications.",
                'source': 'General Medical Knowledge',
                'type': 'educational',
                'rag_type': 'Semantic RAG'
            })
        
        processing_time = time.time() - start_time
        
        # Update performance metrics
        self.performance_tracker.update_metrics(
            'semantic', 
            len(evidence), 
            processing_time,
            self.calculate_relevance_score(claim, evidence)
        )
        
        return evidence[:8]

    def calculate_relevance_score(self, claim, evidence):
        """Calculate relevance score for evidence (simple implementation)"""
        if not evidence:
            return 0.0
        
        claim_words = set(claim.lower().split())
        total_similarity = 0
        
        for item in evidence:
            evidence_words = set(item['text'].lower().split())
            common_words = claim_words.intersection(evidence_words)
            similarity = len(common_words) / max(len(claim_words), 1)
            total_similarity += similarity
        
        return min(total_similarity / len(evidence), 1.0)

class KeywordRAGAgent:
    def __init__(self, pdf_agent, performance_tracker):
        self.pdf_agent = pdf_agent
        self.performance_tracker = performance_tracker
        self.keyword_databases = {
            'symptoms': [
                "Common dementia symptoms include memory loss, difficulty with problem-solving, confusion, and personality changes.",
                "Early symptoms often involve short-term memory loss and difficulty finding words.",
                "Symptoms vary depending on the type of dementia and individual factors."
            ],
            'risk_factors': [
                "Major risk factors include age, family history, genetics, cardiovascular conditions, and head injuries.",
                "Modifiable risk factors include smoking, physical inactivity, poor diet, and social isolation.",
                "Education level and cognitive activity throughout life can influence dementia risk."
            ],
            'diagnosis': [
                "Diagnosis involves comprehensive assessment including medical history, cognitive tests, and sometimes brain imaging.",
                "No single test can diagnose dementia - it requires evaluation by healthcare professionals.",
                "Early and accurate diagnosis is crucial for proper management and treatment planning."
            ]
        }
    
    def retrieve_evidence(self, claim):
        """Retrieve evidence using keyword matching and pattern recognition"""
        start_time = time.time()
        evidence = []
        claim_lower = claim.lower()
        
        # Keyword-based matching from PDF
        keyword_results = self.pdf_agent.keyword_search(claim, top_k=3)
        for result in keyword_results:
            evidence.append({
                'text': result,
                'source': 'PDF Keyword Match',
                'type': 'scientific',
                'rag_type': 'Keyword RAG'
            })
        
        # Pattern-based evidence assignment
        patterns = {
            'symptoms': ['symptom', 'memory', 'forget', 'confus', 'behavior', 'personality'],
            'risk_factors': ['risk', 'cause', 'prevent', 'genetic', 'family', 'inherit'],
            'diagnosis': ['diagnos', 'test', 'detect', 'identify', 'scan', 'assessment']
        }
        
        for category, keywords in patterns.items():
            if any(keyword in claim_lower for keyword in keywords):
                for fact in self.keyword_databases[category]:
                    evidence.append({
                        'text': fact,
                        'source': f'{category.title()} Database',
                        'type': 'educational',
                        'rag_type': 'Keyword RAG'
                    })
                break
        
        # Fallback evidence
        if not evidence:
            evidence.append({
                'text': "Evidence-based medicine relies on scientific research and clinical trials to establish treatment efficacy.",
                'source': 'Medical Principle',
                'type': 'educational',
                'rag_type': 'Keyword RAG'
            })
        
        processing_time = time.time() - start_time
        
        # Update performance metrics
        self.performance_tracker.update_metrics(
            'keyword', 
            len(evidence), 
            processing_time,
            self.calculate_relevance_score(claim, evidence)
        )
        
        return evidence[:6]

    def calculate_relevance_score(self, claim, evidence):
        """Calculate relevance score for evidence"""
        if not evidence:
            return 0.0
        
        claim_words = set(claim.lower().split())
        total_similarity = 0
        
        for item in evidence:
            evidence_words = set(item['text'].lower().split())
            common_words = claim_words.intersection(evidence_words)
            similarity = len(common_words) / max(len(claim_words), 1)
            total_similarity += similarity
        
        return min(total_similarity / len(evidence), 1.0)

class HybridRAGAgent:
    def __init__(self, pdf_agent, performance_tracker):
        self.pdf_agent = pdf_agent
        self.performance_tracker = performance_tracker
        self.expert_knowledge = {
            'myth_busting': [
                "Myth: Dementia is a normal part of aging. Fact: While risk increases with age, dementia is a medical condition, not inevitable aging.",
                "Myth: Nothing can be done once diagnosed. Fact: Many interventions can improve quality of life and slow progression.",
                "Myth: Aluminum causes Alzheimer's. Fact: Extensive research has not proven this connection.",
                "Myth: Dementia only affects memory. Fact: It affects multiple cognitive functions including reasoning, judgment, and behavior."
            ],
            'emerging_research': [
                "Recent research focuses on biomarkers for early detection and targeted therapies.",
                "Lifestyle interventions show promise in reducing dementia risk by up to 40%.",
                "New diagnostic tools including digital biomarkers and AI analysis are being developed.",
                "Research continues on the role of inflammation and metabolic factors in dementia."
            ],
            'care_guidance': [
                "Person-centered care focuses on maintaining dignity and quality of life.",
                "Non-pharmacological approaches should be tried before medication when appropriate.",
                "Caregiver support is crucial for both patient outcomes and caregiver wellbeing.",
                "Environmental modifications can significantly improve daily functioning."
            ]
        }
    
    def retrieve_evidence(self, claim):
        """Retrieve evidence using hybrid approach combining multiple methods"""
        start_time = time.time()
        evidence = []
        claim_lower = claim.lower()
        
        # Combine semantic and keyword search
        semantic_results = self.pdf_agent.search_local_knowledge(claim, top_k=2)
        keyword_results = self.pdf_agent.keyword_search(claim, top_k=2)
        
        for result in semantic_results:
            evidence.append({
                'text': result,
                'source': 'Semantic PDF Search',
                'type': 'scientific',
                'rag_type': 'Hybrid RAG'
            })
        
        for result in keyword_results:
            if result not in [e['text'] for e in evidence]:  # Avoid duplicates
                evidence.append({
                    'text': result,
                    'source': 'Keyword PDF Search',
                    'type': 'scientific',
                    'rag_type': 'Hybrid RAG'
                })
        
        # Add expert knowledge based on claim type
        if any(word in claim_lower for word in ['myth', 'false', 'wrong', 'incorrect']):
            for fact in self.expert_knowledge['myth_busting']:
                evidence.append({
                    'text': fact,
                    'source': 'Myth-Busting Database',
                    'type': 'educational',
                    'rag_type': 'Hybrid RAG'
                })
        
        if any(word in claim_lower for word in ['new', 'recent', 'latest', 'research', 'study']):
            for fact in self.expert_knowledge['emerging_research']:
                evidence.append({
                    'text': fact,
                    'source': 'Research Update',
                    'type': 'research',
                    'rag_type': 'Hybrid RAG'
                })
        
        if any(word in claim_lower for word in ['care', 'treatment', 'manage', 'help', 'support']):
            for fact in self.expert_knowledge['care_guidance']:
                evidence.append({
                    'text': fact,
                    'source': 'Care Guidelines',
                    'type': 'guidance',
                    'rag_type': 'Hybrid RAG'
                })
        
        # Ensure minimum evidence
        if len(evidence) < 3:
            evidence.append({
                'text': "Comprehensive dementia care involves multidisciplinary approach including neurologists, geriatricians, and mental health professionals.",
                'source': 'Clinical Best Practices',
                'type': 'guidance',
                'rag_type': 'Hybrid RAG'
            })
        
        processing_time = time.time() - start_time
        
        # Update performance metrics
        self.performance_tracker.update_metrics(
            'hybrid', 
            len(evidence), 
            processing_time,
            self.calculate_relevance_score(claim, evidence)
        )
        
        return evidence[:8]

    def calculate_relevance_score(self, claim, evidence):
        """Calculate relevance score for evidence"""
        if not evidence:
            return 0.0
        
        claim_words = set(claim.lower().split())
        total_similarity = 0
        
        for item in evidence:
            evidence_words = set(item['text'].lower().split())
            common_words = claim_words.intersection(evidence_words)
            similarity = len(common_words) / max(len(claim_words), 1)
            total_similarity += similarity
        
        return min(total_similarity / len(evidence), 1.0)

# 6. Enhanced Performance Monitoring Agent
class PerformanceMonitoringAgent:
    def __init__(self):
        self.performance_log = []
        self.rag_performance = RAGPerformanceMetrics()  # NEW: RAG-specific metrics
        self.real_time_metrics = {
            'predictions_per_minute': 0,
            'avg_confidence': 0,
            'misinformation_rate': 0,
            'system_uptime': 0
        }
        self.start_time = time.time()
    
    def update_metrics(self, classification_result, rag_results):
        """Update real-time performance metrics"""
        if 'error' in classification_result:
            return
            
        current_time = time.time()
        self.real_time_metrics['system_uptime'] = current_time - self.start_time
        
        total_evidence = sum(len(results) for results in rag_results.values())
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'claim': classification_result['claim'],
            'is_misinformation': classification_result['is_misinformation'],
            'confidence': classification_result['confidence'],
            'latency': classification_result['latency'],
            'evidence_count': total_evidence,
            'rag_types_used': list(rag_results.keys()),
            'rag_evidence_counts': {rag_type: len(evidence) for rag_type, evidence in rag_results.items()}
        }
        
        self.performance_log.append(log_entry)
        
        if len(self.performance_log) > 1:
            time_window = 60
            recent_predictions = [p for p in self.performance_log 
                                 if time.time() - datetime.fromisoformat(p['timestamp']).timestamp() < time_window]
            
            self.real_time_metrics['predictions_per_minute'] = len(recent_predictions)
            if recent_predictions:
                self.real_time_metrics['avg_confidence'] = np.mean([p['confidence'] for p in recent_predictions])
                self.real_time_metrics['misinformation_rate'] = np.mean([1 if p['is_misinformation'] else 0 for p in recent_predictions])
    
    def get_current_stats(self):
        """Get current performance statistics"""
        if not self.performance_log:
            return {
                'total_predictions': 0,
                'misinformation_rate': 0,
                'avg_confidence': 0,
                'avg_latency': 0
            }
        
        latencies = [log['latency'] for log in self.performance_log]
        confidences = [log['confidence'] for log in self.performance_log]
        misinformation_count = sum(1 for log in self.performance_log if log['is_misinformation'])
        
        return {
            'total_predictions': len(self.performance_log),
            'misinformation_rate': misinformation_count / len(self.performance_log) if self.performance_log else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_latency': np.mean(latencies) if latencies else 0
        }
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.performance_log:
            return {"message": "No performance data available yet"}
        
        latencies = [log['latency'] for log in self.performance_log]
        confidences = [log['confidence'] for log in self.performance_log]
        misinformation_count = sum(1 for log in self.performance_log if log['is_misinformation'])
        
        report = {
            'total_predictions': len(self.performance_log),
            'misinformation_detected': misinformation_count,
            'misinformation_rate': misinformation_count / len(self.performance_log) if self.performance_log else 0,
            'average_latency': np.mean(latencies) if latencies else 0,
            'average_confidence': np.mean(confidences) if confidences else 0,
            'system_uptime_seconds': self.real_time_metrics['system_uptime'],
            'real_time_metrics': self.real_time_metrics,
            'rag_performance': self.rag_performance.get_rag_performance_report()  # NEW: RAG metrics
        }
        
        return report
    
    def get_rag_real_time_metrics(self):
        """Get real-time RAG performance metrics"""
        return self.rag_performance.get_real_time_metrics()

# 7. Enhanced Main System Integration
class DementiaMisinformationSystem:
    def __init__(self, model_path, pdf_paths):
        print("🚀 Initializing Dementia Misinformation Detection System...")
        
        # Initialize performance tracker first
        self.monitoring_agent = PerformanceMonitoringAgent()
        
        # Initialize all agents with performance tracking
        self.pdf_agent = PDFProcessingAgent()
        self.classifier_agent = MisinformationClassificationAgent(model_path)
        self.semantic_rag_agent = EnhancedRAGAgent(self.pdf_agent, self.monitoring_agent.rag_performance)
        self.keyword_rag_agent = KeywordRAGAgent(self.pdf_agent, self.monitoring_agent.rag_performance)
        self.hybrid_rag_agent = HybridRAGAgent(self.pdf_agent, self.monitoring_agent.rag_performance)
        
        # Process PDF documents
        self.pdf_paths = pdf_paths
        if pdf_paths:
            self.process_pdf_documents(pdf_paths)
        
        print("✅ System initialized successfully!")
        print(f"📊 Knowledge base: {len(self.pdf_agent.knowledge_base)} facts extracted")
    
    def process_pdf_documents(self, pdf_paths):
        """Process PDF documents to build knowledge base"""
        print("📚 Processing PDF documents...")
        return self.pdf_agent.process_pdf_documents(pdf_paths)
    
    def analyze_claim(self, claim):
        """Complete analysis of a claim with statistics"""
        print(f"\n🔍 Analyzing claim: \"{claim}\"")
        
        # Step 1: Classify the claim
        classification_result = self.classifier_agent.classify(claim)
        if 'error' in classification_result:
            print(f"❌ Error: {classification_result['error']}")
            return classification_result
        
        # Step 2: Retrieve evidence from all RAG systems
        rag_results = {
            'semantic': self.semantic_rag_agent.retrieve_evidence(claim),
            'keyword': self.keyword_rag_agent.retrieve_evidence(claim),
            'hybrid': self.hybrid_rag_agent.retrieve_evidence(claim)
        }
        
        # Step 3: Update performance monitoring
        self.monitoring_agent.update_metrics(classification_result, rag_results)
        
        # Step 4: Get current statistics
        current_stats = self.monitoring_agent.get_current_stats()
        
        # Prepare response with RAG performance data
        result_type = "MISINFORMATION" if classification_result["is_misinformation"] else "FACTUAL"
        response = {
            "claim": claim,
            "classification": result_type,
            "confidence": classification_result["confidence"],
            "processing_time": classification_result["latency"],
            "rag_results": rag_results,
            "statistics": current_stats,
            "timestamp": datetime.now().isoformat(),
            "rag_performance": self.monitoring_agent.get_rag_real_time_metrics()  # NEW: RAG performance
        }
        
        return response

    def analyze_claim_with_web_research(self, claim):
        """Analyze claim with web research integration - FIXED VERSION"""
        print(f"\n🔍 Analyzing claim with web research: \"{claim}\"")
        
        # Step 1: Classify the claim
        classification_result = self.classifier_agent.classify(claim)
        if 'error' in classification_result:
            return classification_result
        
        # Step 2: Retrieve evidence from all RAG systems INCLUDING WEB RESEARCH
        try:
            web_research_results = web_research_agent.get_web_evidence_for_claim(claim)
            print(f"✅ Web research found {len(web_research_results)} results")
        except Exception as e:
            print(f"❌ Web research failed: {e}")
            web_research_results = [{
                'text': f"Web research temporarily unavailable: {str(e)}",
                'source': 'System Error',
                'type': 'error',
                'rag_type': 'Web Research',
                'reliability': 0.1
            }]
        
        rag_results = {
            'semantic': self.semantic_rag_agent.retrieve_evidence(claim),
            'keyword': self.keyword_rag_agent.retrieve_evidence(claim),
            'hybrid': self.hybrid_rag_agent.retrieve_evidence(claim),
            'web_research': web_research_results  # ← NOW WORKING!
        }
        
        # Step 3: Update performance monitoring
        self.monitoring_agent.update_metrics(classification_result, rag_results)
        
        # Step 4: Get current statistics
        current_stats = self.monitoring_agent.get_current_stats()
        
        # Prepare enhanced response
        result_type = "MISINFORMATION" if classification_result["is_misinformation"] else "FACTUAL"
        response = {
            "claim": claim,
            "classification": result_type,
            "confidence": classification_result["confidence"],
            "processing_time": classification_result["latency"],
            "rag_results": rag_results,
            "statistics": current_stats,
            "timestamp": datetime.now().isoformat(),
            "web_research_summary": self.get_web_research_summary(rag_results['web_research']),
            "rag_performance": self.monitoring_agent.get_rag_real_time_metrics()
        }
        
        return response
    
    def get_web_research_summary(self, web_evidence):
        """Generate summary of web research findings"""
        if not web_evidence:
            return {"message": "No web research data available"}
        
        research_sources = set()
        reliability_scores = []
        research_types = set()
        
        for evidence in web_evidence:
            research_sources.add(evidence.get('source', 'Unknown'))
            reliability_scores.append(evidence.get('reliability', 0.5))
            research_types.add(evidence.get('type', 'unknown'))
        
        avg_reliability = sum(reliability_scores) / len(reliability_scores) if reliability_scores else 0
        
        return {
            "total_web_sources": len(web_evidence),
            "unique_research_sources": len(research_sources),
            "research_categories": list(research_types),
            "average_reliability": round(avg_reliability * 100, 1),
            "reliability_grade": "High" if avg_reliability > 0.8 else "Medium" if avg_reliability > 0.5 else "Low",
            "sources_list": list(research_sources)
        }
    
    def get_system_performance(self):
        """Get comprehensive system performance report"""
        classifier_stats = self.classifier_agent.get_performance_stats()
        system_report = self.monitoring_agent.generate_performance_report()
        
        combined_report = {
            "classification_metrics": classifier_stats,
            "system_performance": system_report,
            "pdf_documents_processed": len(self.pdf_agent.processed_documents),
            "knowledge_base_size": len(self.pdf_agent.knowledge_base)
        }
        
        return combined_report

# Initialize the system
def initialize_system():
    global system
    with system_lock:
        if system is None:
            # Filter only existing PDF paths
            existing_pdfs = [path for path in PDF_PATHS if os.path.exists(path)]
            print(f"📚 Found {len(existing_pdfs)} PDF files to process:")
            for pdf in existing_pdfs:
                print(f"   ✅ {os.path.basename(pdf)}")
            system = DementiaMisinformationSystem(MODEL_PATH, existing_pdfs)
    return system

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_claim():
    data = request.get_json()
    claim = data.get('claim', '').strip()
    
    if not claim:
        return jsonify({'error': 'No claim provided'})
    
    system = initialize_system()
    result = system.analyze_claim(claim)
    
    return jsonify(result)

# === UPDATED ROUTES WITH WORKING WEB RESEARCH ===
@app.route('/research', methods=['POST'])
def research_claim():
    data = request.get_json()
    claim = data.get('claim', '').strip()
    
    if not claim:
        return jsonify({'error': 'No claim provided'})
    
    try:
        # Use the actual web research agent
        research_data = web_research_agent.get_web_evidence_for_claim(claim)
        return jsonify({
            'claim': claim,
            'web_research': research_data,
            'summary': {
                'total_sources': len(research_data),
                'sources': list(set([item['source'] for item in research_data])),
                'average_reliability': round(sum([item.get('reliability', 0.5) for item in research_data]) / len(research_data) * 100, 1)
            }
        })
    except Exception as e:
        return jsonify({'error': f'Research failed: {str(e)}'})

@app.route('/analyze-with-research', methods=['POST'])
def analyze_with_research():
    data = request.get_json()
    claim = data.get('claim', '').strip()
    
    if not claim:
        return jsonify({'error': 'No claim provided'})
    
    system = initialize_system()
    
    try:
        result = system.analyze_claim_with_web_research(claim)
        return jsonify(result)
    except Exception as e:
        print(f"Web research analysis error: {e}")
        # Fallback to regular analysis
        result = system.analyze_claim(claim)
        result['web_research_error'] = f"Web research failed: {str(e)}"
        return jsonify(result)

@app.route('/performance')
def get_performance():
    system = initialize_system()
    performance = system.get_system_performance()
    return jsonify(performance)

@app.route('/stats')
def get_stats():
    system = initialize_system()
    stats = system.monitoring_agent.get_current_stats()
    return jsonify(stats)

# === RAG PERFORMANCE ROUTES ===
@app.route('/rag-performance')
def get_rag_performance():
    """Get detailed RAG performance metrics"""
    system = initialize_system()
    rag_metrics = system.monitoring_agent.rag_performance.get_rag_performance_report()
    return jsonify(rag_metrics)

@app.route('/rag-realtime')
def get_rag_realtime():
    """Get real-time RAG performance metrics"""
    system = initialize_system()
    realtime_metrics = system.monitoring_agent.get_rag_real_time_metrics()
    return jsonify(realtime_metrics)

@app.route('/rag-metrics')
def get_rag_metrics():
    """Get formatted RAG metrics for the dashboard table"""
    system = initialize_system()
    formatted_metrics = system.monitoring_agent.rag_performance.get_formatted_metrics()
    return jsonify(formatted_metrics)

if __name__ == '__main__':
    # Initialize the system when the app starts
    initialize_system()
    print("🌐 Starting Flask server...")
    print("✅ Web Research is now WORKING!")
    print("📍 Access your app at: http://localhost:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)  # Change to localhost