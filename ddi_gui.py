#!/usr/bin/env python3
"""
Simple GUI for Drug-Drug Interaction Analysis
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from typing import Optional

# Import the DDI analysis components
try:
    from kg_polypharmacy_risk import PolypharmacyRiskAssessor, KnowledgeGraphLoader
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

try:
    from ddi_chatbot import DDIChatbot, LLMClient
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False


class DDIAnalyzerGUI:
    """Simple GUI for DDI Risk Analysis"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Drug-Drug Interaction Analyzer")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Initialize components
        self.assessor: Optional[PolypharmacyRiskAssessor] = None
        self.chatbot: Optional[DDIChatbot] = None
        self.loading = False
        
        # Create UI
        self._create_menu()
        self._create_main_frame()
        self._create_status_bar()
        
        # Load KG in background
        self._load_knowledge_graph()
    
    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Clear", command=self._clear_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_main_frame(self):
        """Create main application frame"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Drug-Drug Interaction Analyzer",
            font=('Helvetica', 16, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Enter Drugs", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Drug input
        ttk.Label(input_frame, text="Drug names (comma-separated):").pack(anchor=tk.W)
        
        self.drug_entry = ttk.Entry(input_frame, width=70, font=('Helvetica', 11))
        self.drug_entry.pack(fill=tk.X, pady=5)
        self.drug_entry.bind('<Return>', lambda e: self._analyze_drugs())
        
        # Example label
        example_label = ttk.Label(
            input_frame, 
            text="Example: warfarin, aspirin, ibuprofen",
            foreground='gray'
        )
        example_label.pack(anchor=tk.W)
        
        # Buttons frame
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.analyze_btn = ttk.Button(
            btn_frame, 
            text="Analyze Interactions",
            command=self._analyze_drugs
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = ttk.Button(
            btn_frame,
            text="Clear",
            command=self._clear_all
        )
        self.clear_btn.pack(side=tk.LEFT)
        
        # Quick examples
        examples_frame = ttk.Frame(btn_frame)
        examples_frame.pack(side=tk.RIGHT)
        
        ttk.Label(examples_frame, text="Quick examples:").pack(side=tk.LEFT, padx=(0, 5))
        
        examples = [
            ("High Risk", "warfarin, aspirin, ibuprofen"),
            ("Moderate", "metformin, lisinopril"),
            ("Low Risk", "acetaminophen, vitamin d")
        ]
        
        for name, drugs in examples:
            btn = ttk.Button(
                examples_frame,
                text=name,
                command=lambda d=drugs: self._set_drugs(d)
            )
            btn.pack(side=tk.LEFT, padx=2)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=('Courier', 10),
            height=15
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for colored output
        self.results_text.tag_configure('title', font=('Helvetica', 12, 'bold'))
        self.results_text.tag_configure('high', foreground='red', font=('Courier', 10, 'bold'))
        self.results_text.tag_configure('moderate', foreground='orange', font=('Courier', 10, 'bold'))
        self.results_text.tag_configure('low', foreground='green', font=('Courier', 10, 'bold'))
        self.results_text.tag_configure('info', foreground='blue')
        self.results_text.tag_configure('drug', foreground='purple', font=('Courier', 10, 'bold'))
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar(value="Loading knowledge graph...")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _load_knowledge_graph(self):
        """Load knowledge graph in background thread"""
        if not KG_AVAILABLE:
            self.status_var.set("Error: Knowledge graph module not available")
            return
        
        def load():
            self.loading = True
            self._update_status("Loading knowledge graph...")
            try:
                loader = KnowledgeGraphLoader()
                kg = loader.load_from_drugbank_csv()
                self.assessor = PolypharmacyRiskAssessor(kg)
                self._update_status(f"Ready - {len(kg.drugs)} drugs, {len(kg.interactions)} interactions loaded")
            except Exception as e:
                self._update_status(f"Error loading KG: {str(e)[:50]}")
            finally:
                self.loading = False
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _update_status(self, message: str):
        """Thread-safe status update"""
        self.root.after(0, lambda: self.status_var.set(message))
    
    def _set_drugs(self, drugs: str):
        """Set drug entry field"""
        self.drug_entry.delete(0, tk.END)
        self.drug_entry.insert(0, drugs)
        self._analyze_drugs()
    
    def _clear_all(self):
        """Clear all fields"""
        self.drug_entry.delete(0, tk.END)
        self.results_text.delete(1.0, tk.END)
    
    def _analyze_drugs(self):
        """Analyze drug interactions"""
        if self.loading:
            messagebox.showwarning("Loading", "Knowledge graph is still loading. Please wait.")
            return
        
        if not self.assessor:
            messagebox.showerror("Error", "Knowledge graph not loaded.")
            return
        
        # Get drugs from entry
        drug_input = self.drug_entry.get().strip()
        if not drug_input:
            messagebox.showwarning("Input Required", "Please enter drug names.")
            return
        
        # Parse drugs
        drugs = [d.strip().lower() for d in drug_input.replace(';', ',').split(',') if d.strip()]
        
        if len(drugs) < 2:
            messagebox.showwarning("Input Required", "Please enter at least 2 drugs to check interactions.")
            return
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Analyze in background
        self._update_status("Analyzing interactions...")
        
        def analyze():
            try:
                result = self.assessor.assess_regimen(drugs)
                self.root.after(0, lambda: self._display_results(drugs, result))
                self._update_status("Analysis complete")
            except Exception as e:
                self.root.after(0, lambda: self._display_error(str(e)))
                self._update_status("Analysis failed")
        
        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()
    
    def _display_results(self, drugs: list, result):
        """Display analysis results"""
        self.results_text.delete(1.0, tk.END)
        
        # Title
        self.results_text.insert(tk.END, "DRUG-DRUG INTERACTION ANALYSIS\n", 'title')
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Drugs analyzed
        self.results_text.insert(tk.END, "Drugs Analyzed: ", 'info')
        self.results_text.insert(tk.END, f"{', '.join(drugs)}\n\n", 'drug')
        
        # Risk assessment
        self.results_text.insert(tk.END, "RISK ASSESSMENT\n", 'title')
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        
        risk_level = result.risk_level.upper()
        risk_tag = 'low'
        if risk_level in ['HIGH', 'CRITICAL']:
            risk_tag = 'high'
        elif risk_level == 'MODERATE':
            risk_tag = 'moderate'
        
        self.results_text.insert(tk.END, f"Risk Level: ")
        self.results_text.insert(tk.END, f"{risk_level}\n", risk_tag)
        self.results_text.insert(tk.END, f"Risk Score: {result.overall_risk_score:.2f}\n\n")
        
        # Interactions
        self.results_text.insert(tk.END, "INTERACTIONS FOUND\n", 'title')
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        
        if result.ddi_pairs:
            for ddi in result.ddi_pairs:
                severity = ddi.get('severity', 'Unknown')
                drug1 = ddi.get('drug1', 'Unknown')
                drug2 = ddi.get('drug2', 'Unknown')
                description = ddi.get('description', '')
                
                # Color code by severity
                sev_tag = 'low'
                if severity.lower() in ['contraindicated', 'major']:
                    sev_tag = 'high'
                elif severity.lower() == 'moderate':
                    sev_tag = 'moderate'
                
                self.results_text.insert(tk.END, f"\n{drug1} + {drug2}\n", 'drug')
                self.results_text.insert(tk.END, f"  Severity: ")
                self.results_text.insert(tk.END, f"{severity}\n", sev_tag)
                if description:
                    self.results_text.insert(tk.END, f"  Description: {description[:100]}...\n" if len(description) > 100 else f"  Description: {description}\n")
        else:
            self.results_text.insert(tk.END, "No significant interactions found.\n", 'low')
        
        # Summary
        self.results_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.results_text.insert(tk.END, f"Total interactions: {len(result.ddi_pairs)}\n")
    
    def _display_error(self, error: str):
        """Display error message"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error: {error}\n", 'high')
    
    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Drug-Drug Interaction Analyzer\n\n"
            "A knowledge graph-based tool for analyzing\n"
            "potential drug interactions and assessing\n"
            "polypharmacy risk.\n\n"
            "Data source: DrugBank\n"
            "Version: 1.0"
        )


def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Set theme
    style = ttk.Style()
    available_themes = style.theme_names()
    if 'clam' in available_themes:
        style.theme_use('clam')
    
    app = DDIAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
