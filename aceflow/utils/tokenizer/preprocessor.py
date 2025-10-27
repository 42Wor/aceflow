import re
import html
import unicodedata
from typing import List, Callable, Optional

try:
    import contractions
    HAS_CONTRACTIONS = True
except ImportError:
    HAS_CONTRACTIONS = False
    print("Warning: contractions library not installed. Using fallback contraction expansion.")

class Preprocessor:
    def __init__(self, language: str = 'english'):
        self.language = language
        self.pipeline = self._get_default_pipeline()
        self.config = {
            'language': language,
            'steps': [step.__name__ for step in self.pipeline],
            'has_contractions_lib': HAS_CONTRACTIONS
        }
        
        # Common contractions mapping for fallback
        self.contractions_map = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot", 
            "can't've": "cannot have", "'cause": "because", "could've": "could have",
            "couldn't": "could not", "couldn't've": "could not have", 
            "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
            "he'll": "he will", "he'll've": "he will have", "he's": "he is",
            "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
            "how's": "how is", "i'd": "i would", "i'd've": "i would have",
            "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
            "i've": "i have", "isn't": "is not", "it'd": "it would",
            "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
            "it's": "it is", "let's": "let us", "ma'am": "madam",
            "mayn't": "may not", "might've": "might have", "mightn't": "might not",
            "mightn't've": "might not have", "must've": "must have",
            "mustn't": "must not", "mustn't've": "must not have",
            "needn't": "need not", "needn't've": "need not have",
            "o'clock": "of the clock", "oughtn't": "ought not",
            "oughtn't've": "ought not have", "shan't": "shall not",
            "sha'n't": "shall not", "shan't've": "shall not have",
            "she'd": "she would", "she'd've": "she would have",
            "she'll": "she will", "she'll've": "she will have", "she's": "she is",
            "should've": "should have", "shouldn't": "should not",
            "shouldn't've": "should not have", "so've": "so have",
            "so's": "so as", "that'd": "that would", "that'd've": "that would have",
            "that's": "that is", "there'd": "there would",
            "there'd've": "there would have", "there's": "there is",
            "they'd": "they would", "they'd've": "they would have",
            "they'll": "they will", "they'll've": "they will have",
            "they're": "they are", "they've": "they have", "to've": "to have",
            "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
            "we'll": "we will", "we'll've": "we will have", "we're": "we are",
            "we've": "we have", "weren't": "were not", "what'll": "what will",
            "what'll've": "what will have", "what're": "what are",
            "what's": "what is", "what've": "what have", "when's": "when is",
            "when've": "when have", "where'd": "where did", "where's": "where is",
            "where've": "where have", "who'll": "who will",
            "who'll've": "who will have", "who's": "who is", "who've": "who have",
            "why's": "why is", "why've": "why have", "will've": "will have",
            "won't": "will not", "won't've": "will not have",
            "would've": "would have", "wouldn't": "would not",
            "wouldn't've": "would not have", "y'all": "you all",
            "y'all'd": "you all would", "y'all'd've": "you all would have",
            "y'all're": "you all are", "y'all've": "you all have",
            "you'd": "you would", "you'd've": "you would have",
            "you'll": "you will", "you'll've": "you will have",
            "you're": "you are", "you've": "you have"
        }
    
    def _get_default_pipeline(self) -> List[Callable]:
        """Get default preprocessing pipeline"""
        return [
            self.decode_html,
            self.normalize_unicode,
            self.expand_contractions,
            self.clean_special_chars,
            self.normalize_whitespace,
            self.lowercase
        ]
    
    def decode_html(self, text: str) -> str:
        """Decode HTML entities"""
        return html.unescape(text)
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions like don't -> do not"""
        if HAS_CONTRACTIONS:
            try:
                return contractions.fix(text)
            except Exception as e:
                print(f"Warning: Contractions library failed: {e}. Using fallback.")
                return self._expand_contractions_fallback(text)
        else:
            return self._expand_contractions_fallback(text)
    
    def _expand_contractions_fallback(self, text: str) -> str:
        """Fallback contraction expansion using regex"""
        import re
        
        def replace(match):
            contraction = match.group(0).lower()
            return self.contractions_map.get(contraction, contraction)
        
        # Pattern to match common contractions
        pattern = re.compile(r"\b(" + "|".join(self.contractions_map.keys()) + r")\b", re.IGNORECASE)
        return pattern.sub(replace, text)
    
    def clean_special_chars(self, text: str) -> str:
        """Remove or replace special characters"""
        # Keep basic punctuation, remove other special chars
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def lowercase(self, text: str) -> str:
        """Convert to lowercase"""
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        """Remove all punctuation"""
        return re.sub(r'[^\w\s]', '', text)
    
    def remove_numbers(self, text: str) -> str:
        """Remove numbers"""
        return re.sub(r'\d+', '', text)
    
    def custom_replace(self, patterns: List[tuple]) -> Callable:
        """Create custom replacement function"""
        def replace_func(text: str) -> str:
            for pattern, replacement in patterns:
                text = re.sub(pattern, replacement, text)
            return text
        return replace_func
    
    def add_step(self, step: Callable, position: Optional[int] = None):
        """Add a step to the pipeline"""
        if position is None:
            self.pipeline.append(step)
        else:
            self.pipeline.insert(position, step)
        self.config['steps'] = [step.__name__ for step in self.pipeline]
    
    def remove_step(self, step_name: str):
        """Remove a step from the pipeline"""
        self.pipeline = [step for step in self.pipeline if step.__name__ != step_name]
        self.config['steps'] = [step.__name__ for step in self.pipeline]
    
    def process(self, text: str) -> str:
        """Process text through the pipeline"""
        if not isinstance(text, str):
            return ""
        
        for step in self.pipeline:
            text = step(text)
        
        return text
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts"""
        return [self.process(text) for text in texts]
    
    def set_language(self, language: str):
        """Set language for language-specific processing"""
        self.language = language
        self.config['language'] = language
    
    def info(self) -> dict:
        """Get preprocessor information"""
        return {
            'language': self.language,
            'pipeline_steps': self.config['steps'],
            'has_contractions_lib': HAS_CONTRACTIONS,
            'description': f"Preprocessor for {self.language} with {len(self.pipeline)} steps"
        }
    
    def __call__(self, text: str) -> str:
        """Make preprocessor callable"""
        return self.process(text)
    
    def __str__(self) -> str:
        info = self.info()
        return f"Preprocessor(language='{info['language']}', steps={info['pipeline_steps']})"