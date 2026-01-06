"""
Text Preprocessor Module
Handles tokenization, POS tagging, NER using NLTK (no spaCy required)
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
from typing import Dict, List, Any
import re

# Download required NLTK data
def download_nltk_data():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('corpora/words', 'words')
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_data()


class TextPreprocessor:
    """
    Text preprocessing pipeline using NLTK
    Handles: Tokenization, POS Tagging, NER, Sentence splitting
    """
    
    def __init__(self, spacy_model: str = None):
        """Initialize the preprocessor"""
        self._ready = True
        self.stop_words = set(stopwords.words('english'))
    
    def is_ready(self) -> bool:
        """Check if preprocessor is ready"""
        return self._ready
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Main preprocessing function that returns structured text data
        """
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Tokenize into sentences
        sentences = sent_tokenize(cleaned_text)
        
        # Tokenize into words and POS tag
        all_tokens = []
        all_pos_tags = []
        
        for sent in sentences:
            tokens = word_tokenize(sent)
            pos_tags = pos_tag(tokens)
            
            for word, tag in pos_tags:
                all_tokens.append({
                    "text": word,
                    "lemma": word.lower(),
                    "pos": self._convert_pos_tag(tag),
                    "tag": tag,
                    "is_stop": word.lower() in self.stop_words,
                    "is_punct": not word.isalnum()
                })
        
        # Extract named entities
        entities = self._extract_entities(cleaned_text)
        
        # Extract noun phrases
        noun_phrases = self._extract_noun_phrases(cleaned_text)
        
        # Extract verbs
        verbs = [t for t in all_tokens if t["pos"] == "VERB"]
        
        # Extract characters and locations from entities
        characters = [e["text"] for e in entities if e["label"] == "PERSON"]
        locations = [e["text"] for e in entities if e["label"] in ["GPE", "LOC"]]
        
        # Segment into paragraphs
        paragraphs = self._segment_paragraphs(text)
        
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "sentences": sentences,
            "tokens": all_tokens,
            "entities": entities,
            "noun_phrases": noun_phrases,
            "verbs": verbs,
            "dependencies": [],  # Simplified - no dependency parsing without spaCy
            "characters": list(set(characters)),
            "locations": list(set(locations)),
            "paragraphs": paragraphs,
            "word_count": len([t for t in all_tokens if not t["is_punct"]]),
            "sentence_count": len(sentences)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
        return text.strip()
    
    def _convert_pos_tag(self, tag: str) -> str:
        """Convert Penn Treebank tags to universal tags"""
        tag_map = {
            'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
            'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
            'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
            'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
            'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON', 'WP$': 'PRON',
            'DT': 'DET', 'PDT': 'DET', 'WDT': 'DET',
            'IN': 'ADP', 'TO': 'PART', 'RP': 'PART',
            'CC': 'CCONJ', '.': 'PUNCT', ',': 'PUNCT', ':': 'PUNCT',
        }
        return tag_map.get(tag, 'X')
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using NLTK NER"""
        entities = []
        
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            iob_tags = tree2conlltags(named_entities)
            
            current_entity = []
            current_type = None
            start_pos = 0
            
            for i, (word, pos, tag) in enumerate(iob_tags):
                if tag.startswith('B-'):
                    if current_entity:
                        entities.append({
                            "text": " ".join(current_entity),
                            "label": current_type,
                            "start": start_pos,
                            "end": start_pos + len(" ".join(current_entity))
                        })
                    current_entity = [word]
                    current_type = tag[2:]
                    start_pos = text.find(word)
                elif tag.startswith('I-'):
                    current_entity.append(word)
                else:
                    if current_entity:
                        entities.append({
                            "text": " ".join(current_entity),
                            "label": current_type,
                            "start": start_pos,
                            "end": start_pos + len(" ".join(current_entity))
                        })
                    current_entity = []
                    current_type = None
            
            if current_entity:
                entities.append({
                    "text": " ".join(current_entity),
                    "label": current_type,
                    "start": start_pos,
                    "end": start_pos + len(" ".join(current_entity))
                })
                
        except Exception as e:
            print(f"NER extraction error: {e}")
        
        return entities
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases using POS patterns"""
        noun_phrases = []
        
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            current_np = []
            for word, tag in pos_tags:
                if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                    current_np.append(word)
                elif tag in ['JJ', 'JJR', 'JJS'] and not current_np:
                    current_np.append(word)
                else:
                    if current_np:
                        noun_phrases.append(" ".join(current_np))
                        current_np = []
            
            if current_np:
                noun_phrases.append(" ".join(current_np))
                
        except Exception as e:
            print(f"Noun phrase extraction error: {e}")
        
        return noun_phrases
    
    def _segment_paragraphs(self, text: str) -> List[str]:
        """Segment text into paragraphs"""
        paragraphs = re.split(r'\n\n+|\r\n\r\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def get_story_beats(self, text: str) -> List[Dict[str, Any]]:
        """Segment text into story beats for comic generation"""
        processed = self.process(text)
        sentences = processed["sentences"]
        
        beats = []
        current_beat = []
        
        for i, sentence in enumerate(sentences):
            current_beat.append(sentence)
            
            if len(current_beat) >= 2 or i == len(sentences) - 1:
                beat_text = " ".join(current_beat)
                beats.append({
                    "id": len(beats) + 1,
                    "text": beat_text,
                    "characters": [],
                    "locations": [],
                    "actions": []
                })
                current_beat = []
        
        return beats
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords"""
        processed = self.process(text)
        
        content_words = [
            token["lemma"]
            for token in processed["tokens"]
            if token["pos"] in ["NOUN", "VERB", "ADJ", "PROPN"]
            and not token["is_stop"]
            and len(token["text"]) > 2
        ]
        
        word_freq = {}
        for word in content_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
