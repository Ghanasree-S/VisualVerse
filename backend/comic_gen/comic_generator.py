"""
Comic Generator Module
Generates comic strips from narrative text using story segmentation and image generation
"""

import os
import base64
from typing import Dict, Any, List
import requests
from io import BytesIO


class ComicGenerator:
    """
    Comic Strip Generator
    
    Pipeline:
    1. Segment story into scenes/beats
    2. Extract characters and settings for each scene
    3. Generate scene descriptions (prompts for image generation)
    4. Generate images using Stable Diffusion / DALL-E / or other models
    5. Combine panels into comic strip layout
    """
    
    def __init__(self):
        """Initialize the comic generator"""
        self.api_key = os.getenv("STABILITY_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.use_placeholder = True  # Use placeholder images if no API key
        
    async def generate(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comic strip from preprocessed text
        
        Returns:
            Dict with:
            - title: Story title
            - summary: Brief summary
            - panels: List of panel dicts with caption, image_url, characters, setting
        """
        text = preprocessed.get("original_text", "")
        sentences = preprocessed.get("sentences", [])
        characters = preprocessed.get("characters", [])
        locations = preprocessed.get("locations", [])
        
        # Generate title
        title = self._generate_title(text, preprocessed)
        
        # Generate summary
        summary = self._generate_summary(sentences)
        
        # Segment into panels (4-6 panels typically)
        panels = self._segment_into_panels(sentences, characters, locations)
        
        # Generate images for each panel
        for panel in panels:
            panel["image_url"] = await self._generate_panel_image(panel)
        
        return {
            "title": title,
            "summary": summary,
            "panels": panels
        }
    
    def _generate_title(self, text: str, preprocessed: Dict[str, Any]) -> str:
        """Generate a title for the comic"""
        characters = preprocessed.get("characters", [])
        
        # Use first character's name if available
        if characters:
            return f"The Story of {characters[0]}"
        
        # Extract first significant noun phrase
        noun_phrases = preprocessed.get("noun_phrases", [])
        if noun_phrases:
            return f"A Tale of {noun_phrases[0].title()}"
        
        # Default title
        first_words = text.split()[:5]
        return " ".join(first_words) + "..."
    
    def _generate_summary(self, sentences: List[str]) -> str:
        """Generate a brief summary"""
        if not sentences:
            return "A visual story unfolds..."
        
        # Use first sentence as summary
        summary = sentences[0]
        if len(summary) > 150:
            summary = summary[:147] + "..."
        
        return summary
    
    def _segment_into_panels(self, sentences: List[str], 
                            characters: List[str], 
                            locations: List[str]) -> List[Dict[str, Any]]:
        """
        Segment story into comic panels
        
        Creates 4-6 panels with:
        - Opening scene
        - Rising action (1-2 panels)
        - Climax
        - Resolution
        """
        num_panels = min(max(4, len(sentences) // 2), 6)
        
        if len(sentences) < num_panels:
            # If too few sentences, use each sentence as a panel
            panels = []
            for i, sent in enumerate(sentences):
                panels.append(self._create_panel(i + 1, sent, characters, locations))
            return panels
        
        # Distribute sentences across panels
        sentences_per_panel = len(sentences) // num_panels
        panels = []
        
        for i in range(num_panels):
            start_idx = i * sentences_per_panel
            end_idx = start_idx + sentences_per_panel if i < num_panels - 1 else len(sentences)
            
            panel_text = " ".join(sentences[start_idx:end_idx])
            panels.append(self._create_panel(i + 1, panel_text, characters, locations))
        
        return panels
    
    def _create_panel(self, panel_num: int, text: str, 
                     characters: List[str], locations: List[str]) -> Dict[str, Any]:
        """Create a single panel structure"""
        # Generate image prompt from text
        prompt = self._generate_image_prompt(text, characters, locations)
        
        # Create caption (shorter version for display)
        caption = text if len(text) <= 100 else text[:97] + "..."
        
        return {
            "id": f"panel_{panel_num}",
            "panel_number": panel_num,
            "caption": caption,
            "full_text": text,
            "prompt": prompt,
            "characters": characters,
            "setting": locations[0] if locations else "Unknown location",
            "image_url": None  # Will be filled by image generation
        }
    
    def _generate_image_prompt(self, text: str, 
                               characters: List[str], 
                               locations: List[str]) -> str:
        """
        Generate a detailed image prompt for the panel
        
        Format: Comic book style, [scene description], [characters], [setting], [mood]
        """
        # Base style
        style = "Comic book art style, vibrant colors, dynamic composition, "
        
        # Scene description (simplified from text)
        scene = text[:200] if len(text) > 200 else text
        
        # Characters
        char_desc = ""
        if characters:
            char_desc = f"featuring {', '.join(characters[:2])}, "
        
        # Location
        loc_desc = ""
        if locations:
            loc_desc = f"set in {locations[0]}, "
        
        # Mood/atmosphere
        mood = self._detect_mood(text)
        mood_desc = f"{mood} atmosphere"
        
        prompt = f"{style}{char_desc}{loc_desc}{scene}, {mood_desc}"
        
        return prompt
    
    def _detect_mood(self, text: str) -> str:
        """Detect the mood/atmosphere of the text"""
        text_lower = text.lower()
        
        # Check for mood indicators
        if any(word in text_lower for word in ["happy", "joy", "laugh", "smile", "excited"]):
            return "cheerful and bright"
        elif any(word in text_lower for word in ["sad", "cry", "tears", "lonely", "grief"]):
            return "melancholic and somber"
        elif any(word in text_lower for word in ["angry", "rage", "fight", "battle", "war"]):
            return "intense and dramatic"
        elif any(word in text_lower for word in ["fear", "dark", "scary", "horror", "terror"]):
            return "dark and mysterious"
        elif any(word in text_lower for word in ["love", "romance", "heart", "kiss"]):
            return "romantic and warm"
        elif any(word in text_lower for word in ["adventure", "journey", "discover", "explore"]):
            return "adventurous and exciting"
        else:
            return "neutral and balanced"
    
    async def _generate_panel_image(self, panel: Dict[str, Any]) -> str:
        """
        Generate image for a panel
        Returns fast SVG placeholder (can be upgraded to use Stable Diffusion locally)
        """
        # Use instant placeholder for fast loading
        return self._get_placeholder_image(panel["panel_number"], panel.get("caption", ""))
    
    def _get_placeholder_image(self, panel_number: int, caption: str = "") -> str:
        """
        Generate a beautiful SVG placeholder image
        Fast and works without external API
        """
        colors = [
            ("#FF6B6B", "#C44D4D"),  # Red
            ("#4ECDC4", "#36A89F"),  # Teal
            ("#45B7D1", "#2E8DA8"),  # Blue
            ("#96CEB4", "#6BAF8F"),  # Green
            ("#FFEAA7", "#D4C680"),  # Yellow
            ("#DDA0DD", "#B87AB8"),  # Purple
        ]
        
        color, dark_color = colors[(panel_number - 1) % len(colors)]
        
        # Shorten caption for display
        short_caption = caption[:50] + "..." if len(caption) > 50 else caption
        
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
            <defs>
                <linearGradient id="bg{panel_number}" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:{color}"/>
                    <stop offset="100%" style="stop-color:{dark_color}"/>
                </linearGradient>
            </defs>
            <rect width="512" height="512" fill="url(#bg{panel_number})"/>
            <rect x="10" y="10" width="492" height="492" fill="none" stroke="white" stroke-width="4" rx="10"/>
            <text x="256" y="180" font-family="Comic Sans MS, cursive, sans-serif" font-size="72" fill="white" text-anchor="middle" opacity="0.9">🎬</text>
            <text x="256" y="280" font-family="Arial, sans-serif" font-size="36" fill="white" text-anchor="middle" font-weight="bold">PANEL {panel_number}</text>
            <text x="256" y="380" font-family="Arial, sans-serif" font-size="16" fill="white" text-anchor="middle" opacity="0.8">{short_caption}</text>
            <text x="256" y="470" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle" opacity="0.5">VisualVerse Comic Generator</text>
        </svg>'''
        
        # Return as base64 encoded SVG
        import base64
        svg_bytes = svg.encode('utf-8')
        b64 = base64.b64encode(svg_bytes).decode('utf-8')
        return f"data:image/svg+xml;base64,{b64}"
    
    def generate_comic_layout(self, panels: List[Dict[str, Any]], 
                             layout: str = "grid") -> Dict[str, Any]:
        """
        Generate final comic layout configuration
        
        Layouts:
        - grid: 2x2 or 2x3 grid
        - vertical: Single column
        - manga: Right-to-left reading
        """
        num_panels = len(panels)
        
        if layout == "grid":
            if num_panels <= 4:
                rows, cols = 2, 2
            else:
                rows, cols = 2, 3
        elif layout == "vertical":
            rows, cols = num_panels, 1
        elif layout == "manga":
            rows, cols = 2, 2  # Will be reversed in frontend
        else:
            rows, cols = 2, 2
        
        return {
            "layout": layout,
            "rows": rows,
            "cols": cols,
            "panel_order": list(range(len(panels))),
            "reading_direction": "rtl" if layout == "manga" else "ltr"
        }
