"""
Feature Extraction Module for Construction Planning Assistant
Extracts intelligent features from project descriptions to enhance planning
"""
import re
from typing import Dict, List, Any


class FeatureExtractor:
    """Extracts features from project descriptions for intelligent planning"""
    
    def __init__(self):
        # Feature definitions with their impacts
        self.feature_definitions = {
            # Quality & Luxury Features
            'luxury': {
                'keywords': ['luxury', 'premium', 'high-end', 'deluxe', 'upscale', 'elite'],
                'cost_multiplier': 1.5,
                'extra_days': 15,
                'extra_tasks': [
                    'Install premium fixtures and fittings',
                    'Apply high-end finishes and materials',
                    'Install smart home automation system',
                    'Add custom cabinetry and millwork'
                ],
                'description': 'Luxury finishes and premium materials'
            },
            'modern': {
                'keywords': ['modern', 'contemporary', 'smart', 'automated', 'tech', 'digital'],
                'cost_multiplier': 1.2,
                'extra_days': 10,
                'extra_tasks': [
                    'Install smart wiring and home automation',
                    'Set up modular kitchen with modern appliances',
                    'Install energy-efficient lighting systems',
                    'Add contemporary design elements'
                ],
                'description': 'Modern amenities and smart home features'
            },
            'eco': {
                'keywords': ['eco', 'green', 'sustainable', 'environmental', 'solar', 'rainwater'],
                'cost_multiplier': 1.15,
                'extra_days': 12,
                'extra_tasks': [
                    'Install solar panel system',
                    'Set up rainwater harvesting system',
                    'Install energy-efficient insulation',
                    'Add green roofing materials'
                ],
                'description': 'Eco-friendly and sustainable features'
            },
            'basic': {
                'keywords': ['basic', 'simple', 'budget', 'economical', 'minimal'],
                'cost_multiplier': 0.85,
                'extra_days': -5,
                'extra_tasks': [
                    'Standard finishes and fixtures',
                    'Basic electrical and plumbing installation',
                    'Simple flooring and paint'
                ],
                'description': 'Budget-friendly basic construction'
            },
            # Specific Features
            'basement': {
                'keywords': ['basement', 'cellar', 'underground'],
                'cost_multiplier': 1.3,
                'extra_days': 20,
                'extra_tasks': [
                    'Excavate basement area',
                    'Install basement waterproofing',
                    'Construct basement walls and flooring',
                    'Install basement drainage system'
                ],
                'description': 'Basement construction'
            },
            'garage': {
                'keywords': ['garage', 'parking', 'carport'],
                'cost_multiplier': 1.1,
                'extra_days': 8,
                'extra_tasks': [
                    'Construct garage foundation',
                    'Build garage structure and roofing',
                    'Install garage door and opener',
                    'Add garage flooring and drainage'
                ],
                'description': 'Garage or parking structure'
            },
            'pool': {
                'keywords': ['pool', 'swimming', 'swimming pool'],
                'cost_multiplier': 1.25,
                'extra_days': 15,
                'extra_tasks': [
                    'Excavate pool area',
                    'Install pool structure and waterproofing',
                    'Set up filtration and pumping systems',
                    'Add pool decking and safety features'
                ],
                'description': 'Swimming pool construction'
            },
            'garden': {
                'keywords': ['garden', 'landscaping', 'outdoor', 'yard'],
                'cost_multiplier': 1.05,
                'extra_days': 5,
                'extra_tasks': [
                    'Prepare garden soil and drainage',
                    'Install irrigation system',
                    'Add landscaping elements and plants',
                    'Construct garden pathways'
                ],
                'description': 'Landscaping and garden features'
            }
        }
    
    def extract_features(self, description: str) -> Dict[str, Any]:
        """
        Extract features from project description
        
        Args:
            description: Project description string
            
        Returns:
            Dictionary with extracted features and their impacts
        """
        if not description:
            return self._get_default_features()
        
        description_lower = description.lower()
        detected_features = []
        total_cost_multiplier = 1.0
        total_extra_days = 0
        all_extra_tasks = []
        features_detected = []
        
        # Check each feature definition
        for feature_name, feature_config in self.feature_definitions.items():
            if self._contains_keywords(description_lower, feature_config['keywords']):
                detected_features.append(feature_name)
                
                # Apply impacts
                total_cost_multiplier *= feature_config['cost_multiplier']
                total_extra_days += feature_config['extra_days']
                all_extra_tasks.extend(feature_config['extra_tasks'])
                features_detected.append({
                    'feature': feature_name,
                    'description': feature_config['description'],
                    'cost_impact': f"+{int((feature_config['cost_multiplier'] - 1) * 100)}%" if feature_config['cost_multiplier'] > 1 else f"{int((feature_config['cost_multiplier'] - 1) * 100)}%",
                    'days_impact': feature_config['extra_days']
                })
        
        # Ensure minimum values
        total_cost_multiplier = max(0.7, total_cost_multiplier)  # Minimum 70% of base cost
        total_extra_days = max(-10, total_extra_days)  # Minimum -10 days
        
        return {
            'cost_multiplier': total_cost_multiplier,
            'extra_days': total_extra_days,
            'extra_tasks': all_extra_tasks,
            'features_detected': features_detected,
            'detected_features': detected_features,
            'has_features': len(detected_features) > 0
        }
    
    def _contains_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords"""
        return any(keyword in text for keyword in keywords)
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Return default features when no description is provided"""
        return {
            'cost_multiplier': 1.0,
            'extra_days': 0,
            'extra_tasks': [],
            'features_detected': [],
            'detected_features': [],
            'has_features': False
        }
    
    def get_feature_summary(self, features: Dict[str, Any]) -> str:
        """Generate a human-readable summary of detected features"""
        if not features['has_features']:
            return "No special features detected. Using standard construction planning."
        
        summary_parts = []
        for feature in features['features_detected']:
            summary_parts.append(f"• {feature['description']}")
        
        return f"Detected features:\n" + "\n".join(summary_parts)


# Convenience function for backward compatibility
def extract_features(description: str) -> Dict[str, Any]:
    """Extract features from project description"""
    extractor = FeatureExtractor()
    return extractor.extract_features(description)
