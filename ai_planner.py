"""
AI Planning Module for Construction Planning Assistant
Uses Groq LLM to generate intelligent construction insights
"""
import json
import re
from typing import Dict, List, Any, Optional
from config.llm_config import get_groq_client


class AIPlanner:
    """AI-powered construction planning with Groq LLM"""
    
    def __init__(self):
        self.client = get_groq_client()
    
    def generate_ai_insights(self, project_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI-powered construction insights
        
        Args:
            project_params: Dictionary containing project parameters
            
        Returns:
            Dictionary with AI-generated insights
        """
        try:
            # Extract parameters
            area = project_params.get("area", 2000)
            floors = project_params.get("floors", 1)
            building_type = project_params.get("building_type", "Residential")
            quality = project_params.get("quality", "Standard")
            location = project_params.get("location", "Tier 2")
            description = project_params.get("description", "")
            
            # Generate AI insights
            insights = self._call_ai_planner(area, floors, building_type, quality, location, description)
            
            return {
                "status": "success",
                "insights": insights,
                "enhanced_tasks": insights.get("tasks", []),
                "ai_duration_estimate": insights.get("duration", 0),
                "ai_cost_breakdown": insights.get("cost_breakdown", {}),
                "ai_recommendations": insights.get("recommendations", [])
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "insights": {},
                "enhanced_tasks": [],
                "ai_duration_estimate": 0,
                "ai_cost_breakdown": {},
                "ai_recommendations": []
            }
    
    def _call_ai_planner(self, area: int, floors: int, building_type: str, 
                        quality: str, location: str, description: str) -> Dict[str, Any]:
        """Call Groq LLM for construction planning insights"""
        
        prompt = f"""
You are a construction planning expert with deep knowledge of Indian construction practices.

Based on the following project parameters:
- Area: {area} sq ft
- Floors: {floors}
- Building Type: {building_type}
- Quality Grade: {quality}
- Location: {location}
- Description: "{description}"

Provide intelligent construction insights by analyzing the requirements and suggesting optimizations.

Return STRICT JSON format:
{{
  "analysis": {{
    "project_complexity": "simple|moderate|complex",
    "key_considerations": ["consideration1", "consideration2"],
    "risk_factors": ["risk1", "risk2"]
  }},
  "enhanced_tasks": [
    {{
      "name": "Task name",
      "description": "Detailed description",
      "category": "foundation|structural|finishing|utilities|special",
      "priority": "high|medium|low",
      "estimated_days": 5,
      "dependencies": []
    }}
  ],
  "duration_insights": {{
    "estimated_total_days": 120,
    "critical_path_tasks": ["task1", "task2"],
    "potential_delays": ["weather", "material_delivery"],
    "optimization_suggestions": ["suggestion1", "suggestion2"]
  }},
  "cost_insights": {{
    "cost_distribution": {{
      "labor_percentage": 40,
      "materials_percentage": 50,
      "equipment_percentage": 10
    }},
    "cost_optimization_tips": ["tip1", "tip2"],
    "potential_cost_savers": ["saver1", "saver2"]
  }},
  "recommendations": [
    "Recommendation 1 based on project analysis",
    "Recommendation 2 for quality improvement",
    "Recommendation 3 for cost optimization"
  ]
}}

Guidelines:
- Generate 3-5 enhanced tasks based on the description
- Consider Indian construction context and materials
- Provide practical, actionable insights
- Keep estimates realistic for the given parameters
- Focus on quality, cost, and timeline optimization
"""
        
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # Return empty structure if JSON parsing fails
            return self._get_empty_insights()
    
    def _get_empty_insights(self) -> Dict[str, Any]:
        """Return empty insights structure for fallback"""
        return {
            "analysis": {
                "project_complexity": "moderate",
                "key_considerations": [],
                "risk_factors": []
            },
            "enhanced_tasks": [],
            "duration_insights": {
                "estimated_total_days": 0,
                "critical_path_tasks": [],
                "potential_delays": [],
                "optimization_suggestions": []
            },
            "cost_insights": {
                "cost_distribution": {
                    "labor_percentage": 40,
                    "materials_percentage": 50,
                    "equipment_percentage": 10
                },
                "cost_optimization_tips": [],
                "potential_cost_savers": []
            },
            "recommendations": []
        }
    
    def merge_with_rule_based(self, ai_insights: Dict[str, Any], 
                            rule_based_tasks: List[Dict], 
                            rule_based_duration: int,
                            rule_based_cost: Dict) -> Dict[str, Any]:
        """
        Merge AI insights with rule-based calculations
        
        Args:
            ai_insights: AI-generated insights
            rule_based_tasks: Existing rule-based tasks
            rule_based_duration: Rule-based duration
            rule_based_cost: Rule-based cost breakdown
            
        Returns:
            Merged planning results
        """
        if ai_insights.get("status") != "success":
            return {
                "merged_tasks": rule_based_tasks,
                "merged_duration": rule_based_duration,
                "merged_cost": rule_based_cost,
                "ai_enhancement": False
            }
        
        insights = ai_insights.get("insights", {})
        
        # Merge tasks (combine rule-based and AI-enhanced)
        merged_tasks = rule_based_tasks.copy()
        ai_tasks = insights.get("enhanced_tasks", [])
        
        # Add AI tasks with special marker
        for task in ai_tasks:
            task["ai_generated"] = True
            task["source"] = "AI Enhancement"
            merged_tasks.append(task)
        
        # Merge duration (use rule-based as base, add AI insights)
        ai_duration = insights.get("duration_insights", {}).get("estimated_total_days", 0)
        merged_duration = rule_based_duration  # Keep rule-based as primary
        
        # Merge cost (use rule-based, add AI optimization tips)
        merged_cost = rule_based_cost.copy()
        cost_insights = insights.get("cost_insights", {})
        merged_cost["ai_optimization_tips"] = cost_insights.get("cost_optimization_tips", [])
        merged_cost["potential_cost_savers"] = cost_insights.get("potential_cost_savers", [])
        
        return {
            "merged_tasks": merged_tasks,
            "merged_duration": merged_duration,
            "merged_cost": merged_cost,
            "ai_enhancement": True,
            "ai_analysis": insights.get("analysis", {}),
            "ai_recommendations": insights.get("recommendations", []),
            "duration_insights": insights.get("duration_insights", {}),
            "cost_insights": insights.get("cost_insights", {})
        }


# Convenience function for backward compatibility
def generate_ai_insights(project_params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate AI insights for construction planning"""
    planner = AIPlanner()
    return planner.generate_ai_insights(project_params)
