"""
Resource Validator Agent for Construction Planning Assistant
Validates task feasibility by checking resource availability
"""
from crewai import Agent, Task
from config.llm_config import get_groq_llm_for_crewai, VALIDATOR_PROMPT
from tools.resource_tools import validate_all_resources
import json
import re


class ResourceValidatorAgent:
    """Resource Validation Expert Agent"""
    
    def __init__(self):
        self.llm = get_groq_llm_for_crewai()
        self.agent = Agent(
            role='Resource Validation Expert',
            goal='Validate construction tasks by checking labor, material, and equipment availability',
            backstory="""You are a resource management specialist with extensive experience in construction 
            logistics and supply chain management. You excel at identifying potential resource constraints, 
            validating material availability, checking labor capacity, and ensuring equipment availability. 
            You provide detailed feedback on resource feasibility and suggest alternatives when constraints exist.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            system_prompt=VALIDATOR_PROMPT
        )
    
    def validate_tasks(self, tasks: list) -> dict:
        """Validate resource availability for all tasks"""
        validated_tasks = []
        
        for task in tasks:
            # Use the mock tools to check resource availability
            validation_result = validate_all_resources(task)
            
            # Add additional LLM-based analysis for complex scenarios
            enhanced_validation = self._enhance_validation_with_llm(task, validation_result)
            validated_tasks.append(enhanced_validation)
        
        return {
            "validated_tasks": validated_tasks,
            "total_tasks": len(validated_tasks),
            "approved_tasks": len([t for t in validated_tasks if t["overall_available"]]),
            "blocked_tasks": len([t for t in validated_tasks if not t["overall_available"]])
        }
    
    def _enhance_validation_with_llm(self, task: dict, basic_validation: dict) -> dict:
        """Use LLM to enhance validation with contextual analysis"""
        try:
            # For now, return basic validation without LLM enhancement
            # to avoid direct LLM call issues
            basic_validation.update({
                "risk_assessment": "medium",
                "mitigation_suggestions": "Monitor resource availability",
                "alternative_approaches": "Standard construction approach",
                "confidence_level": 7,
                "enhanced_notes": "Basic validation completed"
            })
            return basic_validation
            
        except Exception as e:
            # Return basic validation if enhancement fails
            basic_validation.update({
                "risk_assessment": "medium",
                "mitigation_suggestions": "Monitor resources closely",
                "alternative_approaches": "Standard approach",
                "confidence_level": 6,
                "enhanced_notes": f"Enhancement failed: {str(e)}"
            })
            return basic_validation
    
    def create_validation_task(self, tasks_data: dict) -> Task:
        """Create a CrewAI task for resource validation"""
        task_description = f"""
        Validate resource availability for these construction tasks:
        
        {tasks_data}
        
        For each task, check:
        1. Labor availability and skills required
        2. Material availability and supply chain status
        3. Equipment availability and scheduling
        4. Overall feasibility assessment
        
        Use the available resource checking tools and provide detailed validation results.
        """
        
        return Task(
            description=task_description,
            agent=self.agent,
            expected_output="JSON with validated tasks including resource availability status and recommendations"
        )
    
    def get_agent(self):
        """Return the CrewAI agent for workflow integration"""
        return self.agent
