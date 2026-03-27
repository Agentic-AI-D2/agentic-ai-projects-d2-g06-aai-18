"""
Planner Agent for Construction Planning Assistant
Breaks down high-level construction goals into actionable tasks
"""
from crewai import Agent
from config.llm_config import get_groq_llm_for_crewai, PLANNER_PROMPT
import json
import re


class PlannerAgent:
    """Construction Planning Expert Agent"""
    
    def __init__(self):
        self.llm = get_groq_llm_for_crewai()
        self.agent = Agent(
            role='Construction Planning Expert',
            goal='Break down high-level construction goals into detailed, actionable tasks with proper sequencing and dependencies',
            backstory="""You are a senior construction project manager with 15+ years of experience in 
            residential and commercial construction. You excel at decomposing complex construction projects 
            into manageable tasks, understanding proper sequencing, and identifying critical dependencies. 
            You're familiar with all phases of construction from permitting to final finishing.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            system_prompt=PLANNER_PROMPT
        )
    
    def create_task_breakdown(self, construction_goal: str) -> dict:
        """Create a detailed task breakdown for the given construction goal"""
        try:
            # Use CrewAI's task execution instead of direct LLM calls
            from crewai import Task
            
            task = Task(
                description=f"Break down this construction goal into detailed tasks: {construction_goal}",
                agent=self.agent,
                expected_output="JSON with task breakdown including dependencies and durations"
            )
            
            # Execute the task
            result = task.execute()
            
            # Try to parse the result
            import json
            import re
            
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_content = json_match.group()
                parsed_result = json.loads(json_content)
                return parsed_result
            else:
                # Fallback structure
                return {
                    "goal": construction_goal,
                    "tasks": [
                        {
                            "id": "task_1",
                            "name": "Initial Planning",
                            "description": "Basic planning and assessment",
                            "category": "permits",
                            "estimated_duration_days": 3,
                            "dependencies": []
                        }
                    ],
                    "raw_response": result
                }
                
        except Exception as e:
            # Return basic structure if something goes wrong
            return {
                "goal": construction_goal,
                "tasks": [
                    {
                        "id": "task_1",
                        "name": "Initial Planning",
                        "description": f"Planning for {construction_goal}",
                        "category": "permits",
                        "estimated_duration_days": 5,
                        "dependencies": []
                    }
                ],
                "error": f"Error in planner agent: {str(e)}"
            }
    
    def get_agent(self):
        """Return the CrewAI agent for workflow integration"""
        return self.agent
