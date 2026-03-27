"""
Simplified Construction Planning using direct Groq calls
"""
import os
import json
import re
from datetime import datetime
from typing import Dict, Any
from tools.resource_tools import validate_all_resources
from config.llm_config import get_groq_client


class SimpleConstructionPlanner:
    """Simplified construction planning without CrewAI"""
    
    def __init__(self):
        self.client = get_groq_client()
    
    def plan_construction_project(self, construction_goal: str) -> Dict[str, Any]:
        """Execute the complete construction planning workflow"""
        try:
            # Step 1: Task Breakdown
            tasks = self._create_task_breakdown(construction_goal)
            
            if not tasks or "error" in tasks:
                return self._create_error_response("planning", "Failed to generate tasks", construction_goal)
            
            # Step 2: Resource Validation
            validation_results = self._validate_resources(tasks.get("tasks", []))
            
            # Step 3: Scheduling
            schedule_results = self._create_schedule(validation_results.get("validated_tasks", []))
            
            # Step 4: Compile Results
            final_result = self._compile_final_results(construction_goal, tasks, validation_results, schedule_results)
            
            return final_result
            
        except Exception as e:
            return self._create_error_response("workflow", str(e), construction_goal)
    
    def _create_task_breakdown(self, construction_goal: str) -> dict:
        """Create task breakdown using Groq"""
        try:
            prompt = f"""
            You are a construction planning expert. Break down this construction goal into detailed tasks: "{construction_goal}"
            
            Return JSON format:
            {{
              "goal": "{construction_goal}",
              "tasks": [
                {{
                  "id": "task_1",
                  "name": "Task name",
                  "description": "Detailed description",
                  "category": "permits|site_preparation|foundation|structural|utilities|finishing",
                  "estimated_duration_days": 5,
                  "dependencies": []
                }}
              ]
            }}
            
            Create 8-12 realistic tasks with proper sequencing.
            """
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Could not parse response", "raw": content}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _validate_resources(self, tasks: list) -> dict:
        """Validate resources for all tasks"""
        validated_tasks = []
        
        for task in tasks:
            validation_result = validate_all_resources(task)
            validated_tasks.append(validation_result)
        
        return {
            "validated_tasks": validated_tasks,
            "total_tasks": len(validated_tasks),
            "approved_tasks": len([t for t in validated_tasks if t["overall_available"]]),
            "blocked_tasks": len([t for t in validated_tasks if not t["overall_available"]])
        }
    
    def _create_schedule(self, validated_tasks: list) -> dict:
        """Create project schedule"""
        if not validated_tasks:
            return {"schedule": [], "total_project_duration": 0}
        
        # Sort tasks by category priority
        category_priority = {
            'permits': 1,
            'site_preparation': 2,
            'foundation': 3,
            'structural': 4,
            'utilities': 5,
            'finishing': 6
        }
        
        sorted_tasks = sorted(validated_tasks, key=lambda t: (
            category_priority.get(t.get('category', ''), 99),
            len(t.get('dependencies', []))
        ))
        
        schedule_tasks = []
        current_day = 1
        
        for task in sorted_tasks:
            duration = task.get('estimated_duration_days', 5)
            start_day = current_day
            end_day = start_day + duration - 1
            
            schedule_task = {
                "task_id": task.get('task_id', task.get('id', '')),
                "task_name": task.get('task_name', task.get('name', '')),
                "start_day": start_day,
                "end_day": end_day,
                "duration_days": duration,
                "dependencies_completed": task.get('dependencies', []),
                "critical_path": task.get('category') in ['foundation', 'structural'],
                "validation_status": task.get('validation_status', 'unknown')
            }
            
            schedule_tasks.append(schedule_task)
            current_day += duration
        
        total_duration = max([task["end_day"] for task in schedule_tasks]) if schedule_tasks else 0
        
        return {
            "schedule": schedule_tasks,
            "total_project_duration": total_duration,
            "critical_path_tasks": [task["task_id"] for task in schedule_tasks if task["critical_path"]],
            "optimization_suggestions": ["Monitor resource availability", "Build buffer time for weather"],
            "schedule_confidence": 7
        }
    
    def _compile_final_results(self, goal: str, tasks: dict, validation: dict, schedule: dict) -> Dict[str, Any]:
        """Compile final results"""
        total_tasks = len(tasks.get("tasks", []))
        approved_tasks = validation.get("approved_tasks", 0)
        total_duration = schedule.get("total_project_duration", 0)
        total_cost = self._calculate_total_cost(validation.get("validated_tasks", []))
        
        return {
            "project_metadata": {
                "goal": goal,
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_tasks": total_tasks,
                "total_duration_days": total_duration,
                "total_estimated_cost": total_cost
            },
            "task_breakdown": tasks,
            "resource_validation": validation,
            "project_schedule": schedule,
            "project_health": {
                "approval_rate_percentage": round((approved_tasks / total_tasks * 100) if total_tasks > 0 else 0, 1),
                "schedule_confidence": schedule.get("schedule_confidence", 7),
                "risk_level": "Low" if approved_tasks >= total_tasks * 0.8 else "Medium"
            },
            "summary": {
                "project_overview": f"Construction plan for '{goal}' with {total_tasks} tasks spanning {total_duration} days",
                "key_highlights": [
                    f"{approved_tasks} of {total_tasks} tasks approved for execution",
                    f"Estimated project duration: {total_duration} days",
                    f"Total estimated cost: {total_cost}"
                ]
            },
            "status": "completed"
        }
    
    def _calculate_total_cost(self, validated_tasks: list) -> str:
        """Calculate total estimated cost"""
        total_cost = 0
        for task in validated_tasks:
            cost_str = task.get("total_estimated_cost", "$0")
            try:
                cost_num = int(cost_str.replace("$", "").replace(",", "").replace(" ", ""))
                total_cost += cost_num
            except (ValueError, AttributeError):
                continue
        
        return f"${total_cost:,}"
    
    def _create_error_response(self, error_type: str, error_message: str, goal: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "project_metadata": {
                "goal": goal,
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "error": {
                "type": error_type,
                "message": error_message,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "status": "failed"
        }


# Update the app.py to use this simple version
def plan_construction_project(construction_goal: str) -> Dict[str, Any]:
    """Convenience function"""
    planner = SimpleConstructionPlanner()
    return planner.plan_construction_project(construction_goal)
