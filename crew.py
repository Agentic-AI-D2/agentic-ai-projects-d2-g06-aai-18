"""
CrewAI Workflow Orchestration for Construction Planning Assistant
Manages the multi-agent workflow for construction planning
"""
from crewai import Crew, Task
from agents.planner import PlannerAgent
from agents.validator import ResourceValidatorAgent
from agents.scheduler import SchedulerAgent
import json
import logging
from datetime import datetime
from typing import Dict, Any


class ConstructionPlanningCrew:
    """Main workflow orchestrator for construction planning"""
    
    def __init__(self):
        self.planner = PlannerAgent()
        self.validator = ResourceValidatorAgent()
        self.scheduler = SchedulerAgent()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create the crew
        self.crew = Crew(
            agents=[
                self.planner.get_agent(),
                self.validator.get_agent(),
                self.scheduler.get_agent()
            ],
            verbose=True,
            process="sequential"  # Agents work in sequence
        )
    
    def plan_construction_project(self, construction_goal: str) -> Dict[str, Any]:
        """Execute the complete construction planning workflow"""
        self.logger.info(f"Starting construction planning for: {construction_goal}")
        
        try:
            # Step 1: Task Breakdown (Planner Agent)
            self.logger.info("Step 1: Creating task breakdown...")
            task_breakdown = self.planner.create_task_breakdown(construction_goal)
            
            if "error" in task_breakdown:
                self.logger.error(f"Planner agent error: {task_breakdown['error']}")
                return self._create_error_response("planning", task_breakdown['error'], construction_goal)
            
            tasks = task_breakdown.get("tasks", [])
            if not tasks:
                return self._create_error_response("planning", "No tasks generated", construction_goal)
            
            self.logger.info(f"Generated {len(tasks)} tasks")
            
            # Step 2: Resource Validation (Validator Agent)
            self.logger.info("Step 2: Validating resources...")
            validation_results = self.validator.validate_tasks(tasks)
            
            validated_tasks = validation_results.get("validated_tasks", [])
            approved_count = validation_results.get("approved_tasks", 0)
            blocked_count = validation_results.get("blocked_tasks", 0)
            
            self.logger.info(f"Validation complete: {approved_count} approved, {blocked_count} blocked")
            
            # Step 3: Scheduling (Scheduler Agent)
            self.logger.info("Step 3: Creating schedule...")
            schedule_results = self.scheduler.create_schedule(validated_tasks)
            
            total_duration = schedule_results.get("total_project_duration", 0)
            critical_path_tasks = schedule_results.get("critical_path_tasks", [])
            
            self.logger.info(f"Schedule created: {total_duration} days total")
            
            # Step 4: Compile Final Results
            final_result = self._compile_final_results(
                construction_goal,
                task_breakdown,
                validation_results,
                schedule_results
            )
            
            self.logger.info("Construction planning workflow completed successfully")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Workflow error: {str(e)}")
            return self._create_error_response("workflow", str(e), construction_goal)
    
    def _compile_final_results(self, goal: str, tasks: dict, validation: dict, schedule: dict) -> Dict[str, Any]:
        """Compile all results into a comprehensive project plan"""
        # Extract key metrics
        total_tasks = len(tasks.get("tasks", []))
        approved_tasks = validation.get("approved_tasks", 0)
        blocked_tasks = validation.get("blocked_tasks", 0)
        total_duration = schedule.get("total_project_duration", 0)
        total_cost = self._calculate_total_cost(validation.get("validated_tasks", []))
        
        # Calculate project health metrics
        approval_rate = (approved_tasks / total_tasks * 100) if total_tasks > 0 else 0
        schedule_confidence = schedule.get("schedule_confidence", 7)
        
        return {
            "project_metadata": {
                "goal": goal,
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "workflow_version": "1.0",
                "total_tasks": total_tasks,
                "total_duration_days": total_duration,
                "total_estimated_cost": total_cost
            },
            "task_breakdown": tasks,
            "resource_validation": validation,
            "project_schedule": schedule,
            "project_health": {
                "approval_rate_percentage": round(approval_rate, 1),
                "schedule_confidence": schedule_confidence,
                "risk_level": self._calculate_risk_level(approval_rate, schedule_confidence),
                "blocked_tasks_count": blocked_tasks
            },
            "summary": {
                "project_overview": f"Construction plan for '{goal}' with {total_tasks} tasks spanning {total_duration} days",
                "key_highlights": [
                    f"{approved_tasks} of {total_tasks} tasks approved for execution",
                    f"Estimated project duration: {total_duration} days",
                    f"Critical path includes {len(schedule.get('critical_path_tasks', []))} key tasks",
                    f"Total estimated cost: {total_cost}",
                    f"Project risk level: {self._calculate_risk_level(approval_rate, schedule_confidence)}"
                ],
                "next_steps": [
                    "Review resource constraints for blocked tasks",
                    "Confirm permit requirements and timelines",
                    "Validate material delivery schedules",
                    "Prepare site mobilization plan"
                ]
            },
            "status": "completed"
        }
    
    def _calculate_total_cost(self, validated_tasks: list) -> str:
        """Calculate total estimated cost from validated tasks"""
        total_cost = 0
        for task in validated_tasks:
            cost_str = task.get("total_estimated_cost", "$0")
            try:
                # Extract numeric value from cost string
                cost_num = int(cost_str.replace("$", "").replace(",", "").replace(" ", ""))
                total_cost += cost_num
            except (ValueError, AttributeError):
                continue
        
        return f"${total_cost:,}"
    
    def _calculate_risk_level(self, approval_rate: float, schedule_confidence: int) -> str:
        """Calculate overall project risk level"""
        if approval_rate >= 90 and schedule_confidence >= 8:
            return "Low"
        elif approval_rate >= 70 and schedule_confidence >= 6:
            return "Medium"
        else:
            return "High"
    
    def _create_error_response(self, error_type: str, error_message: str, goal: str) -> Dict[str, Any]:
        """Create a standardized error response"""
        return {
            "project_metadata": {
                "goal": goal,
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "workflow_version": "1.0"
            },
            "error": {
                "type": error_type,
                "message": error_message,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "status": "failed",
            "fallback_suggestions": [
                "Try rephrasing your construction goal with more specific details",
                "Ensure your goal includes the type and size of construction project",
                "Check that all required environment variables are set",
                "Verify internet connectivity and API access"
            ]
        }


# Convenience function for direct usage
def plan_construction_project(construction_goal: str) -> Dict[str, Any]:
    """Convenience function to plan a construction project"""
    crew = ConstructionPlanningCrew()
    return crew.plan_construction_project(construction_goal)
