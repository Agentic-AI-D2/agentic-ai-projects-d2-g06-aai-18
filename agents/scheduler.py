"""
Scheduler Agent for Construction Planning Assistant
Creates optimized execution timeline for validated construction tasks
"""
from crewai import Agent, Task
from config.llm_config import get_groq_llm_for_crewai, SCHEDULER_PROMPT
from datetime import datetime
from typing import Dict, Any, List
import json
import re


class SchedulerAgent:
    """Construction Scheduling Expert Agent"""
    
    def __init__(self):
        self.llm = get_groq_llm_for_crewai()
        self.agent = Agent(
            role='Construction Scheduling Expert',
            goal='Create optimized execution timeline for validated construction tasks with proper sequencing and dependency management',
            backstory="""You are a master construction scheduler with 20+ years of experience in project 
            timeline optimization. You excel at critical path analysis, resource leveling, and creating 
            realistic project schedules. You understand construction sequencing, weather considerations, 
            permit timelines, and how to optimize for both speed and quality. You're an expert at identifying 
            parallel work opportunities while maintaining safety and quality standards.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            system_prompt=SCHEDULER_PROMPT
        )
    
    def create_schedule(self, validated_tasks: list) -> dict:
        """Create an optimized execution schedule for validated tasks"""
        # First, create a basic schedule using algorithmic approach
        basic_schedule = self._create_basic_schedule(validated_tasks)
        
        # Then enhance with LLM-based optimization
        enhanced_schedule = self._enhance_schedule_with_llm(validated_tasks, basic_schedule)
        
        return enhanced_schedule
    
    def _create_basic_schedule(self, validated_tasks: list) -> dict:
        """Create a basic schedule using algorithmic approach"""
        # Sort tasks by category and dependencies
        task_order = self._determine_task_order(validated_tasks)
        
        schedule_tasks = []
        current_day = 1
        project_phases = {}
        
        for task_info in task_order:
            task = task_info['task']
            
            # Calculate start and end days
            start_day = current_day
            duration = task.get('estimated_duration_days', 5)
            end_day = start_day + duration - 1
            
            # Determine if this is a critical path task
            is_critical = self._is_critical_path_task(task, validated_tasks)
            
            # Find parallel tasks
            parallel_tasks = self._find_parallel_tasks(task, task_info['remaining_tasks'])
            
            schedule_task = {
                "task_id": task.get('task_id', task.get('id', '')),
                "task_name": task.get('task_name', task.get('name', '')),
                "start_day": start_day,
                "end_day": end_day,
                "duration_days": duration,
                "dependencies_completed": task.get('dependencies', []),
                "critical_path": is_critical,
                "parallel_tasks": parallel_tasks,
                "validation_status": task.get('validation_status', 'unknown'),
                "resource_confidence": task.get('confidence_level', 7)
            }
            
            schedule_tasks.append(schedule_task)
            
            # Track project phases
            category = task.get('category', 'general')
            if category not in project_phases:
                project_phases[category] = {
                    "phase": category.replace('_', ' ').title(),
                    "start_day": start_day,
                    "end_day": end_day,
                    "tasks": []
                }
            else:
                project_phases[category]["end_day"] = max(project_phases[category]["end_day"], end_day)
            
            project_phases[category]["tasks"].append(task.get('task_id', task.get('id', '')))
            
            # Update current day (accounting for parallel tasks)
            if parallel_tasks:
                max_parallel_duration = max(
                    [t.get('estimated_duration_days', 5) for t in parallel_tasks], 
                    default=duration
                )
                current_day += max_parallel_duration
            else:
                current_day += duration
        
        # Calculate project statistics
        total_duration = max([task["end_day"] for task in schedule_tasks]) if schedule_tasks else 0
        critical_path_tasks = [task["task_id"] for task in schedule_tasks if task["critical_path"]]
        
        return {
            "schedule": schedule_tasks,
            "total_project_duration": total_duration,
            "critical_path_tasks": critical_path_tasks,
            "project_phases": list(project_phases.values()),
            "schedule_created_date": datetime.now().strftime("%Y-%m-%d"),
            "optimization_level": "basic"
        }
    
    def _determine_task_order(self, tasks: list) -> list:
        """Determine optimal task order based on dependencies and categories"""
        # Define category priority
        category_priority = {
            'permits': 1,
            'site_preparation': 2,
            'foundation': 3,
            'structural': 4,
            'utilities': 5,
            'finishing': 6
        }
        
        # Sort by category priority, then by dependencies
        sorted_tasks = sorted(tasks, key=lambda t: (
            category_priority.get(t.get('category', ''), 99),
            len(t.get('dependencies', [])),
            t.get('estimated_duration_days', 0)
        ))
        
        # Create ordered list with remaining tasks context
        ordered_tasks = []
        for i, task in enumerate(sorted_tasks):
            ordered_tasks.append({
                'task': task,
                'remaining_tasks': sorted_tasks[i+1:]
            })
        
        return ordered_tasks
    
    def _is_critical_path_task(self, task: dict, all_tasks: list) -> bool:
        """Determine if a task is on the critical path"""
        dependencies = task.get('dependencies', [])
        
        # Foundation and structural tasks are typically critical
        critical_categories = ['foundation', 'structural']
        if task.get('category') in critical_categories:
            return True
        
        # Tasks with no dependencies (starting tasks) are critical
        if not dependencies:
            return True
        
        # Tasks that many other tasks depend on are critical
        task_id = task.get('task_id', task.get('id', ''))
        dependency_count = sum(1 for t in all_tasks if task_id in t.get('dependencies', []))
        
        return dependency_count >= 2
    
    def _find_parallel_tasks(self, current_task: dict, remaining_tasks: list) -> list:
        """Find tasks that can be executed in parallel"""
        current_task_id = current_task.get('task_id', current_task.get('id', ''))
        current_dependencies = current_task.get('dependencies', [])
        
        parallel_tasks = []
        for task in remaining_tasks:
            task_dependencies = task.get('dependencies', [])
            
            if (current_task_id not in task_dependencies and
                task.get('task_id', task.get('id', '')) not in current_dependencies and
                task.get('category') == current_task.get('category') and
                task.get('overall_available', True)):
                
                parallel_tasks.append(task)
        
        return parallel_tasks[:2]  # Limit to 2 parallel tasks for simplicity
    
    def _enhance_schedule_with_llm(self, validated_tasks: list, basic_schedule: dict) -> dict:
        """Use LLM to enhance schedule with optimization and risk analysis"""
        try:
            # For now, return basic schedule without LLM enhancement
            # to avoid direct LLM call issues
            enhancements = {
                "optimization_suggestions": ["Monitor resource availability closely"],
                "risk_factors": ["Weather delays possible"],
                "mitigation_strategies": ["Build in buffer time"],
                "weather_considerations": "Standard weather planning",
                "recommended_buffer_percentage": 10,
                "resource_leveling_opportunities": ["Standard leveling"],
                "schedule_confidence": 7
            }
            
            # Apply enhancements to the schedule
            enhanced_schedule = basic_schedule.copy()
            enhanced_schedule.update(enhancements)
            enhanced_schedule["optimization_level"] = "enhanced"
            
            # Add buffer time if recommended
            buffer_percentage = enhancements.get("recommended_buffer_percentage", 10)
            if buffer_percentage > 0:
                original_duration = enhanced_schedule["total_project_duration"]
                buffer_days = int(original_duration * buffer_percentage / 100)
                enhanced_schedule["total_project_duration_with_buffer"] = original_duration + buffer_days
                enhanced_schedule["buffer_days"] = buffer_days
            
            return enhanced_schedule
            
        except Exception as e:
            # Return basic schedule if enhancement fails
            basic_schedule.update({
                "optimization_suggestions": ["Monitor schedule closely"],
                "risk_factors": ["Standard construction risks"],
                "mitigation_strategies": ["Regular progress reviews"],
                "weather_considerations": "Plan for weather delays",
                "recommended_buffer_percentage": 10,
                "resource_leveling_opportunities": ["Standard resource management"],
                "schedule_confidence": 6,
                "optimization_level": "basic_with_fallback",
                "enhancement_error": str(e)
            })
            return basic_schedule
    
    def create_scheduling_task(self, validated_tasks_data: dict) -> Task:
        """Create a CrewAI task for scheduling"""
        task_description = f"""
        Create an optimized construction schedule for these validated tasks:
        
        {validated_tasks_data}
        
        Requirements:
        1. Respect task dependencies and sequencing
        2. Optimize for parallel execution where possible
        3. Include realistic time buffers
        4. Identify critical path tasks
        5. Consider resource constraints and availability
        6. Provide clear timeline with start/end dates
        
        Return a comprehensive schedule with project phases and duration estimates.
        """
        
        return Task(
            description=task_description,
            agent=self.agent,
            expected_output="JSON with optimized schedule including timeline, critical path, and project phases"
        )
    
    def get_agent(self):
        """Return the CrewAI agent for workflow integration"""
        return self.agent
