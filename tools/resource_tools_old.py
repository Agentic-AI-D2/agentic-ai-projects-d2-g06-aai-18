"""
Mock Resource Tools for Construction Planning
These tools simulate real-world resource availability checks
"""
import random
from typing import Dict, Any
from datetime import datetime


def check_labor_availability(task: Dict[str, Any]) -> Dict[str, Any]:
    """Mock tool to check labor availability for a construction task"""
    task_name = task.get("name", "").lower()
    task_category = task.get("category", "").lower()
    
    labor_scenarios = {
        "permits": {
            "available": True,
            "details": "Permit specialists and legal team available within 2-3 business days",
            "required_skills": ["permit specialist", "legal advisor"],
            "available_workers": 3
        },
        "site_preparation": {
            "available": True,
            "details": "Site preparation crew (excavators, graders) available immediately",
            "required_skills": ["heavy equipment operator", "site supervisor"],
            "available_workers": 8
        },
        "foundation": {
            "available": random.choice([True, True, False]),
            "details": "Foundation specialists partially available - may need to schedule 1 week out",
            "required_skills": ["concrete specialist", "foundation engineer"],
            "available_workers": 4 if random.random() > 0.33 else 2
        },
        "structural": {
            "available": True,
            "details": "Structural engineers and steelworkers fully available",
            "required_skills": ["structural engineer", "welder", "crane operator"],
            "available_workers": 12
        },
        "utilities": {
            "available": random.choice([True, False]),
            "details": "Licensed electricians and plumbers in high demand - limited availability",
            "required_skills": ["electrician", "plumber", "HVAC technician"],
            "available_workers": 5 if random.random() > 0.5 else 2
        },
        "finishing": {
            "available": True,
            "details": "Finishing crew (carpenters, painters, floorers) readily available",
            "required_skills": ["carpenter", "painter", "flooring specialist"],
            "available_workers": 10
        }
    }
    
    default_response = {
        "available": True,
        "details": "General construction crew available",
        "required_skills": ["construction worker", "foreman"],
        "available_workers": 6
    }
    
    result = labor_scenarios.get(task_category, default_response)
    result.update({
        "check_date": datetime.now().strftime("%Y-%m-%d"),
        "task_name": task_name,
        "estimated_lead_time_days": random.randint(1, 7) if result["available"] else random.randint(7, 14)
    })
    
    return result


def check_material_availability(task: Dict[str, Any]) -> Dict[str, Any]:
    """Mock tool to check material availability for a construction task"""
    task_name = task.get("name", "").lower()
    task_category = task.get("category", "").lower()
    
    material_scenarios = {
        "permits": {
            "available": True,
            "details": "All permit forms and documentation templates in stock",
            "materials": ["permit applications", "legal forms", "site plans"],
            "supplier": "City Municipal Office"
        },
        "site_preparation": {
            "available": True,
            "details": "Gravel, sand, and erosion control materials fully stocked",
            "materials": ["gravel", "sand", "erosion barriers", "survey stakes"],
            "supplier": "Local Construction Supply"
        },
        "foundation": {
            "available": random.choice([True, True, False]),
            "details": "Concrete supply chain experiencing delays - steel rebar available",
            "materials": ["concrete", "steel rebar", "waterproofing membrane", "forms"],
            "supplier": "Regional Concrete Co."
        },
        "structural": {
            "available": True,
            "details": "Steel beams and structural components available with 2-week lead time",
            "materials": ["steel beams", "bolts", "welding materials", "safety equipment"],
            "supplier": "National Steel Suppliers"
        },
        "utilities": {
            "available": random.choice([True, False, False]),
            "details": "Electrical components experiencing supply chain issues",
            "materials": ["electrical conduit", "wiring", "plumbing pipes", "HVAC units"],
            "supplier": "National Utility Suppliers"
        },
        "finishing": {
            "available": True,
            "details": "All finishing materials in stock - multiple supplier options",
            "materials": ["drywall", "paint", "flooring", "fixtures", "trim"],
            "supplier": "Local Building Materials"
        }
    }
    
    default_response = {
        "available": True,
        "details": "Standard construction materials available",
        "materials": ["general construction materials"],
        "supplier": "Multiple suppliers available"
    }
    
    result = material_scenarios.get(task_category, default_response)
    base_cost = random.randint(1000, 10000)
    result.update({
        "check_date": datetime.now().strftime("%Y-%m-%d"),
        "task_name": task_name,
        "estimated_cost": f"${base_cost:,} - ${base_cost * 2:,}",
        "delivery_time_days": random.randint(1, 5) if result["available"] else random.randint(10, 21),
        "quantity_available": "sufficient" if result["available"] else "limited"
    })
    
    return result


def check_equipment_availability(task: Dict[str, Any]) -> Dict[str, Any]:
    """Mock tool to check equipment availability for a construction task"""
    task_name = task.get("name", "").lower()
    task_category = task.get("category", "").lower()
    
    equipment_scenarios = {
        "permits": {
            "available": True,
            "details": "Office equipment and documentation tools available",
            "equipment": ["computers", "printers", "survey equipment"],
            "rental_cost_per_day": "$50"
        },
        "site_preparation": {
            "available": True,
            "details": "Heavy equipment fleet available - excavators, bulldozers ready",
            "equipment": ["excavator", "bulldozer", "grader", "dump trucks"],
            "rental_cost_per_day": "$1,200"
        },
        "foundation": {
            "available": random.choice([True, True, False]),
            "details": "Concrete mixers available - crane rental may require advance booking",
            "equipment": ["concrete mixer", "crane", "concrete pump", "vibrators"],
            "rental_cost_per_day": "$800"
        },
        "structural": {
            "available": True,
            "details": "Cranes and welding equipment fully available",
            "equipment": ["crane", "welding machine", "steel cutter", "safety harnesses"],
            "rental_cost_per_day": "$1,500"
        },
        "utilities": {
            "available": random.choice([True, True]),
            "details": "Specialized tools available - some testing equipment limited",
            "equipment": ["pipe threader", "electrical testers", "HVAC tools"],
            "rental_cost_per_day": "$300"
        },
        "finishing": {
            "available": True,
            "details": "All finishing tools and equipment in good condition",
            "equipment": ["paint sprayers", "flooring tools", "carpentry tools"],
            "rental_cost_per_day": "$200"
        }
    }
    
    default_response = {
        "available": True,
        "details": "Standard construction equipment available",
        "equipment": ["basic tools"],
        "rental_cost_per_day": "$150"
    }
    
    result = equipment_scenarios.get(task_category, default_response)
    result.update({
        "check_date": datetime.now().strftime("%Y-%m-%d"),
        "task_name": task_name,
        "maintenance_status": "good",
        "operator_required": True,
        "booking_lead_time_days": random.randint(1, 3) if result["available"] else random.randint(7, 14)
    })
    
    return result


def validate_all_resources(task: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive resource validation for a task"""
    labor = check_labor_availability(task)
    materials = check_material_availability(task)
    equipment = check_equipment_availability(task)
    
    all_available = labor["available"] and materials["available"] and equipment["available"]
    
    labor_cost = random.randint(500, 5000)
    material_cost = int(materials["estimated_cost"].split("$")[1].split(",")[0].replace(" ", ""))
    equipment_cost = int(equipment["rental_cost_per_day"].replace("$", "").replace(",", "")) * task.get("estimated_duration_days", 1)
    total_cost = labor_cost + material_cost + equipment_cost
    
    return {
        "task_id": task.get("id", ""),
        "task_name": task.get("name", ""),
        "overall_available": all_available,
        "validation_status": "approved" if all_available else "needs_review",
        "labor": labor,
        "materials": materials,
        "equipment": equipment,
        "total_estimated_cost": f"${total_cost:,}",
        "validation_notes": generate_validation_notes(labor, materials, equipment),
        "validation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def generate_validation_notes(labor: Dict, materials: Dict, equipment: Dict) -> str:
    """Generate human-readable validation notes"""
    notes = []
    
    if not labor["available"]:
        notes.append(f"Labor constraints: {labor['details']}")
    
    if not materials["available"]:
        notes.append(f"Material delays: {materials['details']}")
    
    if not equipment["available"]:
        notes.append(f"Equipment limitations: {equipment['details']}")
    
    if labor["available"] and materials["available"] and equipment["available"]:
        notes.append("All resources readily available - task can proceed as scheduled")
    
    return " | ".join(notes) if notes else "All resources available"
