"""
Deterministic Resource Tools for Construction Planning
Rule-based cost and duration calculations without random values
"""
from typing import Dict, Any
from datetime import datetime


# Cost configuration (INR - Indian Rupees)
COST_PER_SQFT = {
    "Basic": 1500,
    "Standard": 2200,
    "Premium": 3200
}

LOCATION_MULTIPLIER = {
    "Metro": 1.3,
    "Tier 2": 1.0,
    "Rural": 0.8
}

# Cost breakdown percentages
LABOR_PERCENTAGE = 0.4
MATERIAL_PERCENTAGE = 0.5
EQUIPMENT_PERCENTAGE = 0.1


def calculate_project_cost(area: int, quality: str, location: str, floors: int = 1) -> Dict[str, Any]:
    """Calculate deterministic project cost based on parameters with floor scaling"""
    
    # Step 1: Add effective floor scaling (slight realism)
    effective_floors = 1 + 0.9 * (floors - 1)
    
    # Step 2: Calculate adjusted built-up area
    total_builtup_area = area * effective_floors
    
    # Step 3: Final cost calculation
    base_cost = total_builtup_area * COST_PER_SQFT[quality]
    final_cost = base_cost * LOCATION_MULTIPLIER[location]
    
    # Step 4: Cost breakdown
    labor_cost = final_cost * LABOR_PERCENTAGE
    material_cost = final_cost * MATERIAL_PERCENTAGE
    equipment_cost = final_cost * EQUIPMENT_PERCENTAGE
    
    return {
        "total_cost": int(final_cost),
        "labor_cost": int(labor_cost),
        "material_cost": int(material_cost),
        "equipment_cost": int(equipment_cost),
        "cost_per_sqft": COST_PER_SQFT[quality],
        "location_factor": LOCATION_MULTIPLIER[location],
        "currency": "INR",
        "effective_floors": round(effective_floors, 2),
        "total_builtup_area": int(total_builtup_area),
        "ground_floor_area": area
    }


def calculate_project_duration(area: int, floors: int, building_type: str = "Residential") -> Dict[str, Any]:
    """Calculate realistic project duration using phase-based construction modeling"""
    
    # Base duration for Indian construction scenarios
    base_duration_map = {
        "Residential": 120,  # 4 months for standard residential
        "Commercial": 180   # 6 months for standard commercial
    }
    
    # Step 2: Assign correct base duration
    selected_base_duration = base_duration_map[building_type]
    
    # Floor factor - each additional floor adds 30% time (diminishing returns)
    floor_factor = 1 + (floors - 1) * 0.3
    
    # Area factor - slight adjustment for area (not proportional)
    area_factor = 1 + ((area / 2000) - 1) * 0.2
    # Clamp to avoid extreme values
    area_factor = max(0.8, min(area_factor, 1.5))
    
    # Step 3: Calculate total duration
    total_duration = int(selected_base_duration * floor_factor * area_factor)
    
    # Phase breakdown using percentage distribution
    foundation_days = int(total_duration * 0.2)
    structure_days = int(total_duration * 0.5)
    finishing_days = total_duration - foundation_days - structure_days  # Ensure sum equals total
    
    return {
        "total_days": total_duration,
        "foundation_days": foundation_days,
        "structure_days": structure_days,
        "finishing_days": finishing_days,
        "base_duration": selected_base_duration,  # Fixed: was showing 0
        "floor_factor": round(floor_factor, 2),
        "area_factor": round(area_factor, 2),
        "building_type": building_type,
        "duration_per_sqft": total_duration / area if area > 0 else 0
    }


def check_labor_availability(task: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic labor availability check"""
    task_category = task.get("category", "").lower()
    
    labor_scenarios = {
        "permits": {
            "available": True,
            "details": "Permit specialists and legal team available within 2-3 business days",
            "required_skills": ["permit specialist", "legal advisor"],
            "available_workers": 3,
            "estimated_lead_time_days": 3
        },
        "site_preparation": {
            "available": True,
            "details": "Site preparation crew (excavators, graders) available immediately",
            "required_skills": ["heavy equipment operator", "site supervisor"],
            "available_workers": 8,
            "estimated_lead_time_days": 1
        },
        "foundation": {
            "available": True,
            "details": "Foundation specialists available - may need to schedule 1 week out",
            "required_skills": ["concrete specialist", "foundation engineer"],
            "available_workers": 4,
            "estimated_lead_time_days": 7
        },
        "structural": {
            "available": True,
            "details": "Structural engineers and steelworkers fully available",
            "required_skills": ["structural engineer", "welder", "crane operator"],
            "available_workers": 12,
            "estimated_lead_time_days": 2
        },
        "utilities": {
            "available": True,
            "details": "Licensed electricians and plumbers available with standard lead time",
            "required_skills": ["electrician", "plumber", "HVAC technician"],
            "available_workers": 5,
            "estimated_lead_time_days": 5
        },
        "finishing": {
            "available": True,
            "details": "Finishing crew (carpenters, painters, floorers) readily available",
            "required_skills": ["carpenter", "painter", "flooring specialist"],
            "available_workers": 10,
            "estimated_lead_time_days": 2
        }
    }
    
    default_response = {
        "available": True,
        "details": "General construction crew available",
        "required_skills": ["construction worker", "foreman"],
        "available_workers": 6,
        "estimated_lead_time_days": 3
    }
    
    result = labor_scenarios.get(task_category, default_response)
    result.update({
        "check_date": datetime.now().strftime("%Y-%m-%d"),
        "task_name": task.get("name", "Unknown task")
    })
    
    return result


def check_material_availability(task: Dict[str, Any], project_cost: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic material availability check with cost allocation"""
    task_category = task.get("category", "").lower()
    
    material_scenarios = {
        "permits": {
            "available": True,
            "details": "All permit forms and documentation templates in stock",
            "materials": ["permit applications", "legal forms", "site plans"],
            "supplier": "City Municipal Office",
            "delivery_time_days": 2
        },
        "site_preparation": {
            "available": True,
            "details": "Gravel, sand, and erosion control materials fully stocked",
            "materials": ["gravel", "sand", "erosion barriers", "survey stakes"],
            "supplier": "Local Construction Supply",
            "delivery_time_days": 3
        },
        "foundation": {
            "available": True,
            "details": "Concrete and steel rebar available with standard scheduling",
            "materials": ["concrete", "steel rebar", "waterproofing membrane", "forms"],
            "supplier": "Regional Concrete Co.",
            "delivery_time_days": 5
        },
        "structural": {
            "available": True,
            "details": "Steel beams and structural components available with 2-week lead time",
            "materials": ["steel beams", "bolts", "welding materials", "safety equipment"],
            "supplier": "National Steel Suppliers",
            "delivery_time_days": 14
        },
        "utilities": {
            "available": True,
            "details": "Electrical and plumbing components available with standard lead time",
            "materials": ["electrical conduit", "wiring", "plumbing pipes", "HVAC units"],
            "supplier": "National Utility Suppliers",
            "delivery_time_days": 7
        },
        "finishing": {
            "available": True,
            "details": "All finishing materials in stock - multiple supplier options",
            "materials": ["drywall", "paint", "flooring", "fixtures", "trim"],
            "supplier": "Local Building Materials",
            "delivery_time_days": 3
        }
    }
    
    default_response = {
        "available": True,
        "details": "Standard construction materials available",
        "materials": ["general construction materials"],
        "supplier": "Multiple suppliers available",
        "delivery_time_days": 5
    }
    
    result = material_scenarios.get(task_category, default_response)
    
    # Allocate material cost based on task category
    total_material_cost = project_cost["material_cost"]
    category_cost_allocation = {
        "permits": 0.02,      # 2% of material cost
        "site_preparation": 0.05,  # 5% of material cost
        "foundation": 0.25,   # 25% of material cost
        "structural": 0.35,   # 35% of material cost
        "utilities": 0.20,    # 20% of material cost
        "finishing": 0.13     # 13% of material cost
    }
    
    allocation = category_cost_allocation.get(task_category, 0.1)
    task_material_cost = int(total_material_cost * allocation)
    
    result.update({
        "check_date": datetime.now().strftime("%Y-%m-%d"),
        "task_name": task.get("name", "Unknown task"),
        "estimated_cost": f"₹{task_material_cost:,}",
        "quantity_available": "sufficient",
        "cost_allocation_percentage": allocation * 100
    })
    
    return result


def check_equipment_availability(task: Dict[str, Any], project_cost: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic equipment availability check with cost allocation"""
    task_category = task.get("category", "").lower()
    
    equipment_scenarios = {
        "permits": {
            "available": True,
            "details": "Office equipment and documentation tools available",
            "equipment": ["computers", "printers", "survey equipment"],
            "rental_cost_per_day": "₹2,000"
        },
        "site_preparation": {
            "available": True,
            "details": "Heavy equipment fleet available - excavators, bulldozers ready",
            "equipment": ["excavator", "bulldozer", "grader", "dump trucks"],
            "rental_cost_per_day": "₹25,000"
        },
        "foundation": {
            "available": True,
            "details": "Concrete mixers available - crane rental available with scheduling",
            "equipment": ["concrete mixer", "crane", "concrete pump", "vibrators"],
            "rental_cost_per_day": "₹15,000"
        },
        "structural": {
            "available": True,
            "details": "Cranes and welding equipment fully available",
            "equipment": ["crane", "welding machine", "steel cutter", "safety harnesses"],
            "rental_cost_per_day": "₹30,000"
        },
        "utilities": {
            "available": True,
            "details": "Specialized tools available - testing equipment ready",
            "equipment": ["pipe threader", "electrical testers", "HVAC tools"],
            "rental_cost_per_day": "₹5,000"
        },
        "finishing": {
            "available": True,
            "details": "All finishing tools and equipment in good condition",
            "equipment": ["paint sprayers", "flooring tools", "carpentry tools"],
            "rental_cost_per_day": "₹3,000"
        }
    }
    
    default_response = {
        "available": True,
        "details": "Standard construction equipment available",
        "equipment": ["basic tools"],
        "rental_cost_per_day": "₹2,500"
    }
    
    result = equipment_scenarios.get(task_category, default_response)
    
    # Allocate equipment cost based on task category and duration
    total_equipment_cost = project_cost["equipment_cost"]
    category_cost_allocation = {
        "permits": 0.05,      # 5% of equipment cost
        "site_preparation": 0.25,  # 25% of equipment cost
        "foundation": 0.20,   # 20% of equipment cost
        "structural": 0.30,   # 30% of equipment cost
        "utilities": 0.10,    # 10% of equipment cost
        "finishing": 0.10     # 10% of equipment cost
    }
    
    allocation = category_cost_allocation.get(task_category, 0.1)
    task_equipment_cost = int(total_equipment_cost * allocation)
    
    result.update({
        "check_date": datetime.now().strftime("%Y-%m-%d"),
        "task_name": task.get("name", "Unknown task"),
        "maintenance_status": "good",
        "operator_required": True,
        "booking_lead_time_days": 3,
        "estimated_total_cost": f"₹{task_equipment_cost:,}",
        "cost_allocation_percentage": allocation * 100
    })
    
    return result


def validate_all_resources(task: Dict[str, Any], project_cost: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive resource validation for a task with deterministic costs"""
    labor = check_labor_availability(task)
    materials = check_material_availability(task, project_cost)
    equipment = check_equipment_availability(task, project_cost)
    
    all_available = labor["available"] and materials["available"] and equipment["available"]
    
    # Calculate labor cost based on task duration and category
    total_labor_cost = project_cost["labor_cost"]
    category_labor_allocation = {
        "permits": 0.05,      # 5% of labor cost
        "site_preparation": 0.15,  # 15% of labor cost
        "foundation": 0.20,   # 20% of labor cost
        "structural": 0.30,   # 30% of labor cost
        "utilities": 0.20,    # 20% of labor cost
        "finishing": 0.10     # 10% of labor cost
    }
    
    task_category = task.get("category", "").lower()
    labor_allocation = category_labor_allocation.get(task_category, 0.1)
    task_labor_cost = int(total_labor_cost * labor_allocation)
    
    # Parse material and equipment costs
    material_cost_str = materials["estimated_cost"].replace("₹", "").replace(",", "")
    equipment_cost_str = equipment["estimated_total_cost"].replace("₹", "").replace(",", "")
    
    try:
        material_cost_num = int(material_cost_str)
        equipment_cost_num = int(equipment_cost_str)
    except ValueError:
        material_cost_num = 0
        equipment_cost_num = 0
    
    total_cost = task_labor_cost + material_cost_num + equipment_cost_num
    
    return {
        "task_id": task.get("id", ""),
        "task_name": task.get("name", ""),
        "overall_available": all_available,
        "validation_status": "approved" if all_available else "needs_review",
        "labor": labor,
        "materials": materials,
        "equipment": equipment,
        "total_estimated_cost": f"₹{total_cost:,}",
        "labor_cost": f"₹{task_labor_cost:,}",
        "material_cost": f"₹{material_cost_num:,}",
        "equipment_cost": f"₹{equipment_cost_num:,}",
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
