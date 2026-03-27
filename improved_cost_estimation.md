# Improved Cost & Duration Estimation Plan

## Current Issues:
- Random cost generation
- No regional pricing data
- LLM guessing durations
- No project complexity factors

## Required Improvements:

### 1. Real Cost Database
```python
# Replace random costs with regional data
REGIONAL_COSTS = {
    "north_america": {
        "labor_per_hour": {
            "carpenter": 45,
            "electrician": 65,
            "plumber": 60,
            "heavy_equipment_operator": 55
        },
        "material_costs_per_unit": {
            "concrete_per_cubic_yard": 125,
            "steel_per_pound": 0.65,
            "lumber_per_board_foot": 0.85
        }
    }
}
```

### 2. Project Parameters Input
Add fields to the UI:
- Building size (sq ft/meters)
- Number of floors
- Quality level (basic/standard/premium)
- Location/region
- Project complexity

### 3. Industry-Standard Duration Formulas
```python
# Replace LLM guesses with construction formulas
def calculate_foundation_duration(size_sqft, soil_type):
    base_days = size_sqft / 1000  # 1 day per 1000 sq ft
    soil_multiplier = {"stable": 1.0, "rocky": 1.5, "soft": 1.3}
    return base_days * soil_multiplier[soil_type]
```

### 4. Enhanced Resource Tools
- Connect to real supplier APIs
- Include seasonal factors
- Add labor union rates
- Consider permit processing times by location

### 5. Risk Factors
- Weather delays by region
- Supply chain disruptions
- Labor availability fluctuations
- Material price volatility

## Implementation Priority:
1. Add project parameter inputs to UI
2. Create regional cost database
3. Implement industry-standard duration calculations
4. Add risk adjustment factors
5. Connect to real data sources
