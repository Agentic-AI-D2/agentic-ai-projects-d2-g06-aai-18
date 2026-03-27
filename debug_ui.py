"""
Debug script to test UI functionality
"""
import os
import sys
from simple_crew import SimpleConstructionPlanner

def test_ui_functionality():
    """Test the exact UI workflow"""
    print("🔍 Testing UI Functionality")
    print("=" * 50)
    
    # Check environment
    api_key = os.getenv("GROQ_API_KEY")
    print(f"API Key: {'✅ Set' if api_key else '❌ Missing'}")
    
    # Test planner initialization
    try:
        planner = SimpleConstructionPlanner()
        print("✅ Planner initialized successfully")
    except Exception as e:
        print(f"❌ Planner initialization failed: {e}")
        return
    
    # Test different construction goals
    test_goals = [
        "Build a small house",
        "Construct a residential home",
        "Site preparation for new development"
    ]
    
    for i, goal in enumerate(test_goals, 1):
        print(f"\n📋 Test {i}: {goal}")
        print("-" * 30)
        
        try:
            result = planner.plan_construction_project(goal)
            
            if result.get("status") == "completed":
                print(f"✅ SUCCESS")
                print(f"   Tasks: {len(result.get('task_breakdown', {}).get('tasks', []))}")
                print(f"   Duration: {result.get('project_metadata', {}).get('total_duration_days', 0)} days")
                print(f"   Cost: {result.get('project_metadata', {}).get('total_estimated_cost', '$0')}")
                
                # Check data structure
                metadata = result.get('project_metadata', {})
                if metadata:
                    print(f"   Goal: {metadata.get('goal', 'N/A')}")
                    print(f"   Created: {metadata.get('created_date', 'N/A')}")
                
            else:
                print(f"❌ FAILED")
                error = result.get('error', {})
                print(f"   Error: {error.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
    
    print("\n🎯 UI Test Complete")
    print("If all tests pass, the issue might be in the Streamlit frontend.")
    print("Try refreshing the browser or checking browser console for errors.")

if __name__ == "__main__":
    test_ui_functionality()
