"""
Simplified test version of the Streamlit app
"""
import streamlit as st
import os
import json
from datetime import datetime
from simple_crew import SimpleConstructionPlanner

# Configure page
st.set_page_config(
    page_title="Construction Planning Test",
    page_icon="🏗️",
    layout="wide"
)

def check_api_key():
    """Check if Groq API key is configured"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY environment variable not set!")
        st.code("export GROQ_API_KEY='your-api-key-here'")
        return False
    return True

def main():
    """Main test application"""
    st.title("🏗️ Construction Planning Assistant - Test Version")
    
    # Check API key
    if not check_api_key():
        st.stop()
    
    st.success("✅ API Key configured successfully!")
    
    # Input section
    st.header("🎯 Test Construction Planning")
    
    construction_goal = st.text_input(
        "Enter construction goal:",
        value="Build a small house",
        help="Try: 'Build a residential home', 'Construct office building', etc."
    )
    
    if st.button("🚀 Test Generate Plan", type="primary"):
        if not construction_goal.strip():
            st.error("Please enter a construction goal!")
            return
        
        st.info(f"🎯 Processing goal: {construction_goal}")
        
        with st.spinner("🤖 AI Agents working..."):
            try:
                # Show initialization step
                st.write("📝 Step 1: Initializing planner...")
                planner = SimpleConstructionPlanner()
                st.success("✅ Planner initialized")
                
                # Show planning step
                st.write("🔨 Step 2: Generating construction plan...")
                results = planner.plan_construction_project(construction_goal)
                st.success("✅ Plan generated successfully!")
                
                # Display results
                if results.get("status") == "completed":
                    st.header("📊 Results")
                    
                    # Metadata
                    metadata = results.get("project_metadata", {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tasks", metadata.get("total_tasks", 0))
                    with col2:
                        st.metric("Duration (Days)", metadata.get("total_duration_days", 0))
                    with col3:
                        st.metric("Estimated Cost", metadata.get("total_estimated_cost", "$0"))
                    
                    # Task breakdown
                    st.subheader("📋 Task Breakdown")
                    tasks = results.get("task_breakdown", {}).get("tasks", [])
                    
                    if tasks:
                        for i, task in enumerate(tasks, 1):
                            with st.expander(f"Task {i}: {task.get('name', 'Unnamed')}"):
                                st.write(f"**Category:** {task.get('category', 'N/A')}")
                                st.write(f"**Duration:** {task.get('estimated_duration_days', 0)} days")
                                st.write(f"**Description:** {task.get('description', 'No description')}")
                    
                    # Success message
                    st.success(f"🎉 Successfully generated construction plan with {len(tasks)} tasks!")
                    
                else:
                    st.error("❌ Failed to generate plan")
                    error = results.get("error", {})
                    st.error(f"Error: {error.get('message', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Exception occurred: {str(e)}")
                st.code("Check the terminal for detailed error logs")

if __name__ == "__main__":
    main()
