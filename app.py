"""
Streamlit UI for Construction Planning Assistant Agent
Main application interface for the AI-powered construction planning system
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from datetime import datetime
from simple_crew import SimpleConstructionPlanner
from pdf_generator import prepare_pdf_data_from_results, generate_pdf
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Configure Streamlit page
st.set_page_config(
    page_title="Construction Planning Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def check_api_key():
    """Check if Groq API key is configured"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY environment variable not set!")
        st.info("Please set the GROQ_API_KEY environment variable and restart the application.")
        st.code("export GROQ_API_KEY='your-api-key-here'")
        return False
    return True


def display_project_metadata(metadata):
    """Display project metadata in a formatted way"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tasks", metadata.get("total_tasks", 0))
    
    with col2:
        duration = metadata.get("total_duration_days", 0)
        st.metric("Duration (Days)", duration)
    
    with col3:
        cost = metadata.get("total_estimated_cost", "₹0")
        st.metric("Estimated Cost", cost)
    
    with col4:
        created_date = metadata.get("created_date", "").split()[0]
        st.metric("Created", created_date)


def display_ai_insights(results):
    """Display AI-powered insights and recommendations"""
    ai_insights = results.get("ai_insights", {})
    
    if ai_insights.get("status") == "success" and ai_insights.get("ai_enhancement", False):
        st.subheader("🤖 AI Insights & Recommendations")
        
        # AI Analysis
        ai_analysis = ai_insights.get("ai_analysis", {})
        if ai_analysis:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                complexity = ai_analysis.get("project_complexity", "moderate")
                st.metric("Project Complexity", complexity.title())
            
            with col2:
                considerations = ai_analysis.get("key_considerations", [])
                st.metric("Key Considerations", len(considerations))
            
            with col3:
                risk_factors = ai_analysis.get("risk_factors", [])
                st.metric("Risk Factors", len(risk_factors))
        
        # AI Recommendations
        recommendations = ai_insights.get("ai_recommendations", [])
        if recommendations:
            st.markdown("**📋 AI Recommendations:**")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        
        # Enhanced Tasks
        enhanced_tasks = ai_insights.get("enhanced_tasks", [])
        if enhanced_tasks:
            st.markdown("**🔧 AI-Generated Enhanced Tasks:**")
            for task in enhanced_tasks:
                with st.expander(f"🤖 {task.get('name', 'AI Task')} ({task.get('estimated_days', 0)} days)"):
                    st.write(f"**Category:** {task.get('category', 'AI')}")
                    st.write(f"**Priority:** {task.get('priority', 'medium').title()}")
                    st.write(f"**Description:** {task.get('description', 'No description')}")
        
        # Duration Insights
        duration_insights = ai_insights.get("duration_insights", {})
        if duration_insights:
            st.markdown("**⏱️ Duration Insights:**")
            col1, col2 = st.columns(2)
            
            with col1:
                critical_tasks = duration_insights.get("critical_path_tasks", [])
                if critical_tasks:
                    st.write("**Critical Path Tasks:**")
                    for task in critical_tasks:
                        st.write(f"• {task}")
                
                potential_delays = duration_insights.get("potential_delays", [])
                if potential_delays:
                    st.write("**Potential Delays:**")
                    for delay in potential_delays:
                        st.write(f"• {delay}")
            
            with col2:
                optimizations = duration_insights.get("optimization_suggestions", [])
                if optimizations:
                    st.write("**Optimization Suggestions:**")
                    for opt in optimizations:
                        st.write(f"• {opt}")
    
    else:
        # Show fallback message
        st.info("🤖 **AI Insights**: AI enhancement is currently unavailable. Using rule-based planning only.")


def display_feature_detection(results):
    """Display feature extraction results"""
    feature_data = results.get("feature_extraction", {})
    
    st.subheader("🔍 Feature Detection Summary")
    
    if feature_data.get("has_features", False):
        # Show detected features
        features_detected = feature_data.get("features_detected", [])
        if features_detected:
            st.markdown("**✨ Detected Features:**")
            
            for feature in features_detected:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{feature['description']}**")
                
                with col2:
                    st.metric("Cost Impact", feature['cost_impact'])
                
                with col3:
                    st.metric("Days Impact", f"{feature['days_impact']}")
        
        # Show cost adjustment summary
        cost_multiplier = feature_data.get("cost_multiplier", 1.0)
        extra_days = feature_data.get("extra_days", 0)
        extra_tasks_count = feature_data.get("extra_tasks_count", 0)
        
        st.markdown("**📊 Feature Impact Summary:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cost Multiplier", f"{cost_multiplier:.2f}x")
        
        with col2:
            st.metric("Extra Days", f"{extra_days} days")
        
        with col3:
            st.metric("Extra Tasks", extra_tasks_count)
        
        # Show feature summary
        feature_summary = feature_data.get("feature_summary", "")
        if feature_summary:
            st.info(f"📋 **Summary**: {feature_summary}")
    
    else:
        st.info("🔍 **No Special Features Detected**: Using standard construction planning based on your parameters.")


def display_cost_breakdown(results):
    """Display detailed cost breakdown in INR with AI enhancements"""
    if "cost_breakdown" in results:
        cost_data = results["cost_breakdown"]
        
        st.subheader("💰 Cost Breakdown (INR)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Cost", 
                cost_data.get("total_cost", "₹0"),
                help="Total project cost including all components"
            )
            st.metric(
                "Cost per sq ft", 
                cost_data.get("cost_per_sqft", "₹0"),
                help="Base cost per square foot before location adjustment"
            )
        
        with col2:
            st.metric(
                "Labor Cost", 
                cost_data.get("labor_cost", "₹0"),
                help=f"40% of total cost - {results.get('project_metadata', {}).get('quality', 'Standard')} quality"
            )
            st.metric(
                "Material Cost", 
                cost_data.get("material_cost", "₹0"),
                help="50% of total cost - includes all construction materials"
            )
        
        with col3:
            st.metric(
                "Equipment Cost", 
                cost_data.get("equipment_cost", "₹0"),
                help="10% of total cost - equipment rental and tools"
            )
            st.metric(
                "Location Factor", 
                f"{cost_data.get('location_factor', 1.0)}x",
                help=f"Location multiplier - {results.get('project_metadata', {}).get('location', 'Tier 2')}"
            )
        
        # Show area breakdown for transparency
        st.markdown("**Area Calculation:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Ground Floor Area", 
                f"{cost_data.get('ground_floor_area', 0)} sq ft",
                help="Input area per floor"
            )
        
        with col2:
            st.metric(
                "Effective Floors", 
                cost_data.get("effective_floors", 1.0),
                help="Floor scaling factor (1 + 0.9 × (floors - 1))"
            )
        
        with col3:
            st.metric(
                "Total Built-up Area", 
                f"{cost_data.get('total_builtup_area', 0)} sq ft",
                help="Ground floor area × Effective floors"
            )
        
        # Show feature-based cost adjustments
        base_cost = cost_data.get("base_cost", "")
        cost_increase = cost_data.get("cost_increase", "")
        feature_multiplier = cost_data.get("feature_multiplier", 1.0)
        
        if base_cost and cost_increase and feature_multiplier != 1.0:
            st.markdown("**💡 Feature-Based Cost Adjustments:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Base Cost", base_cost)
            
            with col2:
                st.metric("Feature Multiplier", f"{feature_multiplier:.2f}x")
            
            with col3:
                st.metric("Cost Increase", cost_increase)
        
        # Show AI optimization tips
        ai_tips = cost_data.get("ai_optimization_tips", [])
        cost_savers = cost_data.get("potential_cost_savers", [])
        
        if ai_tips or cost_savers:
            st.markdown("**🤖 AI Cost Optimization:**")
            
            if ai_tips:
                st.write("**Optimization Tips:**")
                for tip in ai_tips:
                    st.write(f"• {tip}")
            
            if cost_savers:
                st.write("**Potential Cost Savers:**")
                for saver in cost_savers:
                    st.write(f"• {saver}")


def display_duration_breakdown(results):
    """Display detailed duration breakdown with factors and feature adjustments"""
    if "duration_breakdown" in results:
        duration_data = results["duration_breakdown"]
        
        st.subheader("📅 Duration Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Duration", f"{duration_data.get('total_days', 0)} days")
            st.metric("Foundation Phase", f"{duration_data.get('foundation_days', 0)} days")
        
        with col2:
            st.metric("Structural Phase", f"{duration_data.get('structure_days', 0)} days")
            st.metric("Finishing Phase", f"{duration_data.get('finishing_days', 0)} days")
        
        # Show calculation factors
        st.markdown("**Duration Calculation Factors:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Base Duration", 
                f"{duration_data.get('base_duration', 0)} days",
                help=f"Base timeline for {duration_data.get('building_type', 'Residential')} construction"
            )
        
        with col2:
            st.metric(
                "Floor Factor", 
                f"{duration_data.get('floor_factor', 1.0)}x",
                help="Each additional floor adds 30% time"
            )
        
        with col3:
            st.metric(
                "Area Factor", 
                f"{duration_data.get('area_factor', 1.0)}x",
                help="Area adjustment (clamped to realistic range)"
            )
        
        # Show feature-based duration adjustments
        extra_days = duration_data.get("extra_days", 0)
        duration_increase = duration_data.get("duration_increase", 0)
        
        if extra_days != 0:
            st.markdown("**💡 Feature-Based Duration Adjustments:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Extra Days", f"{extra_days} days")
            
            with col2:
                st.metric("Duration Increase", f"{duration_increase} days")
        
        # Duration per sq ft indicator
        days_per_sqft = duration_data.get('days_per_sqft', 0)
        if days_per_sqft > 0:
            st.info(f"📊 Construction Speed: {days_per_sqft:.3f} days per square foot")
        
        # Show AI duration insights
        duration_insights = duration_data.get("duration_insights", {})
        if duration_insights:
            st.markdown("**🤖 AI Duration Insights:**")
            
            critical_tasks = duration_insights.get("critical_path_tasks", [])
            potential_delays = duration_insights.get("potential_delays", [])
            optimizations = duration_insights.get("optimization_suggestions", [])
            
            if critical_tasks or potential_delays or optimizations:
                col1, col2 = st.columns(2)
                
                with col1:
                    if critical_tasks:
                        st.write("**Critical Path Tasks:**")
                        for task in critical_tasks:
                            st.write(f"• {task}")
                    
                    if potential_delays:
                        st.write("**Potential Delays:**")
                        for delay in potential_delays:
                            st.write(f"• {delay}")
                
                with col2:
                    if optimizations:
                        st.write("**Optimization Suggestions:**")
                        for opt in optimizations:
                            st.write(f"• {opt}")
        
        # Add explanation
        st.info("""
        📋 **Duration Methodology**: 
        Duration is estimated using phase-based construction modeling adjusted for floors and area, 
        based on typical Indian residential and commercial project timelines.
        - Base: 120 days (Residential) | 180 days (Commercial)
        - Floor scaling: 30% per additional floor
        - Area adjustment: 20% variation from 2,000 sq ft baseline
        - Phase distribution: Foundation 20% | Structure 50% | Finishing 30%
        """)


def display_estimation_note():
    """Display transparency note about estimation methodology"""
    st.info("""
    📋 **Cost Estimation Methodology**: 
    Cost estimates are based on standard Indian construction benchmarks and rule-based calculations.
    - Costs calculated using industry-standard rates per square foot
    - Location factors account for regional price variations  
    - Quality grades adjust material and finishing standards
    - Labor: 40% | Materials: 50% | Equipment: 10%
    """)


def display_task_breakdown(task_breakdown):
    """Display task breakdown with visualizations"""
    tasks = task_breakdown.get("tasks", [])
    
    if not tasks:
        st.warning("No tasks available")
        return
    
    st.subheader("📋 Task Overview")
    
    # Create DataFrame for visualization
    df = pd.DataFrame(tasks)
    
    # Task categories distribution
    if 'category' in df.columns:
        fig = px.pie(
            df, 
            names='category', 
            title='Task Distribution by Category',
            color_discrete_map={
                'permits': '#FF6B6B',
                'site_preparation': '#4ECDC4',
                'foundation': '#45B7D1',
                'structural': '#96CEB4',
                'utilities': '#FFEAA7',
                'finishing': '#DDA0DD'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Task duration timeline
    if 'estimated_duration_days' in df.columns and 'name' in df.columns:
        df_sorted = df.sort_values('estimated_duration_days')
        fig = px.bar(
            df_sorted,
            x='estimated_duration_days',
            y='name',
            orientation='h',
            title='Task Duration Timeline',
            color='estimated_duration_days',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed task list
    st.subheader("📝 Detailed Task List")
    
    for i, task in enumerate(tasks, 1):
        with st.expander(f"Task {i}: {task.get('name', 'Unnamed Task')} ({task.get('estimated_duration_days', 0)} days)"):
            # Task basic information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**ID:** {task.get('id', 'N/A')}")
                st.write(f"**Category:** {task.get('category', 'N/A').title()}")
            
            with col2:
                st.write(f"**Duration:** {task.get('estimated_duration_days', 0)} days")
                dependencies = task.get('dependencies', [])
                if dependencies:
                    st.write(f"**Dependencies:** {', '.join(dependencies)}")
                else:
                    st.write("**Dependencies:** None")
            
            with col3:
                # Add phase information based on category
                category = task.get('category', '').lower()
                phase_map = {
                    'permits': 'Pre-Construction',
                    'site_preparation': 'Pre-Construction', 
                    'foundation': 'Foundation Phase',
                    'structural': 'Structural Phase',
                    'utilities': 'Structural Phase',
                    'finishing': 'Finishing Phase'
                }
                phase = phase_map.get(category, 'General')
                st.write(f"**Phase:** {phase}")
            
            # Enhanced description section
            st.markdown("---")
            st.write("**📋 Task Description:**")
            description = task.get('description', 'No description available')
            st.write(description)
            
            # Add task considerations
            if category in ['foundation', 'structural']:
                st.info("🔧 **Critical Task:** This task is on the critical path and may impact overall project timeline.")
            elif category == 'permits':
                st.info("📜 **Important:** Ensure all permits are approved before proceeding with construction.")
            elif category == 'utilities':
                st.info("⚡ **Coordination Required:** May require coordination with utility providers.")


def display_resource_validation(validation_results):
    """Display resource validation results"""
    validated_tasks = validation_results.get("validated_tasks", [])
    approved_tasks = validation_results.get("approved_tasks", 0)
    blocked_tasks = validation_results.get("blocked_tasks", 0)
    
    st.subheader("✅ Resource Validation Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tasks", len(validated_tasks))
    
    with col2:
        st.metric("Approved", approved_tasks, delta=None, delta_color="normal")
    
    with col3:
        st.metric("Blocked", blocked_tasks, delta=None, delta_color="inverse")
    
    # Validation status distribution
    if validated_tasks:
        status_counts = {}
        for task in validated_tasks:
            status = task.get('validation_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        fig = px.pie(
            values=list(status_counts.values()),
            names=list(status_counts.keys()),
            title='Validation Status Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed validation results
    st.subheader("🔍 Detailed Validation")
    
    for task in validated_tasks:
        status = task.get('validation_status', 'unknown')
        task_name = task.get('task_name', 'Unnamed Task')
        
        # Color code based on status
        if status == 'approved':
            icon = "✅"
            color = "success"
        elif status == 'needs_review':
            icon = "⚠️"
            color = "warning"
        else:
            icon = "❌"
            color = "error"
        
        with st.expander(f"{icon} {task_name} - {status.upper()}"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Labor availability
                labor = task.get('labor', {})
                labor_available = labor.get('available', False)
                labor_icon = "✅" if labor_available else "❌"
                st.write(f"**Labor:** {labor_icon} {'Available' if labor_available else 'Not Available'}")
                if labor.get('details'):
                    st.write(f"Details: {labor['details']}")
            
            with col2:
                # Material availability
                materials = task.get('materials', {})
                material_available = materials.get('available', False)
                material_icon = "✅" if material_available else "❌"
                st.write(f"**Materials:** {material_icon} {'Available' if material_available else 'Not Available'}")
                if materials.get('details'):
                    st.write(f"Details: {materials['details']}")
            
            # Equipment availability
            equipment = task.get('equipment', {})
            equipment_available = equipment.get('available', False)
            equipment_icon = "✅" if equipment_available else "❌"
            st.write(f"**Equipment:** {equipment_icon} {'Available' if equipment_available else 'Not Available'}")
            if equipment.get('details'):
                st.write(f"Details: {equipment['details']}")
            
            # Cost and notes
            st.write(f"**Estimated Cost:** {task.get('total_estimated_cost', 'N/A')}")
            st.write(f"**Notes:** {task.get('validation_notes', 'No notes available')}")


def display_project_schedule(schedule_results):
    """Display project schedule with timeline visualization"""
    schedule = schedule_results.get("schedule", [])
    total_duration = schedule_results.get("total_project_duration", 0)
    critical_path_tasks = schedule_results.get("critical_path_tasks", [])
    
    st.subheader("📅 Project Schedule")
    
    # Schedule overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Duration", f"{total_duration} days")
    
    with col2:
        st.metric("Critical Path Tasks", len(critical_path_tasks))
    
    with col3:
        buffer_days = schedule_results.get("buffer_days", 0)
        if buffer_days > 0:
            st.metric("Buffer Days", buffer_days)
        else:
            st.metric("Buffer Days", "None")
    
    # Gantt chart visualization
    if schedule:
        st.subheader("📊 Project Timeline (Gantt Chart)")
        
        # Prepare data for Gantt chart
        gantt_data = []
        for task in schedule:
            task_name = task.get('task_name', 'Unnamed')
            duration = task.get('duration_days', 0)
            # Include duration in task name for better visualization
            display_name = f"{task_name} ({duration} days)"
            
            gantt_data.append({
                'Task': display_name,
                'Start': task.get('start_day', 0),
                'Finish': task.get('end_day', 0),
                'Category': task.get('task_id', 'general'),
                'Critical Path': 'Critical' if task.get('critical_path', False) else 'Regular',
                'Duration': duration
            })
        
        df_gantt = pd.DataFrame(gantt_data)
        
        # Create Gantt chart
        fig = px.timeline(
            df_gantt,
            x_start="Start",
            x_end="Finish",
            y="Task",
            color="Critical Path",
            title="Construction Project Timeline",
            color_discrete_map={'Critical': '#FF6B6B', 'Regular': '#4ECDC4'}
        )
        
        fig.update_yaxes(categoryorder='total ascending')
        fig.update_layout(
            xaxis_title="Project Day",
            yaxis_title="Tasks",
            height=max(400, len(schedule) * 30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Task summary table
        st.subheader("📋 Task Summary Table")
        
        # Prepare summary data
        summary_data = []
        for task in schedule:
            summary_data.append({
                'Task Name': task.get('task_name', 'Unnamed'),
                'Duration (days)': task.get('duration_days', 0),
                'Start Day': task.get('start_day', 0),
                'End Day': task.get('end_day', 0),
                'Critical Path': 'Yes' if task.get('critical_path', False) else 'No',
                'Dependencies': ', '.join(task.get('dependencies_completed', [])) if task.get('dependencies_completed') else 'None'
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    # Project phases
    phases = schedule_results.get("project_phases", [])
    if phases:
        st.subheader("🏗️ Project Phases")
        
        for phase in phases:
            with st.expander(f"📋 {phase.get('phase', 'Unnamed Phase')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Start Day:** {phase.get('start_day', 'N/A')}")
                    st.write(f"**End Day:** {phase.get('end_day', 'N/A')}")
                    st.write(f"**Duration:** {phase.get('end_day', 0) - phase.get('start_day', 0) + 1} days")
                
                with col2:
                    tasks = phase.get('tasks', [])
                    st.write(f"**Number of Tasks:** {len(tasks)}")
                    if tasks:
                        st.write("**Task IDs:**")
                        for task_id in tasks[:5]:  # Show first 5 tasks
                            st.write(f"- {task_id}")
                        if len(tasks) > 5:
                            st.write(f"... and {len(tasks) - 5} more")
    
    # Schedule optimization insights
    optimization_suggestions = schedule_results.get("optimization_suggestions", [])
    if optimization_suggestions:
        st.subheader("💡 Schedule Optimization")
        for suggestion in optimization_suggestions:
            st.write(f"• {suggestion}")


def display_project_health(health_metrics):
    """Display project health and risk assessment"""
    st.subheader("🏥 Project Health Assessment")
    
    # Health metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        approval_rate = health_metrics.get("approval_rate_percentage", 0)
        st.metric("Task Approval Rate", f"{approval_rate}%")
    
    with col2:
        confidence = health_metrics.get("schedule_confidence", 0)
        st.metric("Schedule Confidence", f"{confidence}/10")
    
    with col3:
        risk_level = health_metrics.get("risk_level", "Unknown")
        risk_color = {"Low": "normal", "Medium": "inverse", "High": "inverse"}[risk_level]
        st.metric("Risk Level", risk_level, delta=None, delta_color=risk_color)


def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🏗️ Construction Planning Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">AI-powered construction project planning and resource management</p>', unsafe_allow_html=True)
    
    # Check API key
    if not check_api_key():
        st.stop()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Example goals
    example_goals = [
        "Build a residential home",
        "Construct a commercial office building",
        "Site preparation for new development",
        "Foundation planning for warehouse",
        "Permit requirements for renovation project"
    ]
    
    st.sidebar.subheader("📝 Example Goals")
    for goal in example_goals:
        if st.sidebar.button(goal):
            st.session_state.example_goal = goal
    
    # Main input section
    st.header("🎯 Project Input")
    
    # Get goal from session state or input
    if 'example_goal' in st.session_state:
        default_goal = st.session_state.example_goal
        del st.session_state.example_goal
    else:
        default_goal = ""
    
    st.subheader("📋 Project Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input(
            "Total Area (sq ft)", 
            min_value=100, 
            max_value=100000, 
            value=2000, 
            step=100,
            help="Total built-up area in square feet"
        )
        
        floors = st.number_input(
            "Number of Floors", 
            min_value=1, 
            max_value=50, 
            value=2, 
            step=1,
            help="Number of floors including ground floor"
        )
        
        building_type = st.selectbox(
            "Construction Type", 
            ["Residential", "Commercial"],
            help="Type of construction project"
        )
    
    with col2:
        quality = st.selectbox(
            "Quality Grade", 
            ["Basic", "Standard", "Premium"],
            index=1,
            help="Construction quality and finishing level"
        )
        
        location = st.selectbox(
            "Location Type", 
            ["Metro", "Tier 2", "Rural"],
            index=1,
            help="Location category for cost calculation"
        )
        
        construction_goal = st.text_input(
            "Project Description (Optional):",
            placeholder="e.g., 3-bedroom house with modern amenities",
            help="Additional details about your project"
        )
    
    # Create project parameters object
    project_params = {
        "area": area,
        "floors": floors,
        "building_type": building_type,
        "quality": quality,
        "location": location,
        "description": construction_goal or f"{building_type} building - {area} sq ft, {floors} floors"
    }
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        generate_button = st.button("🚀 Generate Plan", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear Results", use_container_width=True)
    
    # Clear results
    if clear_button:
        if 'planning_results' in st.session_state:
            del st.session_state.planning_results
        st.rerun()
    
    # Generate plan
    if generate_button:
        if not area or not floors:
            st.error("Please fill in all required project parameters!")
            return
        
        with st.spinner("🤖 AI Agents are working on your construction plan..."):
            try:
                # Initialize the planner
                planner = SimpleConstructionPlanner()
                
                # Execute planning workflow with project parameters
                results = planner.plan_construction_project(project_params)
                
                # Store results in session state
                st.session_state.planning_results = results
                
                if results.get("status") == "completed":
                    st.success("✅ Construction plan generated successfully!")
                else:
                    st.error("❌ Failed to generate construction plan")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please check your API configuration and try again.")
    
    # Display results
    if 'planning_results' in st.session_state:
        results = st.session_state.planning_results
        
        if results.get("status") == "completed":
            # Project metadata
            st.header("📊 Project Overview")
            metadata = results.get("project_metadata", {})
            display_project_metadata(metadata)
            
            # Feature Detection (NEW)
            display_feature_detection(results)
            
            # Cost breakdown
            display_cost_breakdown(results)
            
            # Duration breakdown
            display_duration_breakdown(results)
            
            # AI Insights (NEW)
            display_ai_insights(results)
            
            # Estimation methodology note
            display_estimation_note()
            
            # Summary section
            summary = results.get("summary", {})
            if summary:
                st.subheader("📋 Project Summary")
                st.write(summary.get("project_overview", ""))
                
                highlights = summary.get("key_highlights", [])
                if highlights:
                    st.write("**Key Highlights:**")
                    for highlight in highlights:
                        st.write(f"• {highlight}")
            
            # Tabbed interface for detailed results
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Tasks", "✅ Validation", "📅 Schedule", "🏥 Health"])
            
            with tab1:
                task_breakdown = results.get("task_breakdown", {})
                display_task_breakdown(task_breakdown)
            
            with tab2:
                validation_results = results.get("resource_validation", {})
                display_resource_validation(validation_results)
            
            with tab3:
                schedule_results = results.get("project_schedule", {})
                display_project_schedule(schedule_results)
            
            with tab4:
                health_metrics = results.get("project_health", {})
                display_project_health(health_metrics)
            
            # Next steps
            next_steps = summary.get("next_steps", [])
            if next_steps:
                st.header("🎯 Recommended Next Steps")
                for i, step in enumerate(next_steps, 1):
                    st.write(f"{i}. {step}")
            
            # Download results
            st.header("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Download JSON"):
                    json_data = json.dumps(results, indent=2)
                    st.download_button(
                        label="Download construction_plan.json",
                        data=json_data,
                        file_name=f"construction_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📋 Download PDF Report"):
                    try:
                        # Prepare data for PDF
                        pdf_data = prepare_pdf_data_from_results(results)
                        
                        # Generate PDF
                        pdf_bytes = generate_pdf(pdf_data)
                        
                        # Download PDF
                        st.download_button(
                            label="Download construction_report.pdf",
                            data=pdf_bytes,
                            file_name=f"construction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
                        st.info("Please try again or contact support if the issue persists.")
        
        else:
            # Display error
            error_info = results.get("error", {})
            st.error(f"❌ Error: {error_info.get('message', 'Unknown error')}")
            
            suggestions = results.get("fallback_suggestions", [])
            if suggestions:
                st.subheader("💡 Suggestions")
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")


if __name__ == "__main__":
    main()
