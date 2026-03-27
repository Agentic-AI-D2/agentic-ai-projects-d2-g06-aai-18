# 🏗️ Construction Planning Assistant Agent

An AI-powered construction project planning system that uses multi-agent architecture to decompose high-level construction goals into actionable tasks, validate resources, and generate optimized schedules.

## 🚀 Features

- **Multi-Agent System**: Uses CrewAI with specialized agents for planning, validation, and scheduling
- **Resource Validation**: Mock tools for checking labor, materials, and equipment availability
- **Intelligent Scheduling**: Creates optimized timelines with critical path analysis
- **Interactive UI**: Beautiful Streamlit interface with visualizations
- **Groq-Powered**: Fast LLM responses using Groq API
- **Export Capabilities**: Download results as JSON or text summaries

## 🏛️ System Architecture

### Multi-Agent Workflow

1. **Planner Agent**: Breaks down construction goals into detailed tasks
2. **Resource Validator Agent**: Checks labor, material, and equipment availability
3. **Scheduler Agent**: Creates optimized execution timelines

### Tech Stack

- **Backend**: Python with CrewAI framework
- **LLM**: Groq API (Llama3-70B model)
- **Frontend**: Streamlit with Plotly visualizations
- **Tools**: Custom mock resource validation tools

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key
- Git (for cloning)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd construction_planner_agent
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Groq API Key

#### Option 1: Environment Variable (Recommended)

```bash
# On Windows:
set GROQ_API_KEY="your-groq-api-key-here"

# On macOS/Linux:
export GROQ_API_KEY="your-groq-api-key-here"
```

#### Option 2: Create .env file

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-groq-api-key-here
```

> 💡 **How to get Groq API Key:**
> 1. Visit [Groq Console](https://console.groq.com/)
> 2. Sign up or log in
> 3. Navigate to API Keys section
> 4. Create a new API key
> 5. Copy the key and use it in the configuration above

## 🚀 Running the Application

### Method 1: Direct Streamlit Run

```bash
streamlit run app.py
```

### Method 2: Python Run

```bash
python app.py
```

The application will start and open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Start the Application

Launch the Streamlit app using one of the methods above.

### 2. Enter Construction Goal

In the main interface, enter your construction goal such as:
- "Build a residential home"
- "Construct a commercial office building"
- "Site preparation for new development"
- "Foundation planning for warehouse"

### 3. Generate Plan

Click the **"🚀 Generate Plan"** button to start the AI workflow.

### 4. Review Results

The system will display:
- **Task Breakdown**: Detailed list of construction tasks with categories and durations
- **Resource Validation**: Availability status for labor, materials, and equipment
- **Project Schedule**: Timeline visualization with Gantt chart
- **Project Health**: Risk assessment and confidence metrics

### 5. Export Results

Download your construction plan as:
- **JSON**: Complete data structure for integration
- **Text Summary**: Human-readable overview

## 🏗️ Project Structure

```
construction_planner_agent/
│
├── app.py                  # Streamlit UI application
├── crew.py                 # CrewAI workflow orchestration
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── agents/                # AI Agent implementations
│   ├── planner.py         # Task decomposition agent
│   ├── validator.py       # Resource validation agent
│   └── scheduler.py       # Project scheduling agent
│
├── tools/                 # Mock resource tools
│   └── resource_tools.py  # Labor, material, equipment checks
│
└── config/                # Configuration files
    └── llm_config.py      # Groq LLM setup and prompts
```

## 🔧 Configuration

### LLM Settings

Edit `config/llm_config.py` to modify:
- Model selection (currently `llama3-70b-8192`)
- Temperature and token limits
- Agent system prompts

### Resource Tools

Customize `tools/resource_tools.py` to:
- Adjust availability probabilities
- Modify cost estimations
- Add new resource categories

## 🎯 Example Use Cases

### Residential Construction
**Input**: "Build a 2-story residential home with 3 bedrooms"

**Output**:
- 12-15 detailed tasks
- Resource availability validation
- 45-60 day timeline
- Critical path identification

### Commercial Projects
**Input**: "Construct a commercial office building"

**Output**:
- 15-20 comprehensive tasks
- Enhanced resource requirements
- 90-120 day schedule
- Risk assessment for complex dependencies

### Site Preparation
**Input**: "Site preparation for new development"

**Output**:
- 8-10 specialized tasks
- Heavy equipment requirements
- 15-30 day timeline
- Weather considerations

## 🐛 Troubleshooting

### Common Issues

#### 1. API Key Error
```
GROQ_API_KEY environment variable not set!
```
**Solution**: Ensure your Groq API key is properly set as an environment variable.

#### 2. Module Import Errors
```
ModuleNotFoundError: No module named 'crewai'
```
**Solution**: Install dependencies using `pip install -r requirements.txt`

#### 3. Streamlit Not Starting
```
Error: Could not find a suitable TLS certificate
```
**Solution**: Run with `streamlit run app.py --server.headless true`

#### 4. LLM Response Errors
```
Failed to parse LLM response
```
**Solution**: Check your internet connection and Groq API status. The system includes fallback mechanisms.

### Debug Mode

Enable verbose logging by setting:
```bash
export CREWAI_VERBOSE=true
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit: `git commit -am 'Add new feature'`
5. Push: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CrewAI**: Multi-agent orchestration framework
- **Groq**: Fast LLM inference API
- **Streamlit**: Interactive web application framework
- **Plotly**: Data visualization library

## 📞 Support

For support and questions:

1. Check the troubleshooting section above
2. Review the code comments for detailed explanations
3. Create an issue in the repository for bugs or feature requests

---

**Built with ❤️ using AI agents for construction planning**
