"""
PDF Generator for Construction Planning Reports
Creates professional PDF reports using reportlab
"""
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, darkblue, darkgreen
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class PDFGenerator:
    """Generates professional PDF reports for construction planning"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the PDF"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=darkblue,
            alignment=TA_CENTER,
            borderWidth=0,
            borderColor=HexColor('#2E86AB')
        ))
        
        # Section heading style
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=darkblue,
            borderWidth=1,
            borderColor=HexColor('#2E86AB'),
            borderPadding=5
        ))
        
        # Subsection style
        self.styles.add(ParagraphStyle(
            name='Subsection',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=15,
            textColor=darkgreen
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leading=14
        ))
    
    def generate_pdf(self, data: Dict[str, Any]) -> bytes:
        """
        Generate a professional PDF report
        
        Args:
            data: Dictionary containing all construction planning data
            
        Returns:
            PDF file as bytes
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        # Build the story (content)
        story = []
        
        # Title
        story.append(Paragraph("Construction Planning Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # 1. Project Information
        story.extend(self._create_project_info_section(data))
        
        # 2. Key Metrics
        story.extend(self._create_key_metrics_section(data))
        
        # 3. Cost Breakdown
        story.extend(self._create_cost_breakdown_section(data))
        
        # 4. Task Breakdown
        story.extend(self._create_task_breakdown_section(data))
        
        # 5. AI Insights (if available)
        if data.get('ai_insights') or data.get('features'):
            story.extend(self._create_ai_insights_section(data))
        
        # 6. Duration Breakdown
        if data.get('duration_breakdown'):
            story.extend(self._create_duration_breakdown_section(data))
        
        # Footer
        story.extend(self._create_footer_section())
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _create_project_info_section(self, data: Dict[str, Any]) -> List:
        """Create project information section"""
        elements = []
        elements.append(Paragraph("Project Information", self.styles['SectionHeading']))
        
        # Project details table
        project_data = [
            ['Project Goal:', data.get('goal', 'N/A')],
            ['Area:', f"{data.get('area', 0)} sq ft"],
            ['Floors:', str(data.get('floors', 0))],
            ['Building Type:', data.get('building_type', 'N/A')],
            ['Location:', data.get('location', 'N/A')],
            ['Quality Grade:', data.get('quality', 'N/A')]
        ]
        
        table = Table(project_data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#F0F8FF')),
            ('TEXTCOLOR', (0, 0), (-1, -1), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#E6E6FA')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_key_metrics_section(self, data: Dict[str, Any]) -> List:
        """Create key metrics section"""
        elements = []
        elements.append(Paragraph("Key Metrics", self.styles['SectionHeading']))
        
        # Metrics data
        total_cost = data.get('total_cost', 0)
        cost_per_sqft = data.get('cost_per_sqft', 0)
        
        metrics_data = [
            ['Total Tasks:', str(data.get('total_tasks', 0))],
            ['Project Duration:', f"{data.get('duration', 0)} days"],
            ['Total Cost:', f"₹{total_cost:,}"],
            ['Cost per sq ft:', f"₹{cost_per_sqft:,}"],
            ['Created Date:', data.get('created_date', 'N/A')]
        ]
        
        table = Table(metrics_data, colWidths=[2.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#F0FFF0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#E6E6FA')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_cost_breakdown_section(self, data: Dict[str, Any]) -> List:
        """Create cost breakdown section"""
        elements = []
        elements.append(Paragraph("Cost Breakdown", self.styles['SectionHeading']))
        
        cost_breakdown = data.get('cost_breakdown', {})
        if not cost_breakdown:
            elements.append(Paragraph("No cost breakdown data available.", self.styles['CustomBody']))
            return elements
        
        # Cost breakdown table
        cost_data = [
            ['Component', 'Amount (₹)', 'Percentage'],
            ['Labor Cost', f"{self._format_currency(cost_breakdown.get('labor_cost', 0))}", '40%'],
            ['Material Cost', f"{self._format_currency(cost_breakdown.get('material_cost', 0))}", '50%'],
            ['Equipment Cost', f"{self._format_currency(cost_breakdown.get('equipment_cost', 0))}", '10%'],
            ['Total Cost', f"{self._format_currency(cost_breakdown.get('total_cost', 0))}", '100%']
        ]
        
        table = Table(cost_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4169E1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), 'white'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, -1), (-1, -1), HexColor('#F0F8FF')),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_task_breakdown_section(self, data: Dict[str, Any]) -> List:
        """Create task breakdown section"""
        elements = []
        elements.append(Paragraph("Task Breakdown", self.styles['SectionHeading']))
        
        tasks = data.get('tasks', [])
        if not tasks:
            elements.append(Paragraph("No task data available.", self.styles['CustomBody']))
            return elements
        
        # Task list
        for i, task in enumerate(tasks, 1):
            task_name = task.get('name', 'Unnamed Task')
            task_duration = task.get('estimated_duration_days', 0)
            task_category = task.get('category', 'N/A')
            task_description = task.get('description', '')
            
            elements.append(Paragraph(f"Task {i}: {task_name}", self.styles['Subsection']))
            
            # Task details
            details = f"• Duration: {task_duration} days<br/>• Category: {task_category.title()}"
            if task_description:
                details += f"<br/>• Description: {task_description}"
            
            elements.append(Paragraph(details, self.styles['CustomBody']))
            elements.append(Spacer(1, 10))
        
        return elements
    
    def _create_ai_insights_section(self, data: Dict[str, Any]) -> List:
        """Create AI insights section"""
        elements = []
        elements.append(Paragraph("AI Insights & Feature Detection", self.styles['SectionHeading']))
        
        # Features detected
        features = data.get('features', [])
        if features:
            elements.append(Paragraph("Detected Features:", self.styles['Subsection']))
            for feature in features:
                feature_desc = feature.get('description', 'Unknown feature')
                cost_impact = feature.get('cost_impact', 'N/A')
                days_impact = feature.get('days_impact', 0)
                
                feature_text = f"• {feature_desc} (Cost Impact: {cost_impact}, Days Impact: {days_impact})"
                elements.append(Paragraph(feature_text, self.styles['CustomBody']))
            elements.append(Spacer(1, 10))
        
        # AI insights
        ai_insights = data.get('ai_insights', {})
        if ai_insights and ai_insights.get('status') == 'success':
            recommendations = ai_insights.get('ai_recommendations', [])
            if recommendations:
                elements.append(Paragraph("AI Recommendations:", self.styles['Subsection']))
                for rec in recommendations:
                    elements.append(Paragraph(f"• {rec}", self.styles['CustomBody']))
                elements.append(Spacer(1, 10))
        
        return elements
    
    def _create_duration_breakdown_section(self, data: Dict[str, Any]) -> List:
        """Create duration breakdown section"""
        elements = []
        elements.append(Paragraph("Duration Breakdown", self.styles['SectionHeading']))
        
        duration_data = data.get('duration_breakdown', {})
        if not duration_data:
            elements.append(Paragraph("No duration breakdown data available.", self.styles['CustomBody']))
            return elements
        
        # Duration phases
        phases_data = [
            ['Phase', 'Duration (Days)'],
            ['Foundation Phase', str(duration_data.get('foundation_days', 0))],
            ['Structural Phase', str(duration_data.get('structure_days', 0))],
            ['Finishing Phase', str(duration_data.get('finishing_days', 0))],
            ['Total Duration', str(duration_data.get('total_days', 0))]
        ]
        
        table = Table(phases_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#32CD32')),
            ('TEXTCOLOR', (0, 0), (-1, 0), 'white'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, -1), (-1, -1), HexColor('#F0FFF0')),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_footer_section(self) -> List:
        """Create footer section"""
        elements = []
        
        # Add some space before footer
        elements.append(Spacer(1, 30))
        
        # Footer line
        elements.append(Paragraph("=" * 50, self.styles['CustomBody']))
        
        # Generation info
        current_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        footer_text = f"Report generated on {current_date}<br/>Construction Planning Assistant - AI-Enhanced Planning System"
        
        elements.append(Paragraph(footer_text, self.styles['CustomBody']))
        
        return elements
    
    def _format_currency(self, amount: int) -> str:
        """Format currency amount"""
        if isinstance(amount, str):
            # Remove any non-numeric characters except commas and dots
            amount_str = ''.join(c for c in amount if c.isdigit() or c in ',.')
            try:
                amount = int(amount_str.replace(',', ''))
            except:
                return amount
        
        return f"{amount:,}"


# Convenience function for backward compatibility
def generate_pdf(data: Dict[str, Any]) -> bytes:
    """
    Generate PDF report for construction planning
    
    Args:
        data: Dictionary containing construction planning data
        
    Returns:
        PDF file as bytes
    """
    generator = PDFGenerator()
    return generator.generate_pdf(data)


def prepare_pdf_data_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare data for PDF generation from planning results
    
    Args:
        results: Planning results from the construction planner
        
    Returns:
        Formatted data for PDF generation
    """
    project_metadata = results.get("project_metadata", {})
    cost_breakdown = results.get("cost_breakdown", {})
    duration_breakdown = results.get("duration_breakdown", {})
    task_breakdown = results.get("task_breakdown", {})
    feature_extraction = results.get("feature_extraction", {})
    ai_insights = results.get("ai_insights", {})
    
    # Extract numeric values from formatted strings
    total_cost_str = cost_breakdown.get("total_cost", "₹0").replace('₹', '').replace(',', '')
    total_cost = int(total_cost_str) if total_cost_str.isdigit() else 0
    
    cost_per_sqft_str = cost_breakdown.get("cost_per_sqft", "₹0").replace('₹', '').replace(',', '')
    cost_per_sqft = int(cost_per_sqft_str) if cost_per_sqft_str.isdigit() else 0
    
    labor_cost_str = cost_breakdown.get("labor_cost", "₹0").replace('₹', '').replace(',', '')
    labor_cost = int(labor_cost_str) if labor_cost_str.isdigit() else 0
    
    material_cost_str = cost_breakdown.get("material_cost", "₹0").replace('₹', '').replace(',', '')
    material_cost = int(material_cost_str) if material_cost_str.isdigit() else 0
    
    equipment_cost_str = cost_breakdown.get("equipment_cost", "₹0").replace('₹', '').replace(',', '')
    equipment_cost = int(equipment_cost_str) if equipment_cost_str.isdigit() else 0
    
    return {
        "goal": project_metadata.get("goal", "Construction Project"),
        "area": project_metadata.get("area", 0),
        "floors": project_metadata.get("floors", 0),
        "building_type": project_metadata.get("building_type", "N/A"),
        "location": project_metadata.get("location", "N/A"),
        "quality": project_metadata.get("quality", "N/A"),
        "total_tasks": project_metadata.get("total_tasks", 0),
        "duration": project_metadata.get("total_duration_days", 0),
        "total_cost": total_cost,
        "cost_per_sqft": cost_per_sqft,
        "created_date": project_metadata.get("created_date", ""),
        "cost_breakdown": {
            "total_cost": total_cost,
            "labor_cost": labor_cost,
            "material_cost": material_cost,
            "equipment_cost": equipment_cost
        },
        "duration_breakdown": duration_breakdown,
        "tasks": task_breakdown.get("tasks", []),
        "features": feature_extraction.get("features_detected", []),
        "ai_insights": ai_insights
    }
