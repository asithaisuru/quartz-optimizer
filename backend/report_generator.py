import os
import json
from fpdf import FPDF
from datetime import datetime
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Setup Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_IMG_DIR = os.path.join(BASE_DIR, "temp_plots")
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Quartz Optimization Report', border=False, ln=True, align='C')
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 10, 'Department of Computer Science - Decision Support System', border=False, ln=True, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_blueprint(rough_path, cut_path, output_filename, view='top'):
    """Generates a 2D engineering drawing of the fit."""
    
    # Load meshes
    rough = trimesh.load(rough_path)
    cut = trimesh.load(cut_path)
    
    # Setup Plot
    plt.figure(figsize=(4, 4))
    plt.axis('equal')
    plt.axis('off')
    
    # Get 2D Projection based on View
    if view == 'top': # XY Plane
        r_pts = rough.vertices[:, [0, 1]]
        c_pts = cut.vertices[:, [0, 1]]
        title = "Top View (Crown)"
    else: # Front View (XZ Plane)
        r_pts = rough.vertices[:, [0, 2]]
        c_pts = cut.vertices[:, [0, 2]]
        title = "Side View (Profile)"

    # Draw Rough Outline (Convex Hull of projection)
    try:
        hull_r = ConvexHull(r_pts)
        # Close the loop
        r_x = np.append(r_pts[hull_r.vertices, 0], r_pts[hull_r.vertices[0], 0])
        r_y = np.append(r_pts[hull_r.vertices, 1], r_pts[hull_r.vertices[0], 1])
        plt.plot(r_x, r_y, 'k-', linewidth=2, label='Rough Stone') # Black Line
        plt.fill(r_x, r_y, 'gray', alpha=0.1)
    except:
        pass

    # Draw Cut Outline
    try:
        hull_c = ConvexHull(c_pts)
        c_x = np.append(c_pts[hull_c.vertices, 0], c_pts[hull_c.vertices[0], 0])
        c_y = np.append(c_pts[hull_c.vertices, 1], c_pts[hull_c.vertices[0], 1])
        plt.plot(c_x, c_y, 'r--', linewidth=1.5, label='Cut Plan') # Red Dashed
        plt.fill(c_x, c_y, 'red', alpha=0.1)
    except:
        pass

    plt.title(title)
    plt.tight_layout()
    
    save_path = os.path.join(TEMP_IMG_DIR, output_filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path

def create_pdf(job_folder, job_id, stats):
    pdf = PDFReport()
    pdf.add_page()
    
    # 1. JOB DETAILS
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Job Details', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f"Job ID: {job_id}", ln=True)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    # 2. ANALYSIS DATA
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Yield Analysis', ln=True)
    
    # Create Table-like structure
    pdf.set_font('Helvetica', '', 10)
    col_w = 40
    
    # Row 1
    pdf.cell(col_w, 8, "Rough Weight:", border=1)
    pdf.cell(col_w, 8, f"{stats['raw_carats']} cts", border=1, ln=True)
    
    # Row 2
    pdf.cell(col_w, 8, "Cut Weight:", border=1)
    pdf.cell(col_w, 8, f"{stats['estimated_cut_carats']} cts", border=1, ln=True)
    
    # Row 3
    pdf.cell(col_w, 8, "Yield %:", border=1)
    # Color code yield
    if stats['yield_percent'] > 30:
        pdf.set_text_color(0, 150, 0) # Green
    else:
        pdf.set_text_color(200, 0, 0) # Red
    pdf.cell(col_w, 8, f"{stats['yield_percent']}%", border=1, ln=True)
    pdf.set_text_color(0, 0, 0) # Reset
    
    # Row 4
    pdf.cell(col_w, 8, "Rec. Shape:", border=1)
    pdf.cell(col_w, 8, f"{stats.get('recommended_shape', 'Custom')}", border=1, ln=True)
    pdf.ln(10)
    
    # ROW 5 (Light Analysis)
    light = stats.get('light_analysis', {'score': 0, 'grade': 'N/A'})
    pdf.cell(col_w, 8, "Light Return:", border=1)
    
    # Color code grade
    if light['score'] > 80: pdf.set_text_color(0, 150, 0)
    elif light['score'] < 50: pdf.set_text_color(200, 0, 0)
    
    pdf.cell(col_w, 8, f"{light['score']}/100 ({light['grade']})", border=1, ln=True)
    pdf.set_text_color(0, 0, 0) # Reset color
    # -----------------------------------

    pdf.ln(10)

    # 3. BLUEPRINTS
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Optimization Blueprints', ln=True)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.cell(0, 5, 'Visual comparison of rough stone limits vs. optimal cut.', ln=True)
    pdf.ln(5)

    # Paths to models
    rough_ply = os.path.join(job_folder, "dense", "visual_aligned_stone.ply")
    cut_ply = os.path.join(job_folder, "dense", "best_cut.ply")

    if os.path.exists(rough_ply) and os.path.exists(cut_ply):
        # Generate Images
        img_top = generate_blueprint(rough_ply, cut_ply, f"{job_id}_top.png", 'top')
        img_side = generate_blueprint(rough_ply, cut_ply, f"{job_id}_side.png", 'side')
        
        # Place Images side by side
        y_pos = pdf.get_y()
        pdf.image(img_top, x=10, y=y_pos, w=90)
        pdf.image(img_side, x=105, y=y_pos, w=90)
        
        # Move cursor down
        pdf.set_y(y_pos + 95)
    else:
        pdf.cell(0, 10, "Error: 3D Models not found for visualization.", ln=True)

    # 4. RECOMMENDATION
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'AI Recommendation', ln=True)
    pdf.set_font('Helvetica', '', 10)
    
    note = f"Based on geometric volume analysis, the {stats.get('recommended_shape')} cut maximizes material retention. " \
           f"The stone allows for a {stats['estimated_cut_carats']} carat gem with a yield of {stats['yield_percent']}%."
    
    pdf.multi_cell(0, 6, note)

    # Save
    output_path = os.path.join(job_folder, "report.pdf")
    pdf.output(output_path)
    return output_path