import os
import json
from fpdf import FPDF
from datetime import datetime
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

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
    rough = trimesh.load(rough_path)
    cut = trimesh.load(cut_path)
    
    plt.figure(figsize=(4, 4))
    plt.axis('equal')
    plt.axis('off')
    
    if view == 'top':
        r_pts = rough.vertices[:, [0, 1]]
        c_pts = cut.vertices[:, [0, 1]]
        title = "Top View (Crown)"
    else: 
        r_pts = rough.vertices[:, [0, 2]]
        c_pts = cut.vertices[:, [0, 2]]
        title = "Side View (Profile)"

    try:
        hull_r = ConvexHull(r_pts)
        r_x = np.append(r_pts[hull_r.vertices, 0], r_pts[hull_r.vertices[0], 0])
        r_y = np.append(r_pts[hull_r.vertices, 1], r_pts[hull_r.vertices[0], 1])
        plt.plot(r_x, r_y, 'k-', linewidth=2)
    except: pass

    try:
        hull_c = ConvexHull(c_pts)
        c_x = np.append(c_pts[hull_c.vertices, 0], c_pts[hull_c.vertices[0], 0])
        c_y = np.append(c_pts[hull_c.vertices, 1], c_pts[hull_c.vertices[0], 1])
        plt.plot(c_x, c_y, 'r--', linewidth=1.5)
        plt.fill(c_x, c_y, 'red', alpha=0.1)
    except: pass

    plt.title(title)
    plt.tight_layout()
    save_path = os.path.join(TEMP_IMG_DIR, output_filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path

def create_pdf(job_folder, job_id, stats):
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Job Details', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f"Job ID: {job_id}", ln=True)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Yield Analysis', ln=True)
    
    col_w = 40
    pdf.set_font('Helvetica', '', 10)
    
    pdf.cell(col_w, 8, "Rough Weight:", border=1)
    pdf.cell(col_w, 8, f"{stats['raw_carats']} cts", border=1, ln=True)
    
    pdf.cell(col_w, 8, "Cut Weight:", border=1)
    pdf.cell(col_w, 8, f"{stats['estimated_cut_carats']} cts", border=1, ln=True)
    
    pdf.cell(col_w, 8, "Yield %:", border=1)
    if stats['yield_percent'] > 30: pdf.set_text_color(0, 150, 0)
    else: pdf.set_text_color(200, 0, 0)
    pdf.cell(col_w, 8, f"{stats['yield_percent']}%", border=1, ln=True)
    pdf.set_text_color(0, 0, 0) 
    
    # Dimensions Row
    if 'dimensions_mm' in stats:
        dims = stats['dimensions_mm']
        dim_str = f"{dims[0]} x {dims[1]} x {dims[2]} mm"
        pdf.cell(col_w, 8, "Dimensions:", border=1)
        pdf.cell(col_w, 8, dim_str, border=1, ln=True)

    pdf.ln(10)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Optimization Blueprints', ln=True)
    
    rough_ply = os.path.join(job_folder, "dense", "visual_aligned_stone.ply")
    cut_ply = os.path.join(job_folder, "dense", "best_cut.ply")

    if os.path.exists(rough_ply) and os.path.exists(cut_ply):
        img_top = generate_blueprint(rough_ply, cut_ply, f"{job_id}_top.png", 'top')
        img_side = generate_blueprint(rough_ply, cut_ply, f"{job_id}_side.png", 'side')
        
        y_pos = pdf.get_y() + 5
        pdf.image(img_top, x=15, y=y_pos, w=80)
        pdf.image(img_side, x=105, y=y_pos, w=80)
        
        pdf.set_y(y_pos + 85)
        
        # Add Dimensions Caption
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 5, f"Visual Fit Confirmation. Est Dimensions: {dim_str}", align='C', ln=True)
        
    else:
        pdf.cell(0, 10, "Error: 3D Models not found for visualization.", ln=True)

    output_path = os.path.join(job_folder, "report.pdf")
    pdf.output(output_path)
    return output_path