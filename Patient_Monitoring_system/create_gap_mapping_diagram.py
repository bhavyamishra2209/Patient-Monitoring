"""
create_gap_mapping_diagram.py
-----------------------------
Generates a publication-grade, high-resolution diagram mapping the 3 core research gaps
to their implemented clinical ML modules and empirical validation outputs.
Saves to artifacts/research_gaps_mapping_diagram.png at 300 DPI.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Set up figure
fig = plt.figure(figsize=(16, 10), dpi=300)
fig.patch.set_facecolor('#070b19')  # Deep clinical dark background

# Custom color palette
c_gap1 = '#38bdf8'    # Sky Blue (Temporal Prediction)
c_gap2 = '#a855f7'    # Purple (Trust & XAI)
c_gap3 = '#10b981'    # Emerald Green (Triage)
c_box_bg = '#0e172a'  # Card Background
c_border = '#334155'  # Slate Border
c_text_main = '#f8fafc'
c_text_sub = '#94a3b8'
c_badge_bg = '#1e293b'

ax = fig.add_subplot(111)
ax.set_facecolor('#070b19')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title & Subtitle Banner
ax.text(50, 95.5, "SentryMed Clinical Pipeline: Research Gaps Mapped to ML Modules & Outputs",
        fontsize=18, fontweight='bold', color=c_text_main, ha='center', va='center')
ax.text(50, 92.5, "End-to-End Clinical Decision Support Architecture Validated on PhysioNet / CinC Challenge 2019 (1,000 Patients, 38,776 Hours)",
        fontsize=10.5, color=c_text_sub, ha='center', va='center')

# Top Flow Banner: Input Data
input_box = patches.FancyBboxPatch((15, 84), 70, 5.5, boxstyle="round,pad=0.8,rounding_size=1.2",
                                   facecolor='#131f38', edgecolor='#38bdf8', linewidth=1.5)
ax.add_patch(input_box)
ax.text(50, 86.8, "DATA INGESTION STREAM: Real-Time ICU Telemetry & Laboratory Profiles",
        fontsize=11, fontweight='bold', color='#38bdf8', ha='center', va='center')
ax.text(50, 84.8, "PhysioNet 2019 (1,000 Patients · 38,776 Hours) & Continuous 6-Bed Streaming Telemetry (patient1–patient6)",
        fontsize=9, color=c_text_sub, ha='center', va='center')

# Arrows from Input Data down to the 3 Gaps
ax.annotate("", xy=(23, 79), xytext=(23, 83),
            arrowprops=dict(arrowstyle="->", color=c_gap1, lw=2.5))
ax.annotate("", xy=(50, 79), xytext=(50, 83),
            arrowprops=dict(arrowstyle="->", color=c_gap2, lw=2.5))
ax.annotate("", xy=(77, 79), xytext=(77, 83),
            arrowprops=dict(arrowstyle="->", color=c_gap3, lw=2.5))

# Horizontal Inter-Gap Pipeline Arrows (Predict -> Trust -> Act)
ax.annotate("", xy=(36.0, 52), xytext=(33.5, 52),
            arrowprops=dict(arrowstyle="->", color='#64748b', lw=2.0, ls='--'))
ax.text(34.8, 54, "Candidate\nAlerts", fontsize=7.2, color='#94a3b8', ha='center', va='center')

ax.annotate("", xy=(66.5, 52), xytext=(64.0, 52),
            arrowprops=dict(arrowstyle="->", color='#64748b', lw=2.0, ls='--'))
ax.text(65.3, 54, "Filtered\nAcuity", fontsize=7.2, color='#94a3b8', ha='center', va='center')

# Helper function to create gap columns
def draw_gap_column(x_left, width, title, gap_num, gap_color,
                    problem_text, module_title, module_items,
                    outputs_title, outputs_items, highlight_metric):
    y_top = 79
    h_col = 66
    
    # Outer Card
    card = patches.FancyBboxPatch((x_left, y_top - h_col), width, h_col,
                                  boxstyle="round,pad=0.8,rounding_size=1.5",
                                  facecolor=c_box_bg, edgecolor=gap_color, linewidth=1.8)
    ax.add_patch(card)
    
    # Header Banner
    h_header = 7.5
    hdr = patches.FancyBboxPatch((x_left + 0.5, y_top - h_header), width - 1.0, h_header - 0.5,
                                boxstyle="round,pad=0.4,rounding_size=0.8",
                                facecolor=c_badge_bg, edgecolor='none')
    ax.add_patch(hdr)
    
    # Gap Badge
    badge = patches.Circle((x_left + 3.0, y_top - 3.8), 1.8, facecolor=gap_color, edgecolor='none')
    ax.add_patch(badge)
    ax.text(x_left + 3.0, y_top - 3.8, str(gap_num), fontsize=11, fontweight='bold', color='#000', ha='center', va='center')
    
    # Header Text
    ax.text(x_left + 5.8, y_top - 3.8, title, fontsize=10.5, fontweight='bold', color=c_text_main, va='center')
    
    # Section 1: Clinical Problem
    curr_y = y_top - 10.5
    ax.text(x_left + 2, curr_y, "CLINICAL GAP / PROBLEM:", fontsize=8.2, fontweight='bold', color=gap_color)
    curr_y -= 2.2
    for line in problem_text:
        ax.text(x_left + 2, curr_y, line, fontsize=8.0, color='#cbd5e1', va='top')
        curr_y -= 2.0
        
    curr_y -= 1.0
    # Divider
    ax.plot([x_left + 2, x_left + width - 2], [curr_y, curr_y], color=c_border, lw=0.8)
    curr_y -= 2.2
    
    # Section 2: Implemented ML Module
    ax.text(x_left + 2, curr_y, "IMPLEMENTED MODULE:", fontsize=8.2, fontweight='bold', color=c_text_main)
    curr_y -= 2.0
    ax.text(x_left + 2, curr_y, module_title, fontsize=8.2, fontweight='bold', color=gap_color)
    curr_y -= 2.4
    for item in module_items:
        ax.text(x_left + 2.5, curr_y, f"• {item}", fontsize=7.8, color='#94a3b8', va='top')
        curr_y -= 2.1
        
    curr_y -= 0.8
    # Divider
    ax.plot([x_left + 2, x_left + width - 2], [curr_y, curr_y], color=c_border, lw=0.8)
    curr_y -= 2.2
    
    # Section 3: Empirical Outputs & Validation
    ax.text(x_left + 2, curr_y, "EVALUATION OUTPUTS & METRICS:", fontsize=8.2, fontweight='bold', color=gap_color)
    curr_y -= 2.4
    for item in outputs_items:
        ax.text(x_left + 2.5, curr_y, f"✓ {item}", fontsize=7.8, color='#e2e8f0', va='top')
        curr_y -= 2.1
        
    # Key Highlight Metric Pill at bottom
    curr_y = y_top - h_col + 3.8
    pill = patches.FancyBboxPatch((x_left + 2, curr_y - 2.0), width - 4, 3.8,
                                 boxstyle="round,pad=0.4,rounding_size=0.6",
                                 facecolor=f"{gap_color}22", edgecolor=gap_color, linewidth=1.2)
    ax.add_patch(pill)
    ax.text(x_left + (width / 2.0), curr_y, highlight_metric, fontsize=9.0, fontweight='bold',
            color=gap_color, ha='center', va='center')


# Column 1: Gap 1 (Temporal Prediction)
draw_gap_column(
    x_left=6, width=27,
    title="Early Deterioration Prediction", gap_num=1, gap_color=c_gap1,
    problem_text=[
        "Single-point vital scores (NEWS2, qSOFA) only",
        "react after physiological decompensation occurs.",
        "Ignores subtle, multi-hour deteriorating trends."
    ],
    module_title="Temporal LightGBM Classifier",
    module_items=[
        "Rolling 4h & 12h vitals summary statistics",
        "Linear slope velocity vectors (HR, SBP, SpO₂)",
        "Derived biomarkers: Shock Index & Mean Arterial BP",
        "Patient-level temporal feature aggregation"
    ],
    outputs_title="Outputs",
    outputs_items=[
        "AUROC: 0.7157 (vs 0.6471 Static LGBM, 0.6427 NEWS2)",
        "Mean Advance Warning Lead Time: +58.87 hours",
        "Sensitivity @ 90% Specificity: 34.6%",
        "Artifacts: gap1_roc_pr.png, gap1_lead_time.png"
    ],
    highlight_metric="AUROC 0.716 · +58.9h Advance Notice"
)

# Column 2: Gap 2 (Trust & Explainability)
draw_gap_column(
    x_left=36.5, width=27,
    title="Alarm Fatigue & Explainable AI", gap_num=2, gap_color=c_gap2,
    problem_text=[
        "ICU monitors trigger on isolated vital spikes;",
        "98.6% of alerts are non-actionable noise, causing",
        "clinician alarm fatigue and silenced monitors."
    ],
    module_title="Two-Stage Trust Filter + TreeSHAP",
    module_items=[
        "Stage 2 Validity Model on candidate alarm stream",
        "Verifies end-organ hypoperfusion vs motion artifact",
        "TreeSHAP local feature attribution vectors",
        "Demographic fairness audit across Age & Gender"
    ],
    outputs_title="Outputs",
    outputs_items=[
        "416 False Alarms Eliminated (24.4% noise reduction)",
        "75.0% True Deterioration Sensitivity Retained",
        "Demographic parity audited across age bands",
        "Artifacts: gap2_shap_summary.png, gap2_fairness.png"
    ],
    highlight_metric="-24.4% False Alerts · SHAP Transparency"
)

# Column 3: Gap 3 (Constraint-Aware Triage)
draw_gap_column(
    x_left=67, width=27,
    title="Resource-Constrained Bed Triage", gap_num=3, gap_color=c_gap3,
    problem_text=[
        "Staffing limits: N beds (e.g. 10) but only K nurse",
        "reviews/hr (e.g. 2). Clinicians cannot examine all",
        "beds simultaneously; FIFO/greedy leads to delays."
    ],
    module_title="Constraint-Aware Ward Simulator",
    module_items=[
        "Dynamic multi-objective ranking algorithm",
        "Balances ML risk + trend acceleration + wait penalty",
        "Prevents patient starvation on non-critical beds",
        "Interactive bed capacity & nurse ratio simulation"
    ],
    outputs_title="Outputs",
    outputs_items=[
        "100% of Septic Patients Reviewed Before Onset",
        "0 Missed Onsets under N=10 beds, K=2 reviews/hr",
        "Mean Clinical Notice Time: +20.0 hours",
        "Artifacts: gap3_capacity_sweep.png, policy_comp.png"
    ],
    highlight_metric="100% Onsets Reviewed · 0 Missed (K=2)"
)

# Bottom Integration Bar: SentryMed ICU Dashboard Output
dash_box = patches.FancyBboxPatch((6, 4.5), 88, 6.0, boxstyle="round,pad=0.8,rounding_size=1.2",
                                  facecolor='#0d1b33', edgecolor='#22d3ee', linewidth=1.5)
ax.add_patch(dash_box)
ax.text(50, 8.8, "UNIFIED INTEGRATION: SentryMed Real-Time Clinical Decision Support Command Center",
        fontsize=10.5, fontweight='bold', color='#22d3ee', ha='center', va='center')
ax.text(50, 6.2, "Live Shuffling Acuity Cards • Real-Time Actionable vs. Suppressed Alert Feed • Interactive SHAP Decision Tree • Dynamic Telemetry Graphs",
        fontsize=8.5, color='#94a3b8', ha='center', va='center')

# Arrows from each gap down to Dashboard
ax.annotate("", xy=(19.5, 11), xytext=(19.5, 13),
            arrowprops=dict(arrowstyle="->", color=c_gap1, lw=2.0))
ax.annotate("", xy=(50, 11), xytext=(50, 13),
            arrowprops=dict(arrowstyle="->", color=c_gap2, lw=2.0))
ax.annotate("", xy=(80.5, 11), xytext=(80.5, 13),
            arrowprops=dict(arrowstyle="->", color=c_gap3, lw=2.0))

plt.tight_layout()
out_path = os.path.join("artifacts", "research_gaps_mapping_diagram.png")
plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.close()

print(f"[create_gap_mapping_diagram] Saved publication diagram to {out_path}")
