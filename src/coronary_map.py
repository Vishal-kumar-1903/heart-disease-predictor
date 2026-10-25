"""
Interactive 2D Coronary Artery Anatomical Visualizer (Dark/Healthcare Mode)
Renders LAD, LCx, and RCA coronary branches highlighting stenosis/calcification levels.
"""

def render_coronary_artery_svg(vessels_colored: int, is_ischemia: bool = False):
    """
    Returns an SVG string representing the heart's coronary artery tree
    highlighted according to fluoroscopy vessels (ca) and thallium status.
    """
    # Color branches based on calcification count (0, 1, 2, 3, 4)
    lad_color = "#ef4444" if vessels_colored >= 1 else "#22c55e"
    lcx_color = "#ef4444" if vessels_colored >= 2 else "#22c55e"
    rca_color = "#ef4444" if vessels_colored >= 3 else "#22c55e"

    lad_status = "Stenosis / Calcification Detected" if vessels_colored >= 1 else "Patent (Normal Flow)"
    lcx_status = "Stenosis / Calcification Detected" if vessels_colored >= 2 else "Patent (Normal Flow)"
    rca_status = "Stenosis / Calcification Detected" if vessels_colored >= 3 else "Patent (Normal Flow)"

    ischemia_badge = ""
    if is_ischemia:
        ischemia_badge = """
        <rect x="25" y="240" width="350" height="28" rx="6" fill="rgba(239, 68, 68, 0.25)" stroke="#ef4444" stroke-width="1.5"/>
        <text x="200" y="258" text-anchor="middle" font-size="11" font-weight="bold" fill="#fecaca">
            ⚠️ Active Reversible Myocardial Ischemia Detected (Thallium)
        </text>
        """

    svg = f"""
    <div style="background: #131f38; border: 1px solid #2d3e5e; border-radius: 14px; padding: 1.2rem; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <div style="font-weight: 800; color: #38bdf8; font-size: 1.1rem; margin-bottom: 4px;">
            🫀 Coronary Artery Tree & Perfusion Map
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">
            Fluoroscopy Calcification Visualization • {vessels_colored} Major Vessel(s) Affected
        </div>
        <svg viewBox="0 0 400 280" style="max-width: 100%; height: auto; font-family: -apple-system, sans-serif;">
            <!-- Heart Outline Background -->
            <path d="M 200,60 C 230,15 310,20 310,95 C 310,165 200,225 200,230 C 200,225 90,165 90,95 C 90,20 170,15 200,60 Z"
                  fill="#0b1329" stroke="#334155" stroke-width="2" stroke-dasharray="4 4"/>

            <!-- Aortic Root -->
            <path d="M 190,50 L 210,50 L 205,80 L 195,80 Z" fill="#64748b"/>
            <text x="200" y="42" text-anchor="middle" font-size="10" font-weight="bold" fill="#94a3b8">Aortic Root</text>

            <!-- LMCA -->
            <path d="M 200,80 Q 185,90 170,95" fill="none" stroke="#94a3b8" stroke-width="6" stroke-linecap="round"/>
            <text x="155" y="85" text-anchor="end" font-size="9" font-weight="bold" fill="#cbd5e1">Left Main (LMCA)</text>

            <!-- LAD Branch -->
            <path d="M 170,95 Q 165,140 185,185 Q 195,205 200,220" fill="none" stroke="{lad_color}" stroke-width="5" stroke-linecap="round"/>
            <circle cx="178" cy="155" r="4.5" fill="{lad_color}"/>
            <text x="135" y="160" text-anchor="end" font-size="10" font-weight="bold" fill="{lad_color}">LAD: {lad_status}</text>

            <!-- LCx Branch -->
            <path d="M 170,95 Q 130,110 115,150 Q 110,175 125,195" fill="none" stroke="{lcx_color}" stroke-width="4.5" stroke-linecap="round"/>
            <circle cx="118" cy="145" r="4.5" fill="{lcx_color}"/>
            <text x="95" y="130" text-anchor="end" font-size="10" font-weight="bold" fill="{lcx_color}">LCx: {lcx_status}</text>

            <!-- RCA Branch -->
            <path d="M 205,80 Q 245,95 260,135 Q 265,175 235,205 Q 215,220 200,225" fill="none" stroke="{rca_color}" stroke-width="5" stroke-linecap="round"/>
            <circle cx="260" cy="140" r="4.5" fill="{rca_color}"/>
            <text x="270" y="145" text-anchor="start" font-size="10" font-weight="bold" fill="{rca_color}">RCA: {rca_status}</text>

            {ischemia_badge}
        </svg>
        <div style="display: flex; justify-content: center; gap: 20px; font-size: 0.82rem; margin-top: 10px;">
            <span style="display: inline-flex; align-items: center; gap: 5px; color: #bbf7d0;">
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #22c55e; display: inline-block;"></span> Patent / Normal
            </span>
            <span style="display: inline-flex; align-items: center; gap: 5px; color: #fecaca;">
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #ef4444; display: inline-block;"></span> Stenosis / Calcification
            </span>
        </div>
    </div>
    """
    return svg
