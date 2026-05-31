import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Fix cache
content = content.replace('def cargar_modelo(ruta_modelo: str):', '@st.cache_resource\ndef cargar_modelo(ruta_modelo: str):')

# 2. Fix CSS
new_css = """        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
            
            /* Configuración general */
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
                background-color: #ffffff !important;
                color: #111111 !important;
            }
            
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                color: #111111 !important;
                letter-spacing: -0.02em;
            }
            
            p, span, label, div {
                color: #111111;
            }
            
            /* Tarjetas de Métricas Premium */
            [data-testid="stMetric"] {
                background: #ffffff !important;
                border: 1px solid #e1e4e8 !important;
                border-radius: 12px !important;
                padding: 16px 20px !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
            }
            [data-testid="stMetric"] label {
                color: #555555 !important;
            }
            [data-testid="stMetric"] div {
                color: #111111 !important;
            }
            
            /* Contenedor de Tarjetas Generales */
            .premium-card {
                background: #ffffff !important;
                border: 1px solid #e1e4e8 !important;
                border-radius: 12px !important;
                padding: 20px !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
                margin-bottom: 20px !important;
            }
            
            /* Botones Estilo Premium */
            .stButton > button {
                background: #f6f8fa !important;
                color: #24292e !important;
                border: 1px solid #d1d5da !important;
                border-radius: 8px !important;
            }
            .stButton > button:hover {
                border-color: #1f6feb !important;
                background: #f3f4f6 !important;
            }
            
            /* Botón Primario */
            .stButton > button[kind="primary"] {
                background: #1f6feb !important;
                color: #ffffff !important;
                border: none !important;
            }
            
            /* Barra lateral */
            section[data-testid="stSidebar"] {
                background-color: #f8f9fa !important;
                border-right: 1px solid #e1e4e8 !important;
            }
            section[data-testid="stSidebar"] * {
                color: #111111 !important;
            }
            
            /* Badges */
            .badge-normal {
                display: inline-block;
                background: #e6ffed;
                border: 1px solid #2ea043;
                color: #2ea043 !important;
                font-family: 'JetBrains Mono', monospace;
                padding: 10px 24px;
                border-radius: 8px;
            }
            
            .badge-anomaly {
                display: inline-block;
                background: #ffebe9;
                border: 1px solid #d73a49;
                color: #d73a49 !important;
                font-family: 'JetBrains Mono', monospace;
                padding: 10px 24px;
                border-radius: 8px;
            }
            
            .alerta-anomalia {
                background: #ffebe9;
                border: 1px solid #d73a49;
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
            }
            
            /* Estilizar selectbox */
            div[data-baseweb="select"] > div {
                background-color: #ffffff !important;
                border: 1px solid #d1d5da !important;
            }
            div[data-baseweb="select"] * {
                color: #111111 !important;
            }
            
            /* Estilizar file uploader */
            div[data-testid="stFileUploader"] {
                border: 2px dashed #d1d5da !important;
                background-color: #f8f9fa !important;
            }
        </style>"""

content = re.sub(r"<style>.*?</style>", new_css, content, flags=re.DOTALL)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
