import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix syntax error in renderizar_historico
content = content.replace('ax.set_facecolor="#161b22"\n    ax.set_facecolor("#161b22")', 'ax.set_facecolor("white")')

# Change graficar_espectrograma colors
content = content.replace('fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0d1117")\n    ax.set_facecolor("#0d1117")', 'fig, ax = plt.subplots(figsize=(8, 3), facecolor="white")\n    ax.set_facecolor("white")')
content = content.replace('ax.set_title("Espectrograma de Mel – Último Chunk", color="white", pad=8)\n    ax.tick_params(colors="white")\n    ax.xaxis.label.set_color("white")\n    ax.yaxis.label.set_color("white")', 'ax.set_title("Espectrograma de Mel – Último Chunk", color="black", pad=8)\n    ax.tick_params(colors="black")\n    ax.xaxis.label.set_color("black")\n    ax.yaxis.label.set_color("black")')
content = content.replace('spine.set_edgecolor("#30363d")', 'spine.set_edgecolor("#cccccc")')

# Change renderizar_historico colors
content = content.replace('fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#0d1117")\n    ax.set_facecolor("white")', 'fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="white")\n    ax.set_facecolor("white")')
content = content.replace('ax.set_xlabel("Chunk #", color="white")\n    ax.set_ylabel("Anomalía (%)", color="white")\n    ax.tick_params(colors="white")\n    ax.legend(facecolor="#161b22", labelcolor="white")', 'ax.set_xlabel("Chunk #", color="black")\n    ax.set_ylabel("Anomalía (%)", color="black")\n    ax.tick_params(colors="black")\n    ax.legend(facecolor="white", labelcolor="black")')

# For the third plot (Evolución de Anomalías)
content = content.replace('fig, ax = plt.subplots(figsize=(8, 3.8), facecolor="#0d1117")\n                        ax.set_facecolor("#161b22")', 'fig, ax = plt.subplots(figsize=(8, 3.8), facecolor="white")\n                        ax.set_facecolor("white")')
content = content.replace('ax.set_xlabel("Ventana (Tiempo)", color="white")\n                        ax.set_ylabel("Anomalía (%)", color="white")\n                        ax.tick_params(colors="white")\n                        ax.legend(facecolor="#161b22", labelcolor="white")', 'ax.set_xlabel("Ventana (Tiempo)", color="black")\n                        ax.set_ylabel("Anomalía (%)", color="black")\n                        ax.tick_params(colors="black")\n                        ax.legend(facecolor="white", labelcolor="black")')

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
