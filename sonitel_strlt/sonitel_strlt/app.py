"""
================================================================================
 SONITEL INDUSTRIAL - Monitor Acústico en Tiempo Real
================================================================================
 Sistema de mantenimiento predictivo acústico para maquinaria industrial.
 Detecta anomalías sonoras en tiempo real mediante un Autoencoder entrenado
 sobre espectrogramas de Mel.

 Arquitectura:
  - Hilo de captura (sounddevice)  →  queue.Queue  →  Hilo principal Streamlit
  - El modelo ML se aplica sobre cada chunk de 3 segundos en el hilo principal.
  - st.rerun() refresca la UI sin bloquear la captura de audio.

 Autor  : Proyecto Sonitel – Edge Computing
 Versión: 1.0.0
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import queue
import tempfile
import threading
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import streamlit as st

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # Backend sin GUI para evitar conflictos con Streamlit


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16000        # Frecuencia de muestreo (Hz) compatible con el modelo
CHUNK_SEGUNDOS: float = 2.112    # Duración de cada fragmento de audio analizado (~2.112 s)
CHUNK_MUESTRAS: int = int(SAMPLE_RATE * CHUNK_SEGUNDOS)  # Total de muestras por chunk (33792)
N_MELS: int = 64                # Número de bandas de Mel para el espectrograma
N_FFT: int = 1024                # Tamaño de la ventana FFT
HOP_LENGTH: int = 512            # Desplazamiento entre ventanas FFT
INTERVALO_REFRESCO: float = 0.5  # Segundos entre refrescos de la UI
COLA_MAX_ITEMS: int = 5          # Máximo de chunks en cola (evita acumulación)

# Clave para almacenar el estado global en la sesión de Streamlit
_KEY_ESTADO = "monitor_estado"


# ──────────────────────────────────────────────────────────────────────────────
# MODELO (PLACEHOLDER / MOCK)
# ──────────────────────────────────────────────────────────────────────────────
class ConvEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.project = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.project(x)
        return x

class ConvDecoderFixed(nn.Module):
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.unproject = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128 * 8 * 8),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.unproject(z)
        x = x.view(-1, 128, 8, 8)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class AcousticBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = ConvEncoder(embedding_dim)
        self.decoder = ConvDecoderFixed(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

def cargar_modelo(ruta_modelo: str):
    """
    Carga el modelo de ML pre-entrenado desde disco.
    """
    if not ruta_modelo:
        return None
    try:
        checkpoint = torch.load(ruta_modelo, map_location="cpu", weights_only=False)
        embedding_dim = checkpoint.get("embedding_dim", 32)
        
        modelo = AcousticBackbone(embedding_dim=embedding_dim)
        
        if "best_state" in checkpoint:
            modelo.load_state_dict(checkpoint["best_state"])
        elif "model_state" in checkpoint:
            modelo.load_state_dict(checkpoint["model_state"])
        else:
            modelo.load_state_dict(checkpoint)
            
        modelo.eval()
        st.toast(f"✅ Modelo cargado ({embedding_dim} dims) desde: {ruta_modelo}", icon="🧠")
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

def inferir_anomalia(modelo, mel_espectrograma: np.ndarray) -> float:
    """
    Ejecuta la inferencia sobre un espectrograma de Mel y devuelve
    el porcentaje de anomalía (0–100 %).
    """
    if modelo is None or modelo == "MODELO_MOCK":
        # ── MOCK: genera un valor aleatorio con deriva lenta para simular lecturas reales
        if "anomalia_prev" not in st.session_state:
            st.session_state["anomalia_prev"] = 20.0
        ruido = np.random.uniform(-8, 8)
        nuevo = float(np.clip(st.session_state["anomalia_prev"] + ruido, 0, 100))
        st.session_state["anomalia_prev"] = nuevo
        return nuevo

    try:
        # Normalización local: media 0, std 1
        mean = mel_espectrograma.mean()
        std = mel_espectrograma.std() + 1e-8
        mel_norm = (mel_espectrograma - mean) / std

        t = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float()
        
        with torch.no_grad():
            reconstruccion = modelo(t)
            error = torch.mean((t - reconstruccion) ** 2).item()
        
        max_error = st.session_state.get("max_error_config", 0.15)
        porcentaje = min(error / max_error * 100, 100.0)
        return porcentaje
    except Exception as e:
        st.toast(f"⚠️ Error de inferencia: {e}", icon="❌")
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO DE AUDIO
# ──────────────────────────────────────────────────────────────────────────────
def audio_a_mel(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Convierte una señal de audio cruda en un espectrograma de Mel normalizado.

    Parámetros
    ----------
    audio : np.ndarray  – señal mono en [-1, 1]
    sr    : int         – frecuencia de muestreo

    Retorna
    -------
    np.ndarray (N_MELS × T) con valores en dB normalizados a [0, 1]
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=20,
        fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normaliza a [0, 1] para la inferencia del modelo
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)

    # Ajustar el eje temporal a exactamente 64 frames para que coincida con el modelo (64x64)
    if mel_norm.shape[1] > 64:
        mel_norm = mel_norm[:, :64]
    elif mel_norm.shape[1] < 64:
        pad_width = 64 - mel_norm.shape[1]
        mel_norm = np.pad(mel_norm, ((0, 0), (0, pad_width)), mode='constant')

    return mel_norm


def graficar_espectrograma(mel: np.ndarray, sr: int = SAMPLE_RATE) -> plt.Figure:
    """
    Genera una figura de Matplotlib con el espectrograma de Mel.

    Parámetros
    ----------
    mel : np.ndarray – espectrograma normalizado (N_MELS × T)
    sr  : int        – frecuencia de muestreo

    Retorna
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    # Convierte de vuelta a dB para visualización
    mel_db = mel * 80 - 80  # desnormaliza aproximada para visualización
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Intensidad (dB)")
    ax.set_title("Espectrograma de Mel – Último Chunk", color="white", pad=8)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# HILO DE CAPTURA DE AUDIO (NO BLOQUEANTE)
# ──────────────────────────────────────────────────────────────────────────────
class CapturaAudio:
    """
    Encapsula la captura de audio en un hilo de fondo usando sounddevice.

    El audio se captura en chunks de CHUNK_MUESTRAS muestras y se deposita
    en una queue.Queue compartida con el hilo principal de Streamlit.
    Si la cola está llena, el chunk más antiguo se descarta para mantener
    la latencia baja.
    """

    def __init__(self, device_id: int | None = None):
        """
        Parámetros
        ----------
        device_id : int o None – índice del dispositivo de entrada.
                    None usa el dispositivo predeterminado del sistema.
        """
        self.device_id = device_id
        self.cola: queue.Queue = queue.Queue(maxsize=COLA_MAX_ITEMS)
        self._hilo: threading.Thread | None = None
        self._activo: threading.Event = threading.Event()
        self._buffer: list[np.ndarray] = []
        self._muestras_acumuladas: int = 0
        self.error_msg: str | None = None

    def _callback_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Callback de sounddevice: se ejecuta en tiempo real en cada bloque de audio.
        Acumula bloques hasta completar CHUNK_MUESTRAS y luego encola el chunk.
        """
        if status:
            # Registra advertencias de underrun/overrun sin detener la captura
            pass

        # Toma el canal mono (columna 0 si hay varios canales)
        audio_mono = indata[:, 0].copy()
        self._buffer.append(audio_mono)
        self._muestras_acumuladas += len(audio_mono)

        if self._muestras_acumuladas >= CHUNK_MUESTRAS:
            chunk = np.concatenate(self._buffer)[:CHUNK_MUESTRAS]
            self._buffer = []
            self._muestras_acumuladas = 0
            # Descarta el chunk más antiguo si la cola está llena
            if self.cola.full():
                try:
                    self.cola.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.cola.put_nowait(chunk)
            except queue.Full:
                pass

    def _bucle_captura(self) -> None:
        """Ejecuta el stream de captura mientras el evento _activo esté activo."""
        self.error_msg = None
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=self.device_id,
                callback=self._callback_audio,
            ):
                # Mantiene el stream abierto hasta que se señale la parada
                while self._activo.is_set():
                    time.sleep(0.05)
        except Exception as e:
            self.error_msg = str(e)
            self._activo.clear()

    def iniciar(self) -> None:
        """Lanza el hilo de captura en background."""
        if self._hilo and self._hilo.is_alive():
            return  # Ya está corriendo
        self._activo.set()
        self._hilo = threading.Thread(
            target=self._bucle_captura,
            daemon=True,
            name="SonitelAudioThread",
        )
        self._hilo.start()

    def detener(self) -> None:
        """Señaliza la parada del hilo de captura y espera su finalización."""
        self._activo.clear()
        if self._hilo:
            self._hilo.join(timeout=3)
        self._buffer = []
        self._muestras_acumuladas = 0

    def obtener_chunk(self) -> np.ndarray | None:
        """
        Extrae el chunk más reciente de la cola (no bloqueante).

        Retorna
        -------
        np.ndarray o None si la cola está vacía.
        """
        try:
            return self.cola.get_nowait()
        except queue.Empty:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE DISPOSITIVOS
# ──────────────────────────────────────────────────────────────────────────────
def obtener_micros() -> dict[str, int]:
    """
    Enumera todos los dispositivos de entrada de audio disponibles en el sistema.

    Retorna
    -------
    dict { "nombre (índice)": índice_de_dispositivo }
    """
    dispositivos = sd.query_devices()
    micros = {}
    for idx, dev in enumerate(dispositivos):
        if dev["max_input_channels"] > 0:
            nombre = f"{dev['name']} (#{idx})"
            micros[nombre] = idx
    return micros


# ──────────────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN DEL ESTADO DE SESIÓN
# ──────────────────────────────────────────────────────────────────────────────
def _inicializar_estado() -> None:
    """Garantiza que todas las claves necesarias existen en st.session_state."""
    defaults = {
        "monitor_activo": False,       # ¿Está capturando audio?
        "captura": None,               # Instancia de CapturaAudio
        "modelo": None,                # Modelo de ML cargado
        "ultimo_mel": None,            # Último espectrograma calculado
        "ultima_anomalia": None,       # Último % de anomalía
        "historico_anomalia": [],      # Lista de valores para la gráfica histórica
        "alerta_disparada": False,     # ¿Se superó el umbral?
        "total_chunks": 0,             # Chunks analizados en sesión
        "anomalia_prev": 20.0,         # Estado interno del mock
        "max_error_config": 0.15,      # MSE máximo para inferencia (escala)
    }
    for clave, valor in defaults.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ DE USUARIO
# ──────────────────────────────────────────────────────────────────────────────
def renderizar_sidebar() -> tuple[float, str | None]:
    """
    Construye la barra lateral de configuración.

    Retorna
    -------
    umbral_anomalia : float  – umbral de alerta en porcentaje (0–100)
    ruta_modelo     : str|None – ruta al fichero del modelo, o None
    """
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/microphone.png",
            width=64,
        )
        st.title("⚙️ Configuración")
        st.markdown("---")

        st.subheader("🎚️ Umbral de Anomalía")
        umbral = st.slider(
            label="Porcentaje de alerta (%)",
            min_value=10,
            max_value=95,
            value=85,
            step=1,
            help=(
                "Si el % de anomalía detectada supera este valor, "
                "la aplicación disparará una alerta visual."
            ),
        )
        st.caption(f"Alerta cuando anomalía > **{umbral}%**")

        st.markdown("---")
        st.subheader("🎚️ Sensibilidad del Modelo")
        max_error = st.slider(
            label="MSE Máximo de Inferencia",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.01,
            help=(
                "Factor de escala (error de reconstrucción máximo). "
                "Reduce este valor para aumentar la sensibilidad."
            ),
        )
        st.session_state["max_error_config"] = max_error
        st.caption(f"Inferencia al 100% cuando MSE = **{max_error:.2f}**")

        st.markdown("---")
        st.subheader("🧠 Modelo ML")
        
        default_model_path = ""
        # Buscar posibles ubicaciones del checkpoint por defecto
        for p in ["checkpoint (3).pt", "sonitel_strlt/checkpoint (3).pt", "../checkpoint (3).pt"]:
            if os.path.exists(p):
                default_model_path = p
                break
                
        ruta_modelo = st.text_input(
            "Ruta al modelo (.keras / .pt)",
            value=default_model_path,
            placeholder="checkpoint (3).pt",
            help=(
                "Deja vacío para usar el modo simulación (mock). "
                "Introduce la ruta absoluta o relativa al fichero del modelo."
            ),
        )
        if st.button("📥 Cargar Modelo", use_container_width=True):
            st.session_state["modelo"] = cargar_modelo(ruta_modelo)

        estado_modelo = "✅ Cargado" if st.session_state["modelo"] else "⚪ Mock (simulación)"
        st.info(f"Estado del modelo: {estado_modelo}")

        st.markdown("---")
        st.subheader("📊 Parámetros de Audio")
        st.markdown(
            f"""
            | Parámetro        | Valor         |
            |-----------------|---------------|
            | Frecuencia (Hz)  | `{SAMPLE_RATE}` |
            | Chunk (seg)      | `{CHUNK_SEGUNDOS}` |
            | Bandas Mel       | `{N_MELS}` |
            | FFT Size         | `{N_FFT}` |
            | Hop Length       | `{HOP_LENGTH}` |
            """
        )

        st.markdown("---")
        st.caption("Sonitel Industrial · Edge Computing v1.0")

    return float(umbral), ruta_modelo if ruta_modelo else None


def renderizar_cabecera() -> None:
    """Renderiza el encabezado principal de la aplicación."""
    col_logo, col_titulo = st.columns([1, 5])
    with col_logo:
        st.markdown("# 🔊")
    with col_titulo:
        st.markdown("# Sonitel · Monitor Acústico Industrial")
        st.caption(
            "Sistema de mantenimiento predictivo en tiempo real · Edge Computing"
        )
    st.markdown("---")


def renderizar_controles(micros: dict[str, int]) -> tuple[bool, bool, int | None]:
    """
    Renderiza los controles de inicio/parada y el selector de micrófono.

    Retorna
    -------
    (iniciar, detener, device_id)
    """
    st.markdown("##### 🎤 Micrófono de entrada")
    col_mic, col_inicio, col_stop = st.columns([3, 1, 1])
    with col_mic:
        seleccion = st.selectbox(
            "🎤 Micrófono de entrada",
            options=list(micros.keys()),
            help="Selecciona el dispositivo de audio antes de iniciar.",
            disabled=st.session_state["monitor_activo"],
            label_visibility="collapsed",
        )
        device_id = micros[seleccion] if seleccion else None

    with col_inicio:
        iniciar = st.button(
            "▶️ Iniciar",
            use_container_width=True,
            disabled=st.session_state["monitor_activo"],
            type="primary",
        )
    with col_stop:
        detener = st.button(
            "⏹️ Detener",
            use_container_width=True,
            disabled=not st.session_state["monitor_activo"],
        )

    return iniciar, detener, device_id


def renderizar_metricas(umbral: float) -> None:
    """
    Renderiza el panel de métricas: estado, anomalía y chunks analizados.

    Parámetros
    ----------
    umbral : float – umbral configurado en la barra lateral
    """
    col1, col2, col3, col4 = st.columns(4)

    # ── Estado del monitor ───────────────────────────────────────────────────
    with col1:
        estado_txt = "🟢 Activo" if st.session_state["monitor_activo"] else "🔴 Detenido"
        st.metric("Estado", estado_txt)

    # ── Porcentaje de anomalía ───────────────────────────────────────────────
    with col2:
        anomalia = st.session_state.get("ultima_anomalia")
        if anomalia is not None:
            delta_txt = "⚠️ ALERTA" if anomalia > umbral else "✅ Normal"
            st.metric(
                label="Índice de Anomalía",
                value=f"{anomalia:.1f}%",
                delta=delta_txt,
                delta_color="inverse",
            )
        else:
            st.metric("Índice de Anomalía", "—")

    # ── Umbral configurado ───────────────────────────────────────────────────
    with col3:
        st.metric("Umbral de Alerta", f"{umbral:.0f}%")

    # ── Total de chunks analizados ───────────────────────────────────────────
    with col4:
        st.metric("Chunks Analizados", st.session_state["total_chunks"])


def renderizar_espectrograma() -> None:
    """Renderiza el espectrograma del último chunk analizado."""
    mel = st.session_state.get("ultimo_mel")
    with st.container():
        st.subheader("📈 Espectrograma de Mel en Tiempo Real")
        if mel is not None:
            fig = graficar_espectrograma(mel)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info(
                "⏳ Esperando primer chunk de audio... "
                "Asegúrate de haber iniciado el monitor."
            )


def renderizar_historico() -> None:
    """Renderiza la gráfica histórica del % de anomalía."""
    historico = st.session_state.get("historico_anomalia", [])
    if len(historico) < 2:
        return

    st.subheader("📉 Histórico de Anomalía (sesión actual)")
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#0d1117")
    ax.set_facecolor="#161b22"
    ax.set_facecolor("#161b22")
    ax.plot(historico, color="#58a6ff", linewidth=1.5, label="% Anomalía")
    ax.axhline(
        y=st.session_state.get("umbral_guardado", 85),
        color="#f85149",
        linestyle="--",
        linewidth=1,
        label="Umbral",
    )
    ax.fill_between(range(len(historico)), historico, alpha=0.15, color="#58a6ff")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Chunk #", color="white")
    ax.set_ylabel("Anomalía (%)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#161b22", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# LÓGICA PRINCIPAL DEL BUCLE DE ANÁLISIS
# ──────────────────────────────────────────────────────────────────────────────
def procesar_cola(umbral: float) -> None:
    """
    Extrae chunks de la cola de audio, calcula el espectrograma y la anomalía,
    y actualiza el estado de sesión de Streamlit.

    Si el % de anomalía supera el umbral, dispara una alerta visual.

    Parámetros
    ----------
    umbral : float – umbral de alerta configurado por el usuario
    """
    captura: CapturaAudio | None = st.session_state.get("captura")
    if captura is None:
        return

    chunk = captura.obtener_chunk()
    if chunk is None:
        return  # Cola vacía, nada que procesar

    # 1. Transformar señal cruda en espectrograma de Mel
    mel = audio_a_mel(chunk)
    st.session_state["ultimo_mel"] = mel

    # 2. Inferir anomalía con el modelo (o mock)
    modelo = st.session_state.get("modelo")
    porcentaje = inferir_anomalia(modelo, mel)
    st.session_state["ultima_anomalia"] = porcentaje
    st.session_state["total_chunks"] += 1

    # 3. Actualizar histórico (máximo 120 valores ≈ 6 minutos a 3s/chunk)
    historico: list = st.session_state["historico_anomalia"]
    historico.append(porcentaje)
    if len(historico) > 120:
        historico.pop(0)

    # 4. Alerta si se supera el umbral
    if porcentaje > umbral:
        st.session_state["alerta_disparada"] = True
        st.toast(
            f"⚠️ ANOMALÍA DETECTADA: {porcentaje:.1f}% > umbral {umbral:.0f}%",
            icon="🚨",
        )
    else:
        st.session_state["alerta_disparada"] = False


# ──────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Función principal de la aplicación Streamlit.

    Ciclo de vida:
      1. Configura la página y el estado inicial.
      2. Renderiza la barra lateral y los controles.
      3. Si el monitor está activo, procesa un chunk de la cola.
      4. Renderiza métricas, espectrograma e histórico.
      5. Espera INTERVALO_REFRESCO segundos y llama a st.rerun() para
         el siguiente ciclo sin bloquear la captura de audio.
    """
    # ── Configuración de la página ─────────────────────────────────────────
    st.set_page_config(
        page_title="Sonitel · Monitor Acústico",
        page_icon="🔊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS personalizado para mejorar la estética oscura con Glassmorphism
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
            
            /* Configuración general */
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
                background-color: #0d1117;
                color: #e6edf3;
            }
            
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                letter-spacing: -0.02em;
            }
            
            /* Tarjetas de Métricas Premium */
            [data-testid="stMetric"] {
                background: rgba(22, 27, 34, 0.6);
                border: 1px solid rgba(48, 54, 61, 0.8);
                border-radius: 12px;
                padding: 16px 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }
            [data-testid="stMetric"]:hover {
                border-color: #1f6feb;
                box-shadow: 0 4px 20px rgba(31, 111, 235, 0.15);
                transform: translateY(-2px);
            }
            
            /* Contenedor de Tarjetas Generales (Glassmorphism) */
            .premium-card {
                background: rgba(22, 27, 34, 0.6);
                border: 1px solid rgba(48, 54, 61, 0.8);
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                backdrop-filter: blur(10px);
                margin-bottom: 20px;
            }
            
            /* Botones Estilo Premium */
            .stButton > button {
                background: #21262d !important;
                color: #c9d1d9 !important;
                border: 1px solid #30363d !important;
                border-radius: 8px !important;
                padding: 8px 16px !important;
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
            }
            .stButton > button:hover {
                border-color: #8b949e !important;
                background: #30363d !important;
                color: #f0f6fc !important;
            }
            
            /* Botón Primario */
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
                color: #ffffff !important;
                border: none !important;
                box-shadow: 0 4px 12px rgba(31, 111, 235, 0.25) !important;
            }
            .stButton > button[kind="primary"]:hover {
                background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
                box-shadow: 0 4px 20px rgba(56, 139, 253, 0.4) !important;
            }
            
            /* Barra lateral */
            section[data-testid="stSidebar"] {
                background-color: #161b22;
                border-right: 1px solid #30363d;
            }
            
            /* Badges de Veredicto Normal y Anomalía */
            .badge-normal {
                display: inline-block;
                background: rgba(46, 204, 113, 0.15);
                border: 1px solid #2ecc71;
                color: #2ecc71;
                font-family: 'JetBrains Mono', monospace;
                font-size: 14px;
                font-weight: 600;
                padding: 10px 24px;
                border-radius: 8px;
                letter-spacing: 0.05em;
                margin-bottom: 15px;
            }
            
            .badge-anomaly {
                display: inline-block;
                background: rgba(231, 76, 60, 0.15);
                border: 1px solid #e74c3c;
                color: #e74c3c;
                font-family: 'JetBrains Mono', monospace;
                font-size: 14px;
                font-weight: 600;
                padding: 10px 24px;
                border-radius: 8px;
                letter-spacing: 0.05em;
                margin-bottom: 15px;
                animation: pulso-rojo 1.5s ease-in-out infinite alternate;
            }
            
            /* Alerta de anomalía en monitor */
            .alerta-anomalia {
                background: linear-gradient(135deg, rgba(248, 81, 73, 0.15), rgba(248, 81, 73, 0.05));
                border: 1px solid #f85149;
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
                animation: pulso 1.5s ease-in-out infinite alternate;
            }
            
            @keyframes pulso {
                from { box-shadow: 0 0 8px rgba(248, 81, 73, 0.3); }
                to   { box-shadow: 0 0 20px rgba(248, 81, 73, 0.6); }
            }
            
            @keyframes pulso-rojo {
                from { box-shadow: 0 0 4px rgba(231, 76, 60, 0.2); }
                to   { box-shadow: 0 0 12px rgba(231, 76, 60, 0.5); }
            }
            
            /* Estilizar selectbox de Streamlit */
            div[data-baseweb="select"] {
                border-radius: 8px;
                background-color: #161b22;
                border: 1px solid #30363d;
            }
            
            /* Estilizar file uploader */
            div[data-testid="stFileUploader"] {
                border: 2px dashed rgba(48, 54, 61, 0.8);
                border-radius: 12px;
                background-color: rgba(22, 27, 34, 0.4);
                padding: 15px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Inicialización del estado ──────────────────────────────────────────
    _inicializar_estado()

    # ── Barra lateral ─────────────────────────────────────────────────────
    umbral, _ = renderizar_sidebar()
    st.session_state["umbral_guardado"] = umbral  # Persiste para el gráfico histórico

    # ── Cabecera ──────────────────────────────────────────────────────────
    renderizar_cabecera()

    # ── Selección de Fuente de Audio ──────────────────────────────────────
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    origen_audio = st.selectbox(
        "🔌 Fuente de Entrada de Audio",
        options=["🎙️ Micrófono en Tiempo Real", "📁 Archivo de Grabación (.wav, .mp3, etc.)"],
        help="Elige si deseas capturar audio en tiempo real o analizar una grabación existente."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Detener el monitor si se cambia a modo Archivo
    if origen_audio != "🎙️ Micrófono en Tiempo Real" and st.session_state["monitor_activo"]:
        captura = st.session_state["captura"]
        if captura:
            captura.detener()
        st.session_state["monitor_activo"] = False
        st.session_state["captura"] = None
        st.toast("⏹️ Monitor de micrófono detenido automáticamente.", icon="ℹ️")

    # ── Modo 1: Micrófono en Tiempo Real ──────────────────────────────────
    if origen_audio == "🎙️ Micrófono en Tiempo Real":
        micros = obtener_micros()
        if not micros:
            st.error(
                "❌ No se encontraron dispositivos de entrada de audio. "
                "Conecta un micrófono e reinicia la aplicación."
            )
        else:
            # Envolver controles en tarjeta premium
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            iniciar, detener, device_id = renderizar_controles(micros)
            st.markdown('</div>', unsafe_allow_html=True)

            if iniciar and not st.session_state["monitor_activo"]:
                captura = CapturaAudio(device_id=device_id)
                captura.iniciar()
                st.session_state["captura"] = captura
                st.session_state["monitor_activo"] = True
                st.session_state["total_chunks"] = 0
                st.session_state["historico_anomalia"] = []
                st.success(f"✅ Monitor iniciado · Dispositivo #{device_id}")

            if detener and st.session_state["monitor_activo"]:
                captura: CapturaAudio = st.session_state["captura"]
                captura.detener()
                st.session_state["monitor_activo"] = False
                st.session_state["captura"] = None
                st.info("⏹️ Monitor detenido.")

            st.markdown("---")

            # Alerta de anomalía
            if st.session_state.get("alerta_disparada"):
                anomalia_actual = st.session_state.get("ultima_anomalia", 0)
                st.markdown(
                    f"""
                    <div class="alerta-anomalia">
                        <h3 style="margin: 0 0 8px 0; color: #ff6b6b;">🚨 ¡ANOMALÍA ACÚSTICA DETECTADA!</h3>
                        <p style="margin: 0; color: #ffb3b3;">
                            El índice de anomalía actual es de <strong>{anomalia_actual:.1f}%</strong>,
                            superando el umbral límite de <strong>{umbral:.0f}%</strong>.
                            Se recomienda la inspección preventiva del equipo.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Métricas
            renderizar_metricas(umbral)

            st.markdown("---")

            # Espectrograma + Info
            col_spec, col_info = st.columns([3, 1], gap="medium")
            with col_spec:
                renderizar_espectrograma()
            with col_info:
                st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
                st.subheader("ℹ️ Diagnóstico")
                anomalia = st.session_state.get("ultima_anomalia")
                if anomalia is not None:
                    nivel = (
                        "🟢 Óptimo" if anomalia < 40
                        else "🟡 Atención" if anomalia < umbral
                        else "🔴 Crítico"
                    )
                    st.markdown(f"**Nivel de Estado:** {nivel}")
                    st.progress(int(min(anomalia, 100)) / 100)
                    st.caption(
                        f"Umbral: {umbral:.0f}% | "
                        f"Actual: {anomalia:.1f}%"
                    )
                else:
                    st.caption("Esperando lecturas de audio...")

                st.markdown("<hr style='margin: 15px 0; border-color: rgba(48,54,61,0.5);'>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    **Configuración:**
                    - Sampling Rate: `{SAMPLE_RATE} Hz`
                    - Ventana: `{CHUNK_SEGUNDOS}s`
                    - Bandas Mel: `{N_MELS}`
                    """
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Gráfico histórico
            st.markdown("---")
            renderizar_historico()

    # ── Modo 2: Cargar Archivo de Audio ───────────────────────────────────
    else:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        modelo = st.session_state.get("modelo")
        if modelo is None:
            st.warning("📥 Por favor, carga el modelo de red neuronal (.pt) en la barra lateral antes de analizar archivos.")
        else:
            uploaded_file = st.file_uploader(
                "Arrastra o selecciona un archivo de grabación de audio (.wav, .mp3, .flac)",
                type=["wav", "mp3", "flac", "ogg"],
                key="file_uploader_key"
            )
        st.markdown('</div>', unsafe_allow_html=True)

        if modelo is not None and uploaded_file is not None:
            # Guardar el archivo temporalmente
            suffix = "." + uploaded_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            with st.spinner("Procesando audio y aplicando inferencia del Autoencoder..."):
                try:
                    # 1. Cargar el audio
                    y, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
                    
                    # 2. Segmentar en ventanas con solapamiento
                    hop_samples = int(CHUNK_MUESTRAS * 0.5)
                    ventanas_anomalias = []
                    ultimo_mel_completo = None
                    
                    start_idx = 0
                    while start_idx + CHUNK_MUESTRAS <= len(y):
                        segment = y[start_idx : start_idx + CHUNK_MUESTRAS]
                        mel = audio_a_mel(segment, sr=SAMPLE_RATE)
                        ultimo_mel_completo = mel
                        
                        pct = inferir_anomalia(modelo, mel)
                        ventanas_anomalias.append(pct)
                        start_idx += hop_samples
                        
                    if len(ventanas_anomalias) == 0:
                        mel = audio_a_mel(y, sr=SAMPLE_RATE)
                        ultimo_mel_completo = mel
                        pct = inferir_anomalia(modelo, mel)
                        ventanas_anomalias.append(pct)

                    # 3. Métricas
                    mean_anom = np.mean(ventanas_anomalias)
                    max_anom = np.max(ventanas_anomalias)
                    n_anomalous = sum(1 for v in ventanas_anomalias if v > umbral)
                    anomaly_ratio = n_anomalous / len(ventanas_anomalias)
                    
                    is_anomaly = mean_anom > umbral

                    # 4. Diseño del Dashboard de Archivos
                    st.markdown("### 📊 Dashboard de Diagnóstico del Archivo")
                    
                    col_veredicto, col_graph_file = st.columns([1.2, 1.8], gap="large")
                    
                    with col_veredicto:
                        st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
                        st.markdown("#### Veredicto de Estado")
                        
                        if is_anomaly:
                            st.markdown('<span class="badge-anomaly">🔴 ANOMALÍA DETECTADA</span>', unsafe_allow_html=True)
                            st.error("El patrón acústico difiere significativamente del comportamiento normal.")
                        else:
                            st.markdown('<span class="badge-normal">🟢 FUNCIONAMIENTO NORMAL</span>', unsafe_allow_html=True)
                            st.success("Patrón acústico estable y dentro del comportamiento de referencia.")
                        
                        st.markdown("<hr style='margin: 15px 0; border-color: rgba(48,54,61,0.5);'>", unsafe_allow_html=True)
                        st.markdown("#### Audio Reproductor")
                        st.audio(uploaded_file)
                        
                        st.markdown("<hr style='margin: 15px 0; border-color: rgba(48,54,61,0.5);'>", unsafe_allow_html=True)
                        st.markdown("#### Resumen del Análisis")
                        
                        # Tabla de resumen
                        st.markdown(
                            f"""
                            | Métrica | Valor |
                            |---|---|
                            | **Anomalía Promedio** | `{mean_anom:.1f}%` |
                            | **Anomalía Máxima** | `{max_anom:.1f}%` |
                            | **Segmentos Analizados** | `{len(ventanas_anomalias)}` |
                            | **Segmentos Anómalos** | `{n_anomalous} ({anomaly_ratio*100:.1f}%)` |
                            | **Umbral de Alerta** | `{umbral:.0f}%` |
                            """
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col_graph_file:
                        # Columna de gráficos
                        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                        st.markdown("#### Espectrograma de Mel del Segmento")
                        if ultimo_mel_completo is not None:
                            fig = graficar_espectrograma(ultimo_mel_completo)
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                        st.markdown("#### Evolución de Anomalías por Ventana")
                        
                        fig, ax = plt.subplots(figsize=(8, 3.8), facecolor="#0d1117")
                        ax.set_facecolor("#161b22")
                        
                        x_indices = range(len(ventanas_anomalias))
                        colores_barras = ["#f85149" if v > umbral else "#58a6ff" for v in ventanas_anomalias]
                        
                        ax.bar(x_indices, ventanas_anomalias, color=colores_barras, alpha=0.8, width=0.8)
                        ax.axhline(umbral, color="#f85149", linestyle="--", linewidth=1.5, label=f"Umbral ({umbral:.0f}%)")
                        ax.axhline(mean_anom, color="#ffc107", linestyle=":", linewidth=2, label=f"Media ({mean_anom:.1f}%)")
                        
                        ax.set_ylim(0, 105)
                        ax.set_xlabel("Ventana (Tiempo)", color="white")
                        ax.set_ylabel("Anomalía (%)", color="white")
                        ax.tick_params(colors="white")
                        ax.legend(facecolor="#161b22", labelcolor="white")
                        for spine in ax.spines.values():
                            spine.set_edgecolor("#30363d")
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Error al procesar el archivo: {e}")
                finally:
                    # Limpiar archivo temporal
                    try:
                        os.remove(temp_path)
                    except:
                        pass

    # ──────────────────────────────────────────────────────────────────────
    # BUCLE DE REFRESCO AUTOMÁTICO
    # Si el monitor está activo, procesa la cola y recarga la UI.
    # El sleep evita consumo innecesario de CPU entre refrescos.
    # ──────────────────────────────────────────────────────────────────────
    if origen_audio == "🎙️ Micrófono en Tiempo Real" and st.session_state["monitor_activo"]:
        captura = st.session_state["captura"]
        if captura and captura.error_msg:
            st.error(f"❌ Error en la captura de audio: {captura.error_msg}")
            captura.detener()
            st.session_state["monitor_activo"] = False
            st.session_state["captura"] = None
            st.rerun()
        else:
            procesar_cola(umbral)
            time.sleep(INTERVALO_REFRESCO)
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
