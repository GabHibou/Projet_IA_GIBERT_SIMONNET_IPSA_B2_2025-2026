import subprocess
import sys
import time
import webbrowser
import signal
from pathlib import Path

# Chemin vers le fichier Streamlit à lancer (situé dans le même dossier)
APP_PATH = Path(__file__).parent / "app_UI_streamlit.py"

# URL locale où l’application Streamlit sera accessible
URL = "http://localhost:8501"

# Lancement de l’application Streamlit comme sous-processus Python
process = subprocess.Popen([
    sys.executable,          # Utilise l’interpréteur Python courant
    "-m", "streamlit",       # Lance Streamlit en tant que module
    "run",
    str(APP_PATH),           # Chemin vers le script Streamlit
    "--server.headless", "true",  # Mode headless (sans UI Streamlit interne)
    "--server.port", "8501"       # Port utilisé par le serveur Streamlit
])

def shutdown(signum=None, frame=None):
    """
    Fonction de fermeture propre de l’application Streamlit.
    Elle est appelée lors d’un CTRL+C ou d’un signal de terminaison.
    """
    print("\n🛑 Fermeture de Streamlit...")

    # Vérifie si le processus est encore actif
    if process.poll() is None:
        process.terminate()  # Demande une terminaison propre
        try:
            process.wait(timeout=5)  # Attend la fin du processus
        except subprocess.TimeoutExpired:
            process.kill()  # Force l’arrêt si le délai est dépassé

    sys.exit(0)

# Capture des signaux système (CTRL+C, arrêt du terminal, etc.)
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# Attente pour laisser le temps au serveur Streamlit de démarrer
time.sleep(2)

# Ouverture automatique de l’application dans le navigateur par défaut
webbrowser.open(URL)

# Attend la fin du processus Streamlit
process.wait()

# Appel final de la fonction de fermeture
shutdown()

