import os
import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    script_path = os.path.join(base_path, "ennoiaCAT_RAG_INT.py")

    sys.argv = [
        "streamlit",
        "run",
        script_path,
        "--server.port=8501",
        "--server.enableXsrfProtection=false",
        "--server.enableCORS=false",
        "--global.developmentMode=false"  # 👈 This disables dev mode so port config works
    ]
    print(sys.argv)  # Debugging: print the arguments being passed to Streamlit
    sys.exit(stcli.main())