Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\lenovo\.gemini\antigravity\scratch\heart-disease-predictor"
WshShell.Run """C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe"" -m streamlit run ""C:\Users\lenovo\.gemini\antigravity\scratch\heart-disease-predictor\app.py"" --server.port 8501 --server.headless false", 0, False
