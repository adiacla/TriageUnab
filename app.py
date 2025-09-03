import gradio as gr
import joblib
from google import genai
from google.genai import types
import pandas as pd


# 1. Cargar modelo entrenado
modelo = joblib.load("model.pkl")   # Asegúrate que este archivo existe en tu carpeta

# 2. Inicializar cliente Gemini
client = genai.Client(api_key="AIzaSyCx90r8SfVxj7CCYKyPT1ScjwYjR6a2aTM")   # <-- pon aquí tu API Key

# 3. Función de predicción
# 3. Función de predicción
def predecir_triage(motivo, edad, fc, fr, pas, sat, temp, gcs):
    try:
        # Crear DataFrame con los mismos nombres de columnas del entrenamiento
        X = pd.DataFrame([{
            "motivo": motivo,
            "edad": edad,
            "fc": fc,
            "fr": fr,
            "pas": pas,
            "sat": sat,
            "temp": temp,
            "gcs": gcs
        }])

        # Predecir
        pred = modelo.predict(X)[0]

        # Definir nivel de alerta según ESI
        if pred == 1:
            alerta = "CRÍTICO"
        elif pred == 2:
            alerta = "ALTO"
        elif pred == 3:
            alerta = "MODERADO"
        else:
            alerta = "BAJO"

        # Prompt a Gemini
        prompt = f"""
Eres un asistente de admisión de urgencias.
Genera un reporte estructurado con los siguientes apartados:

1. Resumen del caso
2. Justificación del ESI
3. Nivel de alerta
4. Acciones inmediatas

Variables del paciente:
- Motivo: {motivo}
- Edad: {edad} años
- FC: {fc} lpm
- FR: {fr} rpm
- PAS: {pas} mmHg
- SatO₂: {sat} %
- Temp: {temp} °C
- GCS: {gcs}
- Nivel ESI predicho: {pred}
- Nivel de alerta: {alerta}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"❌ Error: {str(e)}"

# 4. Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Triage ESI con TriageUNAB")
    with gr.Row():
        motivo = gr.Textbox(label="Motivo de consulta")
    with gr.Row():
        edad = gr.Number(label="Edad (años)", precision=0)
        fc = gr.Number(label="Frecuencia cardiaca (lpm)", precision=0)
        fr = gr.Number(label="Frecuencia respiratoria (rpm)", precision=0)
    with gr.Row():
        pas = gr.Number(label="Presión arterial sistólica (mmHg)", precision=0)
        sat = gr.Number(label="Saturación O₂ (%)", precision=0)
        temp = gr.Number(label="Temperatura (°C)")
        gcs = gr.Number(label="Glasgow (3-15)", precision=0)

    btn = gr.Button("Generar Reporte")
    salida = gr.Textbox(label="Resumen clínico", lines=15)

    btn.click(
        fn=predecir_triage,
        inputs=[motivo, edad, fc, fr, pas, sat, temp, gcs],
        outputs=salida
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
