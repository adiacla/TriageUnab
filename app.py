 import os
import gradio as gr
import joblib
import pandas as pd
from openai import OpenAI  # Reemplazamos genai por OpenAI

# 1. Cargar modelo entrenado de Machine Learning

modelo = joblib.load("./model.pkl") 

# 2. Inicializar cliente de NVIDIA (utilizando la interfaz de OpenAI)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY", "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")
)

# 3. Función de predicción y generación de reporte
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

        # Predecir con el modelo local (.pkl)
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

        # Prompt estructurado para el LLM de NVIDIA
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

        # Llamada a la API de NVIDIA
        completion = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,  # Ajustado para una respuesta de triage (65k puede ser excesivo para esta app)
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 1024
            },
            stream=False
        )

        # Retornar el texto generado por el modelo
        return completion.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"

# 4. Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Triage ESI con TriageUNAB (Powered by NVIDIA)")
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
