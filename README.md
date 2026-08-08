# Demo publica: riesgo tributario con BETO y RAG

Aplicacion Streamlit independiente para la sustentacion del proyecto de PLN. El
repositorio contiene solo datos sinteticos. No se deben agregar registros SIRE,
PLE, RUC, comprobantes ni documentos reales de la empresa.

## Arquitectura de la demo

- Clasificacion triclase con el BETO ajustado: Bajo, Medio y Alto Riesgo.
- Reglas duras que mantienen la alerta cuando existe un estado tributario critico.
- Recuperacion hibrida sobre fragmentos sinteticos con trazabilidad por documento
  y pagina.
- Un unico BETO compartido por clasificacion y embeddings para reducir memoria.

## Prueba local

Desde la carpeta `demo_publica_streamlit`:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Para probar sin duplicar el modelo durante el desarrollo:

```powershell
$env:BETO_MODEL_PATH="..\modelos\beto_riesgo_experto"
python -m streamlit run app.py
```

## Preparacion del repositorio publico

El archivo `model.safetensors` supera el limite normal de GitHub y debe viajar
mediante Git LFS. Copie al paquete solamente los cuatro archivos de inferencia:

```powershell
New-Item -ItemType Directory -Force modelos\beto_riesgo_experto
Copy-Item ..\modelos\beto_riesgo_experto\config.json modelos\beto_riesgo_experto\
Copy-Item ..\modelos\beto_riesgo_experto\model.safetensors modelos\beto_riesgo_experto\
Copy-Item ..\modelos\beto_riesgo_experto\tokenizer.json modelos\beto_riesgo_experto\
Copy-Item ..\modelos\beto_riesgo_experto\tokenizer_config.json modelos\beto_riesgo_experto\
```

Luego cree un repositorio independiente:

```powershell
git init
git lfs install
git add .
git commit -m "Publicar demo academica BETO y RAG"
```

Cree un repositorio vacio en GitHub, agregue el remoto indicado por GitHub y
ejecute `git push -u origin main`.

## Despliegue en Streamlit Community Cloud

1. Ingrese a `https://share.streamlit.io` con la cuenta de GitHub.
2. Seleccione **Create app** y el repositorio publico de la demo.
3. Indique la rama `main` y el archivo principal `app.py`.
4. En opciones avanzadas seleccione Python 3.12 y despliegue.
5. Verifique en el enlace publico las consultas de las pestañas BETO y RAG.

## Limitaciones declaradas

Las metricas corresponden a un holdout academico de 18 casos, derivado de 90
operaciones etiquetadas. El resultado es preliminar y no debe interpretarse como
una validacion productiva o asesoria tributaria.

