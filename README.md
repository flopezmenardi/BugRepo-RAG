# 🐛 BugRepo-RAG

Sistema de **Retrieval-Augmented Generation (RAG)** para análisis inteligente de reportes de bugs utilizando embeddings vectoriales y modelos de lenguaje.

## 📋 Descripción del Proyecto

BugRepo-RAG es un sistema que permite consultar y analizar grandes volúmenes de reportes de bugs de manera inteligente. El sistema convierte los reportes en representaciones vectoriales (embeddings) y utiliza búsqueda semántica para encontrar bugs similares, generando después mini-informes contextualizados usando modelos de lenguaje.

### 🎯 Funcionalidades Principales

- **Extracción de datos** desde APIs (Mozilla Bugzilla)
- **Procesamiento y limpieza** de reportes de bugs
- **Generación de embeddings** usando modelos de OpenAI
- **Almacenamiento vectorial** en base de datos Pinecone
- **Búsqueda semántica** de bugs similares
- **Generación de respuestas** contextualizadas con LLMs

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Extracción     │ -> │   Procesamiento  │ -> │   Embeddings    │
│  de Datos       │    │   y Limpieza     │    │   Vectoriales   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Datos Crudos    │    │ Datos Limpios    │    │ Base Vectorial  │
│ (CSV/JSON)      │    │ Estructurados    │    │ (Pinecone)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Reporte         │ <- │ Generación LLM   │ <- │ Búsqueda        │
│ Final           │    │ (OpenAI GPT)     │    │ Semántica       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Estructura del Proyecto

```
BugRepo-RAG/
├── src/
│   ├── config.py                # Configuración central del sistema
│   ├── pipeline.py              # Orquestador principal 
│   ├── data_extraction/
│   │   ├── extract.py           # Extracción desde Bugzilla API
│   │   └── extract_comments.py  # Extracción de comentarios
│   ├── embeddings/
│   │   ├── embedder.py          # Generación de embeddings OpenAI
│   │   └── indexer.py           # Indexación en Pinecone
│   ├── retrieval/
│   │   └── retriever.py         # Búsqueda vectorial 
│   ├── llm/
│   │   ├── generator.py         # Generación de respuestas 
│   │   └── prompts.py           # Augmentation del bug entrante
│   └── evaluation/
│       └── metrics.py           # Métricas de evaluación 
├── data/
│   └── sample_bugs.csv          # Dataset de bugs
├── outputs/                     # Resultados generados
├── requirements.txt             # Dependencias Python
└── README.md                    
```

## 🚀 Configuración Inicial

### 1. Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### 2. Configuración de Variables de Entorno

Crea un archivo `.env` con tus credenciales:

```env
# OpenAI Configuration
OPENAI_API_KEY=tu_clave_openai_aqui

# Pinecone Configuration
PINECONE_API_KEY=tu_clave_pinecone_aqui
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=bugrepo
```

## 🔧 Uso del Sistema

### Extracción de Datos

```bash
# Opcional: Extraer metadata de bugs desde Bugzilla (se puede usar sample_bugs.csv en lugar de extraer)
python src/data_extraction/extract.py

# Extraer comentarios de bugs existentes
python src/data_extraction/extract_comments.py
```

### Indexación de Bugs (Una sola vez)

```bash
# Modo completo (cambiar test_limit=None en el código)
python src/embeddings/indexer.py
```

## 📊 Componentes Implementados

- **📤 Extracción de datos** - Scripts para APIs de Bugzilla
- **⚙️ Configuración** - Sistema centralizado de configuración
- **🔢 Embeddings** - Generación vectorial con OpenAI (512 dimensiones)
- **📚 Indexación** - Almacenamiento en Pinecone con metadata
- **🔍 Retrieval** - Búsqueda semántica en base vectorial
- **🤖 Generación LLM** - Respuestas contextualizadas con GPT
- **🔄 Pipeline** - Orquestador end-to-end
- **📈 Evaluación** - Métricas de precisión y relevancia

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+** - Lenguaje principal
- **OpenAI** - Embeddings (text-embedding-3-small) y LLMs
- **Pinecone** - Base de datos vectorial
- **Pandas** - Manipulación de datos
- **Requests** - Llamadas a APIs externas

## 📝 Notas de Desarrollo

- **Dimensiones**: Configurado para 512 dimensiones (Pinecone free tier)
- **Batch processing**: Procesamiento en lotes para eficiencia
- **Error handling**: Manejo robusto de errores de API
- **Logging**: Sistema de logs detallado para debugging
