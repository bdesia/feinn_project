#!/bin/bash

POETRY_PYPROJECT="pyproject.toml"
PROJECT_NAME=$(grep 'name =' $POETRY_PYPROJECT | awk -F '"' '{print $2}' | head -1)

# Validar que se haya extraído el nombre del proyecto
if [ -z "$PROJECT_NAME" ]; then
    echo "Error: No se pudo extraer el nombre del proyecto de $POETRY_PYPROJECT"
    exit 1
fi
echo "Nombre del proyecto: $PROJECT_NAME"

PYTHON_VERSION="3.11"

echo "Configurando entorno con Poetry..."

# Usa o crea un virtualenv con la versión específica de Python
poetry env use $PYTHON_VERSION

# Instala todas las dependencias del proyecto
poetry install

# Instala herramientas de desarrollo
poetry add --group dev ipykernel

# Registra el kernel en Jupyter
poetry run python -m ipykernel install --user --name=$PROJECT_NAME --display-name "Python ($PROJECT_NAME)"

# Ahora, obtiene la ruta del entorno virtual precisamente donde está en realidad
VENV_PATH=$(poetry env info --path)

# La carpeta site-packages en ese entorno
SITE_PACKAGES="$VENV_PATH/Lib/site-packages"

# Crear la carpeta si no existe
mkdir -p "$SITE_PACKAGES"

# Ruta actual en formato Windows
ABS_PATH=$(cygpath -w "$(pwd)")

# Crea el archivo .pth en la carpeta correcta del entorno virtual
echo "$ABS_PATH\src" > "$SITE_PACKAGES/$PROJECT_NAME.pth"

echo "Archivo .pth creado en: $SITE_PACKAGES/$PROJECT_NAME.pth"
echo "Contenido:"
cat "$SITE_PACKAGES/$PROJECT_NAME.pth"

echo "El entorno con Poetry está listo y configurado."
