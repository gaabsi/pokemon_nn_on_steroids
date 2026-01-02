# Pour que le script tourne aussi bien en local que dans son conteneur 
if [ -d "/app" ]; then
    BASE_DIR="/app"
    PYTHON="python"
else
    BASE_DIR="$HOME/pokemon_nn_on_steroids"
    PYTHON="$BASE_DIR/venv/bin/python"
fi

export BASE_DIR

# # On récupère les png que j'ai stocké dans un dataset sur huggingface
# $PYTHON src/get_images.py

# # Standardisation des images 
# $PYTHON src/data_prep.py

# # #Entrainement modele 
# $PYTHON src/modele.py

# Evaluation du modele 
$PYTHON src/evaluation.py

