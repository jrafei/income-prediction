
"""
evaluate.py : Contient des fonctions pour évaluer les performances des modèles.
Fonctions : evaluate_model(), plot_confusion_matrix()
"""


from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def plot_confusion_matrix_sns(y_test, y_pred, model_name):
    """
    Affiche la matrice de confusion pour les prédictions d'un modèle.
    """

    print('='*30)
    print(model_name)
    print('='*30, '\n')

    print("Matrice de confusion:")
    c_matrix = confusion_matrix(y_test, y_pred)
    print(c_matrix, '\n') # afficher à l'écran notre matrice de confusion
    print("Rapport de classification:")
    print(classification_report(y_test, y_pred), '\n')
    print('Exactitude: %f' %(accuracy_score(y_test, y_pred)*100), '\n')

    # Affichage et enregistrement de la matrice de confusion avec Seaborn
    plt.figure(figsize=(8, 4))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    save_image(plt,model_name)

   
def save_image(plt,filename):
    # Ajuster les marges pour que le titre soit complètement visible
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Enregistrer le schéma dans le dossier 'images'
    plot_filename = '../images/' + filename + '.png'
    plt.savefig(plot_filename)  # Enregistre le schéma
    plt.show()  # Affiche le schéma
    plt.close()  # Ferme la figure pour éviter les conflits de figure
    

    
