#  Mini RAG – Retrieval Augmented Generation (Version Rapide & Simple)

Ce projet met en place un **mini-RAG léger et performant**, basé uniquement sur :

-  Des fichiers **TXT**
-  Un modèle d’embedding (BAAI/bge-large-en)
-  Une base PostgreSQL + pgvector pour la recherche sémantique
-  Un modèle LLM (Gemini 2.5-flash) pour générer la réponse finale

Ce pipeline est conçu pour être **simple, rapide et compréhensible**, sans traitement PDF ou DOCX.

---

##  Fonctionnalités

- ✔️ Lecture **uniquement** des fichiers `.txt`  
- ✔️ Génération d’embeddings pour chaque document  
- ✔️ Stockage des vecteurs avec **pgvector**  
- ✔️ Recherche sémantique avec un `vector <-> vector`  
- ✔️ Réponse générée par LLM avec le contexte trouvé  
- ✔️ Pipeline clair : *Ingestion → Indexation → Récupération → Réponse*

---

##  Structure du projet
```text
chatbot-RAG/
├── data/               # fichiers .txt
├── notebook/
│   └── prototype.py    # Script principal
├── venv/               # environnement virtuel
├── src/
│   └── .env            # fichier de configuration
├── requirements.txt
└── README.md
```
---


##  Output du projet

<img width="1419" height="608" alt="Screenshot 2025-12-12 111909" src="https://github.com/user-attachments/assets/22958229-4ab2-437e-bce3-a03733d111c2" />
