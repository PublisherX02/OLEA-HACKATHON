# OLEA AI - Intelligent Insurance Advisor 🚀

Bienvenue dans le dépôt officiel de la solution **OLEA AI**, conçue spécifiquement pour la **Phase II du Hackathon DataQuest OLEA**. Ce projet représente une architecture MLOps complète, prête pour la production, intégrant un modèle de Machine Learning haute performance, une sécurité Zero-Trust et une expérience conversationnelle GenAI localisée.

## 🌟 Vision & Valeur Ajoutée (Bonus GenAI)
Plutôt que de simplement renvoyer une prédiction mathématique brute (ex: "Pack 3"), notre solution intègre **NVIDIA LLaMA-70B** pour transformer la décision du modèle ML en un argumentaire commercial chaleureux, personnalisé et **en dialecte tunisien (Tounsi)**. 
Cette surcouche d'IA Générative permet aux courtiers OLEA de proposer instantanément des explications claires et convaincantes, adaptées à la culture locale.

---

## 🏗️ Architecture MLOps & Déploiement

Notre système est divisé en microservices (API Backend + UI Frontend) pour garantir scalabilité, modularité et isolation. 

### 1. API d'Inférence Sécurisée (FastAPI) `security_api.py`
Le moteur de notre application. L'API charge en mémoire le modèle Scikit-Learn (provenant de la Phase I) et expose les endpoints :
*   **Validation Stricte (Pydantic) :** Les données entrantes sont systématiquement filtrées pour empêcher les injections SQL et garantir l'intégrité des types.
*   **Sécurité Zero-Trust :** Intègre la vérification JWT, le Rate Limiting anti-DDoS, une politique CORS stricte et le masquage des informations PII (Identity Numbers) dans les logs d'audit.
*   **Endpoint `/api/ml_predict` :** Aligné *au pixel près* avec le code de la Phase I pour éliminer tout *Training-Serving Skew*.

### 2. Interface Utilisateur (Streamlit) `app.py`
Le portail interactif destiné aux courtiers ou clients OLEA.
*   **Devis Rapide ML :** Saisie dynamique des informations client pour obtenir instantanément la recommandation de Pack assécurologique et l'argumentaire Tounsi de notre agent virtuel *Imani*.
*   **Reconnaissance Vocale (Whisper/Google STT) :** Avec un intercepteur de vocabulaire spécifique à l'assurance tunisienne (ex: correction automatique de "kahraba" en "karhba").

### 3. Conteneurisation (Docker & Compose)
L'application entière est encapsulée via **Docker**. Le fichier `docker-compose.yml` déploie les deux services au sein d'un réseau interne (`olea_network`) isolé de l'extérieur.

---

## 🚀 Comment Lancer l'Application (En 1 Commande)

Grâce à Docker Compose, le déploiement sur n'importe quel serveur ou machine locale est immédiat.

### Prérequis
- [Docker](https://www.docker.com/) et [Docker Compose](https://docs.docker.com/compose/)
- Une clé API NVIDIA (pour le LLaMA-70B)

### Étapes d'installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/votre-utilisateur/olea-ai-hackathon.git
   cd olea-ai-hackathon
   ```

2. **Configurer l'environnement :**
   Créez un fichier `.env` à la racine (ou exportez la variable) avec votre clé API :
   ```bash
   NVIDIA_API_KEY="votre_cle_api_nvidia_ici"
   ```

3. **Lancer les conteneurs :**
   ```bash
   docker-compose up --build -d
   ```

4. **Accéder à l'application :**
   - **Interface Utilisateur (Streamlit) :** Rendez-vous sur [http://localhost:8501](http://localhost:8501)
   - **Documentation de l'API (Swagger UI) :** Rendez-vous sur [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛡️ Focus sur la Sécurité (Zero-Trust)
Nous n'avons pas codé pour un simple hackathon, nous avons codé pour une vraie entreprise. Notre conception intègre :
1. **Rate Limiting :** Bloque automatiquement les requêtes abusives (spam) basées sur l'identité (User ID masqué).
2. **PII Masking :** Les identifiants sensibles n'apparaissent jamais en clair dans l'immouvable `audit.log` (SOC2 compliance approach).
3. **Internal Networking :** Le backend n'expose aucun port au public si déployé derrière un reverse-proxy, le frontend communique avec lui via le réseau privé Docker `olea_network`.

## 📦 Reproduction de la Phase I
Le dossier contient également `solution.py` et `model.pkl` (le modèle de la Phase I). Le `requirements.txt` originel est préservé à vide pour respecter la sandbox EvalDA, tandis que les dépendances opérationnelles de la Phase II sont gérées isolément via `requirements_api.txt` et `requirements_ui.txt`.