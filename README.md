# OLEA AI - Intelligent Insurance Advisor 🚀

Bienvenue dans le dépôt officiel de la solution **OLEA AI**, conçue spécifiquement pour la **Phase II du Hackathon DataQuest OLEA**. Ce projet représente une architecture MLOps complète, prête pour la production, intégrant un modèle de Machine Learning haute performance, une sécurité Zero-Trust et une expérience conversationnelle GenAI exclusive.

## 🌟 Vision & Valeur Ajoutée (Bonus GenAI)
Plutôt que de simplement renvoyer une prédiction mathématique brute (ex: "Pack 3") via un formulaire, notre solution intègre **NVIDIA LLaMA-70B** comme un Agent Commercial Actif (*Imani*).
Cet agent va de manière proactive :
1. Démarrer la conversation en demandant au profil client les métriques exactes nécessaires à la prédiction ML.
2. Déclencher le modèle ML pré-entrainé de la Phase I.
3. Transformer la prédiction mathématique brute en un **argumentaire commercial persuasif, entièrement sur-mesure et rédigé en Tounsi (dialecte tunisien)**.
4. **Réserver un rendez-vous automatiquement (Anti-Bureaucratie) :** Si le client est convaincu par l'argumentaire et accepte, l'agent confirme qu'un formulaire de réservation vient d'être soumis directement à OLEA. 

L'époque des formulaires ML froids et abstraits est révolue : bienvenue dans le courtage conversationnel de demain.

---

## 🏗️ Architecture MLOps & Déploiement

Notre système est divisé en microservices (API Backend + UI Frontend) pour garantir scalabilité, modularité et isolation. 

### 1. API d'Inférence Sécurisée (FastAPI) `security_api.py`
Le moteur sécurisé de notre application :
*   **Validation Stricte (Pydantic) :** Les données entrantes sont systématiquement filtrées pour empêcher les injections SQL.
*   **Sécurité Zero-Trust :** Intègre la vérification JWT, le Rate Limiting (limitation de taux), une politique CORS stricte et le masquage des informations PII.
*   **Endpoint `/api/ml_predict` :** Aligné *au pixel près* avec le code de la Phase I pour éliminer tout *Training-Serving Skew*.

### 2. Interface Utilisateur Conversationnelle (Streamlit) `app.py`
Le portail dynamique :
*   **100% Agents IA :** Plus d'inputs fastidieux ni d'onglets de données. L'utilisateur dialogue directement par texte ou en vocal. L'Agent IA s'occupe de l'extraction de paramètres en tâche de fond.
*   **Reconnaissance Vocale (Whisper/Google STT) :** Un intercepteur de vocabulaire spécifique au marché local (ex: il corrige automatiquement "kahraba" en "karhba").

### 3. Conteneurisation (Docker)
L'application entière est "Dockerisée". Le `docker-compose.yml` déploie les deux services au sein d'un réseau interne (`insat_olea_network`) isolé de l'extérieur.

---

## 🚀 Comment Lancer l'Application (En 1 Commande)

Grâce à Docker Compose, le déploiement sur votre machine est immédiat.

### Prérequis
- [Docker](https://www.docker.com/) et [Docker Compose](https://docs.docker.com/compose/)
- Clé API NVIDIA valide pour le LLaMA-70B

### Étapes d'installation

1. **Configurer l'environnement :**
   Créez un fichier `.env` à la racine de ce dossier avec votre clé API :
   ```bash
   NVIDIA_API_KEY="votre_cle_api_nvidia_ici"
   ```

2. **Lancer les conteneurs :**
   ```bash
   docker-compose up --build -d
   ```

3. **Accéder à l'application :**
   - **Interface Utilisateur (Streamlit) :** Rendez-vous sur [http://localhost:8501](http://localhost:8501)
   - **Documentation de l'API (Swagger) :** Rendez-vous sur [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛡️ Focus Sécurité (Zero-Trust)
Notre conception intègre des mécanismes dignes d'une architecture d'entreprise en production :
1. **Rate Limiting :** Bloque automatiquement les attaques (DDoS/Spam) basées sur l'identité (User ID masqué).
2. **PII Masking :** Les identifiants sensibles n'apparaissent jamais en clair dans l'immouvable `audit.log`.
3. **Internal Networking :** Le backend n'expose aucun port au public s'il est déployé derrière une gateway cloud, le frontend communique avec lui via le réseau Docker privé.

---

## 📦 Reproduction de la Phase I
Le projet assure une portabilité totale de la compétition de base. `solution.py` et `model.pkl` sont la fondation de l'application. Le `requirements.txt` de base est resté délibérément vide pour respecter les conditions strictes de l'arène EvalDA, tandis que les microservices Phase II provisionnent leurs propres dépendances via `requirements_api.txt` et `requirements_ui.txt`.