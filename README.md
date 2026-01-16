# 🫀 Projeto de Treinamento e Análise de Dados Cardio

## 📖 Descrição
Este projeto realiza a análise exploratória e o treinamento de modelos de machine learning para prever risco de doenças cardiovasculares.  
O script inclui etapas de limpeza de dados, visualização, preparação de treino/teste e avaliação de diversos algoritmos.  
Além disso, foi criada uma **API em Python** para disponibilizar o modelo treinado como serviço web.

## 📂 Estrutura do Projeto
- `treinamento-cardio/treinamento-cardio.ipynb` → Notebook principal com todo o pipeline de análise e modelagem.
- `treinamento-cardio/base/cardio_train.csv` → Dataset utilizado (formato CSV, separador `;`).
- `api/pipe_logistic_regression.pkl` → Pipeline do modelo de regressão logística salvo para uso em produção.
- `api/cardio_check.py` → Aplicação FastAPI que sobe o webserver e expõe a API.
- `README.md` → Documentação do projeto.
- `treinamento-cardio.pdf` → Código fonte e gráficos mostrados em PDF.

## 📦 Dependências
O projeto utiliza as seguintes bibliotecas para análise e modelagem:
- **pandas**, **numpy** → Manipulação e análise de dados.
- **matplotlib**, **seaborn** → Visualização gráfica.
- **scikit-learn** → Modelos de machine learning (KNN, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, Logistic Regression).
- **shap** → Interpretação de modelos (SHAP values).
- **joblib** → Serialização do modelo.

Para a API, são necessárias também:
- **uvicorn** → Servidor ASGI para rodar a aplicação.
- **fastapi** → Framework para criação da API.
- **pydantic** → Validação de dados e definição de schemas.

Para subir o webserver da API, execute:

uvicorn cardio_check:app --reload --host 0.0.0.0 --port 8000