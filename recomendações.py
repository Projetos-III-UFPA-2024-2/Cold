import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import create_engine
from flask import Flask, jsonify, request

# Inicializando o aplicativo Flask
app = Flask(__name__)

# Função para carregar dados do banco de dados
def carregar_dados_banco(engine):
    # Consultando as tabelas startup e investidor
    query_startup = "SELECT id_empresa, votos FROM startup_metadado"
    query_invest = "SELECT id_investidor, id_empresa, avaliacao FROM dados"

    # Lendo os dados do banco de dados e convertendo em DataFrame
    startup = pd.read_sql(query_startup, engine)
    invest = pd.read_sql(query_invest, engine)

    return startup, invest

# Conexão com o banco de dados (ajuste a URL conforme suas credenciais)
engine = create_engine('mysql+pymysql://admin:Arcanjo2024@projetoarcanjobd.cfuwmca4ushn.us-east-1.rds.amazonaws.com:3306/projetoarcanjobd')

# Carregando os dados uma única vez quando o servidor Flask iniciar
startup, invest = carregar_dados_banco(engine)

# Processando os dados para criar a tabela pivô
startup = startup[['id_empresa', 'votos']]
invest = invest[['id_investidor', 'id_empresa', 'avaliacao']]
startup.dropna(inplace=True)
invest.dropna(inplace=True)
invest_e_startup = invest.merge(startup, on='id_empresa')
startup_pivot = invest_e_startup.pivot_table(columns='id_empresa', index='id_investidor', values='avaliacao')
startup_pivot.fillna(0, inplace=True)
startup_sparse = csr_matrix(startup_pivot)

# Criando o modelo de recomendação KNN
modelo = NearestNeighbors(algorithm='brute', metric="euclidean")
modelo.fit(startup_sparse)

# Função de recomendação de startups para o investidor
def recomendar_startups(investidor_id, n_recomendacoes=3):
    # Verifica se o investidor existe no dataset
    if investidor_id not in startup_pivot.index:
        return {"error": f"Investidor {investidor_id} não encontrado no dataset."}

    # Pegando o perfil do investidor (as avaliações que ele fez)
    investidor_perfil = startup_pivot.loc[investidor_id].values.reshape(1, -1)
    print(investidor_perfil)

    # Verificar quantas startups foram avaliadas pelo investidor
    avaliadas = np.sum(investidor_perfil != 0)

    #if avaliadas == 0:
        # Retornar as startups mais populares
        #recomendadas = list(startup.sort_values(by='votos', ascending=False)['id_empresa'][:n_recomendacoes])
        #return {"investidor_id": investidor_id, "startups_recomendadas": recomendadas}

    # Ajustar o número de recomendações para não exceder o número de startups avaliadas
    n_recomendacoes = min(n_recomendacoes, avaliadas)
    print(n_recomendacoes)

    try:
        distances, sugestions = modelo.kneighbors(investidor_perfil, n_neighbors=n_recomendacoes)
        startups_recomendadas = startup_pivot.columns[sugestions[0]].tolist()
        return {"investidor_id": investidor_id, "startups_recomendadas": startups_recomendadas}
    except IndexError as e:
        #return {"error": f"Erro ao recomendar para o investidor {investidor_id}: {str(e)}"}
        recomendadas = list(startup.sort_values(by='votos', ascending=False)['id_empresa'][:n_recomendacoes])
        return {"investidor_id": investidor_id, "startups_recomendadas": recomendadas}

# Rota para obter recomendações de startups para um investidor
@app.route("/recomendacao/<int:investidor_id>", methods=["GET"])
def recomendacao(investidor_id):
    # Obtendo o número de recomendações da query string (padrão = 3)
    n_recomendacoes = request.args.get("n", default=3, type=int)

    # Chamando a função de recomendação
    recomendacao_resultado = recomendar_startups(investidor_id, n_recomendacoes)

    # Retornando as recomendações em formato JSON
    return jsonify(recomendacao_resultado)

# Executando o servidor Flask
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
