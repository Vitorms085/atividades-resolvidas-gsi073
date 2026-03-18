from dotenv import load_dotenv
from crewai_tools import TXTSearchTool
from crewai import Agent, Task, Crew, Process, LLM
from crewai.rag.embeddings.providers.onnx.types import ONNXProviderSpec

load_dotenv()

OLLAMA_LLM = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

EMBEDDING_MODEL_CONFIG: ONNXProviderSpec = {
    "provider": "onnx"
}

if __name__ == '__main__':
    ferramenta_rag = TXTSearchTool(
        txt='jesus-cristo.txt',
        config={
            "embedding_model": EMBEDDING_MODEL_CONFIG,
            "vectordb": {
                "provider": "chromadb",	
                "config": {
                    "dir": "./banco_vetorial",
                    "collection_name": "dados_jesus_txt_v2"
                }
            }
        }
    )

    pesquisador = Agent(
        role='Pesquisador de Documentação',
        goal='Extrair informações precisas e cruciais dos documentos fornecidos',
        backstory='Você é um analista meticuloso, especialista em buscar dados em bases de conhecimento complexas.',
        tools=[ferramenta_rag],
        verbose=True,
        allow_delegation=False,
        llm=OLLAMA_LLM
    )

    redator = Agent(
        role='Especialista em Comunicação Técnica',
        goal='Criar um resumo claro e direto baseado nas informações encontradas pelo pesquisador',
        backstory='Você é um redator técnico sênior capaz de transformar dados brutos em relatórios perfeitos.',
        tools=[],
        verbose=True,
        allow_delegation=False,
        llm=OLLAMA_LLM
    )

    tarefa_pesquisa = Task(
        description='Encontre todas as menções sobre Jesus Cristo no documento e extraia os principais pontos.',
        expected_output='Uma lista em bullet points com as informações brutas encontradas.',
        agent=pesquisador
    )

    tarefa_redacao = Task(
        description='Use a lista fornecida pelo pesquisador para escrever um parágrafo explicativo para um usuário leigo.',
        expected_output='Um texto contínuo, claro e sem jargões técnicos complexos.',
        agent=redator
    )

    equipe = Crew(
        agents=[pesquisador, redator],
        tasks=[tarefa_pesquisa, tarefa_redacao],
        process=Process.sequential,
    )

    resultado_final = equipe.kickoff()
    print(resultado_final)