# _ANALISEEEG

Aplicação web construída em Flask para facilitar a visualização e análise de sinais de EEG (Eletroencefalograma) armazenados em arquivos EDF. O sistema permite realizar filtragem, gerar gráficos como espectrograma, densidade espectral de potência (DEP), FFT e análises baseadas em Wavelet Discreta, além de oferecer um visualizador interativo de trechos do exame.

## Funcionalidades

- Upload de múltiplos arquivos EDF com armazenamento seguro.
- Visualização empilhada dos canais de EEG com navegação por trechos.
- Métodos de filtragem padrão ou personalizada (frequências configuráveis).
- Análises no domínio da frequência (FFT, espectrograma, DEP).
- Análise Wavelet Discreta com gráficos de energia relativa e boxplots de coeficientes.
- Extração e exibição de metadados do exame (paciente, data, duração, canais etc.).

## Requisitos

- Python 3.9 ou superior.
- Dependências listadas, instaláveis via `pip`:
  - Flask
  - numpy
  - pandas
  - matplotlib
  - scipy
  - pyedflib
  - PyWavelets (`pywt`)
  - werkzeug

> Sugestão: utilizar um ambiente virtual (`venv`) para isolar as dependências.

## Instalação e Execução

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/TCCGITHUB.git
   cd TCCGITHUB/_ANALISEEEG
   ```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
   Caso não exista um arquivo `requirements.txt`, instale manualmente:
   ```bash
   pip install flask numpy pandas matplotlib scipy pyedflib pywavelets
   ```

4. Execute a aplicação Flask:
   ```bash
   flask --app app.py run
   ```
   ou
   ```bash
   python app.py
   ```

5. Acesse no navegador:
   ```
   http://127.0.0.1:5000/
   ```

## Uso

1. Faça upload dos arquivos `.edf` na página inicial.
2. Selecione o arquivo desejado para visualizar metadados e canais disponíveis.
3. Escolha os canais, defina o método de filtragem e o tipo de análise (FFT ou Wavelet).
4. Visualize os resultados em páginas dedicadas com gráficos gerados dinamicamente.
5. Utilize o visualizador interativo para percorrer trechos do exame.

## Estrutura do Projeto

- `app.py`: aplicação Flask, rotas e fluxo principal.
- `analise.py`: funções de processamento e geração de gráficos.
- `templates/`: páginas HTML renderizadas pelo Flask.
- `static/css/`: estilos da aplicação.
- `uploads/`: pasta criada automaticamente para armazenar os arquivos enviados.

## Autor

- Erik Vinicius Lustosa — contato: erik.silva@ufpi.edu.br

Sinta-se à vontade para sugerir melhorias ou relatar problemas. Contribuições são bem-vindas!