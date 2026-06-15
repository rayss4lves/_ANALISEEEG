import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pyedflib
from scipy.signal import welch, butter, filtfilt
import io
import base64
import pywt
import pandas as pd
from MFDFA import MFDFA


plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

def carregar_dados_edf(caminho_arquivo):
    """Carrega sinais de um arquivo EDF, retornando dados, rótulos e taxa de amostragem.

    Args:
        caminho_arquivo (str): Caminho do arquivo EDF a ser lido.

    Returns:
        tuple[numpy.ndarray, list[str], float]: Matriz de sinais (`n_canais x n_amostras`),
            lista com os rótulos dos canais e a frequência de amostragem dos sinais.
    """
    f = pyedflib.EdfReader(caminho_arquivo)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    signals = np.zeros((n, f.getNSamples()[0]))
    sfreq = f.getSampleFrequency(0)
    for i in np.arange(n):
        signals[i, :] = f.readSignal(i)
    f._close()
    del f
    return signals, signal_labels, sfreq

def remover_ruido(dados, sfreq, lowcut=0.5, highcut=40.0):
    """Aplica filtro Butterworth passa-banda para remover ruídos indesejados.

    Args:
        dados (numpy.ndarray): Matriz de sinais em que cada linha representa um canal.
        sfreq (float): Frequência de amostragem dos sinais.
        lowcut (float, opcional): Frequência de corte inferior, em Hz. Padrão 0.5.
        highcut (float, opcional): Frequência de corte superior, em Hz. Padrão 40.

    Returns:
        numpy.ndarray: Sinais filtrados com a mesma dimensão da entrada.
    """
    b, a = butter(4, [lowcut, highcut], btype='band', fs=sfreq)
    dados_filtrados = filtfilt(b, a, dados, axis=1)
    return dados_filtrados

def aplicar_fft(dados, sfreq):
    """Calcula a FFT de cada canal e devolve frequências positivas e magnitudes.

    Args:
        dados (numpy.ndarray): Matriz de sinais com formato `n_canais x n_amostras`.
        sfreq (float): Frequência de amostragem utilizada na aquisição.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Frequências positivas da FFT e matriz de
            magnitudes correspondentes para cada canal.
    """
    n = len(dados[0])
    freqs = np.fft.rfftfreq(n, d=1/sfreq)
    fft_result = np.fft.rfft(dados, axis=1)
    return freqs, np.abs(fft_result)

def gerar_espectrograma_em_base64(dados, sfreq, labels):
    """Cria espectrogramas para cada canal e codifica as figuras em strings base64.

    Args:
        dados (numpy.ndarray): Matriz de sinais por canal.
        sfreq (float): Frequência de amostragem dos dados.
        labels (list[str]): Nomes dos canais correspondentes aos sinais.

    Returns:
        list[str]: Lista contendo imagens PNG em base64 dos espectrogramas.
    """
    imagens = []
    for i in range(dados.shape[0]):
        plt.figure()
        plt.specgram(dados[i], Fs=sfreq, NFFT=256, noverlap=128, cmap='viridis')
        plt.title(f'Espectrograma - Canal {labels[i]}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Frequência (Hz)')
        plt.colorbar(label='Intensidade (dB)')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=95)
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        imagens.append(img_str)
    return imagens

def gerar_psd_em_base64(dados, sfreq, labels):
    """Calcula a densidade espectral de potência e retorna gráficos codificados em base64.

    Args:
        dados (numpy.ndarray): Matriz de sinais dos canais.
        sfreq (float): Frequência de amostragem dos sinais.
        labels (list[str]): Rótulos dos canais utilizados nos gráficos.

    Returns:
        list[str]: Imagens PNG em base64 representando as curvas da DEP de cada canal.
    """
    imagens = []
    for i in range(dados.shape[0]):
        f, Pxx = welch(dados[i], sfreq, nperseg=1024)
        plt.figure()
        plt.semilogy(f, Pxx)
        plt.title(f'DEP - Canal {labels[i]}')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Potência Espectral (dB/Hz)')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=95)
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        imagens.append(img_str)
    return imagens

def gerar_fft_em_base64(freqs, fft_result, labels):
    """Gera gráficos da magnitude da FFT por canal e codifica em base64.

    Args:
        freqs (numpy.ndarray): Frequências positivas calculadas pela FFT.
        fft_result (numpy.ndarray): Magnitudes da FFT para cada canal.
        labels (list[str]): Nomes dos canais a serem exibidos nos títulos.

    Returns:
        list[str]: Lista com imagens PNG em base64 dos gráficos de FFT.
    """
    imagens = []
    for i in range(fft_result.shape[0]):
        plt.figure()
        plt.plot(freqs, fft_result[i])
        plt.title(f'FFT - Canal {labels[i]}')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Magnitude')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=95)
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        imagens.append(img_str)
    return imagens

def aplicar_dwt(dados, wavelet='db4', level=5):
    """Aplica a Transformada Wavelet Discreta a cada canal dos sinais.

    Args:
        dados (numpy.ndarray): Matriz de sinais (`n_canais x n_amostras`).
        wavelet (str, opcional): Nome da wavelet utilizada. Padrão `db4`.
        level (int, opcional): Número máximo de níveis de decomposição. Padrão 5.

    Returns:
        list[list[numpy.ndarray]]: Lista com os coeficientes da DWT para cada canal.
    """
    coeffs = []
    for canal in dados:
        coeff = pywt.wavedec(canal, wavelet=wavelet, level=level)
        coeffs.append(coeff)
    return coeffs

def mapear_bandas_dwt_para_eeg(n_niveis, sfreq):
    """Relaciona níveis da DWT às bandas clássicas de frequência do EEG.

    Args:
        n_niveis (int): Número de níveis de decomposição wavelet disponíveis.
        sfreq (float): Frequência de amostragem original dos sinais.

    Returns:
        list[str]: Rótulos contendo o nome da banda e o intervalo de frequência estimado.
    """
    labels = []
    faixa_max_delta = sfreq / (2**(n_niveis + 1))
    labels.append(f'Delta (0.0-{faixa_max_delta:.1f} Hz)')
    for j in range(n_niveis, 0, -1):
        faixa_min = sfreq / (2**(j + 1))
        faixa_max = sfreq / (2**j)
        freq_central = (faixa_min + faixa_max) / 2
        banda = ''
        if freq_central >= 60:
            banda = 'Gamma Alta'
        elif 30 <= freq_central < 60:
            banda = 'Gamma Baixa'
        elif 14 <= freq_central < 30:
            banda = 'Beta'
        elif 8 <= freq_central < 14:
            banda = 'Alpha'
        elif 4 <= freq_central < 8:
            banda = 'Theta'
        else:
            banda = 'Delta'
        labels.append(f'{banda} ({faixa_min:.1f}-{faixa_max:.1f} Hz)')
    return labels

def extrair_nome_banda(label):
    """Retorna apenas o nome de uma banda a partir de um rótulo completo.

    Args:
        label (str): Texto no formato `Nome (freq_min-freq_max Hz)`.

    Returns:
        str: Nome da banda sem a parte correspondente às frequências.
    """
    try:
        return label[:label.index('(')].strip()
    except ValueError:
        return label

def gerar_dwt_analise(sinais, coeffs_list, labels, sfreq, wavelet='db4'):
    """Gera gráficos da análise wavelet para cada canal e devolve-os em base64.

    Produz gráficos de energia relativa por banda, boxplots dos coeficientes de detalhe
    e decomposição temporal dos coeficientes wavelet.

    Args:
        sinais (numpy.ndarray): Matriz com os sinais no domínio do tempo.
        coeffs_list (list[list[numpy.ndarray]]): Coeficientes da DWT para cada canal.
        labels (list[str]): Lista com os nomes dos canais.
        sfreq (float): Frequência de amostragem dos sinais.
        wavelet (str, opcional): Wavelet utilizada na decomposição. Padrão `db4`.

    Returns:
        list[dict[str, str]]: Lista de dicionários contendo o rótulo do canal e as
            imagens em base64 dos gráficos gerados.
    """
    resultados = []
    for i, sinal in enumerate(sinais):
        coeffs = coeffs_list[i]
        label = labels[i]
        
        n_niveis = len(coeffs) - 1
        bandas_labels = mapear_bandas_dwt_para_eeg(n_niveis, sfreq)

        #energia_relativa
        energia_niveis = [np.sum(c**2) for c in coeffs]
        energia_total = np.sum(energia_niveis)
        energia_relativa = (energia_niveis / energia_total) * 100
        
        cores_bandas = {
            'Delta': '#3498db', 'Theta': '#9b59b6', 'Alpha': '#2ecc71',
            'Beta': '#f1c40f', 'Gamma Baixa': '#e74c3c', 'Gamma Alta': '#d35400'
        }
        

        nomes_simplificados = []
        for l in bandas_labels:
            try:
                nome = l[:l.index('(')].strip()
                nomes_simplificados.append(nome)
            except ValueError:
                nomes_simplificados.append(l)

        cores_grafico = [cores_bandas.get(nome, 'grey') for nome in nomes_simplificados]


        fig_energia, ax_energia = plt.subplots(figsize=(10, 6))
        ax_energia.bar(range(len(energia_relativa)), energia_relativa, color=cores_grafico)
        
        ax_energia.set_title(f'Energia Relativa por Banda para o Canal {label}', fontsize=14)
        ax_energia.set_ylabel('Energia Relativa (%)', fontsize=12)
        ax_energia.set_xlabel('Bandas de Frequência', fontsize=12)
        ax_energia.grid(axis='y', linestyle=':', alpha=0.7)
        ax_energia.set_xticks([])

    
        legend_info = {}
        for full_label in bandas_labels:
            try:
                band_name = full_label[:full_label.index('(')].strip()
                if band_name not in legend_info:
                    legend_info[band_name] = band_name
            except ValueError:
                if full_label not in legend_info:
                    legend_info[full_label] = full_label

        patches = [plt.Rectangle((0, 0), 1, 1, color=cores_bandas.get(name, 'grey')) for name in legend_info]
        labels_legend = list(legend_info.keys())
        ax_energia.legend(patches, labels_legend, loc='upper right')

        grafico_energia_b64 = fig_to_base64(fig_energia)
        plt.close(fig_energia)


     
        coeficientes_detalhe = coeffs[1:]
        labels_detalhe = bandas_labels[1:]
        labels_detalhe_simplificados = [extrair_nome_banda(l) for l in labels_detalhe]
        
        fig_boxplot, ax_boxplot = plt.subplots(figsize=(12, 7))
        ax_boxplot.boxplot(coeficientes_detalhe, labels=labels_detalhe_simplificados, vert=False, patch_artist=True)
        ax_boxplot.set_title(f'Distribuição dos Coeficientes de Detalhe para o Canal {label}', fontsize=14)
        ax_boxplot.set_xlabel('Amplitude dos Coeficientes', fontsize=12)
        ax_boxplot.grid(axis='x', linestyle=':', alpha=0.7)
        
        grafico_boxplot_b64 = fig_to_base64(fig_boxplot)
        plt.close(fig_boxplot)

   
        tempo_total_sinal = len(sinal) / sfreq
        fig_decomposicao, axes_decomposicao = plt.subplots(len(coeffs), 1, figsize=(18, 2.0 * len(coeffs)))
        
        for j, coef in enumerate(coeffs):
            tempo_coef = np.linspace(0, tempo_total_sinal, num=len(coef))
            axes_decomposicao[j].plot(tempo_coef, coef, color='teal', linewidth=0.8, rasterized=True)
            axes_decomposicao[j].set_title(extrair_nome_banda(bandas_labels[j]), fontsize=12)
            axes_decomposicao[j].set_ylabel('Amplitude', fontsize=10)
            axes_decomposicao[j].grid(True, linestyle=':', alpha=0.6)
            axes_decomposicao[j].set_xlim(0, tempo_total_sinal)

        axes_decomposicao[-1].set_xlabel('Tempo (s)', fontsize=12)
        fig_decomposicao.suptitle(f'Decomposição Wavelet do Canal {label}', fontsize=16, y=1.0)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        
        grafico_decomposicao_b64 = fig_to_base64(fig_decomposicao)
        plt.close(fig_decomposicao)

        resultados.append({
            'label': label,
            'grafico_energia': grafico_energia_b64,
            'grafico_boxplot': grafico_boxplot_b64,
            'grafico_decomposicao': grafico_decomposicao_b64
        })

    return resultados

def fig_to_base64(fig):
    """Converte uma figura Matplotlib em uma string base64 no formato PNG.

    Args:
        fig (matplotlib.figure.Figure): Figura a ser serializada.

    Returns:
        str: Representação em base64 da imagem resultante.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=95)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')




################################ DFA ################################

# def calcular_lag(tam_sinal, n_lags=100, order=2):
#     lag_min = order + 1
#     lag_max = tam_sinal // 4
#     lag = np.logspace(np.log10(lag_min), np.log10(lag_max), n_lags).astype(int)
#     return np.unique(lag)

# def calcular_lag_otimizado(tam_sinal, n_lags=40, order=2):
#     # Começa em 10^0.7 (~5) e termina em N/4 em escala logarítmica
#     lag_max = tam_sinal // 10
#     lag = np.logspace(1.0, np.log10(lag_max), n_lags).astype(int)
#     return np.unique(lag)


def calcular_lag(tam_sinal, scmin=32, scmax=None, scres=51):
    """
    Gera o conjunto de escalas (tamanhos de segmento) para o MF-DFA.
    Baseado no Código Matlab 15 do documento de Ihlen (2012).

    Args:
        tam_sinal (int): O comprimento total da série temporal (N).
        scmin (int): Tamanho mínimo do segmento. Recomendado >= 10 ou 16.
        scmax (int): Tamanho máximo do segmento. Recomendado <= N/10.
        scres (int): Número total de escalas (resolução).

    Returns:
        np.array: Array de escalas com espaçamento logarítmico igual.
    """
    if scmax is None:
        # Recomendação do documento: scmax abaixo de 1/10 do tamanho da amostra
        scmax = tam_sinal // 10
    
    # Garantir que scmax seja razoável
    if scmax <= scmin:
        scmax = scmin * 4 # Valor arbitrário seguro se o sinal for muito curto
        
    # Espaçamento igual entre log2(escala) conforme Código Matlab 15
    exponents = np.linspace(np.log2(scmin), np.log2(scmax), scres)
    scales = np.round(2**exponents).astype(int)
    
    # Garantir escalas únicas (np.round pode repetir valores em escalas pequenas)
    return np.unique(scales)

def calcular_q(q_min=-5.0, q_max=5.0, q_step=0.5):
    q = np.arange(q_min, q_max + q_step, q_step)  # q de q_min a q_max
    q = q[q != 0]  # remover q=0 para evitar log(0) em Fq
    return q

# funcao para processar os sinais 
def processar_sinais(sinal, fs, order=2):

    #cortar o sinal 
    # sinal_cortado = cortar_sinal(sinal, fs)
    tam_sinal = len(sinal)
    print(tam_sinal)
    #calcular lag
    lag = calcular_lag(tam_sinal)
    print(f"Lags: {lag}")

    #calcular q
    q = calcular_q()
    print(f"Valores de q: {q}")
    
    return lag, q

# funcao para aplicar o MFdfa
def aplicar_mfdfa(sinal, fs, lags, q_values, order):
    if sinal.ndim > 1:
        sinal = sinal[0, :]
    order = 1
    # lag, q = processar_sinais(sinal, fs, order)
    #aplicar MFdfa
    lag_out, dfa = MFDFA(sinal, lag=lags, q=q_values, order=order)
    
    #exibir todos os resultados do dfa, o retorno tem a forma de uma matriz, onde cada linha corresponde a um valor de lag e cada coluna corresponde a um valor de q
    print("Resultados do MFDFA:")
    for i, lag_val in enumerate(lag_out):
        print(f"Lag: {lag_val} | Fq: {dfa[i]}")
    # print(f"1 linha de resultado dfa:\n{dfa[0]}")
        
    return {
        "sinal_filtrado": sinal,
        "lag_out": lag_out,
        "dfa": dfa,
        "lag": lags,
        "q": q_values
    }

def gerar_mfdfa_grafico_duplo_log(resultado):
    plt.loglog(resultado["lag_out"], resultado["dfa"], 'o', label='fOU: MFDFA q=2')
    metade = len(resultado["lag_out"])
    np.polyfit(np.log(resultado["lag_out"][:metade]), np.log(resultado["dfa"][:metade]), 1)[0]

    # plt.figure()
    # plt.xlabel('Lag')
    # plt.ylabel('DFA')
    

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=95)
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')

import numpy as np
from sklearn.metrics import r2_score

import scipy.stats as stats 
from scipy.stats import linregress

def analisar_e_validar_mfdfa(lags, Fq_matrix, q_values, use_log2=False):
    # Escolha da base do log
    log_func = np.log2 if use_log2 else np.log
    
    log_lags = log_func(lags)
    resultados = []
    avisos = []
    
    for i, q in enumerate(q_values):
        Fq = Fq_matrix[:, i]
        
        # Filtro de dados igual ao da função de cálculo (Kantelhardt)
        mask = np.isfinite(Fq) & (Fq > 0)
        
        if np.sum(mask) < 3: # min_points
            avisos.append(f"Atenção: Dados insuficientes para q={q:.1f}.")
            continue
            
        x = log_lags[mask]
        y = log_func(Fq[mask])
        
        # Usando linregress para ser idêntico ao cálculo principal
        slope, intercept, r_value, _, _ = linregress(x, y)
        r2 = r_value**2
        
        resultados.append({'q': q, 'h': slope, 'r2': r2})
        
        # Critério de validação
        if round(r2, 3) < 0.95:
            avisos.append(f"Atenção: O ajuste para q={q:.1f} apresentou baixa linearidade (R²={r2:.3f}).")
            
    return resultados, avisos

def calcular_multifractalidade(resultados, escalas, q_values):
    h_q = []
    log_s = np.log(escalas)
    
    # Passo 2: Regressão Linear para encontrar h(q)
    # resultados tem shape (len(escalas), len(q_values))
    for i in range(len(q_values)):
        # Pega a coluna i (todos os valores de escala para o q_i)
        log_Fq = np.log(resultados[:, i])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_s, log_Fq)
        h_q.append(slope)
    
    h_q = np.array(h_q)
    
    # Passo 3: Calcular Tau(q)
    tau_q = q_values * h_q - 1
    
    # Passo 4: Calcular Alpha e f(alpha) via gradiente (Legendre)
    alpha = np.gradient(tau_q, q_values)
    f_alpha = q_values * alpha - tau_q
    
    return h_q, tau_q, alpha, f_alpha

def gerar_graficos_multifractais(h_q, tau_q, alpha, f_alpha, q_values):
    # Cria a figura com dois subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico 1: Espectro de Singularidade
    ax1.plot(alpha, f_alpha, 'o-', color='red', markersize=4)
    ax1.set_title('Espectro de Singularidade $f(\\alpha)$')
    ax1.set_xlabel('$\\alpha$')
    ax1.set_ylabel('$f(\\alpha)$')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Gráfico 2: Expoente de Hurst Generalizado
    ax2.plot(q_values, h_q, 's-', color='blue', markersize=4)
    ax2.set_title('Expoente de Hurst $h(q)$')
    ax2.set_xlabel('$q$')
    ax2.set_ylabel('$h(q)$')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Salva em buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')