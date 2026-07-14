import os
import io
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from pyedflib import EdfReader
import matplotlib.pyplot as plt
from MFDFA import MFDFA
from scipy.signal import firwin, filtfilt, hilbert

def bandpass_filter_fir(dados, sfreq, lowcut=0.5, highcut=40.0, numtaps=301):
    """
    Filtro passa-banda FIR linear-phase + envoltória de amplitude (Hilbert).
    """
    # Cria coeficientes FIR com janela de Hamming
    b = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=sfreq)
    
    # Aplica filtro de fase zero
    dados_filtrados = filtfilt(b, [1.0], dados, axis=-1)
    
    # Extrai envoltória de amplitude
    analytic_signal = hilbert(dados_filtrados, axis=-1)
    amplitude_envelope = np.abs(analytic_signal)
    
    return amplitude_envelope


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
    scmax = tam_sinal // scmax  # Garantir que scmax não ultrapasse N/10
        
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
    # print("Resultados do MFDFA:")
    # for i, lag_val in enumerate(lag_out):
    #     print(f"Lag: {lag_val} | Fq: {dfa[i]}" )
    # print(f"1 linha de resultado dfa:\n{dfa[0]}")
        
    return {
        "sinal_filtrado": sinal,
        "lag_out": lag_out,
        "dfa": dfa,
        "lag": lags,
        "q": q_values
    }


def regressao_loglog(lag, Fq):
    """
    Calcula h(q), intercepto e R² para um único valor de q.
    """


    x = np.log(lag)
    y = np.log(Fq)

    # regressão linear
    coef = np.polyfit(x, y, 1)

    slope = coef[0]
    intercept = coef[1]

    # valores previstos pela reta
    y_fit = np.polyval(coef, x)

    # Soma dos quadrados dos resíduos
    ss_res = np.sum((y - y_fit)**2)

    # Soma total dos quadrados
    ss_tot = np.sum((y - np.mean(y))**2)

    # coeficiente de determinação
    r2 = 1 - ss_res/ss_tot

    return slope, intercept, r2
         

def calcular_multifractalidade(h_q, q_values):
    """
    Calcula tau(q), alpha e f(alpha)
    usando o vetor h(q) já calculado.
    """

    h_q = np.asarray(h_q, dtype=float)

    tau_q = q_values * h_q - 1

    alpha = np.gradient(tau_q, q_values)

    f_alpha = q_values * alpha - tau_q

    return tau_q, alpha, f_alpha

def validar_regressoes(lag_out, dfa, q_values):

    resultados = []
    avisos = []

    for i, q in enumerate(q_values):

        Fq = dfa[:, i]

        mask = (
            np.isfinite(Fq) &
            (Fq > 0)
        )

        if np.sum(mask) < 3:
            avisos.append(
                f"q={q:g}: dados insuficientes para regressão."
            )
            continue

        h, b, r2 = regressao_loglog(
            lag_out[mask],
            Fq[mask]
        )

        resultados.append({
            "q": q,
            "h": h,
            "intercept": b,
            "r2": r2
        })

        if r2 < 0.95:
            avisos.append(
                f"q={q:g}: baixa linearidade (R²={r2:.3f})"
            )

    return resultados, avisos


def carregar_dados_edf(caminho_edf, indice_canal=3):
    """Carrega um canal específico do arquivo EDF."""
    with EdfReader(caminho_edf) as edf:
        fs = edf.getSampleFrequency(indice_canal)
        sinal = edf.readSignal(indice_canal)
    return sinal, fs

import time

if __name__ == "__main__":
    
    arquivos = [
        ("NORB00001", "test/meus_edf/sub-NORB00001_ses-1_task-EEG_eeg.edf"),
        ("NORB000017", "test/meus_edf/sub-NORB00017_ses-1_task-EEG_eeg.edf"),
    ]
    NUM_EXECUCOES = 30

    resultados_tempos = {}
    for nome_arquivo, caminho_edf in arquivos:
        edf = EdfReader(caminho_edf)
        canais = edf.getSignalLabels()
        print("Canais disponíveis no EDF:", canais)
        
        tempos = []

        canal = "O2"
        sinal_real = np.array(edf.readSignal(canais.index(canal)))
        sfreq = edf.getSampleFrequency(canais.index(canal))
        
        for i in range(10):
            print(f"Execução {i+1}")
            
            # Verifica se o canal realmente existe na sua lista original 'canais'
            dados_filtrados = bandpass_filter_fir(sinal_real, sfreq, lowcut=0.5, highcut=40.0, numtaps=301)
            # Gerar vetores conforme o usuário pediu
            lags = calcular_lag(len(dados_filtrados), scmin=128, scmax=10, scres=30)
            q_values = calcular_q(q_min=-5, q_max=5, q_step=0.5)

            # Executa o cálculo (sua função customizada deve aceitar esses args)
            # Assumindo que sua função retorna a matriz Fq e o gráfico
            # Executa o cálculo
            resultado = aplicar_mfdfa(dados_filtrados, sfreq, lags, q_values, 1)
        
            # Realiza a validação
            resultados_calculados, avisos = validar_regressoes(resultado["lag_out"], resultado["dfa"], resultado["q"])
            
            # 1. Calcular h(q), alpha, f(alpha)
            h_q = np.array([r["h"] for r in resultados_calculados])
            tau_q, alpha, f_alpha = calcular_multifractalidade(h_q, resultado['q'])
            delta_alpha = np.max(alpha) - np.min(alpha)



        for i in range(NUM_EXECUCOES):
            print(f"Execução {i+1}")
            inicio = time.perf_counter()
            # Verifica se o canal realmente existe na sua lista original 'canais'
            dados_filtrados = bandpass_filter_fir(sinal_real, sfreq, lowcut=0.5, highcut=40.0, numtaps=301)
            # Gerar vetores conforme o usuário pediu
            lags = calcular_lag(len(dados_filtrados), scmin=128, scmax=10, scres=30)
            q_values = calcular_q(q_min=-5, q_max=5, q_step=0.5)

            # Executa o cálculo (sua função customizada deve aceitar esses args)
            # Assumindo que sua função retorna a matriz Fq e o gráfico
            # Executa o cálculo
            resultado = aplicar_mfdfa(dados_filtrados, sfreq, lags, q_values, 1)
        
            # Realiza a validação
            resultados_calculados, avisos = validar_regressoes(resultado["lag_out"], resultado["dfa"], resultado["q"])
            
            # 1. Calcular h(q), alpha, f(alpha)
            h_q = np.array([r["h"] for r in resultados_calculados])
            tau_q, alpha, f_alpha = calcular_multifractalidade(h_q, resultado['q'])
            delta_alpha = np.max(alpha) - np.min(alpha)

            fim = time.perf_counter()
            tempo_execucao = fim - inicio
            
            tempos.append(tempo_execucao)
            
        edf.close()
        resultados_tempos[nome_arquivo] = np.array(tempos)
    
    os.makedirs("meu_experimento_mfdfa", exist_ok=True)
    
    print("\n===== Estatísticas =====")
    for nome, tempos in resultados_tempos.items():
        print(f"\nArquivo: {nome}")
        print(f"Média: {np.mean(tempos):.4f} s")
        print(f"Desvio padrão: {np.std(tempos):.4f} s")
        print(f"Mínimo: {np.min(tempos):.4f} s")
        print(f"Máximo: {np.max(tempos):.4f} s")
    
    
    df_tempos = pd.DataFrame({
        "Execucao": np.arange(1, NUM_EXECUCOES + 1),
        "NORB00001": resultados_tempos["NORB00001"],
        "NORB000017": resultados_tempos["NORB000017"],
    })

    df_tempos.to_csv(
        "meu_experimento_mfdfa/comparacao_tempos.csv",
        index=False
    )
    
    df_estatisticas = pd.DataFrame({
        "Arquivo": list(resultados_tempos.keys()),
        "Media (s)": [np.mean(t) for t in resultados_tempos.values()],
        "Desvio Padrao (s)": [np.std(t) for t in resultados_tempos.values()],
        "Minimo (s)": [np.min(t) for t in resultados_tempos.values()],
        "Maximo (s)": [np.max(t) for t in resultados_tempos.values()]
    })

    df_estatisticas.to_csv(
        "meu_experimento_mfdfa/estatisticas_tempos.csv",
        index=False
    )

    plt.figure(figsize=(10,5))

    for nome, tempos in resultados_tempos.items():

        plt.plot(
            range(1, NUM_EXECUCOES+1),
            tempos,
            marker="o",
            linewidth=2,
            label=f"{nome}"
        )

        plt.axhline(
            np.mean(tempos),
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label=f"Média {nome}"
        )

    plt.title("Tempo de execução do MFDFA")
    plt.xlabel("Execução")
    plt.ylabel("Tempo (s)")
    plt.xticks(range(1, NUM_EXECUCOES+1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "meu_experimento_mfdfa/comparacao_tempos.png",
        dpi=300
    )

    plt.show()