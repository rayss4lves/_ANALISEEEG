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

def carregar_dados_edf(caminho_edf, indice_canal=3):
    """Carrega um canal específico do arquivo EDF."""
    with EdfReader(caminho_edf) as edf:
        fs = edf.getSampleFrequency(indice_canal)
        sinal = edf.readSignal(indice_canal)
    return sinal, fs


import numpy as np
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



def calcular_lag(tam_sinal, scmin=32, scmax=None, scres=30):
    """Gera o conjunto de escalas tratando strings como 'N/4' ou 'N/10'"""
    if scmax is None:
        scmax = tam_sinal // 10
    elif isinstance(scmax, str):
        if scmax == 'N/4':
            scmax = tam_sinal // 4
        elif scmax == 'N/8':
            scmax = tam_sinal // 8
        elif scmax == 'N/10':
            scmax = tam_sinal // 10
        else:
            scmax = tam_sinal // 10

    if scmax <= scmin:
        scmax = scmin * 4 
        
    exponents = np.linspace(np.log2(scmin), np.log2(scmax), scres)
    scales = np.round(2**exponents).astype(int)
    return np.unique(scales)

def calcular_q(q_min=-5.0, q_max=5.0, q_step=0.5):
    q = np.arange(q_min, q_max + q_step, q_step)
    q = q[q != 0]  # Remove q=0 para evitar problemas numéricos
    return q


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

def salvar_graficos_cenario(
        lag_out,
        dfa,
        h_q,
        tau_q,
        alpha,
        f_alpha,
        q_values,
        nome_cenario,
        diretorio_saida,
        nome_canal, nome_arquivo):

    ########################################
    # 1) Fluctuation Function
    ########################################

    q_plot = [-3, -1, 1, 3]

    # plt.figure(figsize=(7,5))

    for i, q in enumerate(q_values):

        # Plota somente alguns q
        if not np.any(np.isclose(q, q_plot)):
            continue

        Fq = dfa[:, i]

        mask = (
            np.isfinite(Fq) &
            (Fq > 0)
        )

        plt.loglog(
            lag_out[mask],
            Fq[mask],
            'o-',
            markersize=4,
            linewidth=1.5,
            label=f"q={q:g} | h={h_q[i]:.3f}"
        )

    plt.title("Fluctuation Function")
    plt.xlabel("Scale (s)")
    plt.ylabel(r"$F_q(s)$")
    plt.grid(True, which="both")
    plt.legend(title="q", fontsize=8)

    plt.savefig(
        os.path.join(
            diretorio_saida,
            f"{nome_cenario}_{nome_canal}_{nome_arquivo}_FluctuationFunction.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    ########################################
    # 2) Hurst Exponent
    ########################################

    plt.figure(figsize=(7,5))

    plt.plot(
        q_values,
        h_q,
        'o-',
        linewidth=2,
        color='blue'
    )

    plt.title("Generalized Hurst Exponent")
    plt.xlabel("q")
    plt.ylabel("h(q)")
    plt.grid(True)

    plt.savefig(
        os.path.join(
            diretorio_saida,
            f"{nome_cenario}_{nome_canal}_{nome_arquivo}_Hurst.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    ########################################
    # 3) Mass Exponent
    ########################################

    plt.figure(figsize=(7,5))

    plt.plot(
        q_values,
        tau_q,
        'o-',
        linewidth=2,
        color='orange'
    )

    plt.title("Mass Exponent")
    plt.xlabel("q")
    plt.ylabel(r"$\tau(q)$")
    plt.grid(True)

    plt.savefig(
        os.path.join(
            diretorio_saida,
            f"{nome_cenario}_{nome_canal}_{nome_arquivo}_MassExponent.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    ########################################
    # 4) Multifractal Spectrum
    ########################################

    plt.figure(figsize=(7,5))

    plt.plot(
        alpha,
        f_alpha,
        'o-',
        linewidth=2,
        color='red'
    )

    plt.title("Multifractal Spectrum")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$f(\alpha)$")
    plt.grid(True)
    
    # Calcular Delta Alpha
    delta_alpha = np.max(alpha) - np.min(alpha)

    # Escrever no gráfico
    plt.text(
        0.05,
        0.95,
        rf'$\Delta\alpha = {delta_alpha:.3f}$',
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        
    )

    plt.savefig(
        os.path.join(
            diretorio_saida,
            f"{nome_cenario}_{nome_canal}_{nome_arquivo}_Spectrum.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def pipeline_mfdfa_cenarios(sinal, lista_cenarios, diretorio_saida="resultados_mfdfa", nome_canal="canal", nome_arquivo="arquivo"):
    """Loop automatizado sobre múltiplos cenários de parametrização"""
    os.makedirs(diretorio_saida, exist_ok=True)
    resumo_geral = []
    detalhes_q = []
    tam_sinal = len(sinal)
    
    print(f"Iniciando {len(lista_cenarios)} cenários para N={tam_sinal}...\\n")
    
    for cenario in lista_cenarios:
        nome = cenario['nome']
        scmin = cenario.get('scmin', 32)
        scmax_param = cenario.get('scmax', None)
        scres = cenario.get('scres', 30)
        q_min = cenario.get('q_min', -5.0)
        q_max = cenario.get('q_max', 5.0)
        q_step = cenario.get('q_step', 0.5)
        order = cenario.get('order', 1)
        
        print(f"-> Processando: {nome}")
        
        # 1. Configurar parâmetros temporais de escala
        lags = calcular_lag(tam_sinal, scmin=scmin, scmax=scmax_param, scres=scres)
        q_values = calcular_q(q_min=q_min, q_max=q_max, q_step=q_step)
        
        # 2. Executar MFDFA

        sinal_real = bandpass_filter_fir(sinal, sfreq=200, lowcut=8.0, highcut=13.0)
        lag_out, dfa = MFDFA(sinal_real, lag=lags, q=q_values, order=order)
        
        # 3. Validar regressão linear interna por q (Cálculo do R²)
        # plt.figure(figsize=(9,6))

        resultados_validacao = []
        q_plot = [-3,-1,1,3]

        for i, q in enumerate(q_values):

            Fq = dfa[:, i]

            mask = (
                np.isfinite(Fq) &
                (Fq > 0)
            )

            lag = lag_out[mask]
            Fq = Fq[mask]

            h, b, r2 = regressao_loglog(lag, Fq)

            resultados_validacao.append({
                "q": q,
                "h": h,
                "intercept": b,
                "r2": r2
            })

            x = np.log(lag)

            y_fit = h*x + b

            # if np.any(np.isclose(q, q_plot)):

            #     x = np.log(lag)
            #     y_fit = h*x + b

            #     plt.loglog(
            #         lag,
            #         Fq,
            #         'o',
            #         ms=4
            #     )

            #     plt.loglog(
            #         lag,
            #         np.exp(y_fit),
            #         '-',
            #         lw=2,
            #         label=f"q={q:g}"
            #     )
        #imprimir todos os valores de R² para cada q
        # for res in resultados_validacao:
        #     print(f"  q={res['q']}: {res['r2']:.4f}")


        # --------------------------------------------------
        # h(q)
        # --------------------------------------------------

        h_q = np.array(
            [r["h"] for r in resultados_validacao]
        )

        # --------------------------------------------------
        # Espectro multifractal
        # --------------------------------------------------

        tau_q, alpha, f_alpha = calcular_multifractalidade(
            h_q,
            q_values
        )
        if nome == "cenario93_min128_mx10_res50_q5_ord1" and nome_canal == "T4":
            # 5. Exportar Gráficos do cenário atual
            salvar_graficos_cenario(lag_out, dfa, h_q, tau_q, alpha, f_alpha, q_values, nome, diretorio_saida, nome_canal=nome_canal, nome_arquivo=nome_arquivo)
        elif nome == "cenario77_min128_mx4_res50_q5_ord1" and nome_canal == "O2":
            # 5. Exportar Gráficos do cenário atual
            salvar_graficos_cenario(lag_out, dfa, h_q, tau_q, alpha, f_alpha, q_values, nome, diretorio_saida, nome_canal=nome_canal, nome_arquivo=nome_arquivo)
        # Compilar métricas resumidas para o CSV geral
        r2_medio = np.mean([r['r2'] for r in resultados_validacao]) if resultados_validacao else np.nan
        h_q_2 = h_q[np.argmin(np.abs(q_values - 2.0))] if any(np.isclose(q_values, 2.0)) else h_q[len(h_q)//2]
        
        resumo_geral.append({
            'Cenário': nome,
            'Canal': nome_canal,
            'scmin': scmin,
            'scmax_real': np.max(lag_out),
            'scres_real': len(lag_out),
            'Ordem': order,
            'R2_Medio': r2_medio,
            'Hurst_q2': h_q_2,
            'Largura_Espectro_DeltaAlpha': np.max(alpha) - np.min(alpha)
        })
        
        # Salvar as curvas completas ponto a ponto para cada q
        for idx, q_val in enumerate(q_values):
            detalhes_q.append({
                'Cenário': nome,
                'Canal': nome_canal,
                'q': q_val,
                'h(q)': h_q[idx],
                'tau(q)': tau_q[idx],
                'alpha': alpha[idx],
                'f(alpha)': f_alpha[idx],
                'R2': resultados_validacao[idx]['r2'] if idx < len(resultados_validacao) else np.nan
            })
            
    # Salvar resultados estruturados em Tabelas CSV separadas por ponto-e-vírgula
    df_resumo = pd.DataFrame(resumo_geral)
    df_detalhes = pd.DataFrame(detalhes_q)
    
    # Arredondar os valores numéricos para 3 casas decimais
    df_resumo = df_resumo.round(2)
    df_detalhes = df_detalhes.round(2)
    
    return df_resumo, df_detalhes

import itertools

def gerar_cenarios_automaticamente():
    # Definir os valores possíveis para interagir
    scmin_values = [16, 32, 64, 128]
    scmax_values = ["N/4","N/8","N/10"]
    scres_values = [30, 50]
    q_configs = [
        {"q_min": -5.0,  "q_max": 5.0,  "q_step": 0.5,  "q_tag": "q5"},
        {"q_min": -10.0,  "q_max": 10.0,  "q_step": 1,  "q_tag": "q10"},
    ]
    order_values = [1, 2]

    # Gerar todas as combinações cruzadas (interações completas) automaticamente
    lista_cenarios = []
    contador = 1

    for scmin, scmax, scres, q_conf, order in itertools.product(
        scmin_values, scmax_values, scres_values, q_configs, order_values
    ):
        # Criar uma tag limpa para o nome baseada nos parâmetros do cenário
        mx_tag = "mx4" if scmax == "N/4" else "mx8" if scmax == "N/8" else "mx10"
        nome_cenario = f"cenario{contador:02d}_min{scmin}_{mx_tag}_res{scres}_{q_conf['q_tag']}_ord{order}"
        
        # Adicionar o dicionário à lista combinatória
        lista_cenarios.append({
            "nome": nome_cenario,
            "scmin": scmin,
            "scmax": scmax,
            "scres": scres,
            "q_min": q_conf["q_min"],
            "q_max": q_conf["q_max"],
            "q_step": q_conf["q_step"],
            "order": order
        })
        contador += 1

    print(f"Total de cenários gerados automaticamente com sucesso: {len(lista_cenarios)}")

    return lista_cenarios

# --- EXEMPLO PRÁTICO DE USO ---
if __name__ == "__main__":
    
    caminho_edf = "test/meus_edf/sub-NORB00001_ses-1_task-EEG_eeg.edf"
    edf = EdfReader(caminho_edf)
    canais = edf.getSignalLabels()
    print("Canais disponíveis no EDF:", canais)
       
    todos_resumos = []
    todos_detalhes = []

    # 2. Sua lista dinâmica de cenários
    cenarios_para_testar = gerar_cenarios_automaticamente()
    
    canais_desejados = ["O2", "T4"]

    for canal in canais_desejados:
        # Verifica se o canal realmente existe na sua lista original 'canais'
        if canal in canais:
            print(f"Processando canal: {canal}")
            sinal_real = np.array(edf.readSignal(canais.index(canal)))
        
            # 3. Chamar a automação passando a pasta destino desejada
            df_res, df_det = pipeline_mfdfa_cenarios(sinal_real, cenarios_para_testar, diretorio_saida="meu_experimento_mfdfa", nome_canal=canal, nome_arquivo="N1")
            
            todos_resumos.append(df_res)
            todos_detalhes.append(df_det)
            
    df_resumo_final = pd.concat(todos_resumos, ignore_index=True)
    df_detalhes_final = pd.concat(todos_detalhes, ignore_index=True)
    

    df_resumo_final.to_csv(
        "meu_experimento_mfdfa/resultado_geral_cenarios_NORB00001_8_13.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    df_detalhes_final.to_csv(
        "meu_experimento_mfdfa/resultado_detalhado_por_q_NORB00001_8_13.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )