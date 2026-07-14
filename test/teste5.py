import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def resolver_caminho_csv(caminho, diretorio_base=None):
    """Resolve um caminho de CSV relativo ao workspace e retorna o primeiro arquivo existente."""
    if diretorio_base is None:
        diretorio_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if os.path.isabs(caminho):
        return caminho if os.path.exists(caminho) else None

    candidatos = []
    candidatos.append(os.path.join(diretorio_base, caminho))
    candidatos.append(os.path.join(diretorio_base, "meu_experimento_mfdfa", os.path.basename(caminho)))
    candidatos.append(os.path.join(diretorio_base, "meu_experimento_mfdfa", caminho))

    for candidato in candidatos:
        if os.path.exists(candidato):
            return candidato

    return None


def comparar_resultados_csv(
        csv1,
        csv2,
        diretorio_saida,
        nome1="Arquivo 1",
        nome2="Arquivo 2",
        cenario=None,
        canal=None):

    os.makedirs(diretorio_saida, exist_ok=True)

    csv1_resolvido = resolver_caminho_csv(csv1)
    csv2_resolvido = resolver_caminho_csv(csv2)

    if csv1_resolvido is None or csv2_resolvido is None:
        raise FileNotFoundError(
            f"Arquivos CSV não encontrados. Verifique os caminhos: {csv1} e {csv2}"
        )

    # ------------------------------------------------------------------
    # Leitura dos arquivos
    # ------------------------------------------------------------------
    df1 = pd.read_csv(csv1_resolvido, sep=";")
    df2 = pd.read_csv(csv2_resolvido, sep=";")

    # ------------------------------------------------------------------
    # Filtrar cenário
    # ------------------------------------------------------------------
    if cenario is not None:
        df1 = df1[df1["Cenário"] == cenario]
        df2 = df2[df2["Cenário"] == cenario]

    # ------------------------------------------------------------------
    # Filtrar canal
    # ------------------------------------------------------------------
    if canal is not None:
        df1 = df1[df1["Canal"] == canal]
        df2 = df2[df2["Canal"] == canal]

    if df1.empty or df2.empty:
        print("Nenhum dado encontrado para esse cenário/canal.")
        return

    ##########################################################
    # HURST
    ##########################################################

    plt.figure(figsize=(7,5))

    plt.plot(
        df1["q"],
        df1["h(q)"],
        'o-',
        linewidth=2,
        label=nome1
    )

    plt.plot(
        df2["q"],
        df2["h(q)"],
        's-',
        linewidth=2,
        label=nome2
    )

    plt.title("Generalized Hurst Exponent")
    plt.xlabel("q")
    plt.ylabel("h(q)")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(diretorio_saida, f"Comparacao_Hurst_{cenario}_{canal}_0.5_40.0.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    ##########################################################
    # MASS EXPONENT
    ##########################################################

    plt.figure(figsize=(7,5))

    plt.plot(
        df1["q"],
        df1["tau(q)"],
        'o-',
        linewidth=2,
        label=nome1
    )

    plt.plot(
        df2["q"],
        df2["tau(q)"],
        's-',
        linewidth=2,
        label=nome2
    )

    plt.title("Mass Exponent")
    plt.xlabel("q")
    plt.ylabel(r"$\tau(q)$")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(diretorio_saida, f"Comparacao_MassExponent_{cenario}_{canal}_0.5_40.0.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    ##########################################################
    # ESPECTRO MULTIFRACTAL
    ##########################################################

    plt.figure(figsize=(7,5))

    plt.plot(
        df1["alpha"],
        df1["f(alpha)"],
        'o-',
        linewidth=2,
        label=nome1
    )

    plt.plot(
        df2["alpha"],
        df2["f(alpha)"],
        's-',
        linewidth=2,
        label=nome2
    )

    delta_alpha1 = df1["alpha"].max() - df1["alpha"].min()
    delta_alpha2 = df2["alpha"].max() - df2["alpha"].min()

    texto = (
        f"{nome1}: Δα = {delta_alpha1:.3f}\n"
        f"{nome2}: Δα = {delta_alpha2:.3f}"
    )

    plt.text(
        0.03,
        0.97,
        texto,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.title("Multifractal Spectrum")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$f(\alpha)$")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(diretorio_saida, f"Comparacao_Spectrum_{cenario}_{canal}_0.5_40.0.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Gráficos gerados com sucesso!")
    
if __name__ == "__main__":
    
    csv1 = os.path.join("meu_experimento_mfdfa_0.5_40.0", "resultado_detalhado_por_q_NORB00001_0.5_40.0.csv")
    csv2 = os.path.join("meu_experimento_mfdfa_0.5_40.0", "resultado_detalhado_por_q_NORB00006_0.5_40.0.csv")

    comparar_resultados_csv(
        csv1=csv1,
        csv2=csv2,
        diretorio_saida="Graficos_Comparacao",

        nome1="NORB00001",
        nome2="NORB00006",

        cenario="cenario73_min128_mx4_res30_q5_ord1",
        canal="O2"
    )