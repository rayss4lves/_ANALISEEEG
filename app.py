"""Aplicação Flask para upload e análise de sinais EEG em arquivos EDF."""

import json
import numpy as np
import pyedflib
from flask import Flask, flash, jsonify, render_template, request, redirect, url_for, session, Response
import os
import pandas as pd
from werkzeug.utils import secure_filename
from analise import remover_ruido, aplicar_fft, gerar_espectrograma_em_base64, gerar_psd_em_base64, gerar_fft_em_base64, aplicar_dwt, gerar_dwt_analise

app = Flask(__name__)


UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'analise-eeg-chave-muito-secreta' 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)




def get_eeg_data_for_viewer(filepath, start_second=0, duration=10):
    """Lê um trecho do arquivo EDF e retorna dados para visualização empilhada.

    Args:
        filepath (str): Caminho completo do arquivo EDF.
        start_second (float, opcional): Segundo inicial do trecho desejado. Padrão 0.
        duration (float, opcional): Duração do segmento em segundos. Padrão 10.

    Returns:
        tuple[list[dict], list[dict], float | None]: Lista com traçados para Plotly,
            anotações com rótulos de canais e duração total do arquivo em segundos.
            Retorna tupla de `None` quando não é possível ler o arquivo.
    """
    try:
        with pyedflib.EdfReader(filepath) as f:
            n_channels = f.signals_in_file
            sfreq = f.getSampleFrequency(0)
            channel_labels = f.getSignalLabels()
            file_duration = f.getFileDuration()

            if start_second >= file_duration:
                return None, None, None

            start_sample = int(start_second * sfreq)
            samples_to_read = int(duration * sfreq)

            if start_sample + samples_to_read > f.getNSamples()[0]:
                samples_to_read = f.getNSamples()[0] - start_sample
            
            actual_duration = samples_to_read / sfreq
            time_vector = np.linspace(start_second, start_second + actual_duration, samples_to_read)

            max_amplitudes = []
            for i in range(n_channels):
                num_samples_for_spacing = min(int(sfreq * 5), f.getNSamples()[i])
                s_temp = f.readSignal(i, 0, num_samples_for_spacing)
                if len(s_temp) > 0:
                    max_amplitudes.append(np.max(np.abs(s_temp)))
            
            spacing = np.percentile(max_amplitudes, 85) * 2.5 if max_amplitudes else 100
            if spacing == 0: spacing = 100

            traces = []
            annotations = []
            for i in range(n_channels):
                signal = f.readSignal(i, start_sample, samples_to_read)
                offset = (n_channels - 1 - i) * spacing
                
                traces.append({
                    'x': time_vector.tolist(),
                    'y': (signal + offset).tolist(),
                    'type': 'scattergl',
                    'mode': 'lines',
                    'name': channel_labels[i],
                    'line': {'color': 'black', 'width': 1}
                })
                
                annotations.append({
                    'x': start_second, 'y': offset,
                    'xref': 'x', 'yref': 'y',
                    'text': channel_labels[i],
                    'showarrow': False,
                    'xanchor': 'right', 'yanchor': 'middle',
                    'font': {'size': 10}
                })
                
        return traces, annotations, file_duration
    except Exception as e:
        print(f"Erro em get_eeg_data_for_viewer: {e}")
        return None, None, None




@app.route('/')
def index():
    """Página inicial da aplicação, limpa a sessão e exibe a landing page."""
    session.clear()
    return render_template('index.html')

@app.route('/info')
def info_eeg():
    """Renderiza a página com informações educativas sobre EEG."""
    return render_template('info_eeg.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Recebe uploads de arquivos EDF, valida e guarda os nomes na sessão."""
    if 'files' not in request.files:
        flash('Nenhum campo de arquivo encontrado!', 'danger')
        return redirect(request.url)

    uploaded_files = request.files.getlist('files')
    if not uploaded_files or uploaded_files[0].filename == '':
        flash('Nenhum arquivo selecionado!', 'danger')
        return redirect(request.url)
    
    filenames = []
    for file in uploaded_files:
        if file and file.filename.endswith('.edf'):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    
    if not filenames:
        flash('Nenhum arquivo .edf válido foi enviado.', 'danger')
        return redirect(request.url)

    session['uploaded_files'] = filenames
    return redirect(url_for('analise', filename=filenames[0]))


@app.route('/analise/<filename>')
def analise(filename):
    """Renderiza a página principal de análise para um arquivo EDF selecionado."""
    uploaded_files = session.get('uploaded_files')
    if not uploaded_files or filename not in uploaded_files:
        flash('Sessão expirada ou arquivo inválido. Por favor, envie os arquivos novamente.', 'danger')
        return redirect(url_for('index'))

    session['current_file'] = filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    traces, annotations, file_duration = get_eeg_data_for_viewer(filepath, 0, 10)
    
    if traces is None:
        flash(f'Não foi possível ler os dados do arquivo {filename}. Pode estar corrompido.', 'danger')
        return redirect(url_for('index'))

    # --- LEITURA DOS METADADOS COM TRATAMENTO PARA CAMPOS VAZIOS ---
    with pyedflib.EdfReader(filepath) as f:
        patient_name = f.getPatientName() or 'Não Informado'
        data_nasci = f.getBirthdate() or 'Não Informado'
        sexo = {'Male': 'Masculino', 'Female': 'Feminino'}.get(f.getSex(), 'Não Informado')
        
        try:
            record_date, record_time = f.getStartdatetime().strftime("%d/%m/%Y %H:%M").split()
        except Exception:
            record_date = 'Não Informado'
            record_time = 'Não Informado'
        
        exam_duration = round(f.getFileDuration() / 60, 2)
        
        channels_info_table = [
            {
                "name": f.getLabel(i),
                "frequency": f"{f.getSampleFrequency(i)} Hz",
                "physical_min": f.getPhysicalMinimum(i),
                "physical_max": f.getPhysicalMaximum(i),
                "digital_min": f.getDigitalMinimum(i),
                "digital_max": f.getDigitalMaximum(i)
            } for i in range(f.signals_in_file)
        ]
        channels_for_form = [{"name": f.getLabel(i)} for i in range(f.signals_in_file)]

    return render_template('analise.html',
                           uploaded_files=uploaded_files,
                           current_file=filename,
                           channels=channels_for_form,
                           patient_name=patient_name,
                           sexo=sexo,
                           data_nasci=data_nasci,
                           record_date=record_date,
                           record_time=record_time,
                           exam_duration=exam_duration,
                           channels_info_table=channels_info_table,
                           initial_traces=json.dumps(traces),
                           initial_annotations=json.dumps(annotations),
                           file_duration=file_duration)


@app.route('/criar_analise', methods=['POST'])
def criar_analise():
    """Processa parâmetros do formulário, aplica análises e renderiza resultados."""
    nome_arquivo = session.get('current_file')
    if not nome_arquivo:
        flash('Sessão expirada. Por favor, inicie o processo novamente.', 'danger')
        return redirect(url_for('index'))

    filter_type = request.form.get('filter_type')
    metodo = request.form.get('method')
    canais_selecionados = request.form.getlist('channels')

    if not canais_selecionados:
        flash('Nenhum canal foi selecionado para análise!', 'warning')
        return redirect(url_for('analise', filename=nome_arquivo))

    caminho_arquivo = os.path.join(app.config['UPLOAD_FOLDER'], nome_arquivo)
    
    from analise import carregar_dados_edf
    dados, labels, sfreq = carregar_dados_edf(caminho_arquivo)
    
    indices_canais = [labels.index(canal) for canal in canais_selecionados if canal in labels]
    dados_filtrados = dados[indices_canais]
    labels_filtrados = [labels[i] for i in indices_canais]

    if filter_type == 'standard':
        dados_filtrados = remover_ruido(dados_filtrados, sfreq)
    elif filter_type == 'custom':
        try:
            lowcut = float(request.form.get('lowcut'))
            highcut = float(request.form.get('highcut'))
            if lowcut >= highcut or lowcut < 0 or highcut >= sfreq / 2:
                raise ValueError("Valores de frequência inválidos.")
            dados_filtrados = remover_ruido(dados_filtrados, sfreq, lowcut=lowcut, highcut=highcut)
        except (ValueError, TypeError):
            flash('Valores de filtro personalizado inválidos.', 'danger')
            return redirect(url_for('analise', filename=nome_arquivo))
            
    if metodo == 'fft':
        freqs, fft_result = aplicar_fft(dados_filtrados, sfreq)
        espectrogramas = gerar_espectrograma_em_base64(dados_filtrados, sfreq, labels_filtrados)
        psd_graficos = gerar_psd_em_base64(dados_filtrados, sfreq, labels_filtrados)
        fft_graficos = gerar_fft_em_base64(freqs, fft_result, labels_filtrados)
        return render_template('resultados.html', espectrogramas=espectrogramas, psd_graficos=psd_graficos, fft_graficos=fft_graficos)
    elif metodo == 'twc':
        sinais_para_analise = dados_filtrados.copy()
        coeffs = aplicar_dwt(sinais_para_analise)
        dwt_resultados = gerar_dwt_analise(sinais_para_analise, coeffs, labels_filtrados, sfreq)
        return render_template('resultados_wavelet.html', dwt_resultados=dwt_resultados)
    return redirect(url_for('analise', filename=nome_arquivo))


@app.route('/get_eeg_chunk')
def get_eeg_chunk():
    """Retorna via JSON um segmento das séries temporais para o visualizador."""
    filename = request.args.get('filename')
    start_second = request.args.get('start', 0, type=float)
    
    if not filename:
        return jsonify(error="Nome do arquivo não fornecido"), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify(error="Arquivo não encontrado"), 404

    traces, annotations, _ = get_eeg_data_for_viewer(filepath, start_second, 10)
    
    if traces is None:
        return jsonify(error="Fim do arquivo"), 404
        
    return jsonify(traces=traces, annotations=annotations)


if __name__ == '__main__':
    app.run(debug=True)
