import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# ------------------------
# Definição dos universos e funções de pertinência
# ------------------------

# Orçamento: de R$ 4.545 a R$ 50.000
x_orcamento = np.arange(4545, 50001, 1)
orcamento_baixo = fuzz.trimf(x_orcamento, [4545, 4545, 27270])
orcamento_medio = fuzz.trimf(x_orcamento, [4545, 27270, 50000])
orcamento_alto = fuzz.trimf(x_orcamento, [27270, 50000, 50000])

# Tempo (Dias): de 4 a 44 dias
x_tempo = np.arange(4, 45, 1)
tempo_curto = fuzz.trimf(x_tempo, [4, 4, 24])
tempo_medio = fuzz.trimf(x_tempo, [4, 24, 44])
tempo_longo = fuzz.trimf(x_tempo, [24, 44, 44])

# Clima (Graus): de 40°C a 25°C. A escala é inversa (40°C = frio, 25°C = quente)
# A escala original de 0 a 10 para o clima foi remapeada para os graus
# Para manter a lógica original (0=frio, 10=quente), remapeamos os graus
x_clima = np.arange(25, 41, 1)
clima_frio = fuzz.trimf(x_clima, [25, 25, 30])
clima_agradavel = fuzz.trimf(x_clima, [25, 30, 40])
clima_quente = fuzz.trimf(x_clima, [30, 40, 40])

# Distância (Km): de 1.000 a 41.000 km
x_distancia = np.arange(1000, 41001, 1)
distancia_curta = fuzz.trimf(x_distancia, [1000, 1000, 21000])
distancia_media = fuzz.trimf(x_distancia, [1000, 21000, 41000])
distancia_longa = fuzz.trimf(x_distancia, [21000, 41000, 41000])

# Saída: recomendação de destino (índice representando tipos de destino)
x_destino = np.arange(0, 11, 1)
destino_praia = fuzz.trimf(x_destino, [0, 0, 5])
destino_montanha = fuzz.trimf(x_destino, [0, 5, 10])
destino_cidade = fuzz.trimf(x_destino, [5, 10, 10])

# ------------------------
# Entrada do usuário
# ------------------------

orcamento_input = 25000
tempo_input = 30
clima_input = 28
distancia_input = 25000

# ------------------------
# Fuzzificação
# ------------------------

orcamento_nivel_baixo = fuzz.interp_membership(x_orcamento, orcamento_baixo, orcamento_input)
orcamento_nivel_medio = fuzz.interp_membership(x_orcamento, orcamento_medio, orcamento_input)
orcamento_nivel_alto = fuzz.interp_membership(x_orcamento, orcamento_alto, orcamento_input)

tempo_nivel_curto = fuzz.interp_membership(x_tempo, tempo_curto, tempo_input)
tempo_nivel_medio = fuzz.interp_membership(x_tempo, tempo_medio, tempo_input)
tempo_nivel_longo = fuzz.interp_membership(x_tempo, tempo_longo, tempo_input)

clima_nivel_frio = fuzz.interp_membership(x_clima, clima_frio, clima_input)
clima_nivel_agradavel = fuzz.interp_membership(x_clima, clima_agradavel, clima_input)
clima_nivel_quente = fuzz.interp_membership(x_clima, clima_quente, clima_input)

distancia_nivel_curta = fuzz.interp_membership(x_distancia, distancia_curta, distancia_input)
distancia_nivel_media = fuzz.interp_membership(x_distancia, distancia_media, distancia_input)
distancia_nivel_longa = fuzz.interp_membership(x_distancia, distancia_longa, distancia_input)

# ------------------------
# Regras de inferência (adaptadas para incluir distância)
# ------------------------

# Regra 1: Se o orçamento é ALTO E o tempo é LONGO E a distância é LONGA, então o destino é CIDADE.
regra1 = np.fmin(np.fmin(orcamento_nivel_alto, tempo_nivel_longo), distancia_nivel_longa)
ativacao_cidade_1 = np.fmin(regra1, destino_cidade)

# Regra 2: Se o orçamento é MÉDIO E o tempo é MÉDIO E a distância é MÉDIA, então o destino é MONTANHA.
regra2 = np.fmin(np.fmin(orcamento_nivel_medio, tempo_nivel_medio), distancia_nivel_media)
ativacao_montanha_1 = np.fmin(regra2, destino_montanha)

# Regra 3: Se o orçamento é BAIXO E o tempo é CURTO E a distância é CURTA, então o destino é PRAIA.
regra3 = np.fmin(np.fmin(orcamento_nivel_baixo, tempo_nivel_curto), distancia_nivel_curta)
ativacao_praia_1 = np.fmin(regra3, destino_praia)

# Regra 4: Se o clima é QUENTE, então o destino é PRAIA.
ativacao_praia_2 = np.fmin(clima_nivel_quente, destino_praia)

# Regra 5: Se o clima é FRIO, então o destino é MONTANHA.
ativacao_montanha_2 = np.fmin(clima_nivel_frio, destino_montanha)

# ------------------------
# Agregação das regras de saída
# ------------------------

agregado_praia = np.fmax(ativacao_praia_1, ativacao_praia_2)
agregado_montanha = np.fmax(ativacao_montanha_1, ativacao_montanha_2)
agregado_cidade = ativacao_cidade_1

agregado = np.fmax(agregado_praia, np.fmax(agregado_montanha, agregado_cidade))

# ------------------------
# Defuzzificação (Centroide)
# ------------------------

destino_final = fuzz.defuzz(x_destino, agregado, 'centroid')
print(f"Destino recomendado (índice): {destino_final:.2f}")

# ------------------------
# Mapeamento do índice para texto
# ------------------------
if destino_final < 3.5:
    recomendacao_texto = "Praia"
elif destino_final < 7.5:
    recomendacao_texto = "Montanha"
else:
    recomendacao_texto = "Cidade"

print(f"A recomendação textual para sua viagem é: {recomendacao_texto}")


# ------------------------
# Gráficos
# ------------------------
fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(8, 15))

# Orçamento
ax0.plot(x_orcamento, orcamento_baixo, 'b', linewidth=1.5, label='Baixo')
ax0.plot(x_orcamento, orcamento_medio, 'g', linewidth=1.5, label='Médio')
ax0.plot(x_orcamento, orcamento_alto, 'r', linewidth=1.5, label='Alto')
ax0.set_title('Orçamento')
ax0.legend()

# Tempo
ax1.plot(x_tempo, tempo_curto, 'b', linewidth=1.5, label='Curto')
ax1.plot(x_tempo, tempo_medio, 'g', linewidth=1.5, label='Médio')
ax1.plot(x_tempo, tempo_longo, 'r', linewidth=1.5, label='Longo')
ax1.set_title('Tempo disponível')
ax1.legend()

# Clima
ax2.plot(x_clima, clima_frio, 'b', linewidth=1.5, label='Frio')
ax2.plot(x_clima, clima_agradavel, 'g', linewidth=1.5, label='Agradável')
ax2.plot(x_clima, clima_quente, 'r', linewidth=1.5, label='Quente')
ax2.set_title('Clima (Graus Celsius)')
ax2.legend()

# Distância
ax3.plot(x_distancia, distancia_curta, 'b', linewidth=1.5, label='Curta')
ax3.plot(x_distancia, distancia_media, 'g', linewidth=1.5, label='Média')
ax3.plot(x_distancia, distancia_longa, 'r', linewidth=1.5, label='Longa')
ax3.set_title('Distância (Km)')
ax3.legend()

# Destino
ax4.plot(x_destino, destino_praia, 'b', linewidth=1.5, label='Praia')
ax4.plot(x_destino, destino_montanha, 'g', linewidth=1.5, label='Montanha')
ax4.plot(x_destino, destino_cidade, 'r', linewidth=1.5, label='Cidade')
ax4.set_title('Recomendação de Destino')
ax4.legend()

for ax in (ax0, ax1, ax2, ax3, ax4):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()
