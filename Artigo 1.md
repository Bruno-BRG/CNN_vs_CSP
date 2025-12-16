# Paper 1 — EEGNet vs CSP-LDA na Classificação de Imagética Motora (MI) em EEG (BCI IV-2a)



## 1. Objetivo e Pergunta de Pesquisa

**Objetivo:** comparar o desempenho de uma CNN leve (EEGNet) com um método clássico (CSP + LDA) na classificação de **imagética motora** de **mão esquerda vs mão direita** em EEG.



**Pergunta:** a **EEGNet** consegue **igualar ou superar** o baseline **CSP (8 componentes) + LDA**, reduzindo dependência de *feature engineering*?



**Hipótese:** a EEGNet, por aprender filtros espaço-temporais diretamente do EEG, pode obter desempenho comparável/superior ao CSP-LDA, especialmente em sujeitos com padrões mais complexos e não-lineares.



---



## 2. Contribuições do Artigo

- Benchmark controlado entre **CSP-LDA** (baseline clássico) e **EEGNet** (DL leve) em MI.

- Pipeline reprodutível com pré-processamento fixo (**8–30 Hz**, janela **0–4 s** pós-cue).

- Reporte por sujeito usando **Accuracy**, **Kappa** e **Matriz de Confusão**.

- Discussão sobre trade-off entre métodos clássicos (interpretação/simplicidade) e DL leve (flexibilidade/menos engenharia).



---



## 3. Base Teórica Essencial (curta)

- MI é associada a padrões de **ERD/ERS** principalmente em:
  - **Mu** (≈ 8–12 Hz)
  - **Beta** (≈ 18–25 Hz)

- Por isso, adota-se bandpass **8–30 Hz** como faixa padrão de análise para MI.

- **CSP** busca projeções espaciais que maximizam variância discriminativa entre classes; **LDA** faz a separação linear.

- **EEGNet** explora filtros temporais e espaciais (convoluções separáveis), aprendendo representações discriminativas com baixo custo computacional.



---



## 4. Dataset e Tarefa

**Dataset:** BCI Competition IV-2a (EEG multicanal).  

**Tarefa:** classificação binária **mão esquerda vs mão direita**.  

**Observação:** definir claramente se a avaliação é:

- **Subject-dependent** (treino/teste por sujeito)  

ou  

- **Cross-subject** (generalização entre sujeitos)

> **Nota para versão final:** explicitar sessões/partições e garantir que nenhum dado do teste influencia treino, seleção de hiperparâmetros ou ajuste do CSP.



---



## 5. Pré-processamento (padrão do Paper 1)

- **Filtro passa-faixa:** Butterworth 3ª ordem, **8–30 Hz**

- **Epoch/janela:** **0–4 s** após o *cue* (início do estímulo)

- (Opcional, se você usar) **Notch 50/60 Hz** e padronização (z-score) *somente usando o treino*.



---



## 6. Pipeline A — CSP + LDA (Baseline)

1. Aplicar pré-processamento (8–30 Hz, 0–4 s)

2. Extrair **CSP com 8 componentes**

3. Extrair features (ex.: log-variância por componente)

4. Treinar **LDA**

5. Avaliar no conjunto de teste

**Boas práticas:**

- Ajustar CSP **somente no treino**.

- Se usar CV, CSP precisa ser recalculado dentro de cada fold.



---



## 7. Pipeline B — EEGNet (CNN leve)

1. Aplicar pré-processamento (8–30 Hz, 0–4 s)

2. Treinar EEGNet com entradas (canais × tempo)

3. Avaliar no teste

**Relatar no artigo:**

- principais hiperparâmetros (ex.: taxa de aprendizado, épocas, batch)

- número de parâmetros do modelo (para justificar "leve")

- critérios anti-overfitting (early stopping, dropout, etc.)



---



## 8. Protocolo de Avaliação (definir e fixar)

### Split recomendado (exemplos)

- **Treina em uma sessão, testa na outra** (por sujeito)  

ou  

- **k-fold CV** (por sujeito), com controle estrito de vazamento



### Métricas (mínimo)

- **Accuracy**

- **Cohen's Kappa**

- **Matriz de Confusão**

- (Opcional) AUC, F1, Balanced Accuracy



### Reprodutibilidade

- Rodar **3 seeds** e reportar média ± desvio.

- Registrar versão de libs e hardware (GPU/CPU).



---



## 9. Resultados (estrutura sugerida)

### Tabelas

- **Tabela 1:** configuração experimental (filtro, janela, classes, split)

- **Tabela 2:** resultados por sujeito (Accuracy e Kappa) — CSP-LDA vs EEGNet

- **Tabela 3 (opcional):** média geral, desvio, e teste estatístico pareado

### Figuras

- **Figura 1:** diagrama do pipeline (CSP-LDA vs EEGNet)

- **Figura 2:** arquitetura resumida do EEGNet

- **Figura 3:** boxplot/violin de performance por método

- **Figura 4 (opcional):** matrizes de confusão agregadas

### Estatística

- Comparação pareada entre métodos (por sujeito):
  - teste **Wilcoxon** (robusto) ou **t pareado** (se normalidade fizer sentido)



---



## 10. Discussão (o que responder)

- Em quais sujeitos o EEGNet melhora/piora vs CSP-LDA?

- O ganho (se houver) vem de capturar temporalidade/não-linearidade?

- O CSP-LDA ainda é competitivo por simplicidade/interpretabilidade?

- Limitações:
  - banda fixa 8–30 (gancho direto para Paper 2)
  - tarefa binária L vs R (gancho para Paper 3: MI vs ME)



---



## 11. Conclusão (template)

- Resumir: EEGNet (leve) **[igualou/superou/não superou]** CSP-LDA na MI L vs R no BCI IV-2a sob pré-processamento 8–30 Hz e janela 0–4 s.

- Implicação: DL leve pode reduzir *feature engineering* sem perder desempenho (dependendo da generalização e do split adotado).

- Próximos passos: ablação de bandas (Paper 2) e MI vs ME (Paper 3).



---



## 12. To-do imediato (para fechar a versão 1)

- [ ] Definir protocolo de split (sessão→sessão ou k-fold) e escrever isso com precisão

- [ ] Fixar seeds e rodar 3 repetições por método

- [ ] Gerar tabelas por sujeito (acc/kappa)

- [ ] Gerar boxplots + matrizes de confusão

- [ ] Rodar teste estatístico pareado

- [ ] Escrever Introdução e Metodologia com base nesse pipeline



