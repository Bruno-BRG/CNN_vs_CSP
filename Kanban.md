---
kanban-plugin: basic
---
# Roadmap visual (bem tranquilo) — Paper 1: EEGNet vs CSP-LDA (MI L/R)

> Agora com **duas trilhas** rodando em paralelo:
> - **Trilha A (Experimentos)**: produzir evidências (resultados, figuras, tabelas)
> - **Trilha B (Escrita)**: transformar em artigo pronto (IMRaD + revisão)

---

## Visão geral (fluxo)
✅ Dados → ✅ Pré-processo → ✅ Split sem vazamento  
→ ✅ Baseline CSP-LDA → ✅ EEGNet  
→ ✅ Avaliação + estatística → ✅ Figuras/Tabelas  
→ ✅ Escrita IMRaD → ✅ Revisão final → ✅ Submissão/Entrega

---

# Trilha A — Experimentos (evidência)

## Fase 0 — Setup (para não virar bagunça)
**Objetivo:** projeto organizado e reprodutível.
- [x] Estrutura de pastas: `data/raw`, `data/processed`, `src`, `configs`, `results`, `figures`, `tables`
- [x] `configs/paper1.yaml` (banda, janela, split, seeds, hparams)
- [x] padrão de logs e salvamento (CSV/JSON por sujeito + seeds)

✅ Entregável: pastas + config + template de resultados.

---

## Fase 1 — Dados (carregar e confiar)
- [x] Loader por sujeito/sessão
- [x] checar labels (L/R), eventos/cue, fs, shape
- [x] sanity checks (contagem de trials, plots cru, NaN/inf)

✅ Entregável: loader validado + prints/plots de confirmação.

---

## Fase 2 — Pré-processamento
- [x] bandpass 8–30 Hz (Butterworth 3ª ordem)
- [x] epoch 0.7s pós-cue
- [x] normalização (se usar) com **fit no treino**
- [x] PSD antes/depois + checar alinhamento do cue

✅ Entregável: `preprocess()` que devolve `(X, y)` pronto.

---

## Fase 3 — Split (P0 do paper)
- [x] Escolher split: sessão→sessão **OU** k-fold estrito
- [x] escrever regra anti-vazamento (vai pro texto do paper)
- [x] `get_splits(subject)` pronto

✅ Entregável: split fixado + texto do split.

---

## Fase 4 — Baseline CSP + LDA
- [x] CSP (8 comps) fit só no treino
- [x] features (log-variance)
- [x] LDA treino/teste
- [x] salvar por sujeito: acc, kappa, confusão, y_true/y_pred

✅ Entregável: baseline 1 sujeito → depois todos.

---

## Fase 5 — EEGNet
- [x] padronizar shape de entrada
- [x] treinar EEGNet com hparams fixos
- [x] salvar outputs iguais ao baseline
- [x] rodar 3 seeds (mínimo)

✅ Entregável: EEGNet 1 sujeito → depois todos (+ seeds).

---

## Fase 6 — Comparação + estatística
- [x] tabela por sujeito (acc/kappa CSP vs EEGNet)
- [x] média±dp + variação por seed
- [x] teste pareado (Wilcoxon ou t pareado)

✅ Entregável: `results_summary.csv` + p-valor + agregados.

---

## Fase 7 — Figuras e tabelas finais
**Figuras mínimas:**
- [x] pipeline CSP-LDA vs EEGNet
- [x] arquitetura EEGNet (blocos)
- [x] boxplot/violin acc e/ou kappa
- [x] matriz de confusão agregada

**Tabelas mínimas:**
- [x] setup experimental (banda, janela, split, n)
- [x] resultados por sujeito (acc/kappa)
- [x] média±dp + estatística

✅ Entregável: `figures/` + `tables/` prontas pra colar no artigo.

---

# Trilha B — Escrita do artigo (IMRaD + qualidade)

## Fase W0 — Preparar o “esqueleto” do paper (antes de escrever de verdade)
**Objetivo:** criar um arquivo base e só ir preenchendo.
- [x] Criar `paper_draft.md` (ou LaTeX) com:
  - [x] Título provisório
  - [x] Resumo + Palavras-chave (placeholders)
  - [x] Introdução
  - [x] Metodologia
  - [x] Resultados
  - [x] Discussão
  - [x] Conclusão
  - [x] Referências
  - [x] Apêndice (opcional: configs, hparams)

✅ Entregável: arquivo de artigo com seções prontas.

---

## Fase W1 — Escrever primeiro o que é “mecânico” (Metodologia)
> Isso é o mais fácil porque você só descreve o que implementou.

### 1) Dataset e tarefa
- [x] descrever dataset (BCI IV-2a) e a tarefa (L vs R)
- [x] descrever número de sujeitos e canais usados (o que você de fato rodou)

### 2) Pré-processamento
- [x] banda 8–30 Hz, ordem do filtro, notch (se houver)
- [x] janela 0–4 s pós-cue
- [x] normalização (se houver), com regra: fit no treino

### 3) Protocolo de avaliação (parte mais importante)
- [x] descrever split com precisão (sessão→sessão ou CV)
- [x] reforçar anti-vazamento (CSP e normalização só no treino)
- [x] descrever seeds e repetições

### 4) Modelos comparados
- [x] CSP (8 comps) + extração de features + LDA
- [ ] EEGNet (arquitetura resumida + hparams principais)

### 5) Métricas e estatística
- [x] accuracy, kappa, confusão
- [x] teste pareado (Wilcoxon/t)
- [x] como agregou entre sujeitos

✅ Entregável: seção Metodologia completa e reprodutível.

---

## Fase W2 — Resultados (texto curto + figuras/tabelas)
> Regra: aqui você NÃO explica “por quê”, só mostra “o quê”.

- [x] inserir Tabela de resultados por sujeito (acc/kappa)
- [x] inserir média±dp + teste estatístico
- [x] inserir figuras (boxplot + confusão)
- [x] descrever em 5–10 linhas:
  - [x] quem ganhou (em média)
  - [x] quantos sujeitos melhoraram/pioraram
  - [x] estabilidade por seed (se reportar)

✅ Entregável: seção Resultados “publicável”.

---

## Fase W3 — Discussão (onde você vira autor de verdade)
> Aqui você interpreta sem inventar: conecta com ERD, CSP, CNN, limitações.

### Checklist do que discutir
- [ ] Comparação qualitativa:
  - [ ] EEGNet capturou melhor temporalidade? (possível)
  - [ ] CSP-LDA foi competitivo por simplicidade/linearidade?
- [ ] Variabilidade entre sujeitos:
  - [ ] sujeitos “bons” e “ruins” em MI (comentar sem exagerar)
- [ ] Implicações práticas:
  - [ ] EEGNet leve como alternativa para pipelines com menos engenharia
- [ ] Limitações (obrigatório)
  - [ ] banda fixa 8–30
  - [ ] tarefa binária L vs R
  - [ ] protocolo (subject-dependent vs cross-subject)
- [ ] Gancho para próximos trabalhos (sem prometer demais)
  - [ ] Paper 2: ablação de bandas Mu/Beta
  - [ ] Paper 3: MI vs ME (intenção vs ação)

✅ Entregável: Discussão sólida + limitações explícitas.

---

## Fase W4 — Introdução (depois de ter resultados!)
> A intro fica MUITO mais fácil quando você já sabe o que encontrou.

**Estrutura simples (4 parágrafos):**
1. [ ] Contexto: MI em BCI e relevância (reabilitação/controle)
2. [ ] Problema: MI tem baixa SNR e exige engenharia (CSP etc.)
3. [ ] Solução/hipótese: DL leve (EEGNet) pode aprender filtros úteis
4. [ ] Objetivo e contribuições: comparar EEGNet vs CSP-LDA + protocolo reprodutível

✅ Entregável: Introdução com objetivo claro + contribuições listadas.

---

## Fase W5 — Resumo/Abstract e Título (no final)
- [ ] título final (curto e específico)
- [ ] resumo estruturado em 5 partes:
  - [ ] contexto (1 frase)
  - [ ] objetivo (1 frase)
  - [ ] método (2 frases: dataset + split + modelos + métricas)
  - [ ] resultados (1–2 frases com números)
  - [ ] conclusão/implicação (1 frase)
- [ ] 3–6 palavras-chave (ex.: EEG, motor imagery, EEGNet, CSP, LDA, BCI)

✅ Entregável: abstract pronto para submissão.

---

## Fase W6 — Referências e “acabamento” (o que dá nota)
- [ ] padronizar estilo (ABNT/APA) e conferir consistência
- [ ] checar que toda afirmação de background tem citação
- [ ] remover jargão desnecessário e frases vagas
- [ ] conferir todas as figuras:
  - [ ] legenda autoexplicativa
  - [ ] eixos e unidades
  - [ ] resolução (300 dpi se for PDF)
- [ ] conferir tabelas:
  - [ ] formatação consistente
  - [ ] n (número de sujeitos) explícito
- [ ] checar “reprodutibilidade mínima”:
  - [ ] hiperparâmetros no texto ou apêndice
  - [ ] seeds
  - [ ] versão das libs (pelo menos no apêndice)

✅ Entregável: versão final revisada e “limpa”.

---

# Ordem mais eficiente (pra você não travar)
1. [ ] Fixar Split (Fase 3)
2. [ ] Rodar baseline completo (Fase 4)
3. [ ] Rodar EEGNet completo (Fase 5)
4. [ ] Gerar summary + estatística (Fase 6)
5. [ ] Figuras/tabelas finais (Fase 7)
6. [ ] Escrever Metodologia (W1)
7. [ ] Escrever Resultados (W2)
8. [ ] Escrever Discussão (W3)
9. [ ] Escrever Introdução (W4)
10. [ ] Abstract + Título (W5)
11. [ ] Referências + acabamento (W6)

---

# “Definition of Done” (acabou quando…)
- [ ] Rodou **todos os sujeitos** para CSP-LDA e EEGNet
- [ ] Split documentado e sem vazamento
- [ ] Tabela por sujeito (acc/kappa) + média±dp
- [ ] 3 seeds (ou justificativa clara)
- [ ] Figuras e tabelas finais exportadas
- [ ] Draft IMRaD fechado + revisão final
