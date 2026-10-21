# G008 v15: análise das 244 derrotas

## Resultado avaliado

O checkpoint `step-236` venceu 217 de 512 jogos (42,38%), empatou 51 e perdeu
244 contra `g007-step-300`. Esse resultado é exploratório e abaixo do objetivo
G008 de mais de 60% contra `open-space` em pelo menos 512 partidas. O adversário
`open-space` não foi usado nesta avaliação.

## O que foi reproduzido

O relatório contém seeds, resultados e duração, mas não contém ações por turno.
Reproduzi as 244 derrotas localmente, em CPU, com os dois checkpoints ONNX e o
roster determinístico. Os 244 resultados e números de ticks coincidiram com o
relatório. Isso reconcilia resultado e duração; não prova identidade da
trajetória original, que não foi gravada. O resumo e os replays estão em
`artifacts/circuit-ai/diagnostics/g008-v15-244-losses/`.

## Evidência das derrotas

- Em 240/244 derrotas, a ação terminal matou o candidato, mas havia outra ação
  que o mantinha vivo contra a ação real escolhida pelo adversário naquele
  turno. É um contrafactual de um turno, não prova que a alternativa venceria.
- Em duas derrotas não havia movimento imediato seguro. Nas outras duas, havia
  uma saída livre, mas nenhuma alternativa sobreviveu à ação simultânea real.
- Causas terminais do candidato: 206 colisões com trilha, 27 colisões com a
  cabeça adversária e 11 saídas da arena.
- Em 227 derrotas houve ao menos uma decisão anterior localmente dominada: uma
  alternativa manteve pelo menos tantos movimentos livres e tantas células
  alcançáveis após um turno, mantendo fixa a ação do adversário. Foram 584
  ocorrências; isso indica oportunidades locais, mas não estabelece causa raiz.
- A observação canônica, o batch de treino e o cache incremental coincidiram
  em todos os turnos reproduzidos (zero divergências). Não há evidência de erro
  na codificação espacial examinada.

Um exemplo curto é a derrota no seed `1000026`, assento 1, no primeiro turno:
o candidato entrou na célula ocupada pelo adversário embora tivesse duas ações
livres; havia 143 células alcançáveis antes da jogada. Veja
`replays/1000026-seat-1.json`.

## Interpretação e próximo experimento

A evidência aponta para escolha de ação e estratégia de espaço como os alvos
mais promissores. Ela não explica por que a política escolheu cada ação; a nova
telemetria PPO foi adicionada para expor reward, entropia, KL, perdas separadas,
clip fraction, duração e resultados por adversário nos próximos runs.

O coletor já permite um primeiro teste de diversidade: política mais recente,
snapshot inicial congelado e heurísticas de treino (`cruise`, `local-mobility`,
`interceptor`). Ele ainda não carrega vários snapshots históricos. O teste
controlado pode mudar apenas a mistura de adversários, mantendo arquitetura,
checkpoint inicial, hiperparâmetros, orçamento de updates e protocolo de seeds
iguais entre braços, com novas seeds de treino. As métricas novas e avaliações
pareadas devem comparar vitória/empate/derrota contra adversários fortes; o
`open-space` fica para a avaliação final isolada. Não ampliar o orçamento se
não houver ganho reproduzível sem regressão relevante.

Os relatórios exploratórios atuais usaram o mesmo roster de validação em vários
checkpoints, portanto não demonstram generalização a seeds frescas. O split
congelado também reserva `open-space` à avaliação final. Antes de registrar o
controlado, é necessário escolher a rotação de validação sem consumir o final
antes de congelar candidato.
