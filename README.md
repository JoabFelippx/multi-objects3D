# Associação de objetos entre quatro câmeras

`object_matching.py` recebe detecções de quatro câmeras pelo broker IS, associa objetos da mesma classe e publica uma posição no plano do chão para cada grupo válido. O processamento é contínuo: a cada rodada, o script lê a mensagem mais recente de cada câmera e publica um `ObjectAnnotations` em `SkeletonDetector.3D.Annotations`.

## Preparação

Use Python 3.12 e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

O script precisa de `data/calibration/calib_rt1.npz` até `calib_rt4.npz`. Cada arquivo deve conter `cam_id`, `K`, `nK`, `dist`, `rt`, `rt_inv`, `P_original`, `P_undistorted`, `height`, `width` e `roi`. `rt` transforma coordenadas do mundo em coordenadas da câmera; `rt_inv` faz a transformação inversa. Os pontos publicados usam as unidades dessas calibrações, normalmente metros.

Para usar outro diretório de calibração ou outro broker, defina as variáveis de ambiente antes de executar:

```bash
CALIB_PATH_RAW=/caminho/para/calibration \
BROKER_URI=amqp://guest:guest@localhost:5672 \
python object_matching.py
```

Sem essas variáveis, o diretório de calibração é `data/calibration` e o broker é `amqp://guest:guest@10.10.50.176:30000`.

## Entrada e filtro de classes

O script assina `ObjectDetector.1.Detection` até `ObjectDetector.4.Detection`. Cada mensagem deve conter um `ObjectAnnotations` com `objects`; para cada objeto, são usados `label` e `region.vertices` da caixa delimitadora.

Edite a lista `class_target` no início de `object_matching.py` para escolher as classes. O nome precisa ser igual ao `detection.label` publicado pelo detector:

```python
class_target = ["chair", "person"]
# Exemplo para outra classe: ["sports ball"]
```

Classes diferentes nunca são associadas no mesmo grupo. O filtro inicial seleciona apenas as classes presentes nessa lista.

## Como funciona

1. Corrige a distorção dos vértices da caixa com `K`, `dist` e `nK`.
2. Projeta o centro inferior da caixa no chão (`Z = 0`) e transforma o eixo vertical da imagem em uma reta no plano XY.
3. Calcula a interseção das retas de pares da mesma classe vistos por câmeras diferentes. Descarta retas com ângulo menor que 20° e interseções a mais de 0,50 unidade de qualquer um dos dois pontos projetados no chão.
4. Agrupa as interseções com DBSCAN (`eps=0.25`, `min_samples=2`). Quando um grupo contém mais de uma detecção da mesma câmera, mantém a detecção com mais pares válidos e, em caso de empate, a mais próxima do centro das interseções.
5. Estima a posição XY do grupo por mínimos quadrados e a altura a partir do topo das caixas. Descarta grupos sem posição ou sem altura válida; o limite de erro de reprojeção para a altura é 6,65 pixels.

Para cada grupo aceito, publica um objeto com `label` igual à classe e um `keypoint.position`: `x` e `y` são a posição estimada no chão, e `z` é fixado em `0.01`. A altura estimada é mostrada no terminal e usada para validar o grupo, mas não é publicada como `z`. Os objetos recebem IDs distintos dentro da mensagem publicada.

## Experimento para o README

O script `capture_readme_figure.py` recebe imagens de `CameraGateway.<id>.Frame` e detecções de `ObjectDetector.<id>.Detection` para as quatro câmeras. Ele usa `Image.sequence` e `ObjectAnnotations.frame_id` para desenhar cada caixa sobre a imagem que originou aquela detecção. O resultado é uma figura com as quatro imagens, as posições no mundo e os erros em XY e altura.

| Objeto | Classe | Posição real (m) | Altura real (m) |
| --- | --- | --- | ---: |
| C1 | `chair` | `(0, 0, 0)` | 0,970 |
| C2 | `chair` | `(1, 1, 0)` | 0,920 |
| C3 | `chair` | `(-1, 1, 0)` | 0,920 |
| C4 | `chair` | `(-1, -1, 0)` | 0,815 |
| P1 | `person` | `(1, -1, 0)` | 1,810 |

Depois de posicionar os objetos e a pessoa, execute:

```bash
python capture_readme_figure.py
```

O arquivo é salvo em `docs/readme_experiment.png`. Os grupos associados às posições de referência têm a mesma cor nas quatro imagens e nos gráficos; o texto da caixa mostra o grupo e a classe. A comparação associa grupos e posições reais pela menor distância XY, separadamente por classe, com associação única e distância máxima de 0,75 m. O erro XY é a distância entre as posições real e estimada; o erro de altura mostrado é `altura estimada − altura real`. Objetos reais sem grupo são identificados como ausentes. O gráfico XY mantém a área fixa de 4,5 m × 3,5 m, centrada em `(0, 0)`.

![Detecções das quatro câmeras e erros de posição e altura](docs/readme_experiment.png)

## Limitações

- O fluxo contínuo de `object_matching.py` lê a mensagem mais recente de cada câmera, sem sincronização por timestamp. O script da figura associa imagem e detecção por número de frame em cada câmera, mas não sincroniza as quatro câmeras entre si.
- O centro inferior da caixa é uma aproximação do contato com o chão. Uma caixa imprecisa ou um objeto elevado pode deslocar a posição estimada.
- Um grupo precisa ter interseções suficientes para o DBSCAN e uma estimativa de altura válida. Quando nenhum grupo atende a esses critérios, a mensagem publicada contém uma lista de objetos vazia.
- Os IDs são atribuídos novamente em cada mensagem; não há rastreamento de objetos ao longo do tempo.
