# TP4 SIA - Aprendizaje No Supervisado

## 👋 Introducción

Trabajo práctico para la materia de Sistemas de Inteligencia Artificial en el ITBA. Se buscó implementar distintos modelos no supervisados, los cuales fueron la red de Kohonen, la red de Oja y la red de Hopfield. En los mains de las primeras dos redes se usa un dataset "europe.csv" y en el hopfield se ingresan los patrones a guardar y el patrón a consultar.

Este fue el [Enunciado](docs/Enunciado%20TP4.pdf)

### ❗ Requisitos

- Python3 (La aplicación se probó en la versión de Python 3.11.*)
- pip3
- [pipenv](https://pypi.org/project/pipenv)

### 💻 Instalación

En caso de no tener python, descargarlo desde la [página oficial](https://www.python.org/downloads/release/python-3119/)

Utilizando pip (o pip3 en mac/linux) instalar la dependencia de **pipenv**:

```sh
pip install pipenv
```

Parado en la carpeta del proyecto ejecutar:

```sh
pipenv install
```

para instalar las dependencias necesarias en el ambiente virtual.

## 🛠️ Configuración
Para configurar el `main_kohonen.py` se usa el `configs/kohonen_config.json`, se permiten estas configuraciones:
- `k_grid_dimension`: la cantidad de neuronas k en la grilla kxk
- `use_weights_from_inputs`: true: take the data vectors as weights for each neuron in the epoch 0, false if you want random weights for each neuron
- `max_epochs`: max epochs, this is the end condition
- `initial_radius`: initial neighborhood radius
- `decrease_radius`: true if radius should decrease, false if not
- `initial_learn_rate`: initial learning rate
- `decrease_learn_rate`: true if learning rate should decrease, false if not
- `plot_type`: type of heatmap requested for output. Options are: final_entries/euclidean/variable. The first one will give the final activation counts for each neurons along with the labels, the second one will give the euclidean distance between neurons and the last one will give you the heatmap of each neuron focusing only on the variable provided
- `variable_type`: if the option of the last config is "variable", then you need to specify an specific variable for the heatmap, the options are: GDP/Life.expect/Inflation/Area/Pop.growth/Military/Unemployment

Para configurar el `main_oja.py` se usa el `configs/oja_config.json`, se permiten estas configuraciones:
- `max_epochs`: max epochs, this is the end condition
- `ini_learn_rate`: initial learning rate
- `decrease_learn_rate`: true if learning rate should decrease, false if not

Para configurar el `main_hopfield.py` se usa el `configs/hopfield_config.json`, se permiten estas configuraciones:
- `net_patterns`: letter patters to be stored in the neural net
- `max_epochs`: max epochs if no pattern is found
- `query_letter`: letter to be queried to find it's most fitting pattern
- `gauss_noice`: gaussian noice to the query letter
- `output_file`: output path and file name of the resulting gif

## 🏃 Ejecución

Para probar la aplicación, correr:
```shell
pipenv run python <main deseado>
```
Siendo \<main deseado> una de estas opciones:
- `main_kohonen.py`
- `main_oja.py` 
- `main_hopfield.py`

En el caso del primero se ploteará un gráfico de calor seleccionable en la configuración, en el caso del segundo se imprimirán los pesos final que convergieron y en el caso del tercero se guardará un GIF en la ubicación espacificada en la config con la evolución del patrón.

Para abrir el google Colab dónde se realizaron las pruebas ir del primer ejecicio [link](https://colab.research.google.com/drive/1QsD_DbTI9s8SJsIMdVjkUgS7O1IghomW?usp=sharing)
Para abrir el google Colab dónde se realizaron las pruebas ir del segundo ejecicio [link](https://colab.research.google.com/drive/1pW4ddvw9PFHjLyMTi4X5XKtugB0kmHX0?usp=sharing)