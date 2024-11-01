# Requerimientos

Primeramente instale las dependencias necesarias utilizando `pip`:
```bash
pip install -r requirements.txt
```

Seguidamente corra el programa:

```bash
python main.py
```

El programa generara un archivo llamado `output.csv`, el cual contiene el siguiente formato para cada
linea:

```txt
"tweet, sentimiento, negativa, neutral, positiva, defuzz, sent_defuzz, t_fuzz, t_defuzz, t_total"
```

Ademas se imprimiran algunas metricas relacionadas al tiempo de ejecucion y a la presicion del analisis de sentimiento.

El proyecto ya cuenta con datos de test en el archivo test_data.cvs, aun asi puede descargar el dataset utilizado desde
[aqui](https://www.kaggle.com/datasets/krishbaisoya/tweets-sentiment-analysis).
