import pandas as pd
import gspread
from google.colab import auth
from google.auth import default
import urllib.parse
import sqlite3
from gspread_dataframe import set_with_dataframe
import numpy as np
import re
import ast

def convertir_fecha(base, columna):
  """ Convierte el formato de Marca temporal proveniente de FORMS en un formato de Fecha y Hora
    Args:
        base (DataFrame): df a procesar
        columna (str): nombre de la columna tipo Marca temporal

    Returns:
       df[column]: Devuelve la columna formateada del df original.

    Raises:
        ValueError: Se encuentran valores Null en la columna original.

       """
  base[columna] = pd.to_datetime(base[columna], errors="coerce", format='%d/%m/%Y %H:%M:%S') #Fixed indentation: Removed 4 spaces


  null_values = base[columna][base[columna].isnull()]
  if not null_values.empty:
      original_values = base.loc[null_values.index, columna]  # Get original values

      raise ValueError(f"Error: Null values encountered in the datetime column after conversion. Original values: {original_values.tolist()}")

  return base[columna]


def extraer_antes_del_punto(texto):
  """ Convierte el formato de Marca temporal proveniente de FORMS en un formato de Fecha y Hora
    Args:
        base (DataFrame): df a procesar
        columna (str): nombre de la columna tipo Marca temporal

    Returns:
       df[column]: Devuelve la columna formateada del df original.

    Raises:
        ValueError: Se encuentran valores Null en la columna original.

       """
  texto = str(texto)

  match = re.search(r'(\S*)\.', texto)  # Search for non-space characters before a period
  if match:
    return match.group(1)  # Return the matched group (before the period)
  else:
    return texto  # Return the original string if no match is found


def replace_values(df, column, dict1):
    for index, row in df.iterrows():
        if row[column] in dict1:
            target_column = dict1[row[column]]  # Cargado de codigo de salto
            next_column_index = df.columns.get_loc(column) + 1
            while df.columns[next_column_index] != target_column:
                current_column = df.columns[next_column_index]
                df.loc[index, current_column] = "" #Se llena como vacio
                next_column_index += 1
                if next_column_index == target_column:
                    break  #
    return df


def extraer_headers(base): ## con "." como delimitador
  """ Extrae los codigos de pregunta en una tabla de headers del df original, incluyendo subcodigos y los separa en columnas. Los codigos finales de alojan en la columna 'Codigo'
    Args:
        base (DataFrame): df a procesar

    Returns:
      base_headers: data frame con los headers del df original separados en columnas Codigo, Pregunta y Subcodigo

  """
  base_headers=pd.DataFrame(base.columns.values)
  base_headers.columns=['Header']
  # Split the 'Header' column into two, filling with None if there's no '.'
  base_headers[['Codigo','Pregunta']] = base_headers['Header'].str.split(".",n=1,expand=True).reindex(columns=[0, 1]).fillna('')
  base_headers['Subcodigo']=base_headers['Header'].str.extract(r'\[(.*?)\.')
  base_headers.loc[base_headers["Subcodigo"].notna(), "Codigo"] = base_headers.loc[base_headers["Subcodigo"].notna(), "Subcodigo"]

  return base_headers[['Codigo','Pregunta','Subcodigo']]


def agregar_dataframe_a_hoja(sheet_id, dataframe, new_sheet_name,gc):
  """
  Agrega un DataFrame al final de una hoja de cálculo de Google Sheets existente,
  sin borrar los datos existentes.

  Args:
    sheet_id: El ID de la hoja de cálculo de Google Sheets.
    dataframe: El DataFrame de Pandas que se agregará.
    new_sheet_name: El nombre de la hoja donde se agregará el DataFrame.
  """
  from gspread_dataframe import set_with_dataframe
  try:
    sh = gc.open_by_key(sheet_id)
  except gspread.exceptions.SpreadsheetNotFound:
    sh = gc.create(new_sheet_name)
    print(f"Nueva hoja de cálculo '{new_sheet_name}' creada con ID: {sh.id}")
    sheet_id = sh.id

  # Intenta obtener la hoja, si no existe, la crea
  try:
    worksheet = sh.worksheet(new_sheet_name)
  except gspread.exceptions.WorksheetNotFound:
    worksheet = sh.add_worksheet(title=new_sheet_name, rows="1000", cols="20")
    print(f"Hoja de cálculo '{new_sheet_name}' creada.")
    # Si es nueva, incluye encabezados
    set_with_dataframe(worksheet, dataframe)
    print(f"DataFrame agregado a nueva hoja '{new_sheet_name}'.")
    return

  # Si ya existe, agrega al final sin sobrescribir
  existing_data = worksheet.get_all_values()
  start_row = len(existing_data) + 1  # siguiente fila disponible

  # Evita repetir encabezados si ya existen
  include_column_header = start_row == 1

  set_with_dataframe(worksheet, dataframe, row=start_row, include_column_header=include_column_header)
  print(f"DataFrame agregado a partir de la fila {start_row} en la hoja '{new_sheet_name}'.")


def corregir_horas(df, col_hora, col_h1):
    '''
    Corrige la columna H1 si tiene errores de AM/PM.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    col_hora (str): Nombre de la columna con la hora correcta.
    col_h1 (str): Nombre de la columna con la hora posiblemente incorrecta.
    '''
    df = df.copy()

    # Convertir a datetime en formato de 24 horas
    df[col_hora] = pd.to_datetime(df[col_hora], format='%H:%M')
    df[col_h1] = pd.to_datetime(df[col_h1], format='%H:%M')

    # Aplicar la corrección si la diferencia entre Hora y H1 es mayor a 6 horas
    df[f'{col_h1}_corregido'] = df.apply(lambda row: row[col_h1] + pd.Timedelta(hours=12)
                                         if (row[col_hora] - row[col_h1]).total_seconds() > 6 * 3600
                                         else row[col_h1], axis=1)

    # Convertir de nuevo a formato HH:MM
    df[f'{col_h1}_corregido'] = df[f'{col_h1}_corregido'].dt.strftime('%H:%M')

    return df


def coalesce_seccion(df, cols, new_col):
    """
    Crea una nueva columna tomando el primer valor no nulo de una lista de columnas.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    cols (list): Lista de columnas a evaluar en orden.
    new_col (str): Nombre de la nueva columna.

    Retorna:
    pd.DataFrame: DataFrame con la columna 'new_col' agregada.
    """
    df[new_col] = df[cols].apply(lambda x: next((v for v in x if pd.notna(v) and str(v).strip() != ''), None), axis=1)
    return df


def reordenar_menciones(df, bloque,ns):
    """
    Ordena las mencione segun su codigo de limpieza dejando los casos de 99 para la ultima mencion.

    Parámetros:
    df: DataFrame con los datos.
    bloque (df): dataframe con los nombre de las columnas a ordenar ya limpiadas.
    ns: codigo de NS/NC (9 o 99 segun la pregunta)

    Retorna:
    pd.DataFrame: DataFrame con el bloque de menciones ordenado.
    """
    ns=ns.astype(str)

    def reordenar_fila(row):
        # Convertimos todo a string y eliminamos espacios en blanco
        row = row.astype(str).str.strip()

        # Extraer valores distintos de "99" manteniendo su orden original
        valores = [val for val in row if val != ns]

        # Contar cuántos "99" hay
        cantidad_99 = len(row) - len(valores)

        # Rellenar con "99" al final
        return valores + [ns] * cantidad_99

    # Aplicamos la función fila por fila en las columnas del bloque
    df[bloque] = df[bloque].apply(reordenar_fila, axis=1, result_type="expand")
    return df


def autenticar_drive():
    from google.colab import drive
    drive.mount('/content/drive')


def conectar_sqlite_memoria():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    return conn, cursor


def autenticar_google():
    auth.authenticate_user()
    creds, _ = default()
    gc = gspread.authorize(creds)
    return gc


def validar_ids_unicos(df, columna="ID_registro"):
    """
    Verifica que los valores de la columna especificada sean únicos.

    Args:
        df (pd.DataFrame): DataFrame con la columna a validar.
        columna (str): Nombre de la columna que contiene los IDs.

    Raises:
        ValueError: Si se encuentran valores duplicados.
    """
    if not df[columna].is_unique:
        duplicados = df[df.duplicated(columna, keep=False)]
        raise ValueError(
            f"⚠️ ADVERTENCIA: Se encontraron IDs duplicados en la columna '{columna}'.\n\n"
            f"Filas duplicadas:\n{duplicados.to_string(index=False)}"
        )
    else:
        print(f"✅ Todos los valores en la columna '{columna}' son únicos.")


