import numpy as np
import pandas as pd
import os
import io



DATAFRAME_CACHE_PATH = '../data/processed/datasets_merged.xlsx'


def get_dataframe(data_folder):
	if os.path.exists(DATAFRAME_CACHE_PATH):
		return pd.read_excel(DATAFRAME_CACHE_PATH)
	
	main = pd.DataFrame()
	
	for df in _scan_datasets(data_folder):
		main = pd.concat([main, df]).drop_duplicates(ignore_index=True)
	
	main.to_excel(DATAFRAME_CACHE_PATH, index=False)
	return main

# *****************************************************************************************************************

# Obtiene el año y mes (número) a partir de la ruta del archivo xls.
def _get_filename_date(path):
	try: anio = int(os.path.basename(os.path.dirname(path)))
	except: return None

	filename = os.path.basename(path).lower()
	meses = [
		('enero', 'ene'),
		('febrero', 'feb'),
		('marzo', 'mar'),
		('abril', 'abr'),
		('mayo', 'may'),
		('junio', 'jun'),
		('julio', 'jul'),
		('agosto', 'ago'),
		('septiembre', 'sep'),
		('octubre', 'oct'),
		('noviembre', 'nov'),
		('diciembre', 'dic'),
	]
	
	for i,(m0, m1) in enumerate(meses, 1):
		if filename.find(m0) != -1 or filename.find(m1) != -1:
			return anio, i
	
	return None

# Hay archivos xls que tienen varias hojas, por lo que hay que ubicar la correcta que tenga el dataset de interés.
def _get_valid_sheet(path):
	valid_sheets = ('cuadro x establecimiento', 'todos lo sectores', 'cuadro x sector', 'precios sm y ms junio 2022', 'costo x sector', '12. cuadro prom x sector')
	
	xl = pd.read_excel(path, sheet_name=None, header=None)
	sheets = [(s, s.strip().lower()) for s in xl.keys()]
	
	if len(sheets) == 1:
		return xl[sheets[0][0]]
	
	for n in valid_sheets:
		for sh, lsh in sheets:
			if lsh == n:
				return xl[sh]
	
	return None

# Recorta el dataset quitándole filas y columnas innecesarias o vacías tanto arriba como a la izquierda.
# Se realiza esto debido a que el xls fue creado orientado a una buena presentación visual, y no enfocado
# a la manipulación de la data.
def _fix_dataset(df):
	# Las hojas siempre comienzan con la primera columna el producto, y la segunda la medida.
	# El resto de las columnas (supermercado) son los precios de los productos (fila).
	
	df = df.replace({np.nan: None}).dropna(how='all').dropna(axis=1, how='all')
	
	# Recorre las filas de arriba a abajo quitando las que sean todas vacias,
	# o tengan más de 5 celdas vacías.
	while True:
		counts = df.iloc[0].value_counts(dropna=False)
		num_empty = counts[counts.index.isnull()].values
		
		if num_empty.size:
			if num_empty[0] > 5: df = df.iloc[1:]
			else: break
		
		else:
			break
	
	# Escanear cada fila hasta encontrar la celda con el valor 'Producto', y usar su índice para recortar
	# las columnas a la izquierda. Si no lo encuentra, por default el índice es cero (no recorta nada).
	idx = (df.iloc[0].values == 'Producto').argmax()
	
	old_cols = df.columns.values.tolist()
	selected_cols = old_cols[idx:]
	
	# En este punto, la primera fila debe ser la lista de supermercados. Ignrar las 2 primeras celdas (producto, medida).
	supermercados = df[selected_cols].iloc[0].tolist()[2:]
	
	# Se crea un nuevo dataframe con las columnas seleccionadas y quitando la primera fila (nombres de supermercados).
	df = df[selected_cols].iloc[1:]
	new_data = []
	
	for row in df.values:
		# Hay algunas hojas que les colocan una última fila de totales o de disclaimer. Se ignoran.
		if row[1] is None: continue
		producto, medida = row[0:2]
		
		for i,val in enumerate(row[2:]):
			# Ignorar productos que no tienen costo
			if val is None: continue
			
			# Verificar que el costo sea realmente un número. Redondear a moneda de 2 decimales.
			try: costo = int(float(val) * 100)/100
			except: continue
			
			arr_medida = medida.split(" ")
			medida_cantidad = float(arr_medida.pop(0).replace(',', '.'))
			medida_unidad = " ".join(arr_medida).strip()
			
			new_data.append({
				'producto': producto.replace('*', '').strip(),
				'medida_cantidad': medida_cantidad,
				'medida_unidad': medida_unidad,
				'supermercado': supermercados[i],
				'costo': costo,
			})
	
	return pd.DataFrame(new_data)

def _scan_datasets(scan_dir):
	for root, dirs, files in os.walk(scan_dir):
		for fi in files:
			path = os.path.join(root, fi)
			filename_date = _get_filename_date(path)
			
			if filename_date:
				if path.endswith('.xls'):
					df = _get_valid_sheet(path)
					if df is None: continue
					
					df = _fix_dataset(df)
					df[['anio', 'mes']] = filename_date
					
					yield df



if __name__ == '__main__':
	df = get_dataframe('../data/raw')
	print(df)
