import numpy as np
import pandas as pd
import unicodedata
import os
import io



DATAFRAME_CACHE_PATH = '../data/processed/datasets_merged.xlsx'


def get_dataframe(data_folder, force_new=False):
	if force_new:
		try: os.remove(DATAFRAME_CACHE_PATH)
		except: pass

	if os.path.exists(DATAFRAME_CACHE_PATH):
		main = pd.read_excel(DATAFRAME_CACHE_PATH)
	
	else:
		main = pd.DataFrame()
		
		for df in _scan_datasets(data_folder):
			main = pd.concat([main, df]).drop_duplicates(ignore_index=True)
		
		main.to_excel(DATAFRAME_CACHE_PATH, index=False)
	
	
	# --------------------------------------------------------------------
	# Adecuaciones al dataframe para utilizar durante tiempo de ejecución.
	# --------------------------------------------------------------------
	
	# Generar las cadenas de supermercados en base a sus nombres.
	for cad in ('REY', 'RIBA SMITH', 'SUPER 99', 'EL MACHETAZO', 'XTRA', 'EL FUERTE', 'CASA DE LA CARNE'):
		main.loc[main.supermercado.str.contains(cad), 'cadena'] = cad
	
	# para el resto de los supermercados, categorizarlos como 'Otros'
	main.loc[main.cadena.isnull(), 'cadena'] = 'Otros'
	
	# Generación de variables categóricas y sus respectivos ids únicos
	main['supermercado'] = main.supermercado.astype('category')
	main['supermercado_id'] = main.supermercado.cat.codes
	
	main['producto'] = main.producto.astype('category')
	main['producto_id'] = main.producto.cat.codes
	
	main['cadena'] = main.cadena.astype('category')
	main['cadena_id'] = main.cadena.cat.codes
	
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
	valid_sheets = [
		'cuadro x establecimiento',
		'datos abierto dic 2020',
		'todos lo sectores',
		'cuadro x sector',
		'precios sm y ms junio 2022',
		'12. cuadro prom x sector',
		'costo x sector',
	]
	
	xl = pd.read_excel(path, sheet_name=None, header=None)
	sheets = [(s, s.strip().lower()) for s in xl.keys()]
	
	if len(sheets) == 1:
		return xl[sheets[0][0]]
	
	for n in valid_sheets:
		for sh, lsh in sheets:
			if lsh == n:
				return xl[sh]
	
	return None

def _strclean(txt):
	if txt:
		txt = txt.replace("/", " / ")
		
		# quita espacios duplicados
		txt = " ".join(str(txt).split()).strip().upper()
		
		if txt:
			# quitar acentos
			txt = unicodedata.normalize("NFKD", txt)
			return txt.encode("ascii", "ignore").decode("ascii")
	
	return None

def _fix_supermercado(txt):
	txt = _strclean(txt)
	
	if txt:
		txt = txt.replace("-", " ").\
			replace(".", "").\
			replace("M / S", "MINISUPER").\
			replace("ABT", "ABARROTERIA").\
			replace("MINI MARKET", "MINIMARKET").\
			replace("MINI SUPER", "MINISUPER").\
			replace("CASA / CARNE", "CASA DE LA CARNE").\
			replace("/", " ")
		
		return " ".join(str(txt).split()).strip()
	
	return None



# No todos los datasets comienzan en la misma celda de Excel. Inclusive hay algunos que tienen texto sin utilidad
# para el estudio (fechas de generación del reporte, titulos, descripciones, etc.). Es por ello que hay que encontrar
# la celda exacta donde comienza la data.
def _get_starting_coords(df):
	h,w = df.shape
	counts = {}
	
	# Se recorre el dataset por columnas, de izquierda a derecha. A la primera celda que tenga el valor 'Producto',
	# se detiene el loop y se retorna sus coordenadas.
	for i in range(w):
		col = df.iloc[:,i]
		
		for j,v in enumerate(col.values):
			# garantizar que trabajaremos con strings.
			try: v = v.strip() or None
			except: v = None
			if v is None: continue
			
			# Si el valor es el texto 'Producto' se rompe el loop y se devuelve las coordenadas.
			if v == 'Producto':
				return j,i
			
			# Hay datasets que no le colocaron el texto 'Producto' pero aún así el dataset es válido.
			# En teoría si se encuentra un valor tipo string, es porque comienza el listado de productos.
			# Al primer string que se encuentre, se le resta uno, ya que antes de él debió existir el texto
			# 'Producto' que no s escribió.
			elif isinstance(v, str):
				return j-1,i
	
	return 0
	

# Recorta el dataset quitándole filas y columnas innecesarias o vacías tanto arriba como a la izquierda.
# Se realiza esto debido a que el xls fue creado orientado a una buena presentación visual, y no enfocado
# a la manipulación de la data.
def _fix_dataset(df):
	# Las hojas siempre comienzan con la primera columna el producto, y la segunda la medida.
	# El resto de las columnas (supermercado) son los precios de los productos (fila).
	
	# Quitar filas y columnas totalmente vacías
	df = df.replace({np.nan: None}).dropna(how='all').dropna(axis=1, how='all')
	
	# Obtener coordenadas exactas donde comienza el dataset.
	j,i = _get_starting_coords(df)
	df = df.iloc[j:, i:]
	
	# En este punto, ya el dataframe debe comenzar con la columna de productos, seguido de la medida.
	# Para obtener los nombres de los supermercados, se le hace un slice a partir de las 2 primeras columnas (producto, medida).
	# Hay columnas de supermercados que no tienen nombre, por lo que se ignoran.
	
	columnas = df.columns.tolist()
	first_row = df.iloc[0].values.tolist()
	selected_columnas = columnas[0:2]
	supermercados = []
	
	for i,v in enumerate(first_row[2:]):
		if v is None or not isinstance(v, str): continue
		supermercados.append(_fix_supermercado(v))
		selected_columnas.append(columnas[i])
	
	# Se crea un nuevo dataframe con las columnas seleccionadas y quitando la primera fila (nombres de supermercados).
	df = df[selected_columnas].iloc[1:]
	new_data = []
	
	for row in df.values:
		# Hay algunas hojas que les colocan una última fila de totales o de disclaimer. Se ignoran.
		if row[1] is None: continue
		producto, medida = row[0:2]
		
		producto = _strclean(producto)
		if producto is None: continue
		
		for i,val in enumerate(row[2:]):
			# Ignorar productos que no tienen costo
			if val is None: continue
			
			# Verificar que el costo sea realmente un número. Redondear a moneda de 2 decimales.
			try: costo = int(float(val) * 100)/100
			except: continue
			
			arr_medida = medida.split(' ')
			medida_cantidad = float(arr_medida.pop(0).replace(',', '.'))
			medida_unidad = ' '.join(arr_medida).strip()
			
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
					if df is None: continue
					
					df[['anio', 'mes']] = filename_date
					yield df



if __name__ == '__main__':
	df = get_dataframe('../data/raw', force_new=True)
	print(df)
