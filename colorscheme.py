def sunset():
	scheme = type('sunset', (), {})()
	scheme.backdrop = ['#ccccff', '#d8ccea', '#e5cce5', '#f8cccc', '#fbdecc', '#fff0bb', '#f0f0ee']
	scheme.line = ['#1155ff', '#ee8822', '#559900']
	return scheme

def plain():
	scheme = type('sunset', (), {})()
	scheme.backdrop = ['#ffff']
	scheme.line = ['#1155ff', '#ee8822', '#559900']
	return scheme

def default():
	return sunset()
