def sunset():
	return colorscheme('sunset')

def sunrise():
	return colorscheme('sunrise')

def plain():
	return colorscheme('plain')

def default():
	return colorscheme('sunset')

def colorscheme(name = 'sunset'):
	# Initialize an empty object
	scheme = type('colorscheme', (), {})()
	scheme.line = ['#0055ff', '#ee8800', '#559900', '#990000']
	scheme.up = 'k'
	scheme.down = 'r'
	scheme.bar_up = 'g'
	scheme.bar_down = 'r'
	scheme.grid = '#b0b0b0'
	scheme.text = 'k'
	scheme.background = 'w'
    # Backdrop
	if name is 'sunset':
		scheme.backdrop = ['#c9e6e3', '#ffe6a9', '#ebc3bc']
	elif name is 'sunrise2':
		scheme.backdrop = ['#ffffbb', '#ffcccc', '#ccccff']
	elif name is 'sunrise':
		scheme.backdrop = ['#ccccff', '#d8ccea', '#e5cce5', '#f8cccc', '#fbdecc', '#fff0bb', '#f0f0ee']
	elif name is 'plain':
		scheme.backdrop = ['#ffffff']
	elif name is 'night':
		scheme.backdrop = ['#000033', '#003366']
		scheme.text = ['#ffffff']
		scheme.up = '#33ff00'
		scheme.down = '#ff3300'
		scheme.bar_up = '#33ff00'
		scheme.down_up = '#ff3300'
		scheme.grid = '#0066dd'
		scheme.text = 'w'
		scheme.background = 'k'
	return scheme
