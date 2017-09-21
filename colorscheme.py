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
	scheme.line = ['#0055ff', '#ee8800', '#55aa00', '#990000']
	scheme.up = 'k'
	scheme.down = 'r'
	scheme.bar_up = 'g'
	scheme.bar_down = 'r'
	scheme.grid = '#b0b0b0'
	scheme.text = 'k'
	scheme.background = 'w'
	scheme.background_text_color = 'w'
	scheme.background_text_alpha = 0.33
	name = name.lower()
    # Backdrop
	if name == 'sunset':
		scheme.backdrop = ['#c9e6e3', '#ffe6a9', '#ebc3bc']
	elif name == 'sunrise2':
		print('here')
		scheme.backdrop = ['#ffffbb', '#ffcccc', '#ccccff']
	elif name == 'sunrise':
		scheme.backdrop = ['#ccccff', '#d8ccea', '#e5cce5', '#f8cccc', '#fbdecc', '#fff0bb', '#f0f0ee']
	elif name == 'plain':
		scheme.backdrop = ['#ffffff']
	elif name == 'night':
		scheme.backdrop = ['#000033', '#0a3355']
		scheme.text = ['#ffffff']
		scheme.up = '#33ff00'
		scheme.down = '#ff3300'
		scheme.bar_up = '#33ff00'
		scheme.down_up = '#ff3300'
		scheme.grid = '#3355cc'
		scheme.text = 'w'
		scheme.background = 'k'
		scheme.background_text_color = '#0077ff'
		scheme.background_text_alpha = 0.20
	else:
		scheme.backdrop = ['#ffffff', '#ffffff']
	return scheme
