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
	scheme.line = ['#0055ff', '#ee8800', '#55aa00', '#dd0044', '#004499', '#9933bb', '#000000']
	scheme.up = 'k'
	scheme.down = 'r'
	scheme.bar = '#00bbee'
	scheme.bar_up = 'g'
	scheme.bar_down = 'r'
	scheme.text = 'k'
	scheme.grid = 'k'
	scheme.grid_alpha = 0.2
	scheme.background = 'w'
	scheme.background_text = 'w'
	scheme.background_text_alpha = 0.33
	name = name.lower()
    # Backdrop
	if name == 'sunset':
		scheme.backdrop = ['#c9e6e3', '#ffe6a9', '#ebc3bc']
		scheme.background_text = '#dd7700'
		scheme.background_text_alpha = 0.13
	elif name == 'sunrise2':
		print('here')
		scheme.backdrop = ['#ffffbb', '#ffcccc', '#ccccff']
	elif name == 'sunrise' or name == 'default':
		scheme.backdrop = ['#ccccff', '#d8ccea', '#e5cce5', '#f8cccc', '#fbdecc', '#fff0bb', '#f0f0ee']
		scheme.background_text = '#dd3377'
		scheme.background_text_alpha = 0.13
	elif name == 'plain':
		scheme.backdrop = ['#ffffff']
	elif name == 'night':
		scheme.backdrop = ['#000033', '#0a3355']
		scheme.text = ['#ffffff']
		scheme.up = '#33ff00'
		scheme.down = '#ff3300'
		scheme.bar = '#00dd00'
		scheme.bar_up = '#33ff00'
		scheme.down_up = '#ff3300'
		scheme.grid = '#0099ff'
		scheme.grid_alpha = 0.5
		scheme.text = 'w'
		scheme.background = 'k'
		scheme.background_text = '#0077ff'
		scheme.background_text_alpha = 0.20
	else:
		scheme.backdrop = ['#ffffff', '#ffffff']
	return scheme
