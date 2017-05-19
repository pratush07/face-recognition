audioFolder = "/home/kuliza/Documents/face-recognition/WelcomeAudio/"
key = "a98a528ffe324c37853194aee46f1a36"
urlBase = "http://api.voicerss.org/?"

def welcomeAudioPlay(subjectName):
	# text to audio code
	import urllib2,urllib
	import os.path
	savePath = audioFolder + subjectName + ".mp3"
	if not os.path.exists(savePath):
		
		text = "Hello " + subjectName +".Welcome \n.Have a nice day.";
		urlParams = {'hl': 'en-in','key':key, 'src': text, 'f': '44khz_16bit_stereo','r':-1}
		f = urllib2.urlopen(urlBase + urllib.urlencode(urlParams))
		with open(savePath, "wb") as code:
			code.write(f.read())

	# play audio in case of recognition
	import pygame
	pygame.mixer.init()
	pygame.mixer.music.load(savePath)
	pygame.mixer.music.play()
	while pygame.mixer.music.get_busy(): 
	    pygame.time.Clock().tick(10)