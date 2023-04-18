import configparser

class Config:
    configFile = configparser.ConfigParser()
    filename = ''
    aimKey = []
    triggerKey = []
    useTriggerbot = False
    triggerRangeX = 5
    triggerRangeY = 5
    screenWidth = 1920
    screenHeight = 1080
    fovWidth = 320
    fovHeight = 320
    aaRightShift = 0
    sensX = 0.8
    sensY = 0.8
    confidence = 0.6
    quitKey = 0
    reloadKey = 0
    headshot_mode = False
    headshot_offset = 0.38
    cpsDisplay = False
    visuals = True
    target_fps = 60
    trigger = False
    
    def __init__(self, filename):
        self.filename = filename
        self.configFile.read(filename)
        self.configFile = self.configFile['DEFAULT']

        self.aimKey = self.configFile.get('aimkey', '0')
        self.aimKey = [int(key) for key in self.aimKey.split(',')]

        self.triggerKey = self.configFile.get('triggerKey', '0')
        self.triggerKey = [int(key) for key in self.triggerKey.split(',')]

        self.useTriggerbot = self.configFile.getboolean('useTriggerbot', False)

        self.triggerRangeX = self.configFile.getint('triggerRangeX', 5)
        self.triggerRangeY = self.configFile.getint('triggerRangeY', 5)

        self.screenWidth = self.configFile.getint('screenWidth', 320)
        self.screenHeight = self.configFile.getint('screenHeight', 320)

        # Portion of screen to be captured (This forms a square/rectangle around the center of screen)    
        self.fovWidth = self.configFile.getint('fovWidth', 320)
        self.fovHeight = self.configFile.getint('fovHeight', 320)

        # For use in games that are 3rd person and character model interferes with the autoaim
        # EXAMPLE: Fortnite and New World
        self.aaRightShift = self.configFile.getint('aaRightShift', 0)

        # Autoaim mouse movement amplifier
        self.sensX = self.configFile.getfloat('sensX', 0.8)
        self.sensY = self.configFile.getfloat('sensY', 0.8)

        # Person Class Confidence
        self.confidence = self.configFile.getfloat('confidence', 0.6)

        # What key to press to quit and shutdown the autoaim
        self.quitKey = (self.configFile.getint('quitKey', 0))


        self.reloadKey = (self.configFile.getint('reloadKey', 0))

        # If you want to main slightly upwards towards the head
        self.headshot_mode = self.configFile.getboolean('headshot_mode', False)
        self.headshot_offset = self.configFile.getfloat('headshot_offset', 0.38)

        # Displays the Corrections per second in the terminal
        self.cpsDisplay = self.configFile.getboolean('cpsDisplay', False)

        # Set to True if you want to get the visuals
        self.visuals = self.configFile.getboolean('visuals', True)

        self.target_fps = self.configFile.getint('target_fps', 60)

        self.trigger = False
